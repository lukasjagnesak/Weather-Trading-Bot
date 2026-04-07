"""Multi-source weather verification: cross-validate forecasts from multiple providers.

Fetches deterministic forecasts from free weather APIs to verify and strengthen
our ensemble-based probability model. When multiple independent sources agree
on the temperature, we have much higher confidence in the correct outcome.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import httpx

from .config import CITIES

logger = logging.getLogger(__name__)

# In-memory verification cache (deterministic forecasts also update slowly)
_verification_cache: dict[str, tuple[datetime, dict[tuple[str, date], "VerifiedForecast"]]] = {}
_VERIFICATION_CACHE_TTL_MINUTES = 360  # 6 hours — same as ensemble cache


@dataclass
class ForecastSource:
    """A single deterministic forecast from an external provider."""

    provider: str  # "open_meteo", "openweathermap", "weatherapi", "visual_crossing"
    city: str
    target_date: date
    high_temp: float  # daily high in the city's native unit (C or F)
    low_temp: float | None = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerifiedForecast:
    """Aggregated forecast from multiple sources with agreement metrics."""

    city: str
    target_date: date
    sources: list[ForecastSource]
    unit: str  # "celsius" or "fahrenheit"

    @property
    def mean_high(self) -> float:
        return sum(s.high_temp for s in self.sources) / len(self.sources)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def spread(self) -> float:
        """Standard deviation across sources — lower = more agreement."""
        if len(self.sources) < 2:
            return 5.0  # high uncertainty with single source
        mean = self.mean_high
        return (sum((s.high_temp - mean) ** 2 for s in self.sources) / len(self.sources)) ** 0.5

    @property
    def agreement_score(self) -> float:
        """0-1 score: how much sources agree. 1.0 = perfect agreement."""
        if len(self.sources) < 2:
            return 0.3
        spread = self.spread
        # Within 0.5 degree = perfect, degrades as spread increases
        if spread <= 0.5:
            return 1.0
        elif spread <= 1.0:
            return 0.9
        elif spread <= 1.5:
            return 0.75
        elif spread <= 2.0:
            return 0.6
        elif spread <= 3.0:
            return 0.4
        else:
            return 0.2

    @property
    def peak_bucket_center(self) -> float:
        """Best estimate of the temperature — weighted mean across sources."""
        return self.mean_high


# ─── Open-Meteo deterministic (free, no key) ─────────────────────────────────

async def _fetch_open_meteo_deterministic(
    client: httpx.AsyncClient,
    city_key: str,
    target_date: date,
) -> ForecastSource | None:
    """Fetch deterministic (non-ensemble) forecast from Open-Meteo."""
    city = CITIES.get(city_key)
    if not city:
        return None

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"

    try:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "daily": "temperature_2m_max,temperature_2m_min",
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "temperature_unit": temp_unit,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Open-Meteo deterministic failed for %s: %s", city_key, e)
        return None

    daily = data.get("daily", {})
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])

    if not highs or highs[0] is None:
        return None

    return ForecastSource(
        provider="open_meteo",
        city=city_key,
        target_date=target_date,
        high_temp=float(highs[0]),
        low_temp=float(lows[0]) if lows and lows[0] is not None else None,
    )


# ─── Open-Meteo previous day models (GFS, ECMWF, ICON deterministic) ────────

async def _fetch_open_meteo_models(
    client: httpx.AsyncClient,
    city_key: str,
    target_date: date,
) -> list[ForecastSource]:
    """Fetch deterministic forecasts from multiple NWP models via Open-Meteo."""
    city = CITIES.get(city_key)
    if not city:
        return []

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
    sources = []

    for model in models:
        try:
            await asyncio.sleep(1.0)  # rate-limit: ~6 req/s to avoid 429
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "daily": "temperature_2m_max",
                    "models": model,
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                    "temperature_unit": temp_unit,
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Open-Meteo %s failed for %s: %s", model, city_key, e)
            continue

        daily = data.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        if highs and highs[0] is not None:
            sources.append(ForecastSource(
                provider=f"open_meteo_{model}",
                city=city_key,
                target_date=target_date,
                high_temp=float(highs[0]),
            ))

    return sources


# ─── Open-Meteo current weather (for same-day verification) ──────────────────

async def _fetch_current_observation(
    client: httpx.AsyncClient,
    city_key: str,
) -> float | None:
    """Fetch current observed temperature — useful for same-day markets."""
    city = CITIES.get(city_key)
    if not city:
        return None

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"

    try:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "current": "temperature_2m",
                "temperature_unit": temp_unit,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Current obs failed for %s: %s", city_key, e)
        return None

    current = data.get("current", {})
    temp = current.get("temperature_2m")
    return float(temp) if temp is not None else None


# ─── Public API ──────────────────────────────────────────────────────────────

async def verify_forecast(
    city_key: str,
    target_date: date,
    client: httpx.AsyncClient | None = None,
) -> VerifiedForecast:
    """Fetch and cross-validate forecasts from multiple sources.

    Returns a VerifiedForecast with agreement metrics.
    """
    city = CITIES.get(city_key)
    if not city:
        return VerifiedForecast(city=city_key, target_date=target_date, sources=[], unit="celsius")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20.0)

    sources: list[ForecastSource] = []

    try:
        # Source 1: Open-Meteo deterministic (best-guess)
        await asyncio.sleep(1.0)
        det = await _fetch_open_meteo_deterministic(client, city_key, target_date)
        if det:
            sources.append(det)

        # Source 2-4: Individual NWP models (GFS, ECMWF, ICON deterministic)
        model_sources = await _fetch_open_meteo_models(client, city_key, target_date)
        sources.extend(model_sources)

        # Source 5: Current observation (same-day only — acts as floor for high temp)
        today = date.today()
        if target_date == today:
            await asyncio.sleep(1.0)
            current_temp = await _fetch_current_observation(client, city_key)
            if current_temp is not None:
                # Current temp is a lower bound for today's high
                # Add as a source with the current temp as high estimate
                sources.append(ForecastSource(
                    provider="current_observation",
                    city=city_key,
                    target_date=target_date,
                    high_temp=current_temp,
                ))

    finally:
        if own_client:
            await client.aclose()

    if sources:
        logger.info(
            "Verified forecast for %s on %s: %d sources, mean=%.1f, spread=%.1f, agreement=%.0f%%",
            city_key, target_date, len(sources),
            sum(s.high_temp for s in sources) / len(sources),
            (sum((s.high_temp - sum(s2.high_temp for s2 in sources) / len(sources)) ** 2
                 for s in sources) / len(sources)) ** 0.5 if len(sources) > 1 else 0.0,
            VerifiedForecast(city=city_key, target_date=target_date, sources=sources,
                             unit=city.unit).agreement_score * 100,
        )

    return VerifiedForecast(
        city=city_key,
        target_date=target_date,
        sources=sources,
        unit=city.unit,
    )


async def verify_all_cities(
    city_keys: list[str],
    target_dates: list[date],
) -> dict[tuple[str, date], VerifiedForecast]:
    """Fetch verified forecasts for multiple cities and dates."""
    # Check in-memory cache
    cache_key = (
        ",".join(sorted(city_keys))
        + "|" + ",".join(d.isoformat() for d in sorted(target_dates))
    )
    if cache_key in _verification_cache:
        cached_time, cached_results = _verification_cache[cache_key]
        age_minutes = (datetime.utcnow() - cached_time).total_seconds() / 60
        if age_minutes < _VERIFICATION_CACHE_TTL_MINUTES:
            logger.info(
                "Using cached verification forecasts (%.0f min old, TTL=%d min)",
                age_minutes, _VERIFICATION_CACHE_TTL_MINUTES,
            )
            return cached_results

    results: dict[tuple[str, date], VerifiedForecast] = {}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            for city_key in city_keys:
                for target_date in target_dates:
                    try:
                        vf = await verify_forecast(city_key, target_date, client=client)
                        if vf.sources:
                            results[(city_key, target_date)] = vf
                    except Exception as e:
                        logger.debug("Verification failed for %s/%s: %s", city_key, target_date, e)
    except Exception as e:
        logger.warning("Verification batch failed: %s", e)

    if results:
        _verification_cache[cache_key] = (datetime.utcnow(), results)

    logger.info("Verified forecasts for %d city-date combinations", len(results))
    return results


async def fetch_current_observed_high(
    city_key: str,
) -> float | None:
    """Fetch today's observed maximum temperature so far (real-time).

    Uses multiple sources in priority order:
    1. Weather Underground (actual resolution source for Polymarket)
    2. Open-Meteo hourly forecast — filter to hours already elapsed

    Returns the observed high in the city's native unit, or None.
    """
    city = CITIES.get(city_key)
    if not city:
        return None

    today = date.today()

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Source 1: Weather Underground — the ACTUAL resolution source
        try:
            wu_high = await _fetch_wunderground_high(city_key, today, client)
            if wu_high is not None:
                logger.info(
                    "Observed high for %s from WU: %.1f°%s",
                    city_key, wu_high, "F" if city.unit == "fahrenheit" else "C",
                )
                return wu_high
        except Exception as e:
            logger.debug("WU current high failed for %s: %s", city_key, e)

        # Source 2: Open-Meteo hourly forecast — take max of past hours
        try:
            temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": city.latitude,
                    "longitude": city.longitude,
                    "hourly": "temperature_2m",
                    "start_date": today.isoformat(),
                    "end_date": today.isoformat(),
                    "temperature_unit": temp_unit,
                    "timezone": city.timezone,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])

            if times and temps:
                from zoneinfo import ZoneInfo
                local_now = datetime.now(ZoneInfo(city.timezone))
                current_hour_str = local_now.strftime("%Y-%m-%dT%H:00")

                observed_temps = []
                for t, temp in zip(times, temps):
                    if temp is not None and t <= current_hour_str:
                        observed_temps.append(temp)

                if observed_temps:
                    high = max(observed_temps)
                    logger.info(
                        "Observed high for %s from Open-Meteo: %.1f°%s (up to %s local)",
                        city_key, high, "F" if city.unit == "fahrenheit" else "C",
                        local_now.strftime("%H:%M"),
                    )
                    return high
        except Exception as e:
            logger.debug("Open-Meteo current high failed for %s: %s", city_key, e)

    return None


async def _fetch_wunderground_high(
    city_key: str,
    target_date: date,
    client: httpx.AsyncClient,
) -> float | None:
    """Scrape the daily high temperature from Weather Underground.

    This is the **actual resolution source** for Polymarket temperature
    markets, so it should always be preferred over model-reanalysis data
    (Open-Meteo archive) when available.
    """
    import re

    city = CITIES.get(city_key)
    if not city or not city.wunderground_url:
        return None

    url = f"{city.wunderground_url}/date/{target_date.isoformat()}"
    try:
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; WeatherBot/1.0)"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("WU fetch failed for %s/%s: %s", city_key, target_date, e)
        return None

    # WU embeds daily summary data in a JSON blob inside a <script> tag.
    # Look for the "observations" or summary table with Max temperature.
    # The page renders temps in both °F and °C; we extract based on city unit.

    # Strategy 1: Parse the lib-state JSON that WU injects into the page.
    # It typically contains a "observations" key with "tempHigh" or similar.
    json_match = re.search(
        r'"temperature":\s*\{[^}]*"max":\s*(-?\d+(?:\.\d+)?)', html
    )
    if json_match:
        temp_c = float(json_match.group(1))
        if city.unit == "fahrenheit":
            return round(temp_c * 9 / 5 + 32, 1)
        return temp_c

    # Strategy 2: Look for the "Max" row in the summary table.
    # WU shows "Max Temperature" with the value in a <span> with class
    # "wu-value wu-value-to".  The page default unit depends on locale;
    # we look for the Celsius value (<span class="wu-value-to">).
    max_match = re.search(
        r'Max</span>.*?<span class="wu-value wu-value-to">\s*(-?\d+(?:\.\d+)?)',
        html,
        re.DOTALL,
    )
    if max_match:
        temp_c = float(max_match.group(1))
        if city.unit == "fahrenheit":
            return round(temp_c * 9 / 5 + 32, 1)
        return temp_c

    # Strategy 3: Grab any "high" temperature from the JSON-LD or inline data.
    high_match = re.search(
        r'"high":\s*"?(-?\d+(?:\.\d+)?)"?', html
    )
    if high_match:
        val = float(high_match.group(1))
        # WU JSON typically stores Celsius
        if city.unit == "fahrenheit":
            return round(val * 9 / 5 + 32, 1)
        return val

    logger.debug("WU: could not parse high temp from HTML for %s/%s", city_key, target_date)
    return None


async def _fetch_open_meteo_observed(
    city_key: str,
    target_date: date,
    client: httpx.AsyncClient,
) -> float | None:
    """Fetch observed daily high from Open-Meteo Historical Archive (fallback)."""
    city = CITIES.get(city_key)
    if not city:
        return None

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    try:
        resp = await client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "daily": "temperature_2m_max",
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "temperature_unit": temp_unit,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Open-Meteo archive failed for %s/%s: %s", city_key, target_date, e)
        return None

    daily = data.get("daily", {})
    highs = daily.get("temperature_2m_max", [])
    if highs and highs[0] is not None:
        return float(highs[0])
    return None


async def fetch_observed_temperature(
    city_key: str,
    target_date: date,
) -> float | None:
    """Fetch the actual observed daily high temperature for a past date.

    Tries Weather Underground first (the actual Polymarket resolution
    source), then falls back to Open-Meteo historical archive data.
    Returns the daily max temperature in the city's native unit, or None.
    """
    city = CITIES.get(city_key)
    if not city:
        logger.warning("Unknown city key: %s", city_key)
        return None

    async with httpx.AsyncClient(timeout=20.0) as client:
        # Primary: Weather Underground (Polymarket resolution source)
        temp = await _fetch_wunderground_high(city_key, target_date, client)
        if temp is not None:
            logger.info(
                "WU observed high for %s/%s: %.1f°%s",
                city_key, target_date, temp,
                "F" if city.unit == "fahrenheit" else "C",
            )
            return temp

        # Fallback: Open-Meteo archive (model reanalysis, may differ by ±1°)
        temp = await _fetch_open_meteo_observed(city_key, target_date, client)
        if temp is not None:
            logger.info(
                "Open-Meteo observed high for %s/%s: %.1f°%s (WU unavailable)",
                city_key, target_date, temp,
                "F" if city.unit == "fahrenheit" else "C",
            )
            return temp

    logger.warning("Could not fetch observed temp for %s/%s from any source",
                   city_key, target_date)
    return None
