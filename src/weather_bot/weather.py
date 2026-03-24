"""Fetch ensemble weather forecasts from Open-Meteo API."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from .config import CITIES, CityConfig
from .models import EnsembleForecast

logger = logging.getLogger(__name__)

ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Disk-based cache: survives restarts, ensemble models update every 6-12h
_CACHE_DIR = Path.home() / ".weather_bot"
_CACHE_FILE = _CACHE_DIR / "forecast_cache.json"
CACHE_TTL_MINUTES = 360  # 6 hours — ensemble models update every 6-12h
CACHE_STALE_TTL_MINUTES = 720  # 12 hours — fallback when rate-limited


def _save_cache(
    cache_key: str,
    results: dict[tuple[str, date], list[EnsembleForecast]],
) -> None:
    """Persist forecast results to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Load existing cache entries
        cache_data = {}
        if _CACHE_FILE.exists():
            cache_data = json.loads(_CACHE_FILE.read_text())

        # Serialize forecasts
        serialized = {}
        for (city, d), forecasts in results.items():
            key = f"{city}|{d.isoformat()}"
            serialized[key] = [
                {
                    "city": f.city,
                    "target_date": f.target_date.isoformat(),
                    "model_name": f.model_name,
                    "members": f.members,
                    "unit": f.unit,
                    "fetched_at": f.fetched_at.isoformat(),
                }
                for f in forecasts
            ]

        cache_data[cache_key] = {
            "cached_at": datetime.utcnow().isoformat(),
            "forecasts": serialized,
        }
        _CACHE_FILE.write_text(json.dumps(cache_data))
    except Exception as e:
        logger.debug("Failed to save forecast cache: %s", e)


def _load_cache(
    cache_key: str, max_age_minutes: float,
) -> dict[tuple[str, date], list[EnsembleForecast]] | None:
    """Load forecast results from disk if fresh enough."""
    try:
        if not _CACHE_FILE.exists():
            return None
        cache_data = json.loads(_CACHE_FILE.read_text())
        entry = cache_data.get(cache_key)
        if not entry:
            return None

        cached_at = datetime.fromisoformat(entry["cached_at"])
        age_min = (datetime.utcnow() - cached_at).total_seconds() / 60
        if age_min > max_age_minutes:
            return None

        results: dict[tuple[str, date], list[EnsembleForecast]] = {}
        for combo_key, forecast_list in entry["forecasts"].items():
            city, date_str = combo_key.split("|", 1)
            d = date.fromisoformat(date_str)
            results[(city, d)] = [
                EnsembleForecast(
                    city=f["city"],
                    target_date=date.fromisoformat(f["target_date"]),
                    model_name=f["model_name"],
                    members=f["members"],
                    unit=f["unit"],
                    fetched_at=datetime.fromisoformat(f["fetched_at"]),
                )
                for f in forecast_list
            ]

        logger.info("Loaded cached forecasts (%.0f min old, max_age=%d min)", age_min, max_age_minutes)
        return results
    except Exception as e:
        logger.debug("Failed to load forecast cache: %s", e)
        return None


# Ensemble model configs: (model_name, num_members)
ENSEMBLE_MODELS = {
    "gfs_seamless": 31,
    "ecmwf_ifs025": 51,
    "icon_seamless": 40,
}



async def fetch_ensemble_forecast(
    city_key: str,
    target_date: date,
    models: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[EnsembleForecast]:
    """Fetch ensemble forecasts for a city and date from Open-Meteo.

    Returns one EnsembleForecast per model.
    """
    city = CITIES[city_key]
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025"]

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)

    forecasts = []
    try:
        for model in models:
            forecast = await _fetch_single_model(client, city, city_key, target_date, model)
            if forecast:
                forecasts.append(forecast)
    finally:
        if own_client:
            await client.aclose()

    return forecasts


async def _fetch_single_model(
    client: httpx.AsyncClient,
    city: CityConfig,
    city_key: str,
    target_date: date,
    model: str,
) -> EnsembleForecast | None:
    """Fetch a single ensemble model forecast."""
    results = await _fetch_model_batch(client, city, city_key, [target_date], model)
    return results.get(target_date)


async def _fetch_model_batch(
    client: httpx.AsyncClient,
    city: CityConfig,
    city_key: str,
    target_dates: list[date],
    model: str,
) -> dict[date, EnsembleForecast]:
    """Fetch a single model forecast for multiple dates in one request."""
    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    sorted_dates = sorted(target_dates)

    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "daily": "temperature_2m_max",
        "models": model,
        "start_date": sorted_dates[0].isoformat(),
        "end_date": sorted_dates[-1].isoformat(),
        "temperature_unit": temp_unit,
        "timezone": "auto",
    }

    try:
        await asyncio.sleep(1.0)  # brief courtesy delay
        resp = await client.get(ENSEMBLE_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limited on %s/%s — will use cache", model, city_key)
        else:
            logger.error("Failed to fetch %s forecast for %s: %s", model, city_key, e)
        return {}
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Failed to fetch %s forecast for %s: %s", model, city_key, e)
        return {}

    daily = data.get("daily", {})
    if not daily:
        logger.warning("No daily data for %s/%s", model, city_key)
        return {}

    # Build date index from API response
    api_dates = daily.get("time", [])
    date_to_idx = {}
    for i, d_str in enumerate(api_dates):
        try:
            date_to_idx[date.fromisoformat(d_str)] = i
        except ValueError:
            continue

    results: dict[date, EnsembleForecast] = {}
    for target_date in target_dates:
        idx = date_to_idx.get(target_date)
        if idx is None:
            continue

        members = []
        for key, values in daily.items():
            if key.startswith("temperature_2m_max_member") and values and len(values) > idx and values[idx] is not None:
                members.append(float(values[idx]))

        if not members:
            logger.warning("No ensemble members for %s/%s/%s", model, city_key, target_date)
            continue

        logger.info(
            "Fetched %s forecast for %s on %s: %d members, mean=%.1f, spread=%.1f",
            model, city_key, target_date, len(members),
            np.mean(members), np.std(members),
        )

        results[target_date] = EnsembleForecast(
            city=city_key,
            target_date=target_date,
            model_name=model,
            members=members,
            unit=city.unit,
        )

    return results


async def _fetch_multi_city_model(
    client: httpx.AsyncClient,
    city_keys: list[str],
    target_dates: list[date],
    model: str,
) -> dict[tuple[str, date], EnsembleForecast]:
    """Fetch one model for ALL cities+dates in a single API request.

    Open-Meteo supports comma-separated latitude/longitude for
    multi-location queries, returning an array of results.
    Cities are grouped by temperature unit (F vs C).
    """
    if not city_keys:
        return {}

    cities = [(ck, CITIES[ck]) for ck in city_keys]
    sorted_dates = sorted(target_dates)

    # All cities in this batch must share the same unit
    temp_unit = "fahrenheit" if cities[0][1].unit == "fahrenheit" else "celsius"

    params = {
        "latitude": ",".join(str(c.latitude) for _, c in cities),
        "longitude": ",".join(str(c.longitude) for _, c in cities),
        "daily": "temperature_2m_max",
        "models": model,
        "start_date": sorted_dates[0].isoformat(),
        "end_date": sorted_dates[-1].isoformat(),
        "temperature_unit": temp_unit,
        "timezone": "auto",
    }

    try:
        await asyncio.sleep(1.0)  # brief courtesy delay
        resp = await client.get(ENSEMBLE_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limited on %s — will use cache", model)
        else:
            logger.error("Failed to fetch %s multi-city forecast: %s", model, e)
        return {}
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Failed to fetch %s multi-city forecast: %s", model, e)
        return {}

    # For a single city, API returns {daily: ...}
    # For multiple cities, API returns [{daily: ...}, {daily: ...}, ...]
    if isinstance(data, dict):
        locations = [data]
    else:
        locations = data

    results: dict[tuple[str, date], EnsembleForecast] = {}

    for loc_idx, loc_data in enumerate(locations):
        if loc_idx >= len(cities):
            break
        city_key, city = cities[loc_idx]

        daily = loc_data.get("daily", {})
        if not daily:
            continue

        # Build date index
        api_dates = daily.get("time", [])
        date_to_idx = {}
        for i, d_str in enumerate(api_dates):
            try:
                date_to_idx[date.fromisoformat(d_str)] = i
            except ValueError:
                continue

        for target_date in target_dates:
            idx = date_to_idx.get(target_date)
            if idx is None:
                continue

            members = []
            for key, values in daily.items():
                if key.startswith("temperature_2m_max_member") and values and len(values) > idx and values[idx] is not None:
                    members.append(float(values[idx]))

            if not members:
                continue

            logger.info(
                "Fetched %s forecast for %s on %s: %d members, mean=%.1f, spread=%.1f",
                model, city_key, target_date, len(members),
                np.mean(members), np.std(members),
            )

            results[(city_key, target_date)] = EnsembleForecast(
                city=city_key,
                target_date=target_date,
                model_name=model,
                members=members,
                unit=city.unit,
            )

    return results


async def fetch_all_cities(
    city_keys: list[str],
    target_dates: list[date] | None = None,
    models: list[str] | None = None,
) -> dict[tuple[str, date], list[EnsembleForecast]]:
    """Fetch ensemble forecasts for multiple cities and dates.

    Strategy (fail-fast, never block on retries):
    1. Return fresh cache if available (< 6h old)
    2. Try API once per model — no retries on 429
    3. If API fails, use stale cache (up to 12h)
    4. If no cache at all, build synthetic forecasts from deterministic API

    Returns a dict keyed by (city_key, target_date).
    """
    if target_dates is None:
        today = date.today()
        target_dates = [today, today + timedelta(days=1), today + timedelta(days=2)]
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025"]

    # Build cache key from sorted inputs
    cache_key = (
        ",".join(sorted(city_keys))
        + "|" + ",".join(d.isoformat() for d in sorted(target_dates))
        + "|" + ",".join(sorted(models))
    )

    # 1. Return fresh cached result (< 6h old)
    cached = _load_cache(cache_key, CACHE_TTL_MINUTES)
    if cached is not None:
        return cached

    # 2. Try ensemble API (one attempt per model, no retries)
    unit_groups: dict[str, list[str]] = {}
    for ck in city_keys:
        unit = CITIES[ck].unit
        unit_groups.setdefault(unit, []).append(ck)

    results: dict[tuple[str, date], list[EnsembleForecast]] = {}
    rate_limited = False

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in models:
            for unit, group_keys in unit_groups.items():
                batch = await _fetch_multi_city_model(
                    client, group_keys, target_dates, model
                )
                if not batch:
                    rate_limited = True
                for key, forecast in batch.items():
                    results.setdefault(key, []).append(forecast)

    if results:
        _save_cache(cache_key, results)
        logger.info("Cached %d forecast results to disk (TTL=%d min)", len(results), CACHE_TTL_MINUTES)
        return results

    # 3. API failed — use stale cache (up to 12h)
    if rate_limited:
        stale = _load_cache(cache_key, CACHE_STALE_TTL_MINUTES)
        if stale is not None:
            logger.warning(
                "API rate-limited — using stale cached forecasts (up to %d min old)",
                CACHE_STALE_TTL_MINUTES,
            )
            return stale

    # 4. No cache available — build synthetic ensemble from deterministic API
    # The deterministic (non-ensemble) API has separate, more generous rate limits
    logger.warning("No ensemble data or cache — falling back to deterministic API")
    results = await _build_synthetic_ensemble(city_keys, target_dates)
    if results:
        _save_cache(cache_key, results)
        logger.info("Built synthetic ensemble from deterministic API for %d combinations", len(results))

    return results


async def _build_synthetic_ensemble(
    city_keys: list[str],
    target_dates: list[date],
) -> dict[tuple[str, date], list[EnsembleForecast]]:
    """Build synthetic ensemble forecasts from deterministic API.

    Uses the free (non-ensemble) Open-Meteo API which has separate rate limits.
    Creates a synthetic spread around the deterministic forecast.
    """
    results: dict[tuple[str, date], list[EnsembleForecast]] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for city_key in city_keys:
            city = CITIES.get(city_key)
            if not city:
                continue

            temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
            sorted_dates = sorted(target_dates)

            try:
                await asyncio.sleep(0.5)
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "daily": "temperature_2m_max",
                        "start_date": sorted_dates[0].isoformat(),
                        "end_date": sorted_dates[-1].isoformat(),
                        "temperature_unit": temp_unit,
                        "timezone": "auto",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.debug("Deterministic fallback failed for %s: %s", city_key, e)
                continue

            daily = data.get("daily", {})
            api_dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])

            for i, d_str in enumerate(api_dates):
                try:
                    d = date.fromisoformat(d_str)
                except ValueError:
                    continue
                if d not in target_dates or i >= len(highs) or highs[i] is None:
                    continue

                high = float(highs[i])
                # Create synthetic ensemble: spread of ~2° around deterministic
                # This gives reasonable probability distributions
                spread = 2.0 if temp_unit == "celsius" else 3.5
                members = [high + (j - 15) * spread / 15 for j in range(31)]

                forecast = EnsembleForecast(
                    city=city_key,
                    target_date=d,
                    model_name="deterministic_fallback",
                    members=members,
                    unit=city.unit,
                )
                results[(city_key, d)] = [forecast]
                logger.info(
                    "Deterministic fallback for %s on %s: high=%.1f° (synthetic spread=%.1f)",
                    city_key, d, high, spread,
                )

    return results
