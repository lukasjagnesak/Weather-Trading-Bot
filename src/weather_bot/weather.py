"""Fetch ensemble weather forecasts from Open-Meteo API."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

import httpx
import numpy as np

from .config import CITIES, CityConfig
from .models import EnsembleForecast

logger = logging.getLogger(__name__)

ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

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
    max_retries: int = 4,
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

    data = None
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(5.0)  # rate-limit: ensemble API free tier is strict
            resp = await client.get(ENSEMBLE_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                logger.warning("Rate limited on %s/%s, retrying in %ds...", model, city_key, wait)
                await asyncio.sleep(wait)
                continue
            logger.error("Failed to fetch %s forecast for %s: %s", model, city_key, e)
            return {}
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Failed to fetch %s forecast for %s: %s", model, city_key, e)
            return {}

    if data is None:
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
    max_retries: int = 4,
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

    data = None
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(5.0)  # rate-limit for ensemble API free tier
            resp = await client.get(ENSEMBLE_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                logger.warning("Rate limited on %s, retrying in %ds...", model, wait)
                await asyncio.sleep(wait)
                continue
            logger.error("Failed to fetch %s multi-city forecast: %s", model, e)
            return {}
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Failed to fetch %s multi-city forecast: %s", model, e)
            return {}

    if data is None:
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

    Batches all cities into a single multi-location request per model,
    grouped by temperature unit. Typically just 3-4 API requests total.

    Returns a dict keyed by (city_key, target_date).
    """
    if target_dates is None:
        today = date.today()
        target_dates = [today, today + timedelta(days=1), today + timedelta(days=2)]
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025"]

    # Group cities by temperature unit (fahrenheit vs celsius)
    unit_groups: dict[str, list[str]] = {}
    for ck in city_keys:
        unit = CITIES[ck].unit
        unit_groups.setdefault(unit, []).append(ck)

    results: dict[tuple[str, date], list[EnsembleForecast]] = {}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in models:
            for unit, group_keys in unit_groups.items():
                batch = await _fetch_multi_city_model(
                    client, group_keys, target_dates, model
                )
                for key, forecast in batch.items():
                    results.setdefault(key, []).append(forecast)

    return results
