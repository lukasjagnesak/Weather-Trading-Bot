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
    max_retries: int = 3,
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
            await asyncio.sleep(1.5)  # rate-limit for Open-Meteo free tier
            resp = await client.get(ENSEMBLE_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
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


async def fetch_all_cities(
    city_keys: list[str],
    target_dates: list[date] | None = None,
    models: list[str] | None = None,
) -> dict[tuple[str, date], list[EnsembleForecast]]:
    """Fetch ensemble forecasts for multiple cities and dates.

    Batches all dates into a single request per city/model to minimize
    API calls (e.g. 6 cities × 2 models = 12 requests instead of 36).

    Returns a dict keyed by (city_key, target_date).
    """
    if target_dates is None:
        today = date.today()
        target_dates = [today, today + timedelta(days=1), today + timedelta(days=2)]
    if models is None:
        models = ["gfs_seamless", "ecmwf_ifs025"]

    results: dict[tuple[str, date], list[EnsembleForecast]] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for city_key in city_keys:
            city = CITIES[city_key]
            for model in models:
                batch = await _fetch_model_batch(
                    client, city, city_key, target_dates, model
                )
                for td, forecast in batch.items():
                    results.setdefault((city_key, td), []).append(forecast)

    return results
