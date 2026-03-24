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
    # Determine temperature unit for the API
    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"

    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "daily": "temperature_2m_max",
        "models": model,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "temperature_unit": temp_unit,
        "timezone": "auto",
    }

    try:
        await asyncio.sleep(1.0)  # rate-limit: max 2 req/s for Open-Meteo free tier
        resp = await client.get(ENSEMBLE_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Failed to fetch %s forecast for %s on %s: %s", model, city_key, target_date, e)
        return None

    daily = data.get("daily", {})
    if not daily:
        logger.warning("No daily data for %s/%s/%s", model, city_key, target_date)
        return None

    # Extract member values — API returns temperature_2m_max_member01, member02, etc.
    members = []
    for key, values in daily.items():
        if key.startswith("temperature_2m_max_member") and values and values[0] is not None:
            members.append(float(values[0]))

    if not members:
        logger.warning("No ensemble members returned for %s/%s/%s", model, city_key, target_date)
        return None

    logger.info(
        "Fetched %s forecast for %s on %s: %d members, mean=%.1f, spread=%.1f",
        model, city_key, target_date, len(members),
        np.mean(members), np.std(members),
    )

    return EnsembleForecast(
        city=city_key,
        target_date=target_date,
        model_name=model,
        members=members,
        unit=city.unit,
    )


async def fetch_all_cities(
    city_keys: list[str],
    target_dates: list[date] | None = None,
    models: list[str] | None = None,
) -> dict[tuple[str, date], list[EnsembleForecast]]:
    """Fetch ensemble forecasts for multiple cities and dates.

    Returns a dict keyed by (city_key, target_date).
    """
    if target_dates is None:
        today = date.today()
        target_dates = [today, today + timedelta(days=1), today + timedelta(days=2)]

    results: dict[tuple[str, date], list[EnsembleForecast]] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for city_key in city_keys:
            for target_date in target_dates:
                forecasts = await fetch_ensemble_forecast(
                    city_key, target_date, models=models, client=client
                )
                if forecasts:
                    results[(city_key, target_date)] = forecasts

    return results
