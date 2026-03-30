"""Latency arbitrage: detect forecast model updates and trade before markets adjust.

GFS updates 4x daily (00z/06z/12z/18z), available ~3.5-4.5h after init.
ECMWF updates 2x daily (00z/12z), available ~6-8h after init.
ICON updates 4x daily (00z/06z/12z/18z), available ~4-5h after init.

When a new model run shifts the forecast significantly (>1°C),
Polymarket prices still reflect the OLD forecast.  This is our edge.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Approximate UTC hours when model output becomes available on Open-Meteo
# Format: (model_name, init_hour_utc, available_hour_utc)
MODEL_RUN_SCHEDULE = [
    # GFS: init every 6h, available ~3.5-4.5h later
    ("gfs_seamless", 0, 4),
    ("gfs_seamless", 6, 10),
    ("gfs_seamless", 12, 16),
    ("gfs_seamless", 18, 22),
    # ECMWF: init every 12h, available ~6-8h later
    ("ecmwf_ifs025", 0, 7),
    ("ecmwf_ifs025", 12, 19),
    # ICON: init every 6h, available ~4-5h later
    ("icon_seamless", 0, 5),
    ("icon_seamless", 6, 11),
    ("icon_seamless", 12, 17),
    ("icon_seamless", 18, 23),
]

# How long the "update window" lasts after model becomes available (hours)
_UPDATE_WINDOW_HOURS = 2

# Persist previous forecast means for shift detection
_PREV_FILE = Path.home() / ".weather_bot" / "previous_forecasts.json"


def is_model_update_window() -> bool:
    """Check if we're in a model update window (new data likely available).

    Returns True if any major model has recently become available,
    meaning Open-Meteo likely has fresh ensemble data.
    """
    now = datetime.utcnow()
    current_hour = now.hour + now.minute / 60

    for model, _init_h, avail_h in MODEL_RUN_SCHEDULE:
        window_start = avail_h
        window_end = avail_h + _UPDATE_WINDOW_HOURS
        # Handle midnight wrap-around
        if window_end > 24:
            if current_hour >= window_start or current_hour < window_end - 24:
                return True
        elif window_start <= current_hour < window_end:
            return True

    return False


def get_active_model_updates() -> list[str]:
    """Return list of model names that have likely just updated."""
    now = datetime.utcnow()
    current_hour = now.hour + now.minute / 60
    active = []

    for model, _init_h, avail_h in MODEL_RUN_SCHEDULE:
        window_start = avail_h
        window_end = avail_h + _UPDATE_WINDOW_HOURS
        if window_end > 24:
            if current_hour >= window_start or current_hour < window_end - 24:
                active.append(model)
        elif window_start <= current_hour < window_end:
            active.append(model)

    return list(set(active))


def _load_previous() -> dict[str, float]:
    """Load previous forecast means from disk."""
    try:
        if _PREV_FILE.exists():
            return json.loads(_PREV_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_previous(data: dict[str, float]) -> None:
    """Save current forecast means to disk for next comparison."""
    try:
        _PREV_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PREV_FILE.write_text(json.dumps(data))
    except Exception as e:
        logger.debug("Failed to save previous forecasts: %s", e)


def detect_forecast_shifts(
    forecasts: dict[tuple[str, date], list],
) -> dict[tuple[str, date], float]:
    """Compare current forecasts with previous run and detect significant shifts.

    Returns a dict of (city, date) → shift_degrees for shifted forecasts.
    Positive shift = warmer, negative = cooler.
    """
    previous = _load_previous()
    current: dict[str, float] = {}
    shifts: dict[tuple[str, date], float] = {}

    for (city, target_date), forecast_list in forecasts.items():
        if not forecast_list:
            continue

        # Compute weighted mean across all models
        all_members = []
        for fc in forecast_list:
            all_members.extend(fc.members)
        if not all_members:
            continue

        mean = float(np.mean(all_members))
        key = f"{city}|{target_date.isoformat()}"
        current[key] = mean

        prev_mean = previous.get(key)
        if prev_mean is not None:
            shift = mean - prev_mean
            if abs(shift) > 0.01:  # any non-trivial shift
                shifts[(city, target_date)] = shift
                logger.info(
                    "FORECAST SHIFT %s/%s: %.1f° → %.1f° (shift=%+.1f°)",
                    city, target_date, prev_mean, mean, shift,
                )

    # Save current as previous for next comparison
    _save_previous(current)
    return shifts
