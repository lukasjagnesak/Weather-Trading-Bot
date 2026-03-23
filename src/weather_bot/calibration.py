"""EMOS calibration from real Polymarket resolutions.

Uses 460+ resolved temperature events to learn:
1. Per-city optimal spread inflation (sigma multiplier)
2. Per-model bias correction (a_0, a_1 in EMOS regression)
3. Optimal model weights based on historical accuracy
4. City-specific forecast difficulty metrics

The calibration minimizes CRPS (Continuous Ranked Probability Score)
which is the standard scoring rule for probabilistic weather forecasts.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .backtest_real import (
    POLYMARKET_CITY_MAP,
    _parse_bucket_label,
    fetch_polymarket_history,
)
from .config import CITIES

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(__file__).parent / "calibration_params.json"


@dataclass
class CalibrationSample:
    """One data point: model forecast vs actual Polymarket resolution."""

    city: str
    target_date: date
    model_name: str
    model_mean: float  # ensemble mean prediction
    model_std: float  # ensemble standard deviation
    unit: str
    winning_bucket_label: str
    winning_bucket_lower: float
    winning_bucket_upper: float
    all_bucket_labels: list[str] = field(default_factory=list)
    all_bucket_lowers: list[float] = field(default_factory=list)
    all_bucket_uppers: list[float] = field(default_factory=list)
    all_bucket_won: list[bool] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Learned calibration parameters."""

    # Per-city spread inflation
    city_spread_inflation: dict[str, float] = field(default_factory=dict)

    # Per-model bias correction: calibrated_mean = a0 + a1 * raw_mean
    model_bias_a0: dict[str, float] = field(default_factory=dict)
    model_bias_a1: dict[str, float] = field(default_factory=dict)

    # Optimal model weights
    model_weights: dict[str, float] = field(default_factory=dict)

    # Global parameters
    global_spread_inflation: float = 1.2
    min_sigma: float = 0.5

    # Per-city forecast difficulty (higher = harder to predict)
    city_difficulty: dict[str, float] = field(default_factory=dict)

    # Metadata
    n_samples: int = 0
    n_cities: int = 0
    avg_crps: float = 0.0
    avg_crps_before: float = 0.0

    def to_dict(self) -> dict:
        return {
            "city_spread_inflation": self.city_spread_inflation,
            "model_bias_a0": self.model_bias_a0,
            "model_bias_a1": self.model_bias_a1,
            "model_weights": self.model_weights,
            "global_spread_inflation": self.global_spread_inflation,
            "min_sigma": self.min_sigma,
            "city_difficulty": self.city_difficulty,
            "n_samples": self.n_samples,
            "n_cities": self.n_cities,
            "avg_crps": self.avg_crps,
            "avg_crps_before": self.avg_crps_before,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _bucket_log_score(
    mu: float,
    sigma: float,
    bucket_lower: float,
    bucket_upper: float,
    is_lower_tail: bool,
    is_upper_tail: bool,
) -> float:
    """Compute log probability of a bucket under Gaussian(mu, sigma^2).

    Returns negative log probability (lower is better).
    """
    if is_lower_tail:
        p = norm.cdf(bucket_upper, loc=mu, scale=sigma)
    elif is_upper_tail:
        p = 1.0 - norm.cdf(bucket_lower, loc=mu, scale=sigma)
    else:
        p = norm.cdf(bucket_upper, loc=mu, scale=sigma) - norm.cdf(
            bucket_lower, loc=mu, scale=sigma
        )
    p = max(p, 1e-10)
    return -np.log(p)


def _bucket_brier_score(
    mu: float,
    sigma: float,
    buckets_lower: list[float],
    buckets_upper: list[float],
    buckets_won: list[bool],
) -> float:
    """Compute Brier score across all buckets for one event.

    Brier score = mean( (forecast_prob - actual)^2 )
    Lower is better.
    """
    total = 0.0
    n = len(buckets_lower)
    for i in range(n):
        lo, hi, won = buckets_lower[i], buckets_upper[i], buckets_won[i]
        is_lower = lo == float("-inf")
        is_upper = hi == float("inf")

        if is_lower:
            p = norm.cdf(hi, loc=mu, scale=sigma)
        elif is_upper:
            p = 1.0 - norm.cdf(lo, loc=mu, scale=sigma)
        else:
            p = norm.cdf(hi, loc=mu, scale=sigma) - norm.cdf(lo, loc=mu, scale=sigma)
        p = max(min(p, 0.999), 0.001)

        actual = 1.0 if won else 0.0
        total += (p - actual) ** 2

    return total / n if n > 0 else 0.0


async def _fetch_forecast_data(
    client: httpx.AsyncClient,
    city_key: str,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, tuple[float, float]]]:
    """Fetch ensemble forecasts (mean, std) for a city/date range.

    Returns dict[date] -> {model_name: (mean, std)}
    """
    city = CITIES.get(city_key)
    if not city:
        return {}

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    results: dict[date, dict[str, tuple[float, float]]] = defaultdict(dict)

    models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
    past_days = (date.today() - start_date).days + 1

    for model in models:
        try:
            if past_days > 70:
                resp = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "daily": "temperature_2m_max",
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "temperature_unit": temp_unit,
                        "timezone": "auto",
                        "models": model,
                    },
                )
            else:
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "daily": "temperature_2m_max",
                        "models": model,
                        "temperature_unit": temp_unit,
                        "timezone": "auto",
                        "past_days": past_days,
                        "forecast_days": 0,
                    },
                )

            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])

            for i, d_str in enumerate(dates):
                d = date.fromisoformat(d_str)
                if i < len(highs) and highs[i] is not None:
                    temp = float(highs[i])
                    # For deterministic forecasts, estimate std from nearby values
                    spread = _estimate_spread(highs, i, city.unit)
                    results[d][model] = (temp, spread)
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Calibration fetch %s/%s failed: %s", model, city_key, exc)

    return dict(results)


def _estimate_spread(highs: list, idx: int, unit: str) -> float:
    """Estimate forecast uncertainty from day-to-day variability."""
    # Use 3-day window around the target day
    window = []
    for i in range(max(0, idx - 1), min(len(highs), idx + 2)):
        if highs[i] is not None:
            window.append(float(highs[i]))

    if len(window) < 2:
        return 1.5 if unit == "celsius" else 2.5

    return max(float(np.std(window)), 0.5 if unit == "celsius" else 1.0)


async def collect_calibration_samples(
    cities_filter: list[str] | None = None,
) -> list[CalibrationSample]:
    """Collect calibration data: model forecasts matched to Polymarket resolutions."""
    samples = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch resolved Polymarket events
        pm_events = await fetch_polymarket_history(client)
        logger.info("Fetched %d resolved events for calibration", len(pm_events))

        if cities_filter:
            pm_events = [e for e in pm_events if e["city"] in cities_filter]

        # Group by city
        events_by_city: dict[str, list[dict]] = defaultdict(list)
        for ev in pm_events:
            events_by_city[ev["city"]].append(ev)

        for city_key, city_events in events_by_city.items():
            if city_key not in CITIES:
                continue

            city_config = CITIES[city_key]
            city_dates = sorted(set(ev["date"] for ev in city_events))
            min_date = min(city_dates)
            max_date = max(city_dates)

            logger.info(
                "Calibration: fetching %s forecasts (%s to %s)...",
                city_key.upper(),
                min_date,
                max_date,
            )

            forecast_data = await _fetch_forecast_data(
                client, city_key, min_date, max_date
            )

            for ev in city_events:
                target = ev["date"]
                models_on_date = forecast_data.get(target, {})
                if not models_on_date:
                    continue

                winning = ev["winning_bucket"]

                # Parse all buckets
                all_labels = []
                all_lowers = []
                all_uppers = []
                all_won = []
                win_lower = None
                win_upper = None

                for b in ev["buckets"]:
                    tb = _parse_bucket_label(b["label"], city_config.unit)
                    if tb is None:
                        continue
                    all_labels.append(b["label"])
                    all_lowers.append(tb.lower)
                    all_uppers.append(tb.upper)
                    all_won.append(b["yes_won"])
                    if b["label"] == winning:
                        win_lower = tb.lower
                        win_upper = tb.upper

                if win_lower is None:
                    continue

                for model_name, (mean, std) in models_on_date.items():
                    samples.append(
                        CalibrationSample(
                            city=city_key,
                            target_date=target,
                            model_name=model_name,
                            model_mean=mean,
                            model_std=std,
                            unit=city_config.unit,
                            winning_bucket_label=winning,
                            winning_bucket_lower=win_lower,
                            winning_bucket_upper=win_upper,
                            all_bucket_labels=all_labels,
                            all_bucket_lowers=all_lowers,
                            all_bucket_uppers=all_uppers,
                            all_bucket_won=all_won,
                        )
                    )

    logger.info("Collected %d calibration samples", len(samples))
    return samples


def _compute_optimal_spread(
    samples: list[CalibrationSample],
) -> float:
    """Find optimal spread inflation that minimizes Brier score."""

    def objective(params):
        inflation = params[0]
        total_score = 0.0
        n = 0
        for s in samples:
            sigma = max(s.model_std * inflation, 0.5)
            score = _bucket_brier_score(
                s.model_mean, sigma,
                s.all_bucket_lowers, s.all_bucket_uppers, s.all_bucket_won,
            )
            total_score += score
            n += 1
        return total_score / n if n > 0 else 999

    result = minimize(
        objective,
        x0=[1.5],
        bounds=[(0.5, 5.0)],
        method="L-BFGS-B",
    )
    return float(result.x[0])


def _compute_optimal_bias(
    samples: list[CalibrationSample],
    spread_inflation: float,
) -> tuple[float, float]:
    """Find optimal bias correction (a0 + a1 * mean) that minimizes Brier score."""

    def objective(params):
        a0, a1 = params
        total_score = 0.0
        n = 0
        for s in samples:
            mu = a0 + a1 * s.model_mean
            sigma = max(s.model_std * spread_inflation, 0.5)
            score = _bucket_brier_score(
                mu, sigma,
                s.all_bucket_lowers, s.all_bucket_uppers, s.all_bucket_won,
            )
            total_score += score
            n += 1
        return total_score / n if n > 0 else 999

    result = minimize(
        objective,
        x0=[0.0, 1.0],
        bounds=[(-5.0, 5.0), (0.8, 1.2)],
        method="L-BFGS-B",
    )
    return float(result.x[0]), float(result.x[1])


def _compute_optimal_weights(
    samples: list[CalibrationSample],
    spread_inflation: float,
    bias_a0: dict[str, float],
    bias_a1: dict[str, float],
) -> dict[str, float]:
    """Find optimal model weights that minimize Brier score of the blend."""
    models = sorted(set(s.model_name for s in samples))
    if len(models) <= 1:
        return {m: 1.0 for m in models}

    # Group samples by (city, date)
    events: dict[tuple, dict[str, CalibrationSample]] = defaultdict(dict)
    for s in samples:
        events[(s.city, s.target_date.isoformat())][s.model_name] = s

    def objective(raw_weights):
        # Softmax to ensure weights sum to 1 and are positive
        w = np.exp(raw_weights) / np.sum(np.exp(raw_weights))
        weight_map = {models[i]: float(w[i]) for i in range(len(models))}

        total_score = 0.0
        n = 0
        for key, model_samples in events.items():
            if not model_samples:
                continue
            # Use the first sample for bucket info
            ref = next(iter(model_samples.values()))

            # Blended mu and sigma
            mu_blend = 0.0
            sigma_sq_blend = 0.0
            w_total = 0.0

            for mname, s in model_samples.items():
                wt = weight_map.get(mname, 0.0)
                a0 = bias_a0.get(mname, 0.0)
                a1 = bias_a1.get(mname, 1.0)
                mu = a0 + a1 * s.model_mean
                sig = max(s.model_std * spread_inflation, 0.5)

                mu_blend += wt * mu
                sigma_sq_blend += wt * (sig**2 + mu**2)
                w_total += wt

            if w_total > 0:
                mu_blend /= w_total
                sigma_sq_blend = sigma_sq_blend / w_total - mu_blend**2
                sigma_blend = max(np.sqrt(max(sigma_sq_blend, 0.01)), 0.5)
            else:
                continue

            score = _bucket_brier_score(
                mu_blend, sigma_blend,
                ref.all_bucket_lowers, ref.all_bucket_uppers, ref.all_bucket_won,
            )
            total_score += score
            n += 1

        return total_score / n if n > 0 else 999

    # Start with equal weights in log space
    x0 = np.zeros(len(models))
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 500})

    opt_w = np.exp(result.x) / np.sum(np.exp(result.x))
    return {models[i]: round(float(opt_w[i]), 4) for i in range(len(models))}


async def run_calibration(
    cities_filter: list[str] | None = None,
) -> CalibrationResult:
    """Run full EMOS calibration using historical Polymarket data.

    Steps:
    1. Collect (forecast, resolution) pairs from Polymarket + Open-Meteo
    2. Optimize per-city spread inflation
    3. Optimize per-model bias correction
    4. Optimize model weights
    5. Save calibration parameters
    """
    samples = await collect_calibration_samples(cities_filter)

    if not samples:
        logger.warning("No calibration samples collected")
        return CalibrationResult()

    result = CalibrationResult(
        n_samples=len(samples),
        n_cities=len(set(s.city for s in samples)),
    )

    # Step 1: Compute baseline Brier score (default params)
    baseline_scores = []
    for s in samples:
        sigma = max(s.model_std * 1.2, 0.5)
        score = _bucket_brier_score(
            s.model_mean, sigma,
            s.all_bucket_lowers, s.all_bucket_uppers, s.all_bucket_won,
        )
        baseline_scores.append(score)
    result.avg_crps_before = float(np.mean(baseline_scores))
    logger.info("Baseline Brier score: %.4f", result.avg_crps_before)

    # Step 2: Per-city spread inflation
    samples_by_city = defaultdict(list)
    for s in samples:
        samples_by_city[s.city].append(s)

    logger.info("Optimizing per-city spread inflation...")
    for city, city_samples in samples_by_city.items():
        optimal = _compute_optimal_spread(city_samples)
        result.city_spread_inflation[city] = round(optimal, 3)
        logger.info("  %s: spread_inflation=%.3f (n=%d)",
                     city.upper(), optimal, len(city_samples))

    # Global spread inflation (median of city values)
    if result.city_spread_inflation:
        result.global_spread_inflation = round(
            float(np.median(list(result.city_spread_inflation.values()))), 3
        )

    # Step 3: Per-model bias correction
    samples_by_model = defaultdict(list)
    for s in samples:
        samples_by_model[s.model_name].append(s)

    logger.info("Optimizing per-model bias correction...")
    for model, model_samples in samples_by_model.items():
        # Use model-specific spread inflation
        avg_inflation = float(np.mean([
            result.city_spread_inflation.get(s.city, result.global_spread_inflation)
            for s in model_samples
        ]))
        a0, a1 = _compute_optimal_bias(model_samples, avg_inflation)
        result.model_bias_a0[model] = round(a0, 3)
        result.model_bias_a1[model] = round(a1, 3)
        logger.info("  %s: a0=%.3f, a1=%.3f (n=%d)",
                     model, a0, a1, len(model_samples))

    # Step 4: Optimal model weights
    logger.info("Optimizing model weights...")
    result.model_weights = _compute_optimal_weights(
        samples,
        result.global_spread_inflation,
        result.model_bias_a0,
        result.model_bias_a1,
    )
    for model, w in sorted(result.model_weights.items()):
        logger.info("  %s: weight=%.4f", model, w)

    # Step 5: Per-city difficulty (average Brier score with optimal params)
    for city, city_samples in samples_by_city.items():
        inflation = result.city_spread_inflation.get(city, result.global_spread_inflation)
        scores = []
        for s in city_samples:
            a0 = result.model_bias_a0.get(s.model_name, 0.0)
            a1 = result.model_bias_a1.get(s.model_name, 1.0)
            mu = a0 + a1 * s.model_mean
            sigma = max(s.model_std * inflation, 0.5)
            score = _bucket_brier_score(
                mu, sigma,
                s.all_bucket_lowers, s.all_bucket_uppers, s.all_bucket_won,
            )
            scores.append(score)
        result.city_difficulty[city] = round(float(np.mean(scores)), 4)

    # Compute calibrated Brier score
    cal_scores = []
    for s in samples:
        inflation = result.city_spread_inflation.get(s.city, result.global_spread_inflation)
        a0 = result.model_bias_a0.get(s.model_name, 0.0)
        a1 = result.model_bias_a1.get(s.model_name, 1.0)
        mu = a0 + a1 * s.model_mean
        sigma = max(s.model_std * inflation, 0.5)
        score = _bucket_brier_score(
            mu, sigma,
            s.all_bucket_lowers, s.all_bucket_uppers, s.all_bucket_won,
        )
        cal_scores.append(score)
    result.avg_crps = float(np.mean(cal_scores))

    improvement = (1 - result.avg_crps / result.avg_crps_before) * 100
    logger.info("Calibrated Brier score: %.4f (%.1f%% improvement)",
                result.avg_crps, improvement)

    return result


def save_calibration(result: CalibrationResult, path: Path | None = None) -> Path:
    """Save calibration parameters to JSON file."""
    path = path or CALIBRATION_FILE
    path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Saved calibration to %s", path)
    return path


def load_calibration(path: Path | None = None) -> CalibrationResult | None:
    """Load calibration parameters from JSON file."""
    path = path or CALIBRATION_FILE
    if not path.exists():
        logger.debug("No calibration file at %s", path)
        return None
    try:
        data = json.loads(path.read_text())
        return CalibrationResult.from_dict(data)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to load calibration: %s", exc)
        return None


def format_calibration_report(r: CalibrationResult) -> str:
    """Format calibration results as readable report."""
    lines = [
        "",
        "=" * 60,
        "  EMOS CALIBRATION REPORT",
        "=" * 60,
        f"  Samples:  {r.n_samples}",
        f"  Cities:   {r.n_cities}",
        "",
        f"  Brier score (before): {r.avg_crps_before:.4f}",
        f"  Brier score (after):  {r.avg_crps:.4f}",
        f"  Improvement:          {(1 - r.avg_crps / r.avg_crps_before) * 100:.1f}%",
        "",
    ]

    # Global params
    lines.extend([
        "  ── GLOBAL PARAMETERS ──",
        f"  Global spread inflation: {r.global_spread_inflation:.3f}",
        f"  Min sigma:               {r.min_sigma}",
        "",
    ])

    # Model weights
    lines.append("  ── MODEL WEIGHTS ──")
    for model, w in sorted(r.model_weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 40)
        lines.append(f"  {model:<20} {w:.4f}  {bar}")
    lines.append("")

    # Model bias correction
    lines.append("  ── MODEL BIAS CORRECTION ──")
    lines.append(f"  {'Model':<20} {'a0':>8} {'a1':>8}  Interpretation")
    lines.append(f"  {'-'*56}")
    for model in sorted(r.model_bias_a0.keys()):
        a0 = r.model_bias_a0[model]
        a1 = r.model_bias_a1[model]
        interp = []
        if abs(a0) > 0.3:
            interp.append(f"{'warm' if a0 > 0 else 'cold'} bias {abs(a0):.1f}°")
        if abs(a1 - 1.0) > 0.02:
            interp.append(f"{'amplified' if a1 > 1 else 'dampened'}")
        lines.append(
            f"  {model:<20} {a0:>+7.3f} {a1:>7.3f}  "
            f"{'  '.join(interp) if interp else 'well-calibrated'}"
        )
    lines.append("")

    # Per-city spread inflation
    lines.append("  ── PER-CITY SPREAD INFLATION ──")
    lines.append(f"  {'City':<14} {'Inflation':>10} {'Difficulty':>10}  Note")
    lines.append(f"  {'-'*50}")
    for city in sorted(r.city_spread_inflation.keys()):
        inf = r.city_spread_inflation[city]
        diff = r.city_difficulty.get(city, 0)
        note = ""
        if inf > 2.0:
            note = "high uncertainty"
        elif inf < 1.0:
            note = "well-calibrated ensemble"
        elif diff > 0.15:
            note = "hard to predict"
        lines.append(
            f"  {city.upper():<14} {inf:>9.3f} {diff:>10.4f}  {note}"
        )

    lines.extend(["", "=" * 60, ""])
    return "\n".join(lines)
