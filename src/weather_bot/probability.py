"""Convert ensemble forecasts to bucket probabilities using EMOS calibration."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
from scipy.stats import norm

from .models import EnsembleForecast, TemperatureBucket

logger = logging.getLogger(__name__)


def compute_bucket_probabilities(
    forecasts: list[EnsembleForecast],
    buckets: list[TemperatureBucket],
    model_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute probability for each temperature bucket using multi-model ensemble.

    Uses EMOS (Ensemble Model Output Statistics) calibration:
    - Fits a Gaussian predictive distribution from ensemble mean and spread
    - Applies light spread inflation to correct for ensemble under-dispersion
    - Combines multiple models via weighted averaging

    Args:
        forecasts: Ensemble forecasts from different models
        buckets: Temperature buckets from the market
        model_weights: Optional weights per model (default: equal weighting)

    Returns:
        Dict mapping bucket label to probability
    """
    if not forecasts or not buckets:
        return {}

    if model_weights is None:
        # Default: weight ECMWF slightly higher (better calibrated)
        model_weights = {
            "gfs_seamless": 0.4,
            "ecmwf_ifs025": 0.5,
            "icon_seamless": 0.1,
        }

    # Compute per-model probabilities, then combine
    combined_probs: dict[str, float] = {b.label: 0.0 for b in buckets}
    total_weight = 0.0

    for forecast in forecasts:
        weight = model_weights.get(forecast.model_name, 1.0 / len(forecasts))
        members = np.array(forecast.members)
        model_probs = _emos_bucket_probabilities(members, buckets)

        for label, prob in model_probs.items():
            combined_probs[label] += weight * prob
        total_weight += weight

    # Normalize
    if total_weight > 0:
        for label in combined_probs:
            combined_probs[label] /= total_weight

    # Ensure probabilities sum to 1 (renormalize)
    prob_sum = sum(combined_probs.values())
    if prob_sum > 0:
        for label in combined_probs:
            combined_probs[label] /= prob_sum

    return combined_probs


def _emos_bucket_probabilities(
    members: np.ndarray,
    buckets: list[TemperatureBucket],
    spread_inflation: float = 1.2,
) -> dict[str, float]:
    """Compute bucket probabilities using EMOS calibration.

    EMOS fits: T ~ N(mu, sigma^2)
      mu = ensemble_mean (a_0=0, a_1=1 for simplicity without training data)
      sigma = spread_inflation * ensemble_std

    The spread_inflation factor (>1) corrects for known ensemble under-dispersion.
    A value of 1.2 adds 20% to the ensemble spread, which is a standard correction.
    """
    mu = float(np.mean(members))
    sigma = float(np.std(members)) * spread_inflation

    # Minimum sigma to avoid degenerate distributions
    sigma = max(sigma, 0.5)

    probs: dict[str, float] = {}

    for bucket in buckets:
        if bucket.is_lower_tail:
            # P(T <= upper)
            p = float(norm.cdf(bucket.upper, loc=mu, scale=sigma))
        elif bucket.is_upper_tail:
            # P(T >= lower)
            p = float(1.0 - norm.cdf(bucket.lower, loc=mu, scale=sigma))
        else:
            # P(lower <= T < upper)
            p_upper = float(norm.cdf(bucket.upper, loc=mu, scale=sigma))
            p_lower = float(norm.cdf(bucket.lower, loc=mu, scale=sigma))
            p = p_upper - p_lower

        probs[bucket.label] = max(p, 0.001)  # floor at 0.1%

    return probs


def compute_raw_ensemble_probability(
    forecasts: list[EnsembleForecast],
    bucket: TemperatureBucket,
) -> float:
    """Compute raw (uncalibrated) probability by counting ensemble members in bucket.

    Useful as a cross-check against the EMOS-calibrated probability.
    """
    total_members = 0
    in_bucket = 0

    for forecast in forecasts:
        for temp in forecast.members:
            total_members += 1
            if bucket.is_lower_tail:
                if temp <= bucket.upper:
                    in_bucket += 1
            elif bucket.is_upper_tail:
                if temp >= bucket.lower:
                    in_bucket += 1
            else:
                if bucket.lower <= temp < bucket.upper:
                    in_bucket += 1

    if total_members == 0:
        return 0.0
    return in_bucket / total_members


def ensemble_confidence(forecasts: list[EnsembleForecast]) -> float:
    """Compute confidence score (0-1) based on model agreement.

    Higher confidence when models agree (low inter-model spread).
    """
    if len(forecasts) < 2:
        return 0.5

    means = [f.mean for f in forecasts]
    inter_model_spread = float(np.std(means))
    avg_intra_spread = float(np.mean([f.spread for f in forecasts]))

    # Confidence is higher when inter-model spread is small relative to intra-model spread
    # Use a baseline of 1.0 degree to handle zero-spread ensembles
    denominator = max(avg_intra_spread, 1.0)
    agreement_ratio = inter_model_spread / denominator
    confidence = max(0.1, min(1.0, 1.0 - agreement_ratio))

    return confidence
