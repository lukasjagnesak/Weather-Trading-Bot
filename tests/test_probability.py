"""Tests for the probability calculator."""

from datetime import date

import numpy as np
import pytest

from weather_bot.models import EnsembleForecast, TemperatureBucket
from weather_bot.probability import (
    compute_bucket_probabilities,
    compute_raw_ensemble_probability,
    ensemble_confidence,
)


def _make_forecast(members, model="gfs_seamless"):
    return EnsembleForecast(
        city="nyc",
        target_date=date(2026, 3, 20),
        model_name=model,
        members=members,
        unit="fahrenheit",
    )


def _make_buckets():
    """Create a simple set of temperature buckets."""
    return [
        TemperatureBucket("≤55", float("-inf"), 56.0, is_lower_tail=True),
        TemperatureBucket("56-57", 56.0, 58.0),
        TemperatureBucket("58-59", 58.0, 60.0),
        TemperatureBucket("60-61", 60.0, 62.0),
        TemperatureBucket("≥62", 62.0, float("inf"), is_upper_tail=True),
    ]


class TestBucketProbabilities:
    def test_probabilities_sum_to_one(self):
        members = list(np.random.normal(59, 2, 31))
        forecast = _make_forecast(members)
        buckets = _make_buckets()

        probs = compute_bucket_probabilities([forecast], buckets)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_peak_at_mean(self):
        """The bucket containing the ensemble mean should have the highest probability."""
        members = list(np.random.normal(59, 1.5, 31))
        forecast = _make_forecast(members)
        buckets = _make_buckets()

        probs = compute_bucket_probabilities([forecast], buckets)
        # 58-59 bucket should be highest (mean is 59)
        assert probs["58-59"] > probs["≤55"]
        assert probs["58-59"] > probs["≥62"]

    def test_multi_model_combination(self):
        """Combining two models should still sum to 1."""
        gfs = _make_forecast(list(np.random.normal(58, 2, 31)), "gfs_seamless")
        ecmwf = _make_forecast(list(np.random.normal(60, 1.5, 51)), "ecmwf_ifs025")
        buckets = _make_buckets()

        probs = compute_bucket_probabilities([gfs, ecmwf], buckets)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_tight_ensemble_concentrates_probability(self):
        """A tight ensemble (low spread) should concentrate probability."""
        tight = _make_forecast([59.0] * 31)
        wide = _make_forecast(list(np.random.normal(59, 5, 31)))
        buckets = _make_buckets()

        tight_probs = compute_bucket_probabilities([tight], buckets)
        wide_probs = compute_bucket_probabilities([wide], buckets)

        # Tight ensemble should have higher peak probability
        assert max(tight_probs.values()) > max(wide_probs.values())


class TestRawEnsembleProbability:
    def test_all_members_in_bucket(self):
        forecast = _make_forecast([59.0] * 31)
        bucket = TemperatureBucket("58-59", 58.0, 60.0)
        prob = compute_raw_ensemble_probability([forecast], bucket)
        assert prob == 1.0

    def test_no_members_in_bucket(self):
        forecast = _make_forecast([70.0] * 31)
        bucket = TemperatureBucket("58-59", 58.0, 60.0)
        prob = compute_raw_ensemble_probability([forecast], bucket)
        assert prob == 0.0


class TestEnsembleConfidence:
    def test_single_model_returns_default(self):
        forecast = _make_forecast([59.0] * 31)
        assert ensemble_confidence([forecast]) == 0.5

    def test_agreeing_models_high_confidence(self):
        gfs = _make_forecast([59.0] * 31, "gfs_seamless")
        ecmwf = _make_forecast([59.0] * 51, "ecmwf_ifs025")
        conf = ensemble_confidence([gfs, ecmwf])
        assert conf > 0.8

    def test_disagreeing_models_low_confidence(self):
        gfs = _make_forecast([50.0] * 31, "gfs_seamless")
        ecmwf = _make_forecast([70.0] * 51, "ecmwf_ifs025")
        conf = ensemble_confidence([gfs, ecmwf])
        assert conf < 0.5
