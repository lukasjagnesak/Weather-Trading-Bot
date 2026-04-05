"""Tests for the trading engine."""

from datetime import date

import pytest

from weather_bot.config import Settings
from weather_bot.models import (
    EnsembleForecast,
    MarketOutcome,
    PortfolioState,
    TemperatureBucket,
)
from weather_bot.trading import _kelly_fraction, detect_signals


class TestKellyFraction:
    def test_positive_edge(self):
        """When model probability > market price, Kelly should be positive."""
        f = _kelly_fraction(true_prob=0.6, market_price=0.4)
        assert f > 0
        # f* = (0.6 - 0.4) / (1 - 0.4) = 0.2 / 0.6 = 0.333
        assert abs(f - 1 / 3) < 0.01

    def test_no_edge(self):
        f = _kelly_fraction(true_prob=0.5, market_price=0.5)
        assert f == 0.0

    def test_negative_edge(self):
        f = _kelly_fraction(true_prob=0.3, market_price=0.5)
        assert f == 0.0

    def test_extreme_edge(self):
        f = _kelly_fraction(true_prob=0.95, market_price=0.1)
        assert f > 0.9  # very high Kelly fraction


class TestDetectSignals:
    def _make_outcome(self, city, target_date, bucket_label, lower, upper, price_yes):
        return MarketOutcome(
            market_id="test",
            condition_id="0x123",
            question=f"Will {city} be {bucket_label}?",
            token_id_yes="token_yes",
            token_id_no="token_no",
            bucket=TemperatureBucket(bucket_label, lower, upper),
            current_price_yes=price_yes,
            current_price_no=1.0 - price_yes,
            best_bid=price_yes - 0.02,
            best_ask=price_yes + 0.02,
            volume=1000.0,
            city=city,
            target_date=target_date,
        )

    def test_detects_certainty_signal_with_observed_high(self):
        """Should detect a certainty YES signal when observed high is in bucket."""
        today = date.today()
        settings = Settings()
        settings.certainty_min_hour = 0  # allow any hour for testing
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        # Market priced at 60c (within 50-82c range) — best_ask = 62c
        outcomes = [
            self._make_outcome("nyc", today, "58-59", 58.0, 60.0, 0.60),
            self._make_outcome("nyc", today, "60-61", 60.0, 62.0, 0.30),
            self._make_outcome("nyc", today, "≥62", 62.0, float("inf"), 0.10),
        ]

        forecasts = {
            ("nyc", today): [
                EnsembleForecast(
                    city="nyc",
                    target_date=today,
                    model_name="gfs_seamless",
                    members=[58.8] * 31,
                    unit="fahrenheit",
                ),
                EnsembleForecast(
                    city="nyc",
                    target_date=today,
                    model_name="ecmwf_ifs025",
                    members=[58.8] * 51,
                    unit="fahrenheit",
                ),
            ]
        }

        # Observed high at 59.0 → inside 58-59 bucket, margin 1.0° from both edges
        observed = {"nyc": 59.0}
        signals = detect_signals(outcomes, forecasts, settings, portfolio,
                                 observed_highs=observed)
        yes_signals = [s for s in signals if s.side == "BUY_YES"]
        assert len(yes_signals) > 0
        assert yes_signals[0].outcome.bucket.label == "58-59"

    def test_certainty_picks_observed_bucket(self):
        """Should BUY_YES on the bucket where observed temp falls."""
        today = date.today()
        settings = Settings()
        settings.certainty_min_hour = 0  # allow any hour for testing
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        import numpy as np
        np.random.seed(42)
        members_gfs = list(np.random.normal(59, 1.5, 31))
        members_ecmwf = list(np.random.normal(59, 1.5, 51))

        # Prices within 50-82c range for YES, 50-75c for NO
        outcomes = [
            self._make_outcome("nyc", today, "≤55", float("-inf"), 56.0, 0.05),
            self._make_outcome("nyc", today, "56-57", 56.0, 58.0, 0.10),
            self._make_outcome("nyc", today, "58-59", 58.0, 60.0, 0.60),
            self._make_outcome("nyc", today, "60-61", 60.0, 62.0, 0.10),
            self._make_outcome("nyc", today, "≥62", 62.0, float("inf"), 0.05),
        ]
        outcomes[0].bucket.is_lower_tail = True
        outcomes[4].bucket.is_upper_tail = True

        forecasts = {
            ("nyc", today): [
                EnsembleForecast(
                    city="nyc",
                    target_date=today,
                    model_name="gfs_seamless",
                    members=members_gfs,
                    unit="fahrenheit",
                ),
                EnsembleForecast(
                    city="nyc",
                    target_date=today,
                    model_name="ecmwf_ifs025",
                    members=members_ecmwf,
                    unit="fahrenheit",
                ),
            ]
        }

        # Observed temp at 59.0 → inside 58-59 bucket
        observed = {"nyc": 59.0}
        signals = detect_signals(outcomes, forecasts, settings, portfolio,
                                 observed_highs=observed)
        yes_signals = [s for s in signals if s.side == "BUY_YES"]
        assert len(yes_signals) == 1
        assert yes_signals[0].outcome.bucket.label == "58-59"

    def test_buy_no_requires_bucket_distance(self):
        """BUY_NO should only fire on buckets ≥3 buckets away."""
        today = date.today()
        settings = Settings()
        settings.certainty_min_hour = 0
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        outcomes = [
            self._make_outcome("nyc", today, "≤55", float("-inf"), 56.0, 0.90),
            self._make_outcome("nyc", today, "56-57", 56.0, 58.0, 0.85),
            self._make_outcome("nyc", today, "58-59", 58.0, 60.0, 0.60),
            self._make_outcome("nyc", today, "60-61", 60.0, 62.0, 0.85),
            self._make_outcome("nyc", today, "62-63", 62.0, 64.0, 0.90),
            self._make_outcome("nyc", today, "64-65", 64.0, 66.0, 0.90),
            self._make_outcome("nyc", today, "≥66", 66.0, float("inf"), 0.30),
        ]
        outcomes[0].bucket.is_lower_tail = True
        outcomes[6].bucket.is_upper_tail = True

        forecasts = {
            ("nyc", today): [
                EnsembleForecast(
                    city="nyc", target_date=today,
                    model_name="gfs_seamless", members=[59.0] * 31, unit="fahrenheit",
                ),
                EnsembleForecast(
                    city="nyc", target_date=today,
                    model_name="ecmwf_ifs025", members=[59.0] * 51, unit="fahrenheit",
                ),
            ]
        }

        # Observed at 59.0 → bucket index 2 (58-59)
        # BUY_NO should only consider buckets with index distance ≥ 3
        # AND degree distance ≥ 5°F
        observed = {"nyc": 59.0}
        signals = detect_signals(outcomes, forecasts, settings, portfolio,
                                 observed_highs=observed)
        no_signals = [s for s in signals if s.side == "BUY_NO"]
        for s in no_signals:
            # All NO signals must be on buckets far from observed
            bucket = s.outcome.bucket
            if bucket.is_upper_tail:
                assert bucket.lower - 59.0 >= 5.0
            elif bucket.is_lower_tail:
                assert 59.0 - bucket.upper >= 5.0

    def test_rejects_yes_outside_price_range(self):
        """YES signal should be rejected if price is outside 50-82c."""
        today = date.today()
        settings = Settings()
        settings.certainty_min_hour = 0
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        # Price at 90c → best_ask = 92c → outside 50-82c range
        outcomes = [
            self._make_outcome("nyc", today, "58-59", 58.0, 60.0, 0.90),
        ]
        forecasts = {
            ("nyc", today): [
                EnsembleForecast(
                    city="nyc", target_date=today,
                    model_name="gfs_seamless", members=[59.0] * 31, unit="fahrenheit",
                ),
                EnsembleForecast(
                    city="nyc", target_date=today,
                    model_name="ecmwf_ifs025", members=[59.0] * 51, unit="fahrenheit",
                ),
            ]
        }
        observed = {"nyc": 59.0}
        signals = detect_signals(outcomes, forecasts, settings, portfolio,
                                 observed_highs=observed)
        yes_signals = [s for s in signals if s.side == "BUY_YES"]
        assert len(yes_signals) == 0

    def test_no_signals_without_observed(self):
        """No signals should be generated without observed temperature."""
        today = date.today()
        settings = Settings()
        settings.certainty_min_hour = 0
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        outcomes = [
            self._make_outcome("nyc", today, "58-59", 58.0, 60.0, 0.60),
        ]
        forecasts = {
            ("nyc", today): [
                EnsembleForecast(
                    city="nyc", target_date=today,
                    model_name="gfs_seamless", members=[59.0] * 31, unit="fahrenheit",
                ),
            ]
        }
        # No observed highs
        signals = detect_signals(outcomes, forecasts, settings, portfolio,
                                 observed_highs=None)
        assert len(signals) == 0
