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

    def test_detects_mispriced_market(self):
        """Should detect a signal when model probability differs from market price."""
        settings = Settings()
        settings.min_edge_threshold = 0.05
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        # Market priced at 20% but model says 40%
        outcomes = [
            self._make_outcome("nyc", date(2026, 3, 20), "58-59", 58.0, 60.0, 0.20),
            self._make_outcome("nyc", date(2026, 3, 20), "60-61", 60.0, 62.0, 0.30),
            self._make_outcome("nyc", date(2026, 3, 20), "≥62", 62.0, float("inf"), 0.50),
        ]

        # Ensemble centered at 59 -> should make "58-59" bucket more probable
        forecasts = {
            ("nyc", date(2026, 3, 20)): [
                EnsembleForecast(
                    city="nyc",
                    target_date=date(2026, 3, 20),
                    model_name="gfs_seamless",
                    members=[59.0] * 31,
                    unit="fahrenheit",
                )
            ]
        }

        signals = detect_signals(outcomes, forecasts, settings, portfolio)
        # Should find at least one signal
        assert len(signals) > 0
        # The 58-59 bucket should be flagged (model says ~high, market says 20%)
        bucket_labels = [s.outcome.bucket.label for s in signals]
        assert "58-59" in bucket_labels

    def test_forecast_strategy_picks_best_bucket(self):
        """Should BUY_YES on the most probable bucket and BUY_NO only on certain losers."""
        settings = Settings()
        settings.max_position_pct = 0.10
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        # Mean=59, tight spread=1.5 → 58-59 bucket is most probable
        # Use multiple models to avoid deterministic_fallback flag
        import numpy as np
        np.random.seed(42)
        members_gfs = list(np.random.normal(59, 1.5, 31))
        members_ecmwf = list(np.random.normal(59, 1.5, 51))
        outcomes = [
            self._make_outcome("nyc", date(2026, 3, 20), "≤55", float("-inf"), 56.0, 0.05),
            self._make_outcome("nyc", date(2026, 3, 20), "56-57", 56.0, 58.0, 0.10),
            self._make_outcome("nyc", date(2026, 3, 20), "58-59", 58.0, 60.0, 0.15),
            self._make_outcome("nyc", date(2026, 3, 20), "60-61", 60.0, 62.0, 0.10),
            self._make_outcome("nyc", date(2026, 3, 20), "≥62", 62.0, float("inf"), 0.05),
        ]
        outcomes[0].bucket.is_lower_tail = True
        outcomes[4].bucket.is_upper_tail = True

        forecasts = {
            ("nyc", date(2026, 3, 20)): [
                EnsembleForecast(
                    city="nyc",
                    target_date=date(2026, 3, 20),
                    model_name="gfs_seamless",
                    members=members_gfs,
                    unit="fahrenheit",
                ),
                EnsembleForecast(
                    city="nyc",
                    target_date=date(2026, 3, 20),
                    model_name="ecmwf_ifs025",
                    members=members_ecmwf,
                    unit="fahrenheit",
                ),
            ]
        }

        signals = detect_signals(outcomes, forecasts, settings, portfolio)
        # Best bucket (58-59) should be BUY_YES
        yes_signals = [s for s in signals if s.side == "BUY_YES"]
        assert len(yes_signals) == 1
        assert yes_signals[0].outcome.bucket.label == "58-59"
        # NO signals only allowed if NO token is 50-95c
        no_signals = [s for s in signals if s.side == "BUY_NO"]
        for s in no_signals:
            assert 0.50 <= s.outcome.current_price_no <= 0.95, \
                f"BUY_NO should only happen at 50-95c NO price, got {s.outcome.current_price_no}"
