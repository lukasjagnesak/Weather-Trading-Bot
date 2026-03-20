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
            self._make_outcome("nyc", date(2026, 3, 20), "58-59", 57.5, 59.5, 0.20),
            self._make_outcome("nyc", date(2026, 3, 20), "60-61", 59.5, 61.5, 0.30),
            self._make_outcome("nyc", date(2026, 3, 20), "≥62", 61.5, float("inf"), 0.50),
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

    def test_no_signals_when_fairly_priced(self):
        """Should not generate signals when markets are fairly priced."""
        settings = Settings()
        settings.min_edge_threshold = 0.08
        portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)

        # Provide a full set of buckets so probabilities distribute properly
        # With mean=59 and spread=3, the 58-59 bucket gets ~20-25% probability
        import numpy as np
        np.random.seed(42)
        members = list(np.random.normal(59, 3, 31))
        outcomes = [
            self._make_outcome("nyc", date(2026, 3, 20), "≤55", float("-inf"), 55.5, 0.15),
            self._make_outcome("nyc", date(2026, 3, 20), "56-57", 55.5, 57.5, 0.20),
            self._make_outcome("nyc", date(2026, 3, 20), "58-59", 57.5, 59.5, 0.25),
            self._make_outcome("nyc", date(2026, 3, 20), "60-61", 59.5, 61.5, 0.25),
            self._make_outcome("nyc", date(2026, 3, 20), "≥62", 61.5, float("inf"), 0.15),
        ]
        # Set tail bucket flags
        outcomes[0].bucket.is_lower_tail = True
        outcomes[4].bucket.is_upper_tail = True

        forecasts = {
            ("nyc", date(2026, 3, 20)): [
                EnsembleForecast(
                    city="nyc",
                    target_date=date(2026, 3, 20),
                    model_name="gfs_seamless",
                    members=members,
                    unit="fahrenheit",
                )
            ]
        }

        signals = detect_signals(outcomes, forecasts, settings, portfolio)
        # With model probs distributed across 5 buckets and market prices
        # set close to model estimates, edge should be < 8% threshold
        assert len(signals) == 0, f"Expected no signals but got: {[(s.outcome.bucket.label, s.edge) for s in signals]}"
