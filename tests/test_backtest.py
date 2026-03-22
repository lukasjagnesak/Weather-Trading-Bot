"""Tests for the backtesting module."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from weather_bot.backtest import (
    BacktestResult,
    BacktestTrade,
    _compute_bucket_probs,
    _generate_synthetic_buckets,
    _simulate_market_prices,
    _temp_in_bucket,
    format_backtest_report,
)
from weather_bot.models import EnsembleForecast, TemperatureBucket


class TestGenerateSyntheticBuckets:
    def test_celsius_buckets(self):
        buckets = _generate_synthetic_buckets(15.0, "celsius")
        # Should have 11 regular + 2 tail = 13 buckets
        assert len(buckets) == 13
        assert buckets[0].is_lower_tail
        assert buckets[-1].is_upper_tail

    def test_fahrenheit_buckets(self):
        buckets = _generate_synthetic_buckets(60.0, "fahrenheit")
        assert len(buckets) == 13
        assert buckets[0].is_lower_tail
        assert buckets[-1].is_upper_tail

    def test_buckets_cover_actual(self):
        """The actual temperature should fall in exactly one bucket."""
        actual = 15.3
        buckets = _generate_synthetic_buckets(actual, "celsius")
        hits = sum(1 for b in buckets if _temp_in_bucket(actual, b))
        assert hits == 1


class TestTempInBucket:
    def test_normal_bucket_inside(self):
        b = TemperatureBucket(label="15°C", lower=14.5, upper=15.5)
        assert _temp_in_bucket(15.0, b) is True

    def test_normal_bucket_outside(self):
        b = TemperatureBucket(label="15°C", lower=14.5, upper=15.5)
        assert _temp_in_bucket(16.0, b) is False

    def test_lower_tail(self):
        b = TemperatureBucket(label="≤10", lower=float("-inf"), upper=10.5,
                              is_lower_tail=True)
        assert _temp_in_bucket(8.0, b) is True
        assert _temp_in_bucket(11.0, b) is False

    def test_upper_tail(self):
        b = TemperatureBucket(label="≥20", lower=19.5, upper=float("inf"),
                              is_upper_tail=True)
        assert _temp_in_bucket(22.0, b) is True
        assert _temp_in_bucket(18.0, b) is False


class TestComputeBucketProbs:
    def test_probabilities_sum_to_one(self):
        forecasts = [
            EnsembleForecast(
                city="nyc", target_date=date(2026, 3, 20),
                model_name="gfs_seamless",
                members=[60 + np.random.normal(0, 2) for _ in range(31)],
                unit="fahrenheit",
            ),
        ]
        buckets = _generate_synthetic_buckets(60.0, "fahrenheit")
        probs = _compute_bucket_probs(forecasts, buckets)
        assert pytest.approx(sum(probs.values()), abs=0.01) == 1.0

    def test_peak_near_mean(self):
        """Highest probability bucket should be near ensemble mean."""
        members = [15.0 + np.random.normal(0, 1) for _ in range(51)]
        forecasts = [
            EnsembleForecast(
                city="london", target_date=date(2026, 3, 20),
                model_name="ecmwf_ifs025",
                members=members,
                unit="celsius",
            ),
        ]
        buckets = _generate_synthetic_buckets(15.0, "celsius")
        probs = _compute_bucket_probs(forecasts, buckets)

        # Find bucket with highest probability
        max_bucket = max(probs, key=probs.get)
        assert "15" in max_bucket


class TestSimulateMarketPrices:
    def test_prices_sum_to_one(self):
        buckets = _generate_synthetic_buckets(20.0, "celsius")
        prices = _simulate_market_prices(20.0, buckets)
        assert pytest.approx(sum(prices.values()), abs=0.01) == 1.0

    def test_peak_near_actual(self):
        buckets = _generate_synthetic_buckets(20.0, "celsius")
        prices = _simulate_market_prices(20.0, buckets)
        max_bucket = max(prices, key=prices.get)
        assert "20" in max_bucket


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult()
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.total_pnl == 0.0

    def test_with_trades(self):
        trades = [
            BacktestTrade(
                city="nyc", target_date=date(2026, 3, 20),
                bucket_label="60-62", bucket_lower=59.5, bucket_upper=62.5,
                side="BUY_YES", edge=0.1, position_size_usd=20.0,
                actual_temp=61.0, temp_in_bucket=True,
                outcome="won", pnl=15.0,
            ),
            BacktestTrade(
                city="london", target_date=date(2026, 3, 20),
                bucket_label="15°C", bucket_lower=14.5, bucket_upper=15.5,
                side="BUY_NO", edge=0.12, position_size_usd=15.0,
                actual_temp=15.0, temp_in_bucket=True,
                outcome="lost", pnl=-15.0,
            ),
        ]
        r = BacktestResult(trades=trades)
        assert r.total_trades == 2
        assert r.wins == 1
        assert r.losses == 1
        assert r.win_rate == 0.5
        assert r.total_pnl == 0.0


class TestFormatBacktestReport:
    def test_empty_report(self):
        r = BacktestResult(cities=["nyc"])
        report = format_backtest_report(r)
        assert "No trades" in report

    def test_report_with_trades(self):
        trades = [
            BacktestTrade(
                city="nyc", target_date=date(2026, 3, 20),
                bucket_label="60-62", bucket_lower=59.5, bucket_upper=62.5,
                side="BUY_YES", model_probability=0.6, market_price=0.4,
                edge=0.2, position_size_usd=20.0,
                actual_temp=61.0, temp_in_bucket=True,
                outcome="won", pnl=30.0,
            ),
        ]
        r = BacktestResult(
            trades=trades,
            start_date=date(2026, 3, 20),
            end_date=date(2026, 3, 20),
            cities=["nyc"],
        )
        report = format_backtest_report(r)
        assert "Win rate" in report
        assert "100.0%" in report
        assert "BUY_YES" in report
        assert "NYC" in report
