"""Tests for the evaluation module: trade recording, resolution, and metrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from weather_bot.evaluation import (
    PerformanceMetrics,
    format_performance_report,
    get_daily_pnl,
    get_performance_metrics,
    get_trades,
    record_trade,
    resolve_trade,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database for testing."""
    return tmp_path / "test_trades.db"


def _make_trade(
    city="nyc",
    date="2026-03-20",
    bucket="58-59",
    side="BUY_YES",
    model_prob=65.0,
    market_prob=50.0,
    edge=15.0,
    size_usd=20.0,
    executed=True,
    verified_sources=0,
    verified_temp=None,
    verified_agreement=0,
    confidence=0.8,
) -> dict:
    """Create a mock trade result dict."""
    return {
        "city": city,
        "date": date,
        "bucket": bucket,
        "side": side,
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": edge,
        "size_usd": size_usd,
        "executed": executed,
        "confidence": confidence,
        "verified_temp": verified_temp,
        "verified_sources": verified_sources,
        "verified_agreement": verified_agreement,
    }


class TestRecordTrade:
    def test_record_basic_trade(self, db_path):
        trade = _make_trade()
        trade_id = record_trade(trade, trading_mode="paper", db_path=db_path)
        assert trade_id == 1

    def test_record_multiple_trades(self, db_path):
        for i in range(5):
            trade = _make_trade(city=f"city{i}")
            tid = record_trade(trade, db_path=db_path)
            assert tid == i + 1

    def test_record_verified_trade(self, db_path):
        trade = _make_trade(verified_sources=3, verified_temp=58.5, verified_agreement=90)
        trade_id = record_trade(trade, db_path=db_path)
        assert trade_id == 1

        records = get_trades(db_path=db_path)
        assert records[0].signal_type == "verified"

    def test_record_edge_trade(self, db_path):
        trade = _make_trade(verified_sources=0)
        record_trade(trade, db_path=db_path)

        records = get_trades(db_path=db_path)
        assert records[0].signal_type == "edge"


class TestResolveTrade:
    def test_resolve_yes_win(self, db_path):
        """BUY_YES wins when actual temp is in bucket."""
        trade = _make_trade(side="BUY_YES", bucket="58-59", market_prob=50.0, size_usd=20.0)
        trade_id = record_trade(trade, db_path=db_path)

        # Need to set bucket bounds manually since no signal object
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (trade_id,))
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=58.5, db_path=db_path)
        assert result["outcome"] == "won"
        assert result["pnl"] > 0
        assert result["temp_in_bucket"] is True

    def test_resolve_yes_loss(self, db_path):
        """BUY_YES loses when actual temp is outside bucket."""
        trade = _make_trade(side="BUY_YES", bucket="58-59", size_usd=20.0)
        trade_id = record_trade(trade, db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (trade_id,))
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=62.0, db_path=db_path)
        assert result["outcome"] == "lost"
        assert result["pnl"] == -20.0

    def test_resolve_no_win(self, db_path):
        """BUY_NO wins when actual temp is NOT in bucket."""
        trade = _make_trade(side="BUY_NO", bucket="58-59", market_prob=40.0, size_usd=15.0)
        trade_id = record_trade(trade, db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (trade_id,))
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=62.0, db_path=db_path)
        assert result["outcome"] == "won"
        assert result["pnl"] > 0

    def test_resolve_no_loss(self, db_path):
        """BUY_NO loses when actual temp IS in bucket."""
        trade = _make_trade(side="BUY_NO", bucket="58-59", market_prob=40.0, size_usd=15.0)
        trade_id = record_trade(trade, db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (trade_id,))
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=58.5, db_path=db_path)
        assert result["outcome"] == "lost"
        assert result["pnl"] == -15.0

    def test_resolve_lower_tail_bucket(self, db_path):
        """Lower tail bucket: temp < upper means in bucket."""
        trade = _make_trade(side="BUY_YES", bucket="<=57", size_usd=10.0)
        trade_id = record_trade(trade, db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE trades SET bucket_lower=NULL, bucket_upper=57, is_lower_tail=1 WHERE id=?",
            (trade_id,),
        )
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=55.0, db_path=db_path)
        assert result["outcome"] == "won"

    def test_resolve_upper_tail_bucket(self, db_path):
        """Upper tail bucket: temp >= lower means in bucket."""
        trade = _make_trade(side="BUY_YES", bucket=">=65", size_usd=10.0)
        trade_id = record_trade(trade, db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE trades SET bucket_lower=65, bucket_upper=NULL, is_upper_tail=1 WHERE id=?",
            (trade_id,),
        )
        conn.commit()
        conn.close()

        result = resolve_trade(trade_id, actual_temp=67.0, db_path=db_path)
        assert result["outcome"] == "won"


class TestGetTrades:
    def test_get_all_trades(self, db_path):
        for i in range(3):
            record_trade(_make_trade(city=f"city{i}"), db_path=db_path)

        trades = get_trades(db_path=db_path)
        assert len(trades) == 3

    def test_filter_by_city(self, db_path):
        record_trade(_make_trade(city="nyc"), db_path=db_path)
        record_trade(_make_trade(city="london"), db_path=db_path)
        record_trade(_make_trade(city="nyc"), db_path=db_path)

        trades = get_trades(city="nyc", db_path=db_path)
        assert len(trades) == 2

    def test_filter_by_outcome(self, db_path):
        trade_id = record_trade(_make_trade(), db_path=db_path)

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (trade_id,))
        conn.commit()
        conn.close()

        resolve_trade(trade_id, actual_temp=58.5, db_path=db_path)
        record_trade(_make_trade(), db_path=db_path)  # pending

        won = get_trades(outcome="won", db_path=db_path)
        pending = get_trades(outcome="pending", db_path=db_path)
        assert len(won) == 1
        assert len(pending) == 1

    def test_limit_and_offset(self, db_path):
        for i in range(10):
            record_trade(_make_trade(city=f"city{i}"), db_path=db_path)

        trades = get_trades(limit=3, db_path=db_path)
        assert len(trades) == 3


class TestPerformanceMetrics:
    def _setup_trades(self, db_path):
        """Create and resolve several trades for metrics testing."""
        import sqlite3

        # 3 winning trades, 2 losing trades
        scenarios = [
            ("nyc", "BUY_YES", 58.5, 58, 59, 20.0, 50.0),   # win
            ("london", "BUY_YES", 15.0, 14, 16, 25.0, 40.0),  # win
            ("paris", "BUY_NO", 22.0, 20, 21, 15.0, 60.0),    # win (NO, temp outside)
            ("tokyo", "BUY_YES", 30.0, 25, 27, 20.0, 50.0),   # lose
            ("seoul", "BUY_NO", 10.0, 9, 11, 10.0, 55.0),     # lose (NO, temp inside)
        ]

        for city, side, actual, lower, upper, size, mkt_prob in scenarios:
            trade = _make_trade(
                city=city, side=side, size_usd=size,
                market_prob=mkt_prob, date="2026-03-18",
            )
            tid = record_trade(trade, db_path=db_path)

            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE trades SET bucket_lower=?, bucket_upper=? WHERE id=?",
                (lower, upper, tid),
            )
            conn.commit()
            conn.close()

            resolve_trade(tid, actual_temp=actual, db_path=db_path)

    def test_basic_metrics(self, db_path):
        self._setup_trades(db_path)
        m = get_performance_metrics(db_path=db_path)

        assert m.total_trades == 5
        assert m.resolved_trades == 5
        assert m.wins == 3
        assert m.losses == 2
        assert m.win_rate == pytest.approx(0.6)
        assert m.total_invested == pytest.approx(90.0)

    def test_pnl_positive_for_wins(self, db_path):
        self._setup_trades(db_path)
        m = get_performance_metrics(db_path=db_path)

        # Total P&L should reflect wins minus losses
        assert m.total_pnl != 0
        assert m.best_trade_pnl > 0
        assert m.worst_trade_pnl < 0

    def test_streaks(self, db_path):
        self._setup_trades(db_path)
        m = get_performance_metrics(db_path=db_path)

        assert m.max_win_streak >= 1
        assert m.max_loss_streak >= 1

    def test_empty_metrics(self, db_path):
        m = get_performance_metrics(db_path=db_path)
        assert m.total_trades == 0
        assert m.win_rate == 0.0

    def test_filter_by_city(self, db_path):
        self._setup_trades(db_path)
        m = get_performance_metrics(city="nyc", db_path=db_path)
        assert m.total_trades == 1
        assert m.wins == 1


class TestFormatReport:
    def test_format_empty(self):
        m = PerformanceMetrics()
        report = format_performance_report(m)
        assert "No resolved trades" in report

    def test_format_with_data(self, db_path):
        # Record and resolve one winning trade
        import sqlite3

        trade = _make_trade(size_usd=20.0, market_prob=50.0)
        tid = record_trade(trade, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (tid,))
        conn.commit()
        conn.close()

        resolve_trade(tid, actual_temp=58.5, db_path=db_path)

        m = get_performance_metrics(db_path=db_path)
        report = format_performance_report(m)

        assert "Win rate" in report
        assert "100.0%" in report
        assert "P&L" in report


class TestDailyPnl:
    def test_daily_pnl_aggregation(self, db_path):
        import sqlite3

        for i, dt in enumerate(["2026-03-18", "2026-03-18", "2026-03-19"]):
            trade = _make_trade(date=dt, city=f"city{i}", size_usd=10.0, market_prob=50.0)
            tid = record_trade(trade, db_path=db_path)

            conn = sqlite3.connect(str(db_path))
            conn.execute("UPDATE trades SET bucket_lower=58, bucket_upper=59 WHERE id=?", (tid,))
            conn.commit()
            conn.close()

            # First two win, third loses
            actual = 58.5 if i < 2 else 62.0
            resolve_trade(tid, actual_temp=actual, db_path=db_path)

        daily = get_daily_pnl(days=30, db_path=db_path)
        assert len(daily) == 2
        assert daily[0]["date"] == "2026-03-18"
        assert daily[0]["trades"] == 2
        assert daily[0]["wins"] == 2
        assert daily[1]["date"] == "2026-03-19"
        assert daily[1]["trades"] == 1
        assert daily[1]["wins"] == 0
