"""Trade history, outcome resolution, and performance evaluation.

Provides persistent SQLite storage for all trades, resolves outcomes
by checking actual temperatures, and computes performance metrics
(win rate, P&L, ROI, Sharpe ratio, etc.).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Default DB location: ~/.weather_bot/trades.db
_DEFAULT_DB_DIR = Path.home() / ".weather_bot"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "trades.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    city TEXT NOT NULL,
    target_date TEXT NOT NULL,
    bucket_label TEXT NOT NULL,
    bucket_lower REAL,
    bucket_upper REAL,
    is_lower_tail INTEGER DEFAULT 0,
    is_upper_tail INTEGER DEFAULT 0,
    side TEXT NOT NULL,
    model_probability REAL NOT NULL,
    market_probability REAL NOT NULL,
    edge REAL NOT NULL,
    position_size_usd REAL NOT NULL,
    price_paid REAL NOT NULL,
    shares REAL NOT NULL,
    confidence REAL NOT NULL,
    verified_temp REAL,
    verified_sources INTEGER DEFAULT 0,
    verified_agreement REAL DEFAULT 0,
    signal_type TEXT DEFAULT 'edge',
    market_id TEXT DEFAULT '',
    executed INTEGER DEFAULT 1,
    resolved INTEGER DEFAULT 0,
    outcome TEXT DEFAULT 'pending',
    actual_temp REAL,
    payout REAL DEFAULT 0,
    pnl REAL DEFAULT 0,
    resolved_at TEXT,
    trading_mode TEXT DEFAULT 'paper'
);

CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(target_date);
CREATE INDEX IF NOT EXISTS idx_trades_city ON trades(city);
CREATE INDEX IF NOT EXISTS idx_trades_resolved ON trades(resolved);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
"""


def _get_db_path() -> Path:
    """Get the database path, creating directory if needed."""
    db_path = Path(os.environ.get("WEATHER_BOT_DB", str(_DEFAULT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    path = db_path or _get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


@dataclass
class TradeRecord:
    """A single trade record from the database."""

    id: int
    timestamp: str
    city: str
    target_date: str
    bucket_label: str
    side: str
    model_probability: float
    market_probability: float
    edge: float
    position_size_usd: float
    price_paid: float
    shares: float
    confidence: float
    signal_type: str
    executed: bool
    resolved: bool
    outcome: str  # "pending", "won", "lost"
    actual_temp: float | None
    payout: float
    pnl: float
    trading_mode: str

    @property
    def is_won(self) -> bool:
        return self.outcome == "won"

    @property
    def roi_pct(self) -> float:
        if self.position_size_usd <= 0:
            return 0.0
        return (self.pnl / self.position_size_usd) * 100


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics."""

    total_trades: int = 0
    resolved_trades: int = 0
    pending_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_invested: float = 0.0
    total_payout: float = 0.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    avg_edge: float = 0.0
    avg_pnl_per_trade: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    # By signal type
    edge_trades: int = 0
    edge_wins: int = 0
    verified_trades: int = 0
    verified_wins: int = 0
    # By side
    yes_trades: int = 0
    yes_wins: int = 0
    no_trades: int = 0
    no_wins: int = 0


def record_trade(
    trade_result: dict,
    signal=None,
    trading_mode: str = "paper",
    db_path: Path | None = None,
) -> int:
    """Record a trade to the database. Returns the trade ID."""
    conn = _get_connection(db_path)
    try:
        # Determine price paid and shares
        market_prob = trade_result.get("market_prob", 0) / 100.0
        size_usd = trade_result.get("size_usd", 0)
        side = trade_result.get("side", "")

        if side == "BUY_YES":
            price_paid = market_prob
        else:
            price_paid = 1.0 - market_prob

        shares = size_usd / price_paid if price_paid > 0 else 0

        # Determine signal type from verification data
        signal_type = "verified" if trade_result.get("verified_sources", 0) >= 2 else "edge"

        # Get bucket bounds from signal if available
        bucket_lower = None
        bucket_upper = None
        is_lower_tail = 0
        is_upper_tail = 0
        market_id = ""
        if signal:
            bucket_lower = signal.outcome.bucket.lower
            bucket_upper = signal.outcome.bucket.upper
            is_lower_tail = int(signal.outcome.bucket.is_lower_tail)
            is_upper_tail = int(signal.outcome.bucket.is_upper_tail)
            market_id = signal.outcome.market_id

        cursor = conn.execute(
            """INSERT INTO trades (
                timestamp, city, target_date, bucket_label,
                bucket_lower, bucket_upper, is_lower_tail, is_upper_tail,
                side, model_probability, market_probability,
                edge, position_size_usd, price_paid, shares,
                confidence, verified_temp, verified_sources,
                verified_agreement, signal_type, market_id,
                executed, trading_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                trade_result["city"],
                trade_result["date"],
                trade_result["bucket"],
                bucket_lower,
                bucket_upper,
                is_lower_tail,
                is_upper_tail,
                side,
                trade_result.get("model_prob", 0) / 100.0,
                market_prob,
                trade_result.get("edge", 0) / 100.0,
                size_usd,
                price_paid,
                round(shares, 2),
                trade_result.get("confidence", 0),
                trade_result.get("verified_temp"),
                trade_result.get("verified_sources", 0),
                trade_result.get("verified_agreement", 0) / 100.0
                if trade_result.get("verified_agreement")
                else 0,
                signal_type,
                market_id,
                int(trade_result.get("executed", False)),
                trading_mode,
            ),
        )
        conn.commit()
        trade_id = cursor.lastrowid
        logger.info("Trade #%d recorded: %s %s %s %s $%.2f",
                     trade_id, trade_result["city"], trade_result["date"],
                     trade_result["bucket"], side, size_usd)
        return trade_id
    finally:
        conn.close()


def resolve_trade(
    trade_id: int,
    actual_temp: float,
    db_path: Path | None = None,
) -> dict:
    """Resolve a trade by checking if actual temp falls in the bucket.

    Returns dict with outcome, payout, pnl.
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            raise ValueError(f"Trade #{trade_id} not found")

        # Determine if temperature is in the bucket
        bucket_lower = row["bucket_lower"]
        bucket_upper = row["bucket_upper"]
        is_lower_tail = row["is_lower_tail"]
        is_upper_tail = row["is_upper_tail"]

        if is_lower_tail:
            temp_in_bucket = actual_temp < bucket_upper
        elif is_upper_tail:
            temp_in_bucket = actual_temp >= bucket_lower
        else:
            temp_in_bucket = bucket_lower <= actual_temp < bucket_upper

        side = row["side"]
        size_usd = row["position_size_usd"]
        price_paid = row["price_paid"]
        shares = row["shares"]

        # For BUY_YES: win if temp is in bucket (payout = $1/share)
        # For BUY_NO: win if temp is NOT in bucket (payout = $1/share)
        if side == "BUY_YES":
            won = temp_in_bucket
        else:
            won = not temp_in_bucket

        if won:
            payout = shares * 1.0  # $1 per share
            pnl = payout - size_usd
            outcome = "won"
        else:
            payout = 0.0
            pnl = -size_usd
            outcome = "lost"

        conn.execute(
            """UPDATE trades SET
                resolved = 1,
                outcome = ?,
                actual_temp = ?,
                payout = ?,
                pnl = ?,
                resolved_at = ?
            WHERE id = ?""",
            (outcome, actual_temp, round(payout, 2), round(pnl, 2),
             datetime.utcnow().isoformat(), trade_id),
        )
        conn.commit()

        result = {
            "trade_id": trade_id,
            "outcome": outcome,
            "actual_temp": actual_temp,
            "bucket": row["bucket_label"],
            "side": side,
            "temp_in_bucket": temp_in_bucket,
            "size_usd": size_usd,
            "payout": round(payout, 2),
            "pnl": round(pnl, 2),
        }
        logger.info("Trade #%d resolved: %s | actual=%.1f° | P&L=$%.2f",
                     trade_id, outcome.upper(), actual_temp, pnl)
        return result
    finally:
        conn.close()


async def resolve_pending_trades(db_path: Path | None = None) -> list[dict]:
    """Resolve all pending trades whose target_date has passed.

    Fetches actual observed temperatures and resolves each trade.
    """
    conn = _get_connection(db_path)
    try:
        today = date.today()
        rows = conn.execute(
            """SELECT id, city, target_date, bucket_label
               FROM trades
               WHERE resolved = 0 AND target_date < ?
               ORDER BY target_date""",
            (today.isoformat(),),
        ).fetchall()

        if not rows:
            logger.info("No pending trades to resolve")
            return []

        logger.info("Resolving %d pending trades...", len(rows))

        # Group by city+date to minimize API calls
        from .verification import fetch_observed_temperature

        results = []
        temp_cache: dict[tuple[str, str], float | None] = {}

        for row in rows:
            cache_key = (row["city"], row["target_date"])
            if cache_key not in temp_cache:
                try:
                    actual = await fetch_observed_temperature(
                        row["city"],
                        date.fromisoformat(row["target_date"]),
                    )
                    temp_cache[cache_key] = actual
                except Exception as e:
                    logger.warning("Could not fetch actual temp for %s/%s: %s",
                                   row["city"], row["target_date"], e)
                    temp_cache[cache_key] = None

            actual_temp = temp_cache[cache_key]
            if actual_temp is None:
                logger.warning("Skipping trade #%d: no actual temperature available", row["id"])
                continue

            result = resolve_trade(row["id"], actual_temp, db_path)
            results.append(result)

        return results
    finally:
        conn.close()


def get_existing_trade_keys(db_path: Path | None = None) -> set[tuple[str, str, str]]:
    """Return set of (city, target_date, bucket_label) for all pending trades.

    Used to prevent duplicate bets on the same market outcome after restart.
    Any existing bet on a bucket (YES or NO) blocks further trades on that bucket.
    """
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT city, target_date, bucket_label FROM trades WHERE resolved = 0",
        ).fetchall()
        return {(r["city"], r["target_date"], r["bucket_label"]) for r in rows}
    finally:
        conn.close()


def get_trades(
    city: str | None = None,
    target_date: str | None = None,
    outcome: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Path | None = None,
) -> list[TradeRecord]:
    """Query trades from the database with optional filters."""
    conn = _get_connection(db_path)
    try:
        conditions = []
        params: list = []

        if city:
            conditions.append("city = ?")
            params.append(city)
        if target_date:
            conditions.append("target_date = ?")
            params.append(target_date)
        if outcome:
            conditions.append("outcome = ?")
            params.append(outcome)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [_row_to_record(r) for r in rows]
    finally:
        conn.close()


def get_performance_metrics(
    days: int | None = None,
    city: str | None = None,
    db_path: Path | None = None,
) -> PerformanceMetrics:
    """Compute aggregated performance metrics."""
    conn = _get_connection(db_path)
    try:
        conditions = ["executed = 1"]
        params: list = []

        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            conditions.append("target_date >= ?")
            params.append(cutoff)
        if city:
            conditions.append("city = ?")
            params.append(city)

        where = f"WHERE {' AND '.join(conditions)}"

        rows = conn.execute(
            f"SELECT * FROM trades {where} ORDER BY timestamp ASC", params
        ).fetchall()

        if not rows:
            return PerformanceMetrics()

        m = PerformanceMetrics()
        m.total_trades = len(rows)

        pnl_list: list[float] = []
        win_streak = 0
        loss_streak = 0
        gross_wins = 0.0
        gross_losses = 0.0

        for row in rows:
            size = row["position_size_usd"]
            m.total_invested += size
            m.avg_edge += row["edge"]

            if row["resolved"]:
                m.resolved_trades += 1
                pnl = row["pnl"]
                pnl_list.append(pnl)
                m.total_payout += row["payout"]
                m.total_pnl += pnl

                if row["outcome"] == "won":
                    m.wins += 1
                    gross_wins += pnl
                    win_streak += 1
                    loss_streak = 0
                    m.max_win_streak = max(m.max_win_streak, win_streak)
                    m.best_trade_pnl = max(m.best_trade_pnl, pnl)
                else:
                    m.losses += 1
                    gross_losses += abs(pnl)
                    loss_streak += 1
                    win_streak = 0
                    m.max_loss_streak = max(m.max_loss_streak, loss_streak)
                    m.worst_trade_pnl = min(m.worst_trade_pnl, pnl)

                # By signal type
                if row["signal_type"] == "verified":
                    m.verified_trades += 1
                    if row["outcome"] == "won":
                        m.verified_wins += 1
                else:
                    m.edge_trades += 1
                    if row["outcome"] == "won":
                        m.edge_wins += 1

                # By side
                if row["side"] == "BUY_YES":
                    m.yes_trades += 1
                    if row["outcome"] == "won":
                        m.yes_wins += 1
                else:
                    m.no_trades += 1
                    if row["outcome"] == "won":
                        m.no_wins += 1
            else:
                m.pending_trades += 1

        # Compute derived metrics
        if m.resolved_trades > 0:
            m.win_rate = m.wins / m.resolved_trades
            m.avg_pnl_per_trade = m.total_pnl / m.resolved_trades
            if m.wins > 0:
                m.avg_win = gross_wins / m.wins
            if m.losses > 0:
                m.avg_loss = -gross_losses / m.losses

        if m.total_invested > 0:
            m.roi_pct = (m.total_pnl / m.total_invested) * 100

        if m.total_trades > 0:
            m.avg_edge = (m.avg_edge / m.total_trades) * 100

        if gross_losses > 0:
            m.profit_factor = gross_wins / gross_losses

        # Sharpe ratio (annualized, assuming daily returns)
        if len(pnl_list) >= 2:
            import statistics
            mean_pnl = statistics.mean(pnl_list)
            std_pnl = statistics.stdev(pnl_list)
            if std_pnl > 0:
                m.sharpe_ratio = (mean_pnl / std_pnl) * (252 ** 0.5)

        return m
    finally:
        conn.close()


def get_city_breakdown(
    days: int | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """Get performance breakdown by city."""
    conn = _get_connection(db_path)
    try:
        conditions = ["executed = 1", "resolved = 1"]
        params: list = []

        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            conditions.append("target_date >= ?")
            params.append(cutoff)

        where = f"WHERE {' AND '.join(conditions)}"

        rows = conn.execute(
            f"""SELECT city,
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome='won' THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       SUM(position_size_usd) as invested,
                       AVG(edge) as avg_edge
                FROM trades {where}
                GROUP BY city
                ORDER BY total_pnl DESC""",
            params,
        ).fetchall()

        return [
            {
                "city": r["city"],
                "trades": r["trades"],
                "wins": r["wins"],
                "win_rate": r["wins"] / r["trades"] if r["trades"] > 0 else 0,
                "total_pnl": round(r["total_pnl"], 2),
                "roi_pct": round((r["total_pnl"] / r["invested"]) * 100, 1)
                if r["invested"] > 0
                else 0,
                "avg_edge": round(r["avg_edge"] * 100, 1),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_daily_pnl(
    days: int = 30,
    db_path: Path | None = None,
) -> list[dict]:
    """Get daily P&L for the last N days."""
    conn = _get_connection(db_path)
    try:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            """SELECT target_date,
                      COUNT(*) as trades,
                      SUM(CASE WHEN outcome='won' THEN 1 ELSE 0 END) as wins,
                      SUM(pnl) as pnl,
                      SUM(position_size_usd) as invested
               FROM trades
               WHERE resolved = 1 AND executed = 1 AND target_date >= ?
               GROUP BY target_date
               ORDER BY target_date""",
            (cutoff,),
        ).fetchall()

        cumulative = 0.0
        result = []
        for r in rows:
            cumulative += r["pnl"]
            result.append({
                "date": r["target_date"],
                "trades": r["trades"],
                "wins": r["wins"],
                "pnl": round(r["pnl"], 2),
                "cumulative_pnl": round(cumulative, 2),
            })
        return result
    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> TradeRecord:
    """Convert a database row to a TradeRecord."""
    return TradeRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        city=row["city"],
        target_date=row["target_date"],
        bucket_label=row["bucket_label"],
        side=row["side"],
        model_probability=row["model_probability"],
        market_probability=row["market_probability"],
        edge=row["edge"],
        position_size_usd=row["position_size_usd"],
        price_paid=row["price_paid"],
        shares=row["shares"],
        confidence=row["confidence"],
        signal_type=row["signal_type"],
        executed=bool(row["executed"]),
        resolved=bool(row["resolved"]),
        outcome=row["outcome"],
        actual_temp=row["actual_temp"],
        payout=row["payout"],
        pnl=row["pnl"],
        trading_mode=row["trading_mode"],
    )


def format_performance_report(m: PerformanceMetrics) -> str:
    """Format performance metrics as a readable text report."""
    lines = [
        "=" * 55,
        "  WEATHER BOT — PERFORMANCE REPORT",
        "=" * 55,
        "",
        f"  Total trades:      {m.total_trades}",
        f"  Resolved:          {m.resolved_trades}  ({m.pending_trades} pending)",
        "",
    ]

    if m.resolved_trades > 0:
        lines.extend([
            f"  Wins / Losses:     {m.wins} / {m.losses}",
            f"  Win rate:          {m.win_rate:.1%}",
            "",
            f"  Total invested:    ${m.total_invested:,.2f}",
            f"  Total payout:      ${m.total_payout:,.2f}",
            f"  Total P&L:         ${m.total_pnl:+,.2f}",
            f"  ROI:               {m.roi_pct:+.1f}%",
            "",
            f"  Avg P&L/trade:     ${m.avg_pnl_per_trade:+.2f}",
            f"  Best trade:        ${m.best_trade_pnl:+.2f}",
            f"  Worst trade:       ${m.worst_trade_pnl:+.2f}",
            f"  Avg win:           ${m.avg_win:+.2f}",
            f"  Avg loss:          ${m.avg_loss:+.2f}",
            "",
            f"  Win streak (max):  {m.max_win_streak}",
            f"  Loss streak (max): {m.max_loss_streak}",
            f"  Profit factor:     {m.profit_factor:.2f}",
            f"  Sharpe ratio:      {m.sharpe_ratio:.2f}",
            f"  Avg edge:          {m.avg_edge:.1f}%",
        ])

        # Signal type breakdown
        lines.extend([
            "",
            "  — By Signal Type —",
        ])
        if m.edge_trades > 0:
            ew = m.edge_wins / m.edge_trades
            lines.append(f"  Edge-based:        {m.edge_trades} trades, {ew:.1%} win rate")
        if m.verified_trades > 0:
            vw = m.verified_wins / m.verified_trades
            lines.append(f"  Verified:          {m.verified_trades} trades, {vw:.1%} win rate")

        # Side breakdown
        lines.extend([
            "",
            "  — By Side —",
        ])
        if m.yes_trades > 0:
            yw = m.yes_wins / m.yes_trades
            lines.append(f"  BUY_YES:           {m.yes_trades} trades, {yw:.1%} win rate")
        if m.no_trades > 0:
            nw = m.no_wins / m.no_trades
            lines.append(f"  BUY_NO:            {m.no_trades} trades, {nw:.1%} win rate")
    else:
        lines.append("  No resolved trades yet.")

    lines.extend(["", "=" * 55])
    return "\n".join(lines)
