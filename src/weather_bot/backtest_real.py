"""Backtest engine using REAL Polymarket historical data.

Fetches resolved temperature events from the Gamma API, combines with
our ensemble model forecasts, and evaluates what the bot would have traded
against actual market prices and resolutions.

Unlike the simulated backtest, this uses:
- Real Polymarket bucket definitions and resolutions
- Real trading volumes (as proxy for market depth)
- Estimated market prices derived from volumes + resolution
- Actual weather outcomes from Polymarket resolution
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
import numpy as np
from scipy.stats import norm

from .config import CITIES
from .models import EnsembleForecast, TemperatureBucket

logger = logging.getLogger(__name__)

# Map Polymarket city names to our config keys
POLYMARKET_CITY_MAP = {
    "new york": "nyc", "nyc": "nyc",
    "london": "london", "paris": "paris",
    "tokyo": "tokyo", "seoul": "seoul",
    "shanghai": "shanghai", "ankara": "ankara",
    "toronto": "toronto", "chicago": "chicago",
    "dallas": "dallas", "atlanta": "atlanta",
    "miami": "miami", "wellington": "wellington",
    "taipei": "taipei", "lucknow": "lucknow",
    "sydney": "sydney", "seattle": "seattle",
    "denver": "denver",
}


@dataclass
class RealBacktestTrade:
    """A trade evaluated against real Polymarket data."""

    city: str
    target_date: date
    bucket_label: str
    side: str  # BUY_YES or BUY_NO
    model_probability: float  # Our model's probability for this bucket
    market_price: float  # Estimated market price (from volume distribution)
    edge: float  # model_prob - market_price
    position_size_usd: float
    bucket_won: bool  # Did this bucket resolve YES on Polymarket?
    outcome: str  # "won" or "lost"
    pnl: float
    bucket_volume: float  # Real volume on this bucket
    event_total_volume: float  # Total volume across all buckets


@dataclass
class RealBacktestResult:
    """Aggregated results from real Polymarket backtest."""

    trades: list[RealBacktestTrade] = field(default_factory=list)
    events_processed: int = 0
    events_with_signals: int = 0
    events_total: int = 0
    start_date: date = date.today()
    end_date: date = date.today()
    bankroll: float = 1000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "won")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "lost")

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.trades) if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_invested(self) -> float:
        return sum(t.position_size_usd for t in self.trades)

    @property
    def roi_pct(self) -> float:
        if self.total_invested <= 0:
            return 0.0
        return (self.total_pnl / self.total_invested) * 100


def _parse_bucket_label(label: str, unit: str) -> TemperatureBucket | None:
    """Parse a Polymarket groupItemTitle into a TemperatureBucket.

    Examples:
      '27°F or below'  -> lower tail
      '38°F or higher' -> upper tail
      '28-29°F'        -> range bucket
      '5°C'            -> single degree (celsius)
      '2°C or below'   -> lower tail celsius
      '8°C or higher'  -> upper tail celsius
    """
    label = label.strip()

    # Lower tail: "27°F or below" / "2°C or below"
    # WU rounds to whole degrees: "≤27" includes 27 → upper=28
    m = re.match(r"(-?\d+)°[FC]?\s+or\s+below", label, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return TemperatureBucket(
            label=label, lower=float("-inf"), upper=val + 1,
            is_lower_tail=True,
        )

    # Upper tail: "38°F or higher" / "8°C or higher"
    # "≥38" means 38 and above → lower=38
    m = re.match(r"(-?\d+)°[FC]?\s+or\s+higher", label, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return TemperatureBucket(
            label=label, lower=val, upper=float("inf"),
            is_upper_tail=True,
        )

    # Range: "28-29°F" / "54-55°F"
    # WU whole degrees: "28-29" means [28, 30)
    m = re.match(r"(-?\d+)-(-?\d+)°?[FC]?", label)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        return TemperatureBucket(
            label=label, lower=low, upper=high + 1,
        )

    # Single degree celsius: "5°C" → [5, 6)
    m = re.match(r"(-?\d+)°C", label)
    if m:
        val = float(m.group(1))
        return TemperatureBucket(
            label=label, lower=val, upper=val + 1,
        )

    logger.debug("Could not parse bucket label: %s", label)
    return None


def _estimate_market_prices(
    buckets: list[dict],
    total_volume: float,
) -> dict[str, float]:
    """Estimate pre-resolution market prices from bucket volumes.

    Volume is a strong proxy for market interest. Higher volume buckets
    tend to have been priced closer to the center of the distribution.
    We use volume share as a rough price estimate, then normalize.

    This isn't perfect but is the best we can do without historical
    order book snapshots.
    """
    if total_volume <= 0:
        # Equal distribution
        n = len(buckets)
        return {b["label"]: 1.0 / n for b in buckets}

    prices = {}
    for b in buckets:
        # Volume share as price proxy — but apply sqrt to compress
        # (high volume buckets get disproportionate volume from limit orders)
        vol_share = b["volume"] / total_volume if total_volume > 0 else 1.0 / len(buckets)
        prices[b["label"]] = max(vol_share, 0.01)

    # Normalize to sum to 1
    s = sum(prices.values())
    if s > 0:
        for k in prices:
            prices[k] /= s

    return prices


async def fetch_polymarket_history(
    client: httpx.AsyncClient,
) -> list[dict]:
    """Fetch all resolved temperature events from Polymarket Gamma API."""
    all_events = []
    offset = 0

    while True:
        resp = await client.get(
            "https://gamma-api.polymarket.com/events",
            params={
                "tag_slug": "temperature",
                "limit": "100",
                "closed": "true",
                "offset": str(offset),
            },
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break
        all_events.extend(events)
        offset += len(events)
        if len(events) < 100:
            break

    logger.info("Fetched %d resolved temperature events from Polymarket", len(all_events))

    # Parse into structured format
    parsed = []
    for e in all_events:
        title = e.get("title", "")
        end_date_str = e.get("endDate", "")[:10]

        # Extract city
        city_key = None
        for name, key in POLYMARKET_CITY_MAP.items():
            if name in title.lower():
                city_key = key
                break
        if not city_key:
            continue

        try:
            target_date = date.fromisoformat(end_date_str)
        except ValueError:
            continue

        buckets = []
        winning_bucket = None
        total_volume = 0.0

        for m in e.get("markets", []):
            outcome_prices = m.get("outcomePrices", ["0.5", "0.5"])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, ValueError):
                    outcome_prices = ["0.5", "0.5"]

            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

            vol_raw = m.get("volume", 0)
            vol = float(vol_raw) if vol_raw else 0.0

            bucket_label = m.get("groupItemTitle", "")
            total_volume += vol

            bucket_info = {
                "label": bucket_label,
                "yes_won": yes_price > 0.5,
                "volume": vol,
            }
            buckets.append(bucket_info)

            if yes_price > 0.5:
                winning_bucket = bucket_label

        if not buckets or winning_bucket is None:
            continue

        parsed.append({
            "city": city_key,
            "date": target_date,
            "title": title,
            "buckets": buckets,
            "winning_bucket": winning_bucket,
            "total_volume": total_volume,
        })

    return parsed


async def _fetch_model_forecasts(
    client: httpx.AsyncClient,
    city_key: str,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, float]]:
    """Fetch historical NWP model forecasts for a city/date range.

    Returns dict[date] -> {model_name: predicted_temp}
    """
    city = CITIES.get(city_key)
    if not city:
        return {}

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    results: dict[date, dict[str, float]] = defaultdict(dict)

    models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
    past_days_count = (date.today() - start_date).days + 1
    use_archive = past_days_count > 70

    for model in models:
        try:
            if use_archive:
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
                        "past_days": past_days_count,
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
                    results[d][model] = float(highs[i])
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Model %s fetch failed for %s: %s", model, city_key, exc)

    return dict(results)


def _models_to_ensemble(
    model_temps: dict[str, float],
    city_unit: str,
) -> list[EnsembleForecast]:
    """Convert deterministic model forecasts to synthetic ensemble."""
    forecasts = []
    rng = np.random.RandomState(42)
    spread = 1.5 if city_unit == "celsius" else 2.5

    for model_name, temp in model_temps.items():
        members = [temp + rng.normal(0, spread) for _ in range(31)]
        forecasts.append(EnsembleForecast(
            city="", target_date=date.today(),
            model_name=model_name, members=members, unit=city_unit,
        ))
    return forecasts


def _compute_all_bucket_probs(
    forecasts: list[EnsembleForecast],
    buckets: list[TemperatureBucket],
    city: str | None = None,
    use_calibration: bool = True,
) -> dict[str, float]:
    """Compute probabilities for all buckets using the shared probability module."""
    from .probability import compute_bucket_probabilities
    return compute_bucket_probabilities(
        forecasts, buckets, city=city if use_calibration else None,
    )


async def run_real_backtest(
    bankroll: float = 1000.0,
    min_edge: float = 0.08,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.02,
    cities_filter: list[str] | None = None,
    use_calibration: bool = True,
) -> RealBacktestResult:
    """Run backtest against real Polymarket resolved temperature markets.

    Steps per event:
    1. Fetch resolved event from Gamma API (real buckets, real resolution)
    2. Fetch our model's forecast for that city/date
    3. Compute our model probability for each bucket
    4. Estimate market price from volume distribution
    5. Detect signals where model disagrees with market
    6. Resolve against Polymarket's actual resolution
    """
    result = RealBacktestResult(bankroll=bankroll)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Fetch all resolved Polymarket temperature events
        logger.info("Fetching resolved Polymarket temperature events...")
        pm_events = await fetch_polymarket_history(client)
        result.events_total = len(pm_events)

        if not pm_events:
            logger.warning("No resolved Polymarket events found")
            return result

        # Filter by cities if specified
        if cities_filter:
            pm_events = [e for e in pm_events if e["city"] in cities_filter]

        logger.info("Processing %d events...", len(pm_events))

        # Group events by city to batch-fetch forecasts
        events_by_city: dict[str, list[dict]] = defaultdict(list)
        for ev in pm_events:
            events_by_city[ev["city"]].append(ev)

        for city_key, city_events in events_by_city.items():
            if city_key not in CITIES:
                logger.debug("City %s not in our config, skipping", city_key)
                continue

            city_config = CITIES[city_key]
            city_dates = sorted(set(ev["date"] for ev in city_events))
            min_date = min(city_dates)
            max_date = max(city_dates)

            logger.info("Fetching forecasts for %s (%s to %s, %d events)...",
                         city_key.upper(), min_date, max_date, len(city_events))

            # Batch fetch model forecasts for the entire date range
            model_data = await _fetch_model_forecasts(
                client, city_key, min_date, max_date
            )

            for ev in city_events:
                target_date = ev["date"]
                model_temps = model_data.get(target_date, {})

                if not model_temps:
                    logger.debug("No model data for %s/%s", city_key, target_date)
                    continue

                result.events_processed += 1

                # Build ensemble from model forecasts
                forecasts = _models_to_ensemble(model_temps, city_config.unit)

                # Parse Polymarket buckets and compute our probabilities
                pm_buckets = ev["buckets"]
                total_vol = ev["total_volume"]
                winning_bucket = ev["winning_bucket"]

                # Estimate market prices from volume distribution
                market_prices = _estimate_market_prices(pm_buckets, total_vol)

                # Compute model probability for each bucket
                parsed_buckets = []
                for b_info in pm_buckets:
                    tb = _parse_bucket_label(b_info["label"], city_config.unit)
                    if tb:
                        parsed_buckets.append((b_info, tb))

                # Use shared calibration-aware probability module
                tb_list = [tb for _, tb in parsed_buckets]
                model_probs = _compute_all_bucket_probs(
                    forecasts, tb_list,
                    city=city_key,
                    use_calibration=use_calibration,
                )

                had_signal = False

                for b_info, tb in parsed_buckets:
                    label = b_info["label"]
                    model_p = model_probs.get(label, 0)
                    market_p = market_prices.get(label, 0.5)
                    bucket_won = b_info["yes_won"]

                    # Edge calculation
                    edge_yes = model_p - market_p
                    edge_no = (1 - model_p) - (1 - market_p)  # = market_p - model_p

                    if edge_yes >= edge_no and edge_yes > min_edge:
                        side = "BUY_YES"
                        edge = edge_yes
                        price = market_p
                        true_prob = model_p
                    elif edge_no > min_edge:
                        side = "BUY_NO"
                        edge = edge_no
                        price = 1 - market_p
                        true_prob = 1 - model_p
                    else:
                        continue

                    if price <= 0.01 or price >= 0.99:
                        continue

                    # Kelly sizing
                    kelly_f = (true_prob - price) / (1 - price)
                    if kelly_f <= 0:
                        continue

                    adj_kelly = kelly_f * kelly_fraction
                    pos_size = min(adj_kelly * bankroll, max_position_pct * bankroll)
                    if pos_size < 1.0:
                        continue

                    shares = pos_size / price

                    if side == "BUY_YES":
                        won = bucket_won
                    else:
                        won = not bucket_won

                    payout = shares * 1.0 if won else 0.0
                    pnl = payout - pos_size if won else -pos_size

                    trade = RealBacktestTrade(
                        city=city_key,
                        target_date=target_date,
                        bucket_label=label,
                        side=side,
                        model_probability=model_p,
                        market_price=market_p,
                        edge=edge,
                        position_size_usd=round(pos_size, 2),
                        bucket_won=bucket_won,
                        outcome="won" if won else "lost",
                        pnl=round(pnl, 2),
                        bucket_volume=b_info["volume"],
                        event_total_volume=total_vol,
                    )
                    result.trades.append(trade)
                    had_signal = True

                if had_signal:
                    result.events_with_signals += 1

    if result.trades:
        dates = [t.target_date for t in result.trades]
        result.start_date = min(dates)
        result.end_date = max(dates)

    return result


def format_real_backtest_report(r: RealBacktestResult) -> str:
    """Format real backtest results as a comprehensive report."""
    lines = [
        "",
        "=" * 70,
        "  REAL POLYMARKET BACKTEST REPORT",
        "=" * 70,
        f"  Period:          {r.start_date} to {r.end_date}",
        f"  Bankroll:        ${r.bankroll:,.2f}",
        f"  Events total:    {r.events_total} (from Polymarket API)",
        f"  Events matched:  {r.events_processed} (had model forecast data)",
        f"  Events w/signal: {r.events_with_signals} (model disagreed with market)",
        "",
    ]

    if not r.trades:
        lines.append("  No trades generated. Model agreed with market on all events.")
        return "\n".join(lines)

    # ─── Summary ───
    lines.extend([
        "  ── SUMMARY ──",
        f"  Total trades:   {r.total_trades}",
        f"  Wins / Losses:  {r.wins} / {r.losses}",
        f"  Win rate:       {r.win_rate:.1%}",
        "",
        f"  Total invested: ${r.total_invested:,.2f}",
        f"  Total P&L:      ${r.total_pnl:+,.2f}",
        f"  ROI:            {r.roi_pct:+.1f}%",
        "",
    ])

    # ─── By City ───
    city_stats: dict[str, dict] = {}
    for t in r.trades:
        if t.city not in city_stats:
            city_stats[t.city] = {
                "trades": 0, "wins": 0, "pnl": 0.0,
                "invested": 0.0, "volume": 0.0,
            }
        s = city_stats[t.city]
        s["trades"] += 1
        if t.outcome == "won":
            s["wins"] += 1
        s["pnl"] += t.pnl
        s["invested"] += t.position_size_usd
        s["volume"] += t.event_total_volume

    lines.append(f"  {'City':<12} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'ROI%':>7} {'AvgMktVol':>10}")
    lines.append(f"  {'-'*55}")
    for city, s in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        roi = (s["pnl"] / s["invested"] * 100) if s["invested"] > 0 else 0
        avg_vol = s["volume"] / s["trades"] if s["trades"] > 0 else 0
        lines.append(
            f"  {city.upper():<12} {s['trades']:>6} {wr:>5.0%} "
            f"${s['pnl']:>+8.2f} {roi:>+6.1f}% ${avg_vol:>9,.0f}"
        )

    # ─── By Side ───
    yes_trades = [t for t in r.trades if t.side == "BUY_YES"]
    no_trades = [t for t in r.trades if t.side == "BUY_NO"]
    lines.extend([
        "",
        "  ── BY SIDE ──",
        f"  BUY_YES: {len(yes_trades)} trades, "
        f"{sum(1 for t in yes_trades if t.outcome == 'won')}/{len(yes_trades)} won"
        f" ({sum(1 for t in yes_trades if t.outcome == 'won') / len(yes_trades):.0%})"
        if yes_trades else f"  BUY_YES: 0 trades",
        f"    P&L: ${sum(t.pnl for t in yes_trades):+,.2f}"
        if yes_trades else "",
        f"  BUY_NO:  {len(no_trades)} trades, "
        f"{sum(1 for t in no_trades if t.outcome == 'won')}/{len(no_trades)} won"
        f" ({sum(1 for t in no_trades if t.outcome == 'won') / len(no_trades):.0%})"
        if no_trades else f"  BUY_NO:  0 trades",
        f"    P&L: ${sum(t.pnl for t in no_trades):+,.2f}"
        if no_trades else "",
    ])

    # ─── Edge Analysis ───
    avg_edge = sum(t.edge for t in r.trades) / len(r.trades) * 100
    avg_edge_won = (sum(t.edge for t in r.trades if t.outcome == "won") /
                    max(r.wins, 1)) * 100
    avg_edge_lost = (sum(t.edge for t in r.trades if t.outcome == "lost") /
                     max(r.losses, 1)) * 100
    lines.extend([
        "",
        "  ── EDGE ANALYSIS ──",
        f"  Avg edge (all):     {avg_edge:.1f}%",
        f"  Avg edge (winners): {avg_edge_won:.1f}%",
        f"  Avg edge (losers):  {avg_edge_lost:.1f}%",
    ])

    # ─── Daily P&L ───
    daily: dict[date, dict] = {}
    for t in r.trades:
        if t.target_date not in daily:
            daily[t.target_date] = {"pnl": 0.0, "trades": 0, "invested": 0.0}
        daily[t.target_date]["pnl"] += t.pnl
        daily[t.target_date]["trades"] += 1
        daily[t.target_date]["invested"] += t.position_size_usd

    if daily:
        daily_pnls = [d["pnl"] for d in daily.values()]
        profitable_days = sum(1 for p in daily_pnls if p > 0)
        losing_days = sum(1 for p in daily_pnls if p < 0)
        flat_days = sum(1 for p in daily_pnls if p == 0)

        # Sharpe-like ratio (daily)
        if len(daily_pnls) > 1:
            mean_daily = np.mean(daily_pnls)
            std_daily = np.std(daily_pnls)
            sharpe = (mean_daily / std_daily * np.sqrt(365)) if std_daily > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for d in sorted(daily.keys()):
            cumulative += daily[d]["pnl"]
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        lines.extend([
            "",
            "  ── DAILY STATISTICS ──",
            f"  Trading days:    {len(daily)}",
            f"  Profitable days: {profitable_days}/{len(daily)} ({profitable_days / len(daily):.0%})",
            f"  Losing days:     {losing_days}/{len(daily)} ({losing_days / len(daily):.0%})",
            f"  Best day:        ${max(daily_pnls):+,.2f}",
            f"  Worst day:       ${min(daily_pnls):+,.2f}",
            f"  Avg daily P&L:   ${np.mean(daily_pnls):+,.2f}",
            f"  Sharpe ratio:    {sharpe:.2f} (annualized)",
            f"  Max drawdown:    ${max_dd:,.2f}",
        ])

    # ─── Volume Analysis ───
    total_market_vol = sum(t.event_total_volume for t in r.trades)
    avg_bucket_vol = sum(t.bucket_volume for t in r.trades) / len(r.trades)
    lines.extend([
        "",
        "  ── MARKET VOLUME ──",
        f"  Total market vol (events traded): ${total_market_vol:,.0f}",
        f"  Avg bucket volume per trade:      ${avg_bucket_vol:,.0f}",
    ])

    # ─── Cumulative P&L by date ───
    lines.extend(["", "  ── DAILY P&L ──"])
    lines.append(f"  {'Date':<12} {'Trades':>6} {'P&L':>10} {'Cumulative':>12}")
    lines.append(f"  {'-'*44}")
    cumulative = 0.0
    for d in sorted(daily.keys()):
        cumulative += daily[d]["pnl"]
        lines.append(
            f"  {d!s:<12} {daily[d]['trades']:>6} "
            f"${daily[d]['pnl']:>+8.2f} ${cumulative:>+10.2f}"
        )

    # ─── Sample Trades ───
    lines.extend(["", "  ── SAMPLE TRADES (last 20) ──"])
    lines.append(
        f"  {'Date':<12} {'City':<10} {'Bucket':<18} {'Side':<9} "
        f"{'Model%':>6} {'Mkt%':>6} {'Edge%':>6} {'Result':>6} {'P&L':>9}"
    )
    lines.append(f"  {'-'*85}")
    for t in r.trades[-20:]:
        mark = "W" if t.outcome == "won" else "L"
        lines.append(
            f"  {t.target_date!s:<12} {t.city.upper():<10} {t.bucket_label:<18} "
            f"{t.side:<9} {t.model_probability * 100:>5.1f}% {t.market_price * 100:>5.1f}% "
            f"{t.edge * 100:>5.1f}% {mark:>6} ${t.pnl:>+8.2f}"
        )

    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)
