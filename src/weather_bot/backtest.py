"""Backtesting engine: simulate historical trading to evaluate bot accuracy.

Fetches historical weather data and simulates what the bot would have traded,
then compares against actual observed temperatures to compute real performance.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
import numpy as np
from scipy.stats import norm

from .config import CITIES, Settings
from .models import EnsembleForecast, TemperatureBucket

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A simulated trade from backtesting."""

    city: str
    target_date: date
    bucket_label: str
    bucket_lower: float
    bucket_upper: float
    is_lower_tail: bool = False
    is_upper_tail: bool = False
    side: str = ""  # BUY_YES or BUY_NO
    model_probability: float = 0.0
    market_price: float = 0.0
    edge: float = 0.0
    position_size_usd: float = 0.0
    actual_temp: float | None = None
    temp_in_bucket: bool = False
    outcome: str = "pending"  # won / lost
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Aggregated backtesting results."""

    trades: list[BacktestTrade] = field(default_factory=list)
    start_date: date = date.today()
    end_date: date = date.today()
    cities: list[str] = field(default_factory=list)
    bankroll: float = 1000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def resolved(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.outcome != "pending"]

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "won")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "lost")

    @property
    def win_rate(self) -> float:
        r = self.resolved
        return self.wins / len(r) if r else 0.0

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


async def _fetch_historical_ensemble(
    client: httpx.AsyncClient,
    city_key: str,
    target_date: date,
    model: str,
) -> EnsembleForecast | None:
    """Fetch historical ensemble forecast for a past date.

    Uses Open-Meteo's previous-day ensemble endpoint which provides
    the forecast that was available BEFORE the target date.
    """
    city = CITIES.get(city_key)
    if not city:
        return None

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"

    try:
        resp = await client.get(
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "daily": "temperature_2m_max",
                "models": model,
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "temperature_unit": temp_unit,
                "timezone": "auto",
                # Request the forecast from 1 day before target
                "past_days": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Historical ensemble %s failed for %s/%s: %s",
                      model, city_key, target_date, e)
        return None

    daily = data.get("daily", {})
    members = []
    for key, values in daily.items():
        if key.startswith("temperature_2m_max") and "member" in key:
            if values:
                # Find the value for our target date
                dates = daily.get("time", [])
                for i, d in enumerate(dates):
                    if d == target_date.isoformat() and i < len(values) and values[i] is not None:
                        members.append(float(values[i]))

    if not members:
        return None

    return EnsembleForecast(
        city=city_key,
        target_date=target_date,
        model_name=model,
        members=members,
        unit=city.unit,
    )


async def _fetch_historical_actual(
    client: httpx.AsyncClient,
    city_key: str,
    target_date: date,
) -> float | None:
    """Fetch actual observed daily high temperature for a past date."""
    city = CITIES.get(city_key)
    if not city:
        return None

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"

    try:
        resp = await client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": city.latitude,
                "longitude": city.longitude,
                "daily": "temperature_2m_max",
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
                "temperature_unit": temp_unit,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.debug("Historical actual failed for %s/%s: %s",
                      city_key, target_date, e)
        return None

    daily = data.get("daily", {})
    highs = daily.get("temperature_2m_max", [])
    if highs and highs[0] is not None:
        return float(highs[0])
    return None


def _generate_synthetic_buckets(actual_temp: float, unit: str) -> list[TemperatureBucket]:
    """Generate synthetic temperature buckets around the actual temperature.

    Mimics how Polymarket creates buckets: typically 1-degree ranges
    centered around the expected temperature.
    """
    # Round to nearest integer
    center = round(actual_temp)
    buckets = []

    if unit == "fahrenheit":
        # Fahrenheit markets typically have 2-degree ranges
        for offset in range(-5, 6):
            low = center + (offset * 2) - 1
            high = center + (offset * 2) + 1
            buckets.append(TemperatureBucket(
                label=f"{low}-{high}",
                lower=low - 0.5,
                upper=high + 0.5,
            ))
    else:
        # Celsius markets have 1-degree buckets
        for offset in range(-5, 6):
            val = center + offset
            buckets.append(TemperatureBucket(
                label=f"{val}°C",
                lower=val - 0.5,
                upper=val + 0.5,
            ))

    # Add tail buckets
    lowest = buckets[0].lower
    highest = buckets[-1].upper
    buckets.insert(0, TemperatureBucket(
        label=f"≤{int(lowest)}",
        lower=float("-inf"),
        upper=lowest,
        is_lower_tail=True,
    ))
    buckets.append(TemperatureBucket(
        label=f"≥{int(highest)}",
        lower=highest,
        upper=float("inf"),
        is_upper_tail=True,
    ))

    return buckets


def _compute_bucket_probs(
    forecasts: list[EnsembleForecast],
    buckets: list[TemperatureBucket],
) -> dict[str, float]:
    """Compute EMOS-calibrated probabilities for buckets (same as production)."""
    model_weights = {
        "gfs_seamless": 0.4,
        "ecmwf_ifs025": 0.5,
        "icon_seamless": 0.1,
    }

    combined: dict[str, float] = {b.label: 0.0 for b in buckets}
    total_w = 0.0

    for f in forecasts:
        w = model_weights.get(f.model_name, 1.0 / len(forecasts))
        members = np.array(f.members)
        mu = float(np.mean(members))
        sigma = max(float(np.std(members)) * 1.2, 0.5)

        for b in buckets:
            if b.is_lower_tail:
                p = float(norm.cdf(b.upper, loc=mu, scale=sigma))
            elif b.is_upper_tail:
                p = float(1.0 - norm.cdf(b.lower, loc=mu, scale=sigma))
            else:
                p = float(norm.cdf(b.upper, loc=mu, scale=sigma) -
                          norm.cdf(b.lower, loc=mu, scale=sigma))
            combined[b.label] += w * max(p, 0.001)
        total_w += w

    if total_w > 0:
        for k in combined:
            combined[k] /= total_w

    s = sum(combined.values())
    if s > 0:
        for k in combined:
            combined[k] /= s

    return combined


def _simulate_market_prices(
    actual_temp: float,
    buckets: list[TemperatureBucket],
) -> dict[str, float]:
    """Simulate approximate market prices based on actual temp.

    Assumes the market is somewhat efficient but has noise/mispricing.
    Uses a wider sigma than reality to simulate imperfect market pricing.
    """
    # Market uses wider distribution (less informed than our model)
    sigma = 2.5

    prices = {}
    for b in buckets:
        if b.is_lower_tail:
            p = float(norm.cdf(b.upper, loc=actual_temp, scale=sigma))
        elif b.is_upper_tail:
            p = float(1.0 - norm.cdf(b.lower, loc=actual_temp, scale=sigma))
        else:
            p = float(norm.cdf(b.upper, loc=actual_temp, scale=sigma) -
                      norm.cdf(b.lower, loc=actual_temp, scale=sigma))
        # Add noise to simulate market inefficiency
        prices[b.label] = max(0.02, min(0.98, p))

    # Normalize
    s = sum(prices.values())
    if s > 0:
        for k in prices:
            prices[k] /= s

    return prices


def _temp_in_bucket(temp: float, bucket: TemperatureBucket) -> bool:
    """Check if temperature falls within bucket."""
    if bucket.is_lower_tail:
        return temp < bucket.upper
    elif bucket.is_upper_tail:
        return temp >= bucket.lower
    else:
        return bucket.lower <= temp < bucket.upper


async def run_backtest(
    cities: list[str],
    start_date: date,
    end_date: date,
    bankroll: float = 1000.0,
    min_edge: float = 0.08,
    kelly_fraction: float = 0.25,
    max_position_pct: float = 0.02,
) -> BacktestResult:
    """Run a backtest over a date range.

    For each city and date:
    1. Fetch the ensemble forecast that was available before that date
    2. Fetch the actual observed temperature
    3. Simulate market prices (since we can't get historical Polymarket prices)
    4. Apply the same signal detection and Kelly sizing as the live bot
    5. Resolve against actual temperatures

    Note: Market prices are simulated since historical Polymarket prices aren't
    freely available. The backtest measures FORECAST ACCURACY and signal quality,
    not exact historical P&L.
    """
    result = BacktestResult(
        start_date=start_date,
        end_date=end_date,
        cities=cities,
        bankroll=bankroll,
    )

    current_date = start_date
    days_processed = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_date <= end_date:
            for city in cities:
                if city not in CITIES:
                    continue

                city_config = CITIES[city]

                # Fetch actual temperature
                actual = await _fetch_historical_actual(client, city, current_date)
                if actual is None:
                    logger.debug("No actual temp for %s/%s", city, current_date)
                    continue

                # Fetch ensemble forecasts (what the bot would have had)
                forecasts = []
                for model in ["gfs_seamless", "ecmwf_ifs025"]:
                    f = await _fetch_historical_ensemble(
                        client, city, current_date, model
                    )
                    if f:
                        forecasts.append(f)

                if not forecasts:
                    logger.debug("No forecasts for %s/%s", city, current_date)
                    continue

                # Generate buckets and compute probabilities
                buckets = _generate_synthetic_buckets(actual, city_config.unit)
                model_probs = _compute_bucket_probs(forecasts, buckets)
                market_prices = _simulate_market_prices(actual, buckets)

                # Detect signals (same logic as live bot)
                for bucket in buckets:
                    model_p = model_probs.get(bucket.label, 0)
                    market_p = market_prices.get(bucket.label, 0.5)

                    edge_yes = model_p - market_p
                    edge_no = (1 - model_p) - (1 - market_p)

                    # Pick best side
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

                    if price <= 0 or price >= 1:
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
                    in_bucket = _temp_in_bucket(actual, bucket)

                    if side == "BUY_YES":
                        won = in_bucket
                    else:
                        won = not in_bucket

                    payout = shares * 1.0 if won else 0.0
                    pnl = payout - pos_size if won else -pos_size

                    trade = BacktestTrade(
                        city=city,
                        target_date=current_date,
                        bucket_label=bucket.label,
                        bucket_lower=bucket.lower,
                        bucket_upper=bucket.upper,
                        is_lower_tail=bucket.is_lower_tail,
                        is_upper_tail=bucket.is_upper_tail,
                        side=side,
                        model_probability=model_p,
                        market_price=market_p,
                        edge=edge,
                        position_size_usd=round(pos_size, 2),
                        actual_temp=actual,
                        temp_in_bucket=in_bucket,
                        outcome="won" if won else "lost",
                        pnl=round(pnl, 2),
                    )
                    result.trades.append(trade)

            days_processed += 1
            if days_processed % 5 == 0:
                logger.info("Backtest progress: %d/%d days",
                            days_processed,
                            (end_date - start_date).days + 1)

            current_date += timedelta(days=1)

    return result


def format_backtest_report(r: BacktestResult) -> str:
    """Format a backtest result as a readable report."""
    lines = [
        "=" * 60,
        "  BACKTEST REPORT",
        f"  Period: {r.start_date} to {r.end_date}",
        f"  Cities: {', '.join(c.upper() for c in r.cities)}",
        f"  Bankroll: ${r.bankroll:,.2f}",
        "=" * 60,
        "",
    ]

    if not r.trades:
        lines.append("  No trades generated during this period.")
        return "\n".join(lines)

    # Summary
    lines.extend([
        f"  Total trades:   {r.total_trades}",
        f"  Wins / Losses:  {r.wins} / {r.losses}",
        f"  Win rate:       {r.win_rate:.1%}",
        "",
        f"  Total invested: ${r.total_invested:,.2f}",
        f"  Total P&L:      ${r.total_pnl:+,.2f}",
        f"  ROI:            {r.roi_pct:+.1f}%",
        "",
    ])

    # By city
    city_stats: dict[str, dict] = {}
    for t in r.trades:
        if t.city not in city_stats:
            city_stats[t.city] = {"trades": 0, "wins": 0, "pnl": 0.0, "invested": 0.0}
        city_stats[t.city]["trades"] += 1
        if t.outcome == "won":
            city_stats[t.city]["wins"] += 1
        city_stats[t.city]["pnl"] += t.pnl
        city_stats[t.city]["invested"] += t.position_size_usd

    lines.append(f"  {'City':<12} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'ROI%':>7}")
    lines.append(f"  {'-'*44}")
    for city, s in sorted(city_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr = s["wins"] / s["trades"] if s["trades"] > 0 else 0
        roi = (s["pnl"] / s["invested"] * 100) if s["invested"] > 0 else 0
        lines.append(
            f"  {city.upper():<12} {s['trades']:>6} {wr:>5.0%} "
            f"${s['pnl']:>+8.2f} {roi:>+6.1f}%"
        )

    # By side
    yes_trades = [t for t in r.trades if t.side == "BUY_YES"]
    no_trades = [t for t in r.trades if t.side == "BUY_NO"]
    lines.extend([
        "",
        "  — By Side —",
        f"  BUY_YES: {len(yes_trades)} trades, "
        f"{sum(1 for t in yes_trades if t.outcome == 'won')}/{len(yes_trades)} won, "
        f"P&L ${sum(t.pnl for t in yes_trades):+.2f}",
        f"  BUY_NO:  {len(no_trades)} trades, "
        f"{sum(1 for t in no_trades if t.outcome == 'won')}/{len(no_trades)} won, "
        f"P&L ${sum(t.pnl for t in no_trades):+.2f}",
    ])

    # Avg edge and calibration
    avg_edge = sum(t.edge for t in r.trades) / len(r.trades) * 100 if r.trades else 0
    lines.extend([
        "",
        f"  Avg edge:       {avg_edge:.1f}%",
    ])

    # Daily P&L
    daily: dict[date, float] = {}
    for t in r.trades:
        daily.setdefault(t.target_date, 0.0)
        daily[t.target_date] += t.pnl

    if daily:
        daily_pnls = list(daily.values())
        profitable_days = sum(1 for p in daily_pnls if p > 0)
        lines.extend([
            f"  Trading days:   {len(daily)}",
            f"  Profitable days: {profitable_days}/{len(daily)} ({profitable_days / len(daily):.0%})",
            f"  Best day:       ${max(daily_pnls):+.2f}",
            f"  Worst day:      ${min(daily_pnls):+.2f}",
        ])

    # Sample trades
    lines.extend(["", "  — Sample Trades (last 10) —"])
    lines.append(f"  {'Date':<12} {'City':<8} {'Bucket':<10} {'Side':<9} {'Edge%':>6} {'Actual':>7} {'Result':>6} {'P&L':>8}")
    lines.append(f"  {'-'*70}")
    for t in r.trades[-10:]:
        emoji = "W" if t.outcome == "won" else "L"
        lines.append(
            f"  {t.target_date!s:<12} {t.city.upper():<8} {t.bucket_label:<10} "
            f"{t.side:<9} {t.edge * 100:>5.1f}% "
            f"{t.actual_temp:>6.1f}° {emoji:>6} ${t.pnl:>+7.2f}"
        )

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
