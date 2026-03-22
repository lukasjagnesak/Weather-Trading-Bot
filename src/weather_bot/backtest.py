"""Backtesting engine: simulate historical trading to evaluate bot accuracy.

Fetches historical weather forecasts (deterministic models) and actual observed
temperatures, simulates trading signals, and resolves against real outcomes.

Strategy: For each past day, we fetch what multiple NWP models predicted
(GFS, ECMWF, ICON) and the actual observed high. We create synthetic ensemble
forecasts from model disagreement, generate market-like buckets, simulate
market prices, and apply the same EMOS/Kelly logic as the live bot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
import numpy as np
from scipy.stats import norm

from .config import CITIES
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


async def _fetch_historical_data(
    client: httpx.AsyncClient,
    city_key: str,
    start_date: date,
    end_date: date,
) -> dict[date, dict]:
    """Fetch historical model forecasts and actual temps for a date range.

    Uses Open-Meteo archive API to get:
    - Actual observed daily max temperature
    - Individual NWP model forecasts (GFS, ECMWF, ICON deterministic)

    Returns dict[date] -> {"actual": float, "models": {name: temp}}
    """
    city = CITIES.get(city_key)
    if not city:
        return {}

    temp_unit = "fahrenheit" if city.unit == "fahrenheit" else "celsius"
    results: dict[date, dict] = {}

    # 1) Fetch actual observed temperatures (archive API)
    try:
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
                results[d] = {"actual": float(highs[i]), "models": {}}
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch archive for %s: %s", city_key, e)
        return {}

    # 2) Fetch model forecasts via the forecast API
    #    Use past_days + forecast_days=0 to get recent historical model data
    past_days_count = (date.today() - start_date).days + 1
    models = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
    for model in models:
        try:
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
                if d in results and i < len(highs) and highs[i] is not None:
                    results[d]["models"][model] = float(highs[i])
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Model %s forecast failed for %s: %s", model, city_key, e)

    return results


def _models_to_ensemble(
    model_temps: dict[str, float],
    city_unit: str,
) -> list[EnsembleForecast]:
    """Convert deterministic model forecasts to synthetic ensemble forecasts.

    Each model's single forecast is expanded into synthetic ensemble members
    using typical forecast uncertainty (±1-2 degrees spread).
    """
    forecasts = []
    rng = np.random.RandomState(42)  # fixed seed for reproducibility

    # Typical forecast uncertainty in degrees
    spread = 1.5 if city_unit == "celsius" else 2.5

    for model_name, temp in model_temps.items():
        # Create synthetic members around the model's forecast
        members = [temp + rng.normal(0, spread) for _ in range(31)]
        forecasts.append(EnsembleForecast(
            city="",
            target_date=date.today(),
            model_name=model_name,
            members=members,
            unit=city_unit,
        ))

    return forecasts


def _generate_synthetic_buckets(actual_temp: float, unit: str) -> list[TemperatureBucket]:
    """Generate synthetic temperature buckets around the actual temperature.

    Mimics how Polymarket creates buckets: typically 1-degree ranges
    centered around the expected temperature.
    """
    center = round(actual_temp)
    buckets = []

    if unit == "fahrenheit":
        for offset in range(-5, 6):
            low = center + (offset * 2) - 1
            high = center + (offset * 2) + 1
            buckets.append(TemperatureBucket(
                label=f"{low}-{high}",
                lower=low - 0.5,
                upper=high + 0.5,
            ))
    else:
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
    model_temps: dict[str, float],
    buckets: list[TemperatureBucket],
    noise_sigma: float = 1.5,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate market prices as if set by less-informed traders.

    The market is modeled as:
    - Centered on the model consensus, but with a random offset (±1-2°)
      representing the market's slightly different information
    - Much wider sigma (3x our model's) — less confident
    This creates realistic edge opportunities where our sharper model
    has genuinely different probabilities than the market.
    """
    rng = np.random.RandomState(seed)
    mean_temp = sum(model_temps.values()) / len(model_temps)

    # Market center is offset from model mean (simulates different info sources)
    market_center = mean_temp + rng.normal(0, 1.5)
    # Market is much less precise than our ensemble model
    market_sigma = max(noise_sigma * 3.0, 3.5)

    prices = {}
    for b in buckets:
        if b.is_lower_tail:
            p = float(norm.cdf(b.upper, loc=market_center, scale=market_sigma))
        elif b.is_upper_tail:
            p = float(1.0 - norm.cdf(b.lower, loc=market_center, scale=market_sigma))
        else:
            p = float(norm.cdf(b.upper, loc=market_center, scale=market_sigma) -
                      norm.cdf(b.lower, loc=market_center, scale=market_sigma))
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
    1. Fetch deterministic model forecasts (GFS, ECMWF, ICON)
    2. Fetch the actual observed temperature
    3. Create synthetic ensemble from model forecasts
    4. Simulate market prices (wider uncertainty than our model)
    5. Apply the same signal detection and Kelly sizing as the live bot
    6. Resolve against actual temperatures

    Note: Market prices are simulated since historical Polymarket prices aren't
    freely available. The backtest measures FORECAST ACCURACY and signal quality.
    """
    result = BacktestResult(
        start_date=start_date,
        end_date=end_date,
        cities=cities,
        bankroll=bankroll,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        for city in cities:
            if city not in CITIES:
                logger.warning("Unknown city: %s", city)
                continue

            city_config = CITIES[city]
            logger.info("Fetching historical data for %s (%s to %s)...",
                        city.upper(), start_date, end_date)

            # Batch fetch all dates for this city
            historical = await _fetch_historical_data(
                client, city, start_date, end_date
            )

            if not historical:
                logger.warning("No historical data for %s", city)
                continue

            logger.info("Got data for %d days for %s", len(historical), city.upper())

            for target_date, day_data in sorted(historical.items()):
                actual = day_data["actual"]
                model_temps = day_data["models"]

                if not model_temps:
                    logger.debug("No model forecasts for %s/%s", city, target_date)
                    continue

                # Create synthetic ensemble from model forecasts
                forecasts = _models_to_ensemble(model_temps, city_config.unit)

                # Generate buckets centered around model consensus
                model_mean = sum(model_temps.values()) / len(model_temps)
                buckets = _generate_synthetic_buckets(model_mean, city_config.unit)

                # Compute probabilities using our model
                model_probs = _compute_bucket_probs(forecasts, buckets)

                # Simulate market prices (wider distribution = less precise market)
                spread = np.std(list(model_temps.values())) if len(model_temps) > 1 else 1.5
                # Use a deterministic seed based on city+date for reproducibility
                seed = hash((city, target_date.isoformat())) % (2**31)
                market_prices = _simulate_market_prices(
                    model_temps, buckets, noise_sigma=max(spread, 1.0), seed=seed
                )

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
                        target_date=target_date,
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

    # Forecast accuracy: how often did the model correctly identify the bucket?
    correct_bucket = sum(1 for t in r.trades if t.side == "BUY_YES" and t.temp_in_bucket)
    yes_count = len(yes_trades)
    no_correct = sum(1 for t in r.trades if t.side == "BUY_NO" and not t.temp_in_bucket)
    no_count = len(no_trades)
    lines.extend([
        "",
        "  — Forecast Accuracy —",
        f"  BUY_YES correct: {correct_bucket}/{yes_count}" +
        (f" ({correct_bucket / yes_count:.0%})" if yes_count else ""),
        f"  BUY_NO correct:  {no_correct}/{no_count}" +
        (f" ({no_correct / no_count:.0%})" if no_count else ""),
    ])

    # Avg edge
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
    lines.extend(["", "  — Sample Trades (last 15) —"])
    lines.append(f"  {'Date':<12} {'City':<8} {'Bucket':<10} {'Side':<9} {'Edge%':>6} {'Actual':>7} {'Result':>6} {'P&L':>8}")
    lines.append(f"  {'-'*70}")
    for t in r.trades[-15:]:
        emoji = "W" if t.outcome == "won" else "L"
        lines.append(
            f"  {t.target_date!s:<12} {t.city.upper():<8} {t.bucket_label:<10} "
            f"{t.side:<9} {t.edge * 100:>5.1f}% "
            f"{t.actual_temp:>6.1f}° {emoji:>6} ${t.pnl:>+7.2f}"
        )

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
