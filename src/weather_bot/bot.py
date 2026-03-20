"""Main bot loop: orchestrates the full trading pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from .config import Settings
from .market import fetch_active_temperature_markets
from .models import PortfolioState
from .risk import apply_risk_controls
from .trading import detect_signals, execute_signal
from .weather import fetch_all_cities

logger = logging.getLogger(__name__)


async def run_scan(
    settings: Settings,
    portfolio: PortfolioState,
) -> list[dict]:
    """Run a single scan cycle: fetch data, find edges, execute trades.

    Returns a list of trade result dicts for reporting.
    """
    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(3)]

    # Step 1: Fetch active temperature markets from Polymarket
    logger.info("Fetching active temperature markets...")
    outcomes = await fetch_active_temperature_markets(
        target_cities=settings.active_cities,
        target_dates=target_dates,
    )
    if not outcomes:
        logger.info("No active temperature markets found")
        return []

    logger.info("Found %d tradeable outcomes across %d cities",
                len(outcomes),
                len(set(o.city for o in outcomes)))

    # Step 2: Determine which cities/dates we need forecasts for
    needed_keys = set()
    for outcome in outcomes:
        needed_keys.add((outcome.city, outcome.target_date))

    city_keys = list(set(k[0] for k in needed_keys))
    dates = list(set(k[1] for k in needed_keys))

    # Step 3: Fetch ensemble weather forecasts
    logger.info("Fetching ensemble forecasts for %d city-date combinations...", len(needed_keys))
    forecasts = await fetch_all_cities(
        city_keys=city_keys,
        target_dates=dates,
        models=settings.ensemble_models,
    )
    logger.info("Got forecasts for %d city-date combinations", len(forecasts))

    # Step 4: Detect trading signals (edge > threshold)
    signals = detect_signals(outcomes, forecasts, settings, portfolio)
    if not signals:
        logger.info("No trading signals found (no sufficient edge)")
        return []

    logger.info("Found %d raw signals", len(signals))

    # Step 5: Apply risk controls
    signals = apply_risk_controls(signals, portfolio, settings)
    if not signals:
        logger.info("All signals filtered by risk controls")
        return []

    logger.info("Executing %d signals after risk controls", len(signals))

    # Step 6: Execute trades
    dry_run = settings.trading_mode == "paper"
    results = []

    for signal in signals:
        success = await execute_signal(signal, settings, dry_run=dry_run)
        results.append({
            "city": signal.outcome.city,
            "date": signal.outcome.target_date.isoformat(),
            "bucket": signal.outcome.bucket.label,
            "side": signal.side,
            "model_prob": round(signal.model_probability * 100, 1),
            "market_prob": round(signal.market_probability * 100, 1),
            "edge": round(signal.edge * 100, 1),
            "size_usd": signal.position_size_usd,
            "executed": success,
        })

    return results


async def run_loop(settings: Settings, portfolio: PortfolioState) -> None:
    """Run the bot in continuous loop mode."""
    logger.info(
        "Starting weather trading bot (mode=%s, bankroll=$%.2f, interval=%ds)",
        settings.trading_mode,
        settings.bankroll,
        settings.scan_interval,
    )

    while True:
        try:
            results = await run_scan(settings, portfolio)
            if results:
                _print_results_table(results)
            else:
                logger.info("Scan complete, no trades")
        except Exception as e:
            logger.error("Scan failed: %s", e, exc_info=True)

        logger.info("Sleeping %d seconds until next scan...", settings.scan_interval)
        await asyncio.sleep(settings.scan_interval)


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted table of trade results."""
    header = f"{'City':<12} {'Date':<12} {'Bucket':<12} {'Side':<9} {'Model%':>7} {'Mkt%':>6} {'Edge%':>6} {'Size$':>7} {'OK':>3}"
    logger.info("\n" + header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            f"{r['city']:<12} {r['date']:<12} {r['bucket']:<12} {r['side']:<9} "
            f"{r['model_prob']:>6.1f}% {r['market_prob']:>5.1f}% {r['edge']:>5.1f}% "
            f"${r['size_usd']:>6.2f} {'✓' if r['executed'] else '✗':>3}"
        )
