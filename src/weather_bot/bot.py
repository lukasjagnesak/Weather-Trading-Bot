"""Main bot loop: orchestrates the full trading pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

from .config import Settings
from .evaluation import (
    get_existing_trade_keys,
    get_performance_metrics,
    record_trade,
    resolve_pending_trades,
)
from .market import fetch_active_temperature_markets
from .models import PortfolioState
from .risk import apply_risk_controls
from .telegram import format_daily_report, format_scan_summary, format_trade_alert, format_resolution_alert, send_telegram
from .trading import detect_signals, execute_signal
from .latency import detect_forecast_shifts, get_active_model_updates, is_model_update_window
from .verification import fetch_current_observed_high, verify_all_cities
from .weather import fetch_all_cities

logger = logging.getLogger(__name__)


def _fetch_onchain_balance(settings: Settings) -> float | None:
    """Fetch actual USDC collateral balance from Polymarket.

    Returns the balance in USD, or None if unavailable (paper mode, no key, etc.)
    """
    if settings.trading_mode != "live" or not settings.polymarket_private_key:
        return None

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import BalanceAllowanceParams

        client = ClobClient(
            settings.polymarket_clob_url,
            key=settings.polymarket_private_key,
            chain_id=137,
            signature_type=settings.polymarket_signature_type,
            funder=settings.polymarket_funder_address,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        params = BalanceAllowanceParams(asset_type="COLLATERAL")
        bal = client.get_balance_allowance(params)
        collateral = int(bal.get("balance", "0")) / 1_000_000
        return collateral
    except Exception as e:
        logger.warning("Failed to fetch on-chain balance: %s", e)
        return None


async def _fetch_portfolio_value(settings: Settings) -> tuple[float, float] | None:
    """Fetch cash balance and total portfolio value (cash + open positions).

    Returns (cash, portfolio_value) or None if unavailable.
    """
    cash = _fetch_onchain_balance(settings)
    if cash is None:
        return None

    # Fetch open positions value using our funder address
    positions_value = 0.0
    if settings.polymarket_funder_address:
        try:
            from .copytrading import fetch_wallet_positions
            positions = await fetch_wallet_positions(settings.polymarket_funder_address)
            for pos in positions:
                # Position value = shares * current price
                positions_value += pos.size * pos.current_price
        except Exception as e:
            logger.debug("Failed to fetch position values: %s", e)

    portfolio_value = cash + positions_value
    logger.info(
        "PORTFOLIO: cash=$%.2f + positions=$%.2f = total=$%.2f",
        cash, positions_value, portfolio_value,
    )
    return cash, portfolio_value


async def run_scan(
    settings: Settings,
    portfolio: PortfolioState,
) -> list[dict]:
    """Run a single scan cycle: fetch data, find edges, execute trades.

    Returns a list of trade result dicts for reporting.
    """
    today = date.today()
    target_dates = [today + timedelta(days=i) for i in range(2)]

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
    # During model update windows, force fresh data to catch forecast shifts
    in_update_window = is_model_update_window()
    if in_update_window:
        active_models = get_active_model_updates()
        logger.info("MODEL UPDATE WINDOW — forcing fresh fetch (models updating: %s)",
                     ", ".join(active_models))

    logger.info("Fetching ensemble forecasts for %d city-date combinations...", len(needed_keys))
    forecasts = await fetch_all_cities(
        city_keys=city_keys,
        target_dates=dates,
        models=settings.ensemble_models,
        force_refresh=in_update_window and settings.latency_arb_enabled,
    )
    logger.info("Got forecasts for %d city-date combinations", len(forecasts))

    if not forecasts:
        logger.warning("No forecast data available — skipping this scan")
        return []

    # Step 3a: Detect forecast shifts for latency arbitrage
    forecast_shifts: dict[tuple[str, date], float] = {}
    if settings.latency_arb_enabled:
        forecast_shifts = detect_forecast_shifts(forecasts)
        significant = {k: v for k, v in forecast_shifts.items()
                       if abs(v) >= settings.latency_arb_min_shift}
        if significant:
            for (city, d), shift in significant.items():
                logger.info("LATENCY OPPORTUNITY: %s/%s shifted %+.1f°", city, d, shift)

    # Step 3b: Cross-validate with multi-source deterministic forecasts
    logger.info("Fetching multi-source verification forecasts...")
    verified = await verify_all_cities(city_keys, dates)
    logger.info("Got verified forecasts for %d city-date combinations", len(verified))

    # Step 3c: Fetch real-time observed highs for today's certainty bets
    observed_highs: dict[str, float] = {}
    if settings.certainty_enabled:
        logger.info("Fetching real-time observed temperatures for certainty strategy...")
        for ck in city_keys:
            try:
                obs = await fetch_current_observed_high(ck)
                if obs is not None:
                    observed_highs[ck] = obs
            except Exception as e:
                logger.debug("Failed to fetch observed high for %s: %s", ck, e)
        if observed_highs:
            logger.info("Got observed highs for %d cities: %s",
                        len(observed_highs),
                        ", ".join(f"{k}={v:.1f}" for k, v in observed_highs.items()))

    # Step 4: Detect trading signals (edge + outcome verification)
    signals = detect_signals(outcomes, forecasts, settings, portfolio, verified=verified,
                             observed_highs=observed_highs,
                             forecast_shifts=forecast_shifts)
    if not signals:
        logger.info("No trading signals found (no sufficient edge)")
        return []

    logger.info("Found %d raw signals", len(signals))

    # Step 4b: Filter out signals for markets we already bet on
    existing_keys = get_existing_trade_keys()
    before_dedup = len(signals)
    signals = [
        s for s in signals
        if (s.outcome.city, s.outcome.target_date.isoformat(),
            s.outcome.bucket.label) not in existing_keys
    ]
    if before_dedup != len(signals):
        logger.info("Filtered %d duplicate signals (already have pending trades)",
                     before_dedup - len(signals))

    if not signals:
        logger.info("No new signals after deduplication")
        return []

    # Step 5: Apply risk controls
    signals = apply_risk_controls(signals, portfolio, settings)
    if not signals:
        logger.info("All signals filtered by risk controls")
        return []

    logger.info("Executing %d signals after risk controls", len(signals))

    # Step 6: Execute trades and record to database
    dry_run = settings.trading_mode == "paper"
    results = []

    for signal in signals:
        success = await execute_signal(signal, settings, dry_run=dry_run)
        # Attach verification data if available
        vf_key = (signal.outcome.city, signal.outcome.target_date)
        vf = verified.get(vf_key)

        trade_result = {
            "city": signal.outcome.city,
            "date": signal.outcome.target_date.isoformat(),
            "bucket": signal.outcome.bucket.label,
            "side": signal.side,
            "model_prob": round(signal.model_probability * 100, 1),
            "market_prob": round(signal.market_probability * 100, 1),
            "edge": round(signal.edge * 100, 1),
            "size_usd": signal.position_size_usd,
            "executed": success,
            "confidence": signal.confidence,
            "verified_temp": round(vf.mean_high, 1) if vf else None,
            "verified_sources": vf.source_count if vf else 0,
            "verified_agreement": round(vf.agreement_score * 100) if vf else 0,
        }
        results.append(trade_result)

        # Record trade to persistent database
        if success:
            try:
                record_trade(
                    trade_result,
                    signal=signal,
                    trading_mode=settings.trading_mode,
                )
            except Exception as e:
                logger.warning("Failed to record trade: %s", e)

        # Send Telegram alert for each executed trade
        if settings.telegram_trade_alerts and success:
            try:
                await send_telegram(format_trade_alert(trade_result), settings)
            except Exception as e:
                logger.warning("Telegram trade alert failed: %s", e)

    return results


async def run_loop(settings: Settings, portfolio: PortfolioState) -> None:
    """Run the bot in continuous loop mode."""
    logger.info(
        "Starting weather trading bot (mode=%s, bankroll=$%.2f, interval=%ds)",
        settings.trading_mode,
        settings.bankroll,
        settings.scan_interval,
    )

    # Track daily report state
    last_report_date: date | None = None
    all_trades_today: list[dict] = []

    while True:
        now = datetime.utcnow()
        today = now.date()

        # Reset daily trades at midnight
        if last_report_date is not None and last_report_date != today:
            all_trades_today = []

        # Send daily report at configured hour
        if (
            settings.telegram_daily_report
            and now.hour >= settings.telegram_daily_report_hour
            and last_report_date != today
        ):
            # Resolve pending trades from past dates
            try:
                resolved = await resolve_pending_trades()
                if resolved:
                    logger.info("Resolved %d pending trades", len(resolved))
                    for r in resolved:
                        portfolio.daily_pnl += r["pnl"]
                        portfolio.bankroll += r["pnl"]
                        portfolio.peak_bankroll = max(
                            portfolio.peak_bankroll, portfolio.bankroll
                        )
                    # Send Telegram alert for each resolved trade
                    try:
                        # Fetch fresh balance for resolution alert
                        res_result = await _fetch_portfolio_value(settings)
                        res_cash = res_result[0] if res_result else None
                        res_pv = res_result[1] if res_result else None
                        msg = format_resolution_alert(
                            resolved, portfolio.bankroll,
                            cash=res_cash, portfolio_value=res_pv,
                        )
                        await send_telegram(msg, settings)
                    except Exception as e:
                        logger.warning("Telegram resolution alert failed: %s", e)
            except Exception as e:
                logger.warning("Failed to resolve pending trades: %s", e)

            try:
                from .copytrading import load_wallets

                tracked_count = len([w for w in load_wallets() if w.enabled])
            except Exception:
                tracked_count = 0

            # Get performance metrics for the report
            try:
                perf = get_performance_metrics()
            except Exception:
                perf = None

            # Fetch fresh balance for daily report
            try:
                dr_result = await _fetch_portfolio_value(settings)
                dr_cash = dr_result[0] if dr_result else None
                dr_pv = dr_result[1] if dr_result else None
            except Exception:
                dr_cash, dr_pv = None, None

            report = format_daily_report(
                trades=all_trades_today,
                bankroll=portfolio.bankroll,
                daily_pnl=portfolio.daily_pnl,
                peak_bankroll=portfolio.peak_bankroll,
                tracked_wallets=tracked_count,
                performance=perf,
                cash=dr_cash,
                portfolio_value=dr_pv,
            )
            try:
                await send_telegram(report, settings)
                logger.info("Daily report sent to Telegram")
            except Exception as e:
                logger.warning("Failed to send daily report: %s", e)

            last_report_date = today

        # Sync bankroll with on-chain balance (live mode only)
        cash_balance = None
        portfolio_value = None
        try:
            result = await _fetch_portfolio_value(settings)
            if result is not None:
                cash_balance, portfolio_value = result
                old_bankroll = portfolio.bankroll
                portfolio.bankroll = cash_balance
                portfolio.peak_bankroll = max(portfolio.peak_bankroll, portfolio_value)
                if abs(cash_balance - old_bankroll) > 0.01:
                    logger.info(
                        "BANKROLL SYNC: $%.2f → $%.2f (cash=$%.2f, positions=$%.2f)",
                        old_bankroll, cash_balance, cash_balance,
                        portfolio_value - cash_balance,
                    )
        except Exception as e:
            logger.debug("Balance sync failed: %s", e)

        # Run scan
        try:
            results = await run_scan(settings, portfolio)
            if results:
                _print_results_table(results)
                all_trades_today.extend(results)
            else:
                logger.info("Scan complete, no trades")

            # Send scan summary to Telegram
            if settings.telegram_trade_alerts:
                try:
                    scan_msg = format_scan_summary(
                        signals_count=len(results),
                        markets_count=0,
                        cities=settings.active_cities,
                        cash=cash_balance,
                        portfolio_value=portfolio_value,
                    )
                    await send_telegram(scan_msg, settings)
                except Exception as e:
                    logger.debug("Telegram scan summary failed: %s", e)
        except Exception as e:
            logger.error("Scan failed: %s", e, exc_info=True)

        # Dynamic scan interval: faster during model update windows
        if settings.latency_arb_enabled and is_model_update_window():
            interval = settings.scan_interval_model_update
            logger.info("Model update window — next scan in %d seconds", interval)
        else:
            interval = settings.scan_interval
            logger.info("Sleeping %d seconds until next scan...", interval)
        await asyncio.sleep(interval)


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
