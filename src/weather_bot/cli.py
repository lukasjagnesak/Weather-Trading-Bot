"""CLI entry point for the weather trading bot."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from .bot import run_loop, run_scan
from .config import Settings
from .models import PortfolioState


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@click.group()
def main():
    """Polymarket Weather Trading Bot."""
    pass


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--mode", type=click.Choice(["paper", "live"]), default=None,
              help="Trading mode (overrides .env)")
@click.option("--bankroll", type=float, default=None, help="Bankroll in USD")
@click.option("--interval", type=int, default=None, help="Scan interval in seconds")
@click.option("--cities", type=str, default=None,
              help="Comma-separated city keys (e.g., nyc,london,paris)")
def run(verbose: bool, mode: str | None, bankroll: float | None,
        interval: int | None, cities: str | None):
    """Run the bot in continuous loop mode."""
    _setup_logging(verbose)
    settings = Settings()

    if mode:
        settings.trading_mode = mode
    if bankroll:
        settings.bankroll = bankroll
    if interval:
        settings.scan_interval = interval
    if cities:
        settings.active_cities = [c.strip() for c in cities.split(",")]

    portfolio = PortfolioState(
        bankroll=settings.bankroll,
        peak_bankroll=settings.bankroll,
    )

    asyncio.run(run_loop(settings, portfolio))


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--cities", type=str, default=None,
              help="Comma-separated city keys")
def scan(verbose: bool, cities: str | None):
    """Run a single scan and print results (no continuous loop)."""
    _setup_logging(verbose)
    settings = Settings()
    settings.trading_mode = "paper"  # always paper for single scan

    if cities:
        settings.active_cities = [c.strip() for c in cities.split(",")]

    portfolio = PortfolioState(
        bankroll=settings.bankroll,
        peak_bankroll=settings.bankroll,
    )

    results = asyncio.run(run_scan(settings, portfolio))

    if not results:
        click.echo("No trading signals found.")
        return

    click.echo(f"\nFound {len(results)} signal(s):\n")
    for r in results:
        marker = ">>>" if r["edge"] > 15 else "  >"
        click.echo(
            f"{marker} {r['city'].upper():<10} {r['date']}  "
            f"{r['bucket']:<12} {r['side']:<9}  "
            f"model={r['model_prob']:.1f}%  mkt={r['market_prob']:.1f}%  "
            f"edge={r['edge']:.1f}%  size=${r['size_usd']:.2f}"
        )


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--cities", type=str, default=None,
              help="Comma-separated city keys")
def markets(verbose: bool, cities: str | None):
    """List active temperature markets on Polymarket."""
    _setup_logging(verbose)

    async def _list():
        from .market import fetch_active_temperature_markets
        target_cities = [c.strip() for c in cities.split(",")] if cities else None
        outcomes = await fetch_active_temperature_markets(target_cities=target_cities)

        if not outcomes:
            click.echo("No active temperature markets found.")
            return

        click.echo(f"\nFound {len(outcomes)} active temperature market(s):\n")
        current_city = ""
        for o in sorted(outcomes, key=lambda x: (x.city, x.target_date, x.bucket.lower)):
            if o.city != current_city:
                current_city = o.city
                click.echo(f"\n  {current_city.upper()} ({o.target_date})")
                click.echo(f"  {'Bucket':<15} {'Price':>7} {'Bid':>7} {'Ask':>7} {'Volume':>10}")
                click.echo(f"  {'-'*50}")

            click.echo(
                f"  {o.bucket.label:<15} "
                f"{o.current_price_yes:>6.3f} "
                f"{o.best_bid:>6.3f} "
                f"{o.best_ask:>6.3f} "
                f"${o.volume:>9.0f}"
            )

    asyncio.run(_list())


@main.command()
@click.argument("city")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def forecast(city: str, verbose: bool):
    """Show ensemble forecast for a city."""
    _setup_logging(verbose)

    async def _show():
        from datetime import date, timedelta

        import numpy as np

        from .weather import fetch_ensemble_forecast

        city_key = city.lower()
        today = date.today()

        for day_offset in range(3):
            target = today + timedelta(days=day_offset)
            forecasts = await fetch_ensemble_forecast(city_key, target)

            if not forecasts:
                click.echo(f"  {target}: No forecast available")
                continue

            click.echo(f"\n  {city_key.upper()} - {target}")
            for f in forecasts:
                members = np.array(f.members)
                click.echo(
                    f"    {f.model_name:<20} "
                    f"mean={np.mean(members):.1f}  "
                    f"std={np.std(members):.1f}  "
                    f"min={np.min(members):.1f}  "
                    f"max={np.max(members):.1f}  "
                    f"members={len(members)}"
                )

    asyncio.run(_show())


@main.group(name="copy")
def copy_group():
    """Copy-trading: track whale wallets and mirror positions."""
    pass


@copy_group.command(name="add")
@click.argument("addresses", nargs=-1, required=True)
@click.option("--label", "-l", default="", help="Label for the wallet(s)")
def copy_add(addresses: tuple[str, ...], label: str):
    """Add wallet address(es) to the copy-trading watchlist."""
    from .copytrading import add_wallet

    for addr in addresses:
        try:
            wallet = add_wallet(addr, label=label)
            click.echo(f"  + {wallet.address}  {wallet.label or '(no label)'}")
        except ValueError as e:
            click.echo(f"  ! {e}", err=True)


@copy_group.command(name="remove")
@click.argument("address")
def copy_remove(address: str):
    """Remove a wallet from the copy-trading watchlist."""
    from .copytrading import remove_wallet

    if remove_wallet(address):
        click.echo(f"  Removed {address.lower()}")
    else:
        click.echo(f"  Wallet {address.lower()} not found", err=True)


@copy_group.command(name="list")
def copy_list():
    """List all tracked wallets."""
    from .copytrading import load_wallets

    wallets = load_wallets()
    if not wallets:
        click.echo("No wallets tracked. Add one with: weather-bot copy add <address>")
        return

    click.echo(f"\nTracked wallets ({len(wallets)}):\n")
    click.echo(f"  {'Address':<44} {'Label':<20} {'Enabled':>7}")
    click.echo(f"  {'-'*74}")
    for w in wallets:
        status = "yes" if w.enabled else "no"
        click.echo(f"  {w.address:<44} {w.label or '-':<20} {status:>7}")


@copy_group.command(name="positions")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--wallet", "-w", default=None, help="Filter by specific wallet address")
def copy_positions(verbose: bool, wallet: str | None):
    """Fetch and display positions for all tracked wallets."""
    _setup_logging(verbose)

    async def _fetch():
        from .copytrading import (
            fetch_all_tracked_positions,
            fetch_wallet_positions,
            load_wallets,
            summarize_positions,
        )

        if wallet:
            positions = await fetch_wallet_positions(wallet.lower())
            if not positions:
                click.echo(f"No positions found for {wallet[:10]}...")
                return
            click.echo(f"\nPositions for {wallet[:10]}... ({len(positions)}):\n")
            click.echo(f"  {'Market':<50} {'Side':<5} {'Size':>8} {'Avg$':>6} {'Now$':>6} {'PnL$':>8}")
            click.echo(f"  {'-'*86}")
            for p in sorted(positions, key=lambda x: abs(x.size), reverse=True):
                click.echo(
                    f"  {p.title[:48]:<50} {p.outcome:<5} "
                    f"{p.size:>8.1f} {p.avg_price:>5.3f} {p.current_price:>5.3f} "
                    f"{'%+.2f' % p.pnl:>8}"
                )
            return

        all_positions = await fetch_all_tracked_positions()
        if not all_positions or all(len(p) == 0 for p in all_positions.values()):
            click.echo("No positions found across tracked wallets.")
            return

        # Show per-wallet summary
        wallets = {w.address: w for w in load_wallets()}
        for addr, positions in all_positions.items():
            if not positions:
                continue
            w = wallets.get(addr)
            name = w.label if w and w.label else addr[:10] + "..."
            click.echo(f"\n  {name} ({len(positions)} positions):")
            for p in sorted(positions, key=lambda x: abs(x.size), reverse=True)[:10]:
                click.echo(
                    f"    {p.title[:45]:<47} {p.outcome:<4} "
                    f"size={p.size:>7.1f}  @{p.avg_price:.3f}  now={p.current_price:.3f}"
                )

        # Show consensus summary
        summary = summarize_positions(all_positions)
        consensus = [s for s in summary if s["wallet_count"] >= 2]
        if consensus:
            click.echo(f"\n  Consensus positions (held by 2+ wallets):\n")
            click.echo(f"  {'Market':<45} {'Side':<5} {'Wallets':>7} {'TotalSz':>9} {'Price':>6}")
            click.echo(f"  {'-'*75}")
            for c in consensus:
                click.echo(
                    f"  {c['title'][:43]:<45} {c['outcome']:<5} "
                    f"{c['wallet_count']:>7} {c['total_size']:>9.1f} "
                    f"{c['current_price']:>5.3f}"
                )

    asyncio.run(_fetch())


@main.command()
@click.option("--days", "-d", type=int, default=None,
              help="Limit to last N days (default: all-time)")
@click.option("--city", "-c", type=str, default=None,
              help="Filter by city key")
def stats(days: int | None, city: str | None):
    """Show trading performance statistics (win rate, P&L, ROI, etc.)."""
    from .evaluation import (
        format_performance_report,
        get_city_breakdown,
        get_daily_pnl,
        get_performance_metrics,
    )

    period = f"last {days} days" if days else "all-time"
    if city:
        period += f" ({city.upper()})"

    m = get_performance_metrics(days=days, city=city)
    click.echo(format_performance_report(m))

    # City breakdown (only if not filtering by city)
    if not city:
        breakdown = get_city_breakdown(days=days)
        if breakdown:
            click.echo(f"\n  {'City':<12} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'P&L':>10} {'ROI%':>7} {'Edge%':>7}")
            click.echo(f"  {'-'*56}")
            for c in breakdown:
                wr = f"{c['win_rate']:.0%}"
                click.echo(
                    f"  {c['city'].upper():<12} {c['trades']:>6} {c['wins']:>5} "
                    f"{wr:>6} ${c['total_pnl']:>+8.2f} {c['roi_pct']:>+6.1f}% "
                    f"{c['avg_edge']:>6.1f}%"
                )

    # Daily P&L summary (last 10 days)
    daily = get_daily_pnl(days=days or 30)
    if daily:
        click.echo(f"\n  {'Date':<12} {'Trades':>6} {'Wins':>5} {'P&L':>10} {'Cumulative':>12}")
        click.echo(f"  {'-'*48}")
        for d in daily[-10:]:
            click.echo(
                f"  {d['date']:<12} {d['trades']:>6} {d['wins']:>5} "
                f"${d['pnl']:>+8.2f} ${d['cumulative_pnl']:>+10.2f}"
            )


@main.command()
@click.option("--city", "-c", type=str, default=None, help="Filter by city key")
@click.option("--outcome", "-o", type=click.Choice(["pending", "won", "lost"]),
              default=None, help="Filter by outcome")
@click.option("--limit", "-n", type=int, default=20, help="Number of trades to show")
def trades(city: str | None, outcome: str | None, limit: int):
    """List recorded trades with their outcomes."""
    from .evaluation import get_trades

    records = get_trades(city=city, outcome=outcome, limit=limit)

    if not records:
        click.echo("No trades found.")
        return

    click.echo(f"\n  {'#':>4} {'Date':<12} {'City':<10} {'Bucket':<12} {'Side':<9} "
               f"{'Edge%':>6} {'Size$':>7} {'Result':>8} {'P&L$':>8}")
    click.echo(f"  {'-'*80}")

    for t in records:
        if t.outcome == "won":
            result = "\u2705 won"
        elif t.outcome == "lost":
            result = "\u274c lost"
        else:
            result = "\u23f3 ..."

        click.echo(
            f"  {t.id:>4} {t.target_date:<12} {t.city.upper():<10} "
            f"{t.bucket_label:<12} {t.side:<9} "
            f"{t.edge * 100:>5.1f}% ${t.position_size_usd:>6.2f} "
            f"{result:>8} ${t.pnl:>+7.2f}"
        )

    # Summary line
    total_pnl = sum(t.pnl for t in records if t.resolved)
    wins = sum(1 for t in records if t.outcome == "won")
    losses = sum(1 for t in records if t.outcome == "lost")
    pending = sum(1 for t in records if t.outcome == "pending")
    click.echo(f"\n  Total: {wins}W / {losses}L / {pending}P | P&L: ${total_pnl:+.2f}")


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def resolve(verbose: bool):
    """Resolve pending trades by fetching actual temperatures."""
    _setup_logging(verbose)

    async def _resolve():
        from .evaluation import resolve_pending_trades

        results = await resolve_pending_trades()

        if not results:
            click.echo("No pending trades to resolve (or actual temps not yet available).")
            return

        click.echo(f"\nResolved {len(results)} trade(s):\n")
        for r in results:
            emoji = "\u2705" if r["outcome"] == "won" else "\u274c"
            click.echo(
                f"  {emoji} #{r['trade_id']} {r['side']} {r['bucket']} "
                f"| actual={r['actual_temp']:.1f}\u00b0 "
                f"| P&L=${r['pnl']:+.2f}"
            )

        total_pnl = sum(r["pnl"] for r in results)
        wins = sum(1 for r in results if r["outcome"] == "won")
        click.echo(f"\n  {wins}/{len(results)} won | Total P&L: ${total_pnl:+.2f}")

    asyncio.run(_resolve())


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--cities", "-c", type=str, default=None,
              help="Comma-separated city keys (default: all)")
@click.option("--save/--no-save", default=True, help="Save calibration to file")
def calibrate(verbose: bool, cities: str | None, save: bool):
    """Calibrate EMOS model using real Polymarket resolutions.

    Learns optimal spread inflation, model bias correction, and model weights
    from 460+ resolved temperature events. Saves parameters for live trading.
    """
    _setup_logging(verbose)

    async def _run():
        from .calibration import (
            format_calibration_report,
            run_calibration,
            save_calibration,
        )

        city_list = [c.strip() for c in cities.split(",")] if cities else None

        click.echo("\nCalibrating EMOS parameters from Polymarket history...")
        if city_list:
            click.echo(f"Cities: {', '.join(c.upper() for c in city_list)}")
        click.echo("")

        result = await run_calibration(cities_filter=city_list)
        click.echo(format_calibration_report(result))

        if save:
            path = save_calibration(result)
            click.echo(f"  Saved calibration to {path}")

    asyncio.run(_run())


@main.command(name="backtest-real")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--cities", "-c", type=str, default=None,
              help="Comma-separated city keys (default: all available)")
@click.option("--bankroll", type=float, default=1000.0, help="Simulated bankroll")
@click.option("--min-edge", type=float, default=0.08, help="Minimum edge threshold")
@click.option("--kelly", type=float, default=0.25, help="Kelly fraction (0-1)")
@click.option("--max-pos", type=float, default=0.02, help="Max position % of bankroll")
def backtest_real(verbose: bool, cities: str | None, bankroll: float,
                  min_edge: float, kelly: float, max_pos: float):
    """Backtest against REAL Polymarket resolved temperature markets.

    Uses actual Polymarket bucket definitions, resolutions, and volumes.
    Market prices are estimated from volume distribution.
    """
    _setup_logging(verbose)

    async def _run():
        from .backtest_real import format_real_backtest_report, run_real_backtest

        city_list = [c.strip() for c in cities.split(",")] if cities else None

        click.echo("\nFetching resolved Polymarket temperature events...")
        click.echo(f"Bankroll: ${bankroll:.2f} | Min edge: {min_edge:.0%} | "
                   f"Kelly: {kelly:.0%} | Max position: {max_pos:.0%}")
        if city_list:
            click.echo(f"Cities: {', '.join(c.upper() for c in city_list)}")
        click.echo("")

        result = await run_real_backtest(
            bankroll=bankroll,
            min_edge=min_edge,
            kelly_fraction=kelly,
            max_position_pct=max_pos,
            cities_filter=city_list,
        )

        click.echo(format_real_backtest_report(result))

    asyncio.run(_run())


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--days", "-d", type=int, default=7,
              help="Number of days to backtest (default: 7)")
@click.option("--cities", "-c", type=str, default="nyc,london,tokyo",
              help="Comma-separated city keys")
@click.option("--bankroll", type=float, default=1000.0, help="Simulated bankroll")
@click.option("--min-edge", type=float, default=0.08, help="Minimum edge threshold")
def backtest(verbose: bool, days: int, cities: str, bankroll: float, min_edge: float):
    """Backtest the bot against historical weather data.

    Simulates what the bot would have traded over the past N days
    and shows win rate, P&L, and accuracy metrics.

    Note: Market prices are simulated. Results measure forecast accuracy,
    not exact historical returns.
    """
    _setup_logging(verbose)

    async def _backtest():
        from datetime import date, timedelta

        from .backtest import format_backtest_report, run_backtest

        city_list = [c.strip() for c in cities.split(",")]
        end = date.today() - timedelta(days=2)  # need 2 days for data availability
        start = end - timedelta(days=days - 1)

        click.echo(f"\nBacktesting {days} days ({start} to {end})")
        click.echo(f"Cities: {', '.join(c.upper() for c in city_list)}")
        click.echo(f"Bankroll: ${bankroll:.2f} | Min edge: {min_edge:.0%}")
        click.echo("Fetching historical data...\n")

        result = await run_backtest(
            cities=city_list,
            start_date=start,
            end_date=end,
            bankroll=bankroll,
            min_edge=min_edge,
        )

        click.echo(format_backtest_report(result))

    asyncio.run(_backtest())


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def preflight(verbose: bool):
    """Run pre-flight checks before live trading.

    Validates: API access, wallet config, market availability,
    forecast data, risk parameters, and Telegram setup.
    """
    _setup_logging(verbose)
    settings = Settings()
    all_ok = True

    def check(name: str, ok: bool, detail: str = ""):
        nonlocal all_ok
        status = "\u2705" if ok else "\u274c"
        msg = f"  {status} {name}"
        if detail:
            msg += f" — {detail}"
        click.echo(msg)
        if not ok:
            all_ok = False

    click.echo("\n  PRE-FLIGHT CHECKS\n  " + "=" * 40 + "\n")

    # 1. Trading mode
    click.echo(f"  Mode: {settings.trading_mode.upper()}")
    click.echo(f"  Bankroll: ${settings.bankroll:.2f}")
    click.echo("")

    # 2. Polymarket credentials
    has_key = bool(settings.polymarket_private_key)
    has_funder = bool(settings.polymarket_funder_address)
    check("Private key configured", has_key,
          "set" if has_key else "MISSING — set POLYMARKET_PRIVATE_KEY in .env")
    check("Funder address configured", has_funder,
          settings.polymarket_funder_address[:10] + "..." if has_funder
          else "MISSING — set POLYMARKET_FUNDER_ADDRESS in .env")

    # 3. py-clob-client + wallet verification
    try:
        import py_clob_client  # noqa: F401
        check("py-clob-client installed", True)
    except ImportError:
        check("py-clob-client installed", False,
              "pip install py-clob-client")

    # 3b. Verify private key matches funder address
    if has_key:
        try:
            from eth_account import Account
            acct = Account.from_key(settings.polymarket_private_key)
            derived_addr = acct.address
            funder_match = (
                derived_addr.lower() == settings.polymarket_funder_address.lower()
            )
            check(
                "Private key → address match",
                funder_match,
                f"key derives {derived_addr}"
                + (
                    ""
                    if funder_match
                    else f" but funder is {settings.polymarket_funder_address}"
                ),
            )
        except ImportError:
            check("Private key → address match", False,
                  "eth-account not installed — pip install eth-account")
        except Exception as e:
            check("Private key → address match", False, str(e))

    # 3c. Verify CLOB client connectivity and API creds
    if has_key and has_funder:
        try:
            from py_clob_client.client import ClobClient

            clob = ClobClient(
                settings.polymarket_clob_url,
                key=settings.polymarket_private_key,
                chain_id=137,
                signature_type=settings.polymarket_signature_type,
                funder=settings.polymarket_funder_address,
            )
            api_creds = clob.create_or_derive_api_creds()
            clob.set_api_creds(api_creds)
            check("CLOB API creds derived", True,
                  f"API key {api_creds.api_key[:12]}...")
        except Exception as e:
            check("CLOB API creds derived", False, str(e))

    # 4. Polymarket API access
    async def _check_api():
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{settings.polymarket_gamma_url}/events",
                                params={"tag_slug": "temperature", "limit": "1"})
                r.raise_for_status()
                events = r.json()
                check("Polymarket API reachable", True,
                      f"{len(events)} temperature event(s)")
        except Exception as e:
            check("Polymarket API reachable", False, str(e))

    asyncio.run(_check_api())

    # 5. Weather API
    async def _check_weather():
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.open-meteo.com/v1/forecast",
                                params={"latitude": 40.78, "longitude": -73.87,
                                        "current": "temperature_2m"})
                r.raise_for_status()
                temp = r.json().get("current", {}).get("temperature_2m")
                check("Open-Meteo API reachable", True, f"NYC current: {temp}°")
        except Exception as e:
            check("Open-Meteo API reachable", False, str(e))

    asyncio.run(_check_weather())

    # 6. Telegram
    has_telegram = bool(settings.telegram_bot_token and settings.telegram_chat_id)
    check("Telegram configured", has_telegram,
          "alerts + daily report" if has_telegram
          else "optional — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

    # 7. Risk parameters
    click.echo("")
    click.echo("  Risk Parameters:")
    click.echo(f"    Min edge:         {settings.min_edge_threshold:.0%}")
    click.echo(f"    Kelly fraction:   {settings.kelly_fraction:.0%}")
    click.echo(f"    Max position:     {settings.max_position_pct:.0%} of bankroll "
               f"(${settings.max_position_pct * settings.bankroll:.2f})")
    click.echo(f"    Daily loss limit: {settings.daily_loss_limit_pct:.0%} "
               f"(${settings.daily_loss_limit_pct * settings.bankroll:.2f})")
    click.echo(f"    Max drawdown:     {settings.max_drawdown_pct:.0%}")
    click.echo(f"    Scan interval:    {settings.scan_interval}s")
    click.echo(f"    Cities:           {', '.join(settings.active_cities)}")

    # 8. Trade database
    from .evaluation import _get_db_path
    db_path = _get_db_path()
    check("\n  Trade database path", True, str(db_path))

    # Summary
    click.echo("\n  " + "=" * 40)
    if settings.trading_mode == "paper":
        click.echo("  Mode is PAPER — no real money will be used.")
        click.echo("  To go live: set TRADING_MODE=live in .env")
    elif all_ok:
        click.echo("  \u2705 All checks passed — ready for LIVE trading!")
        click.echo(f"  \u26a0\ufe0f  Bankroll: ${settings.bankroll:.2f} of REAL money")
        click.echo("  Start with: weather-bot run --mode live")
    else:
        click.echo("  \u274c Some checks failed — fix issues above before going live.")
    click.echo("")


@copy_group.command(name="enable")
@click.argument("address")
def copy_enable(address: str):
    """Enable copy-trading for a wallet."""
    from .copytrading import toggle_wallet

    if toggle_wallet(address, enabled=True):
        click.echo(f"  Enabled {address.lower()}")
    else:
        click.echo(f"  Wallet not found", err=True)


@copy_group.command(name="disable")
@click.argument("address")
def copy_disable(address: str):
    """Disable copy-trading for a wallet (keep in list but don't track)."""
    from .copytrading import toggle_wallet

    if toggle_wallet(address, enabled=False):
        click.echo(f"  Disabled {address.lower()}")
    else:
        click.echo(f"  Wallet not found", err=True)


if __name__ == "__main__":
    main()
