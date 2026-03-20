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
