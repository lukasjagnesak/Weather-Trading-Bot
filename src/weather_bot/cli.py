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


if __name__ == "__main__":
    main()
