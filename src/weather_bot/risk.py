"""Risk management for the trading bot."""

from __future__ import annotations

import logging

from .config import Settings
from .models import PortfolioState, Signal

logger = logging.getLogger(__name__)


def check_portfolio_limits(
    portfolio: PortfolioState,
    settings: Settings,
) -> bool:
    """Check if portfolio-level risk limits allow new trades.

    Returns True if trading is allowed, False if limits are breached.
    """
    # Daily loss limit
    if portfolio.bankroll > 0:
        daily_loss_pct = -portfolio.daily_pnl / portfolio.bankroll
        if daily_loss_pct >= settings.daily_loss_limit_pct:
            logger.warning(
                "Daily loss limit reached: %.1f%% (limit: %.1f%%)",
                daily_loss_pct * 100,
                settings.daily_loss_limit_pct * 100,
            )
            return False

    # Max drawdown
    if portfolio.drawdown >= settings.max_drawdown_pct:
        logger.warning(
            "Max drawdown reached: %.1f%% (limit: %.1f%%)",
            portfolio.drawdown * 100,
            settings.max_drawdown_pct * 100,
        )
        return False

    return True


def filter_correlated_signals(
    signals: list[Signal],
    max_same_day: int = 20,
    max_same_city: int = 10,
) -> list[Signal]:
    """Filter signals to limit correlated exposure.

    - Max N positions for markets resolving on the same day
    - Max M positions for the same city

    Limits are generous because verified signals are high-confidence
    and we want to bet on every bucket where we expect to win.
    """
    filtered: list[Signal] = []
    day_counts: dict[str, int] = {}
    city_counts: dict[str, int] = {}

    for signal in signals:
        day_key = signal.outcome.target_date.isoformat()
        city_key = signal.outcome.city

        if day_counts.get(day_key, 0) >= max_same_day:
            logger.debug("Skipping %s: same-day limit reached", signal.outcome.question[:40])
            continue

        if city_counts.get(city_key, 0) >= max_same_city:
            logger.debug("Skipping %s: same-city limit reached", signal.outcome.question[:40])
            continue

        filtered.append(signal)
        day_counts[day_key] = day_counts.get(day_key, 0) + 1
        city_counts[city_key] = city_counts.get(city_key, 0) + 1

    return filtered


def adjust_for_drawdown(
    signals: list[Signal],
    portfolio: PortfolioState,
    settings: Settings,
) -> list[Signal]:
    """Reduce position sizes when in drawdown.

    At 50% of max drawdown, reduce sizes by 25%.
    At 75% of max drawdown, reduce sizes by 50%.
    """
    if settings.max_drawdown_pct <= 0:
        return signals

    drawdown_ratio = portfolio.drawdown / settings.max_drawdown_pct

    if drawdown_ratio < 0.5:
        return signals

    if drawdown_ratio < 0.75:
        scale = 0.75
    else:
        scale = 0.50

    logger.info("Drawdown at %.0f%% of limit, scaling positions to %.0f%%",
                drawdown_ratio * 100, scale * 100)

    adjusted = []
    for signal in signals:
        adjusted_signal = Signal(
            outcome=signal.outcome,
            model_probability=signal.model_probability,
            market_probability=signal.market_probability,
            edge=signal.edge,
            side=signal.side,
            kelly_fraction=signal.kelly_fraction * scale,
            position_size_usd=round(signal.position_size_usd * scale, 2),
            confidence=signal.confidence,
        )
        if adjusted_signal.position_size_usd >= 1.0:
            adjusted.append(adjusted_signal)

    return adjusted


def apply_risk_controls(
    signals: list[Signal],
    portfolio: PortfolioState,
    settings: Settings,
) -> list[Signal]:
    """Apply all risk controls to a list of signals.

    1. Check portfolio limits (may halt all trading)
    2. Filter correlated signals
    3. Adjust sizes for drawdown
    4. Cap total exposure
    """
    if not check_portfolio_limits(portfolio, settings):
        logger.warning("Portfolio limits breached, no new trades allowed")
        return []

    signals = filter_correlated_signals(signals)
    signals = adjust_for_drawdown(signals, portfolio, settings)

    # Cap total new exposure at 40% of bankroll (daily budget)
    max_total_exposure = portfolio.bankroll * 0.40
    total_exposure = 0.0
    capped: list[Signal] = []

    for signal in signals:
        if total_exposure + signal.position_size_usd > max_total_exposure:
            remaining = max_total_exposure - total_exposure
            if remaining >= 1.0:
                signal = Signal(
                    outcome=signal.outcome,
                    model_probability=signal.model_probability,
                    market_probability=signal.market_probability,
                    edge=signal.edge,
                    side=signal.side,
                    kelly_fraction=signal.kelly_fraction,
                    position_size_usd=round(remaining, 2),
                    confidence=signal.confidence,
                )
                capped.append(signal)
            break
        capped.append(signal)
        total_exposure += signal.position_size_usd

    return capped
