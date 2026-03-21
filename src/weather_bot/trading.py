"""Trading engine: edge detection, Kelly sizing, and order execution."""

from __future__ import annotations

import logging
from datetime import date

from .config import Settings
from .models import EnsembleForecast, MarketOutcome, PortfolioState, Signal
from .probability import compute_bucket_probabilities, ensemble_confidence

logger = logging.getLogger(__name__)


def detect_signals(
    outcomes: list[MarketOutcome],
    forecasts_by_key: dict[tuple[str, date], list[EnsembleForecast]],
    settings: Settings,
    portfolio: PortfolioState,
) -> list[Signal]:
    """Scan all market outcomes and generate trading signals where edge exists.

    For each market outcome:
    1. Look up the ensemble forecasts for the same city/date
    2. Compute model probability for the temperature bucket
    3. Compare to market price
    4. If edge > threshold, compute Kelly-sized position
    """
    signals: list[Signal] = []

    # Group outcomes by (city, date) to compute probabilities once per group
    grouped: dict[tuple[str, date], list[MarketOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.city, outcome.target_date)
        grouped.setdefault(key, []).append(outcome)

    for (city, target_date), city_outcomes in grouped.items():
        forecast_key = (city, target_date)
        forecast_list = forecasts_by_key.get(forecast_key)
        if not forecast_list:
            logger.debug("No forecasts for %s/%s, skipping", city, target_date)
            continue

        # Compute probabilities for all buckets at once
        buckets = [o.bucket for o in city_outcomes]
        model_probs = compute_bucket_probabilities(forecast_list, buckets)
        confidence = ensemble_confidence(forecast_list)

        for outcome in city_outcomes:
            # Skip markets with no real price discovery
            # Exact 0.500 means no orders have been matched yet
            if outcome.current_price_yes == 0.5 and outcome.current_price_no == 0.5:
                continue
            if outcome.volume < 100:
                continue

            model_prob = model_probs.get(outcome.bucket.label, 0.0)
            market_prob = outcome.current_price_yes

            # Calculate edge for both sides
            edge_yes = model_prob - market_prob
            edge_no = (1.0 - model_prob) - outcome.current_price_no

            # Pick the side with the larger edge
            if edge_yes >= edge_no and edge_yes > 0:
                edge = edge_yes
                side = "BUY_YES"
                effective_price = market_prob
            elif edge_no > 0:
                edge = edge_no
                side = "BUY_NO"
                effective_price = outcome.current_price_no
            else:
                continue  # no edge

            # Check minimum edge threshold
            if edge < settings.min_edge_threshold:
                continue

            # Kelly criterion for position sizing
            kelly_f = _kelly_fraction(
                model_prob if side == "BUY_YES" else 1.0 - model_prob,
                effective_price,
            )
            if kelly_f <= 0:
                continue

            # Apply fractional Kelly and confidence scaling
            adjusted_kelly = kelly_f * settings.kelly_fraction * confidence

            # Position size in USD
            position_size = adjusted_kelly * portfolio.bankroll

            # Cap at max position size
            max_position = settings.max_position_pct * portfolio.bankroll
            position_size = min(position_size, max_position)

            # Minimum viable trade size
            if position_size < 1.0:
                continue

            signal = Signal(
                outcome=outcome,
                model_probability=model_prob,
                market_probability=market_prob,
                edge=edge,
                side=side,
                kelly_fraction=adjusted_kelly,
                position_size_usd=round(position_size, 2),
                confidence=confidence,
            )
            signals.append(signal)

            logger.info(
                "Signal: %s %s | %s on %s | model=%.1f%% market=%.1f%% edge=%.1f%% size=$%.2f",
                side,
                outcome.bucket.label,
                city,
                target_date,
                model_prob * 100,
                market_prob * 100,
                edge * 100,
                position_size,
            )

    # Sort by edge (highest first)
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def _kelly_fraction(true_prob: float, market_price: float) -> float:
    """Compute Kelly fraction for a binary prediction market bet.

    For buying YES at price c with true probability p:
        f* = (p - c) / (1 - c)

    For buying NO at price (1-c) with true probability (1-p):
        f* = ((1-p) - (1-c)) / c = (c - p) / c

    This is simplified because prediction markets always pay $1 on correct outcome.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    f = (true_prob - market_price) / (1.0 - market_price)
    return max(f, 0.0)


async def execute_signal(
    signal: Signal,
    settings: Settings,
    dry_run: bool = True,
) -> bool:
    """Execute a trading signal on Polymarket.

    In paper mode (dry_run=True), just logs the trade.
    In live mode, uses py-clob-client to place the order.
    """
    if dry_run or settings.trading_mode == "paper":
        logger.info(
            "PAPER TRADE: %s %s @ $%.3f | size=$%.2f | edge=%.1f%%",
            signal.side,
            signal.outcome.question[:60],
            signal.market_probability,
            signal.position_size_usd,
            signal.edge * 100,
        )
        return True

    # Live trading
    if not settings.polymarket_private_key:
        logger.error("Cannot execute live trade: no private key configured")
        return False

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        client = ClobClient(
            settings.polymarket_clob_url,
            key=settings.polymarket_private_key,
            chain_id=137,
            signature_type=settings.polymarket_signature_type,
            funder=settings.polymarket_funder_address,
        )
        client.set_api_creds(client.create_or_derive_api_creds())

        # Determine token and price
        if signal.side == "BUY_YES":
            token_id = signal.outcome.token_id_yes
            price = signal.outcome.best_ask
        else:
            token_id = signal.outcome.token_id_no
            price = signal.outcome.current_price_no

        # Calculate number of shares
        size = signal.position_size_usd / price if price > 0 else 0
        if size < 1:
            logger.warning("Trade size too small: %.2f shares", size)
            return False

        order = OrderArgs(
            token_id=token_id,
            price=round(price, 2),
            size=round(size, 1),
            side=BUY,
        )
        signed_order = client.create_order(order)
        response = client.post_order(signed_order, OrderType.GTC)

        logger.info(
            "LIVE TRADE executed: %s %s | %.1f shares @ $%.3f | response: %s",
            signal.side,
            signal.outcome.question[:60],
            size,
            price,
            response,
        )
        return True

    except ImportError:
        logger.error("py-clob-client not installed. Run: pip install py-clob-client")
        return False
    except Exception as e:
        logger.error("Trade execution failed: %s", e)
        return False
