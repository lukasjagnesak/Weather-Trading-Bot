"""Trading engine: edge detection, Kelly sizing, and order execution.

Two signal strategies:
1. EDGE-BASED: traditional mispricing detection (model prob vs market price)
2. OUTCOME-FOCUSED: when we have verified forecast data, we know the answer
   for EVERY bucket — bet YES on the correct one, NO on all others.
"""

from __future__ import annotations

import logging
from datetime import date

from .config import Settings
from .models import EnsembleForecast, MarketOutcome, PortfolioState, Signal
from .probability import compute_bucket_probabilities, ensemble_confidence
from .verification import VerifiedForecast

logger = logging.getLogger(__name__)


def detect_signals(
    outcomes: list[MarketOutcome],
    forecasts_by_key: dict[tuple[str, date], list[EnsembleForecast]],
    settings: Settings,
    portfolio: PortfolioState,
    verified: dict[tuple[str, date], VerifiedForecast] | None = None,
) -> list[Signal]:
    """Scan all market outcomes and generate trading signals.

    When verified forecast is available:
    - Correct bucket → BUY_YES
    - Every other bucket → BUY_NO
    - No minimum edge for verified picks (the verification IS the edge)

    Without verification: classic edge-based approach with min_edge_threshold.
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

        # Get verification data for this city/date
        vf = verified.get(forecast_key) if verified else None
        has_verification = vf is not None and vf.source_count >= 2

        if has_verification:
            verification_agreement = vf.agreement_score
            # Blend ensemble confidence with verification agreement
            if vf.source_count >= 3:
                confidence = 0.5 * confidence + 0.5 * verification_agreement
            logger.info(
                "Verified forecast for %s/%s: %.1f° (%d sources, spread=%.1f, agreement=%.0f%%)",
                city, target_date, vf.mean_high, vf.source_count,
                vf.spread, vf.agreement_score * 100,
            )

        for outcome in city_outcomes:
            model_prob = model_probs.get(outcome.bucket.label, 0.0)
            market_prob = outcome.current_price_yes

            # ─── VERIFIED MODE: bet on every bucket ─────────────────────
            if has_verification:
                signal = _verified_signal(
                    outcome, model_prob, market_prob, vf, confidence,
                    settings, portfolio, city, target_date,
                )
                if signal:
                    signals.append(signal)
                continue

            # ─── EDGE MODE: traditional mispricing (no verification) ────
            # Skip markets with no price discovery
            if outcome.current_price_yes == 0.5 and outcome.current_price_no == 0.5:
                continue
            if outcome.volume < 100:
                continue

            signal = _edge_signal(
                outcome, model_prob, market_prob, confidence,
                settings, portfolio, city, target_date,
            )
            if signal:
                signals.append(signal)

    # Sort by edge (highest first)
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


def _verified_signal(
    outcome: MarketOutcome,
    model_prob: float,
    market_prob: float,
    vf: VerifiedForecast,
    confidence: float,
    settings: Settings,
    portfolio: PortfolioState,
    city: str,
    target_date: date,
) -> Signal | None:
    """Generate signal for a verified bucket — always bets YES or NO.

    When we have verified temp data, we know the answer:
    - Temp is in this bucket → BUY_YES (if market underprices)
    - Temp is NOT in this bucket → BUY_NO (if market overprices YES)
    """
    verified_temp = vf.mean_high
    bucket = outcome.bucket

    in_bucket = _temp_in_bucket(verified_temp, bucket)
    distance = _distance_from_bucket(verified_temp, bucket)

    if in_bucket:
        # ─── CORRECT BUCKET: BUY_YES ─────────────────────────────────
        side = "BUY_YES"
        # True probability based on how well sources agree
        # In the correct bucket: high agreement → near-certain YES
        true_prob = min(0.95, model_prob * 0.5 + vf.agreement_score * 0.5)
        true_prob = max(true_prob, model_prob)  # never less than model says
        effective_price = market_prob
        edge = true_prob - effective_price

        if edge <= 0.01:  # market already priced correctly
            logger.debug(
                "Verified YES %s %s but no edge (model=%.0f%% market=%.0f%%)",
                city, bucket.label, true_prob * 100, market_prob * 100,
            )
            return None

        boost = 1.0 + vf.agreement_score  # 1.6x - 2.0x
        tag = "VERIFIED-YES"

    else:
        # ─── WRONG BUCKET: BUY_NO ────────────────────────────────────
        side = "BUY_NO"
        # True probability of NO = how sure we are temp is NOT here
        # Closer to bucket = less certain; far away = very certain
        # With tight agreement (low spread), even nearby buckets are safe
        distance_certainty = min(1.0, distance / max(vf.spread * 3, 3.0))
        true_prob_no = min(0.97, 0.6 + 0.37 * distance_certainty * vf.agreement_score)
        true_prob_no = max(true_prob_no, 1.0 - model_prob)  # never less than model

        effective_price = outcome.current_price_no
        edge = true_prob_no - effective_price

        if edge <= 0.01:  # market already priced correctly
            logger.debug(
                "Verified NO %s %s but no edge (prob_no=%.0f%% price_no=%.0f%%)",
                city, bucket.label, true_prob_no * 100, effective_price * 100,
            )
            return None

        # Boost scales with distance: nearby=1.0x, far=1.5x
        boost = 1.0 + min(0.5, distance / 20.0)
        true_prob = 1.0 - true_prob_no  # for Signal model_probability (prob of YES)
        tag = "VERIFIED-NO"

    # Kelly sizing
    if side == "BUY_YES":
        kelly_f = _kelly_fraction(true_prob, effective_price)
    else:
        kelly_f = _kelly_fraction(true_prob_no, effective_price)

    if kelly_f <= 0:
        return None

    adjusted_kelly = kelly_f * settings.kelly_fraction * confidence * boost

    position_size = adjusted_kelly * portfolio.bankroll
    max_pct = min(settings.max_position_pct * 2, 0.05)  # up to 5% for verified
    position_size = min(position_size, max_pct * portfolio.bankroll)

    if position_size < 1.0:
        return None

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

    logger.info(
        "[%s] %s %s | %s %s | verified=%.1f° dist=%.1f | "
        "prob=%.0f%% market=%.0f%% edge=%.1f%% $%.2f",
        tag, side, outcome.bucket.label, city, target_date,
        verified_temp, distance,
        (true_prob if side == "BUY_YES" else true_prob_no) * 100,
        effective_price * 100, edge * 100, position_size,
    )

    return signal


def _edge_signal(
    outcome: MarketOutcome,
    model_prob: float,
    market_prob: float,
    confidence: float,
    settings: Settings,
    portfolio: PortfolioState,
    city: str,
    target_date: date,
) -> Signal | None:
    """Generate signal based on ensemble edge only (no verification data)."""
    edge_yes = model_prob - market_prob
    edge_no = (1.0 - model_prob) - outcome.current_price_no

    if edge_yes >= edge_no and edge_yes > 0:
        edge = edge_yes
        side = "BUY_YES"
        effective_price = market_prob
    elif edge_no > 0:
        edge = edge_no
        side = "BUY_NO"
        effective_price = outcome.current_price_no
    else:
        return None

    if edge < settings.min_edge_threshold:
        return None

    kelly_f = _kelly_fraction(
        model_prob if side == "BUY_YES" else 1.0 - model_prob,
        effective_price,
    )
    if kelly_f <= 0:
        return None

    adjusted_kelly = kelly_f * settings.kelly_fraction * confidence
    position_size = adjusted_kelly * portfolio.bankroll
    max_position = settings.max_position_pct * portfolio.bankroll
    position_size = min(position_size, max_position)

    if position_size < 1.0:
        return None

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

    logger.info(
        "[EDGE] %s %s | %s %s | model=%.0f%% market=%.0f%% edge=%.1f%% $%.2f",
        side, outcome.bucket.label, city, target_date,
        model_prob * 100, market_prob * 100, edge * 100, position_size,
    )

    return signal


def _temp_in_bucket(temp: float, bucket) -> bool:
    """Check if a temperature falls within a bucket's range."""
    if bucket.is_lower_tail:
        return temp < bucket.upper
    elif bucket.is_upper_tail:
        return temp >= bucket.lower
    else:
        return bucket.lower <= temp < bucket.upper


def _distance_from_bucket(temp: float, bucket) -> float:
    """How far (in degrees) is the temperature from the nearest bucket edge."""
    if bucket.is_lower_tail:
        return max(0.0, temp - bucket.upper)
    elif bucket.is_upper_tail:
        return max(0.0, bucket.lower - temp)
    else:
        if temp < bucket.lower:
            return bucket.lower - temp
        elif temp >= bucket.upper:
            return temp - bucket.upper
        return 0.0  # inside bucket


def _kelly_fraction(true_prob: float, market_price: float) -> float:
    """Compute Kelly fraction for a binary prediction market bet.

    For buying YES at price c with true probability p:
        f* = (p - c) / (1 - c)

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
