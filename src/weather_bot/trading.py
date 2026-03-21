"""Trading engine: edge detection, Kelly sizing, and order execution.

Two signal strategies:
1. EDGE-BASED: traditional mispricing detection (model prob vs market price)
2. OUTCOME-FOCUSED: identify the CORRECT bucket via multi-source verification,
   then bet on it if the market price offers value
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

    Combines two approaches:
    1. Ensemble edge detection (as before)
    2. Outcome-focused: find the correct bucket via cross-source verification
       and bet BUY_YES on the peak probability bucket if market underprices it
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

        # Boost confidence if multi-source verification agrees
        if vf and vf.source_count >= 3:
            verification_agreement = vf.agreement_score
            # Blend ensemble confidence with verification agreement
            confidence = 0.5 * confidence + 0.5 * verification_agreement
            logger.info(
                "Verification for %s/%s: %d sources, mean=%.1f, spread=%.1f, agreement=%.0f%%",
                city, target_date, vf.source_count, vf.mean_high,
                vf.spread, vf.agreement_score * 100,
            )

        # Find the PEAK bucket (most likely outcome)
        peak_label = max(model_probs, key=model_probs.get) if model_probs else None
        peak_prob = model_probs.get(peak_label, 0) if peak_label else 0

        for outcome in city_outcomes:
            # Skip markets with no real price discovery
            if outcome.current_price_yes == 0.5 and outcome.current_price_no == 0.5:
                continue
            if outcome.volume < 100:
                continue

            model_prob = model_probs.get(outcome.bucket.label, 0.0)
            market_prob = outcome.current_price_yes

            # ─── Verification: determine if this bucket is YES or NO ─────
            is_verified_yes = False  # verified temp IS in this bucket
            is_verified_no = False   # verified temp is FAR from this bucket
            verified_boost = 1.0

            if vf and vf.source_count >= 2 and vf.agreement_score >= 0.6:
                verified_temp = vf.mean_high
                bucket = outcome.bucket
                spread = vf.spread

                # Check if verified temperature falls IN this bucket
                in_bucket = _temp_in_bucket(verified_temp, bucket)

                if in_bucket:
                    is_verified_yes = True
                    verified_boost = 1.0 + vf.agreement_score
                    logger.info(
                        "VERIFIED YES: %s %s — temp %.1f IN bucket %s "
                        "(agreement=%.0f%%, boost=%.1fx)",
                        city, target_date, verified_temp, bucket.label,
                        vf.agreement_score * 100, verified_boost,
                    )
                else:
                    # Check how FAR the verified temp is from this bucket
                    distance = _distance_from_bucket(verified_temp, bucket)
                    # If verified temp is > 2 std deviations (spread) away
                    # from the bucket, it's a strong NO signal
                    safe_margin = max(spread * 2, 2.0)  # at least 2 degrees
                    if distance >= safe_margin:
                        is_verified_no = True
                        # Stronger boost when temp is very far from bucket
                        verified_boost = 1.0 + min(vf.agreement_score, distance / 10.0)
                        logger.info(
                            "VERIFIED NO: %s %s — temp %.1f is %.1f away from bucket %s "
                            "(margin=%.1f, agreement=%.0f%%, boost=%.1fx)",
                            city, target_date, verified_temp, distance, bucket.label,
                            safe_margin, vf.agreement_score * 100, verified_boost,
                        )

            is_verified = is_verified_yes or is_verified_no

            # ─── Edge calculation (both sides) ──────────────────────────────
            edge_yes = model_prob - market_prob
            edge_no = (1.0 - model_prob) - outcome.current_price_no

            # Verified YES → force BUY_YES if model agrees
            if is_verified_yes and model_prob > market_prob:
                edge = edge_yes
                side = "BUY_YES"
                effective_price = market_prob
            # Verified NO → force BUY_NO if model agrees
            elif is_verified_no and (1.0 - model_prob) > outcome.current_price_no:
                edge = edge_no
                side = "BUY_NO"
                effective_price = outcome.current_price_no
            # Fallback: pick whichever side has more edge
            elif edge_yes >= edge_no and edge_yes > 0:
                edge = edge_yes
                side = "BUY_YES"
                effective_price = market_prob
            elif edge_no > 0:
                edge = edge_no
                side = "BUY_NO"
                effective_price = outcome.current_price_no
            else:
                continue  # no edge

            # Minimum edge threshold — lower for verified picks
            min_edge = settings.min_edge_threshold
            if is_verified:
                min_edge = min(min_edge, 0.05)  # 5% for verified picks

            if edge < min_edge:
                continue

            # Kelly criterion for position sizing
            kelly_f = _kelly_fraction(
                model_prob if side == "BUY_YES" else 1.0 - model_prob,
                effective_price,
            )
            if kelly_f <= 0:
                continue

            # Apply fractional Kelly, confidence, and verification boost
            adjusted_kelly = kelly_f * settings.kelly_fraction * confidence * verified_boost

            # Position size in USD
            position_size = adjusted_kelly * portfolio.bankroll

            # Cap at max position size (higher cap for verified picks)
            max_pct = settings.max_position_pct
            if is_verified:
                max_pct = min(max_pct * 2, 0.05)  # up to 5% for verified
            max_position = max_pct * portfolio.bankroll
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

            tag = "VERIFIED" if is_verified else "EDGE"
            logger.info(
                "[%s] Signal: %s %s | %s on %s | model=%.1f%% market=%.1f%% "
                "edge=%.1f%% conf=%.0f%% size=$%.2f",
                tag, side, outcome.bucket.label, city, target_date,
                model_prob * 100, market_prob * 100, edge * 100,
                confidence * 100, position_size,
            )

    # Sort by: verified picks first, then by edge
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


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
