"""Trading engine: edge detection, Kelly sizing, and order execution.

Two signal strategies:
1. EDGE-BASED: traditional mispricing detection (model prob vs market price)
2. OUTCOME-FOCUSED: when we have verified forecast data, we know the answer
   for EVERY bucket — bet YES on the correct one, NO on all others.

SAFETY: This bot MUST NEVER withdraw, transfer, or send funds to any wallet.
The only permitted on-chain action is placing BUY orders on Polymarket.
All deposits and withdrawals must be done manually by the user.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np

from .config import CITIES, Settings
from .models import EnsembleForecast, MarketOutcome, PortfolioState, Signal
from .probability import compute_bucket_probabilities, ensemble_confidence
from .verification import VerifiedForecast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAFETY: Forbidden operations — the bot must NEVER move funds out of the
# account.  This list covers known py_clob_client / on-chain methods that
# could withdraw, transfer, or approve spending of user funds.
# ---------------------------------------------------------------------------
_BLOCKED_CLIENT_METHODS = frozenset({
    "withdraw",
    "transfer",
    "transfer_from",
    "send_transaction",
    "send_raw_transaction",
    "approve",
    "cancel_all",
    "withdraw_collateral",
    "withdraw_funds",
})


class _SafeClobClient:
    """Thin wrapper that blocks any withdraw / transfer calls."""

    def __init__(self, inner):  # noqa: ANN001
        self._inner = inner

    def __getattr__(self, name: str):  # noqa: ANN204
        if name in _BLOCKED_CLIENT_METHODS:
            raise PermissionError(
                f"BLOCKED: '{name}' is forbidden — this bot must never "
                f"withdraw or transfer funds.  All withdrawals must be "
                f"done manually by the user."
            )
        return getattr(self._inner, name)


def detect_signals(
    outcomes: list[MarketOutcome],
    forecasts_by_key: dict[tuple[str, date], list[EnsembleForecast]],
    settings: Settings,
    portfolio: PortfolioState,
    verified: dict[tuple[str, date], VerifiedForecast] | None = None,
    observed_highs: dict[str, float] | None = None,
    forecast_shifts: dict[tuple[str, date], float] | None = None,
) -> list[Signal]:
    """Scan all market outcomes and generate trading signals.

    ONLY certainty strategy is active — edge strategy is disabled because
    forecast-based bets have consistently lost money (buying YES at 2-6c,
    buying wrong-side tokens, etc.)

    Certainty strategy (today only, after 3 PM local):
    - Uses real-time OBSERVED high temperature (from WU, not forecast)
    - BUY_YES on the bucket where the observed temp falls
    - BUY_NO on buckets far from the observed temp (≥5° away)
    """
    signals: list[Signal] = []

    # Group outcomes by (city, date) to compute probabilities once per group
    grouped: dict[tuple[str, date], list[MarketOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.city, outcome.target_date)
        grouped.setdefault(key, []).append(outcome)

    today = date.today()

    for (city, target_date), city_outcomes in grouped.items():
        # ── CERTAINTY ONLY: skip if not today ──
        if target_date != today:
            continue

        if not settings.certainty_enabled:
            continue

        # Check if it's past the minimum hour in local time
        city_cfg = CITIES.get(city)
        if city_cfg:
            from zoneinfo import ZoneInfo
            local_now = datetime.now(ZoneInfo(city_cfg.timezone))
            if local_now.hour < settings.certainty_min_hour:
                continue

        # REQUIRE observed high — no forecast-based bets
        observed_high = observed_highs.get(city) if observed_highs else None
        if observed_high is None:
            logger.debug("Certainty skipped for %s: no observed high", city)
            continue

        logger.info(
            "CERTAINTY scan %s/%s: observed_high=%.1f°",
            city, target_date, observed_high,
        )

        # ── Find which bucket the observed temp falls into → BUY_YES ──
        for outcome in city_outcomes:
            if _temp_in_bucket(observed_high, outcome.bucket):
                # Observed temp IS in this bucket → BUY_YES
                cert = _certainty_signal(
                    outcome, 0.0, outcome.current_price_yes,
                    "BUY_YES", 1.0, settings, portfolio, city, target_date,
                    observed_highs=observed_highs,
                )
                if cert:
                    signals.append(cert)
                break  # only one bucket can match

        # ── BUY_NO on buckets far from observed temp ──
        for outcome in city_outcomes:
            if _temp_in_bucket(observed_high, outcome.bucket):
                continue  # never bet NO on the bucket we're in

            no_price = outcome.current_price_no
            # Only bet NO in the 70-85c range
            if not (0.70 <= no_price <= settings.certainty_max_price_no):
                continue

            cert = _certainty_signal(
                outcome, 0.0, no_price,
                "BUY_NO", 1.0, settings, portfolio, city, target_date,
                observed_highs=observed_highs,
            )
            if cert:
                signals.append(cert)

    # Sort: YES bets first, then NO by edge
    signals.sort(key=lambda s: (0 if s.side == "BUY_YES" else 1, -s.edge))
    return signals


def _forecast_signal(
    outcome: MarketOutcome,
    model_prob: float,
    token_price: float,
    side: str,
    confidence: float,
    settings: Settings,
    portfolio: PortfolioState,
    city: str,
    target_date: date,
    vf: "VerifiedForecast | None" = None,
    is_latency: bool = False,
) -> Signal | None:
    """Generate a signal based on forecast probability.

    For BUY_YES: we buy the YES token → profit if bucket wins
    For BUY_NO: we buy the NO token → profit if bucket loses
    """
    MAX_PRICE = 0.95  # never pay more than 95c

    if side == "BUY_YES":
        true_prob = model_prob
        effective_price = token_price
        # Use Gaussian probability from verification if available
        if vf is not None:
            sigma = max(vf.spread, 1.5)
            true_prob = _gaussian_bucket_prob(vf.mean_high, sigma, outcome.bucket)
        edge = true_prob - effective_price
    else:  # BUY_NO
        true_prob = 1.0 - model_prob
        effective_price = token_price
        if vf is not None:
            sigma = max(vf.spread, 1.5)
            true_prob = 1.0 - _gaussian_bucket_prob(vf.mean_high, sigma, outcome.bucket)
        edge = true_prob - effective_price

    if edge <= 0 or effective_price >= MAX_PRICE or effective_price <= 0.01:
        return None

    # Minimum edge threshold — lower for latency arbitrage signals
    min_edge = settings.latency_arb_edge_threshold if is_latency else settings.min_edge_threshold
    if edge < min_edge:
        return None

    # Kelly sizing
    kelly_f = _kelly_fraction(true_prob, effective_price)
    if kelly_f <= 0:
        return None

    adjusted_kelly = kelly_f * settings.kelly_fraction * confidence
    position_size = adjusted_kelly * portfolio.bankroll
    max_position = settings.max_position_pct * portfolio.bankroll
    position_size = min(position_size, max_position)

    # Polymarket minimum is 5 shares; use best_ask (actual execution price)
    # for YES trades since that's what execute_signal pays
    exec_price = outcome.best_ask if side == "BUY_YES" else outcome.current_price_no
    min_usd = max(5.0 * exec_price, 1.0)
    if position_size < min_usd:
        return None

    if is_latency:
        tag = "LATENCY-YES" if side == "BUY_YES" else "LATENCY-NO"
    else:
        tag = "FORECAST-YES" if side == "BUY_YES" else "FORECAST-NO"

    logger.info(
        "[%s] %s %s | %s %s | prob=%.0f%% price=%.0fc edge=%.1f%% $%.2f",
        tag, side, outcome.bucket.label, city, target_date,
        true_prob * 100, effective_price * 100, edge * 100, position_size,
    )

    return Signal(
        outcome=outcome,
        model_probability=model_prob,
        market_probability=outcome.current_price_yes,
        edge=edge,
        side=side,
        kelly_fraction=adjusted_kelly,
        position_size_usd=round(position_size, 2),
        confidence=confidence,
    )



def _certainty_signal(
    outcome: MarketOutcome,
    model_prob: float,
    token_price: float,
    side: str,
    confidence: float,
    settings: Settings,
    portfolio: PortfolioState,
    city: str,
    target_date: date,
    vf: "VerifiedForecast | None" = None,
    observed_highs: dict[str, float] | None = None,
) -> Signal | None:
    """Generate a certainty-strategy signal.

    Uses the real-time observed high temperature to decide.
    All preconditions (today only, after 15:00, observed high exists)
    are checked by detect_signals() before calling this function.
    """
    # Get observed high (guaranteed to exist by caller)
    observed_high = observed_highs.get(city) if observed_highs else None
    if observed_high is None:
        return None

    # ── BUY_YES: observed temp MUST be inside the bucket ──
    # We only bet YES when we KNOW the temperature is in this range.
    if side == "BUY_YES":
        if not _temp_in_bucket(observed_high, outcome.bucket):
            logger.debug(
                "Certainty YES rejected: observed %.1f° NOT in bucket %s",
                observed_high, outcome.bucket.label,
            )
            return None

    # ── BUY_NO safety: validate observed temp is far from bucket ──
    # After 3 PM, temps can still rise 1-2° — need large safety margin.
    if side == "BUY_NO":
        distance = _distance_from_bucket(observed_high, outcome.bucket)
        in_bucket = _temp_in_bucket(observed_high, outcome.bucket)
        if in_bucket:
            return None
        if distance < 5.0:
            logger.debug(
                "Certainty NO rejected: observed %.1f° only %.1f° from bucket %s (need ≥5°)",
                observed_high, distance, outcome.bucket.label,
            )
            return None

    effective_price = token_price

    # For BUY_YES: observed temp IS in the bucket (validated by caller).
    # We know with high confidence this is the right bucket.
    # For BUY_NO: observed temp is ≥5° from bucket (validated above).
    # In both cases, we set true_prob directly based on certainty level.
    if side == "BUY_YES":
        # Temp is in bucket — after 3 PM, very likely to stay.
        # Use 90% confidence (temp can still shift ~1° but usually stays).
        true_prob = 0.90
    else:
        # Temp is ≥5° away from bucket — very unlikely to reach it.
        # Use Gaussian to compute actual probability of NOT being in bucket.
        sigma = 1.5  # conservative — allows for late afternoon shift
        bucket_prob = _gaussian_bucket_prob(observed_high, sigma, outcome.bucket)
        true_prob = 1.0 - bucket_prob

    logger.info(
        "Certainty %s %s | observed=%.1f° bucket=%s | prob=%.0f%% price=%.0fc",
        side, city, observed_high, outcome.bucket.label,
        true_prob * 100, effective_price * 100,
    )

    # Price caps
    if effective_price >= settings.certainty_max_price or effective_price <= 0.01:
        return None

    # BUY_NO: cap price at 85c (not 95c) — at higher prices the risk/reward
    # is terrible (e.g. 90c → risk 90c to win 10c = 9:1 against you).
    # Minimum 15c profit margin ensures reasonable payoff.
    if side == "BUY_NO":
        if effective_price > settings.certainty_max_price_no:
            logger.debug(
                "Certainty NO rejected: price %.0fc too high (max %.0fc for NO)",
                effective_price * 100, settings.certainty_max_price_no * 100,
            )
            return None
        if effective_price < 0.70:
            return None

    # Fixed position size for certainty bets — minimum $1
    # NO bets use smaller size (higher risk per trade)
    pct = settings.certainty_position_pct_no if side == "BUY_NO" else settings.certainty_position_pct
    position_size = max(pct * portfolio.bankroll, 1.0)

    # Polymarket minimum is 5 shares — ensure we meet it
    exec_price = outcome.best_ask if side == "BUY_YES" else outcome.current_price_no
    min_usd = max(5.0 * exec_price, 1.0)
    if position_size < min_usd:
        position_size = min_usd  # bump up to minimum

    if position_size > portfolio.bankroll:
        return None

    edge = true_prob - effective_price
    if edge <= 0:
        logger.debug("Certainty rejected: negative edge %.1f%% for %s/%s %s",
                      edge * 100, city, target_date, outcome.bucket.label)
        return None

    profit_margin = 1.0 - effective_price

    src = "observed" if observed_high is not None else "forecast"
    tag = "CERTAINTY-YES" if side == "BUY_YES" else "CERTAINTY-NO"
    logger.info(
        "[%s] %s %s | %s %s | prob=%.0f%% price=%.0fc margin=%.0fc $%.2f (%s)",
        tag, side, outcome.bucket.label, city, target_date,
        true_prob * 100, effective_price * 100, profit_margin * 100, position_size, src,
    )

    return Signal(
        outcome=outcome,
        model_probability=model_prob,
        market_probability=outcome.current_price_yes,
        edge=edge,
        side=side,
        kelly_fraction=settings.certainty_position_pct,
        position_size_usd=round(position_size, 2),
        confidence=confidence,
    )


def _gaussian_bucket_prob(mean: float, sigma: float, bucket) -> float:
    """Probability that temperature falls in this bucket, using Gaussian.

    Computes P(lower <= T < upper) where T ~ N(mean, sigma²).
    For tail buckets, computes P(T < upper) or P(T >= lower).
    """
    from scipy.stats import norm

    if bucket.is_lower_tail:
        return float(norm.cdf(bucket.upper, loc=mean, scale=sigma))
    elif bucket.is_upper_tail:
        return float(1.0 - norm.cdf(bucket.lower, loc=mean, scale=sigma))
    else:
        p_upper = float(norm.cdf(bucket.upper, loc=mean, scale=sigma))
        p_lower = float(norm.cdf(bucket.lower, loc=mean, scale=sigma))
        return p_upper - p_lower


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

        raw_client = ClobClient(
            settings.polymarket_clob_url,
            key=settings.polymarket_private_key,
            chain_id=137,
            signature_type=settings.polymarket_signature_type,
            funder=settings.polymarket_funder_address,
        )
        raw_client.set_api_creds(raw_client.create_or_derive_api_creds())
        # Wrap with safety guard — blocks withdraw/transfer calls
        client = _SafeClobClient(raw_client)

        # Determine token and price
        if signal.side == "BUY_YES":
            token_id = signal.outcome.token_id_yes
            price = signal.outcome.best_ask
        else:
            token_id = signal.outcome.token_id_no
            price = signal.outcome.current_price_no

        logger.debug(
            "Order details: side=%s token_id=%s price=%.3f condition_id=%s market_id=%s",
            signal.side, token_id[:40], price, signal.outcome.condition_id, signal.outcome.market_id,
        )

        # Calculate number of shares
        size = signal.position_size_usd / price if price > 0 else 0
        if signal.position_size_usd < 1.0:
            logger.warning("Trade size too small: $%.2f (Polymarket min=$1)", signal.position_size_usd)
            return False

        # Polymarket uses tick sizes of 0.001; ensure price is valid
        price = round(price, 3)
        if price < 0.01 or price > 0.99:
            logger.warning("Price %.3f outside tradeable range, skipping", price)
            return False

        order = OrderArgs(
            token_id=token_id,
            price=price,
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
