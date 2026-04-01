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

    Forecast-first strategy:
    1. Determine the most probable temperature bucket per city/date
    2. BUY_YES on that bucket (only if YES token is cheap = high upside)
    3. BUY_NO on buckets the forecast says are wrong (NO price 50-95c, model ≥70% NO)

    Certainty strategy (today only, after 3 PM local):
    4. Use real-time observed high to bet on known outcomes

    Both YES and NO bets are driven by the verified multi-source forecast.
    """
    signals: list[Signal] = []

    # Group outcomes by (city, date) to compute probabilities once per group
    grouped: dict[tuple[str, date], list[MarketOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.city, outcome.target_date)
        grouped.setdefault(key, []).append(outcome)

    today = date.today()

    for (city, target_date), city_outcomes in grouped.items():
        forecast_key = (city, target_date)
        forecast_list = forecasts_by_key.get(forecast_key)
        if not forecast_list:
            logger.debug("No forecasts for %s/%s, skipping", city, target_date)
            continue

        # Log ensemble forecast details per model so we can verify correctness
        for fc in forecast_list:
            logger.info(
                "FORECAST %s %s/%s: model=%s members=%d predicted_daily_high=%.1f spread=%.1f range=[%.1f..%.1f]",
                fc.unit, city, target_date, fc.model_name,
                len(fc.members), np.mean(fc.members), np.std(fc.members),
                np.min(fc.members), np.max(fc.members),
            )

        # Skip edge trading on deterministic fallback — synthetic ensembles
        # have fake spread that doesn't reflect real forecast uncertainty.
        # Only certainty strategy (with observed temps) is allowed on fallback.
        is_deterministic = any(
            fc.model_name == "deterministic_fallback" for fc in forecast_list
        )

        # ── RISK GUARD: Edge only on D+0 ─────────────────────────
        # D+1 forecast error is too high — edge strategy only trades today.
        is_today = (target_date == today)

        # ── RISK GUARD: Require model agreement for edge ─────────
        # Only allow edge bets when all ensemble models agree (spread < 2°).
        model_means = [float(np.mean(fc.members)) for fc in forecast_list]
        model_spread = max(model_means) - min(model_means) if len(model_means) > 1 else 0.0
        models_agree = model_spread < 2.0
        if not models_agree:
            logger.info(
                "Models disagree for %s/%s: spread=%.1f° (%.1f vs %.1f) — skipping edge",
                city, target_date, model_spread, min(model_means), max(model_means),
            )

        # Edge trading allowed only when: today + not deterministic + models agree
        edge_allowed = is_today and not is_deterministic and models_agree

        # ── LATENCY ARBITRAGE: detect forecast shifts ─────────
        # When a new model run shifts the forecast, markets haven't adjusted yet.
        # Use lower edge threshold and allow D+1 for shifted forecasts.
        shift = forecast_shifts.get((city, target_date), 0.0) if forecast_shifts else 0.0
        has_significant_shift = abs(shift) >= settings.latency_arb_min_shift
        if has_significant_shift:
            logger.info(
                "LATENCY ARB %s/%s: forecast shifted %+.1f° — using lower edge threshold (%.0f%%)",
                city, target_date, shift, settings.latency_arb_edge_threshold * 100,
            )
            # Allow edge trading even on D+1 if there's a significant shift
            if not is_deterministic and models_agree:
                edge_allowed = True

        # Compute probabilities for all buckets at once
        buckets = [o.bucket for o in city_outcomes]
        model_probs = compute_bucket_probabilities(forecast_list, buckets)
        confidence = ensemble_confidence(forecast_list)

        # Get verification data for this city/date
        vf = verified.get(forecast_key) if verified else None
        has_verification = vf is not None and vf.source_count >= 2

        if has_verification:
            if vf.source_count >= 3:
                confidence = 0.5 * confidence + 0.5 * vf.agreement_score
            logger.info(
                "Verified forecast for %s/%s: %.1f° (%d sources, spread=%.1f, agreement=%.0f%%)",
                city, target_date, vf.mean_high, vf.source_count,
                vf.spread, vf.agreement_score * 100,
            )

        # ── Find the BEST bucket (highest model probability) ──────────
        best_outcome = None
        best_prob = 0.0
        for outcome in city_outcomes:
            p = model_probs.get(outcome.bucket.label, 0.0)
            if p > best_prob:
                best_prob = p
                best_outcome = outcome

        if best_outcome and best_prob > 0.20:
            # BUY_YES on the most probable bucket — the core bet
            signal = None
            if edge_allowed:
                signal = _forecast_signal(
                    best_outcome, best_prob, best_outcome.current_price_yes,
                    "BUY_YES", confidence, settings, portfolio, city, target_date,
                    vf=vf, is_latency=has_significant_shift,
                )
                if signal:
                    signals.append(signal)
            else:
                logger.debug("Skipping edge BUY_YES for %s/%s: %s",
                             city, target_date,
                             "D+1" if not is_today else
                             "deterministic" if is_deterministic else "models disagree")

            # Certainty strategy runs independently (not as fallback)
            if settings.certainty_enabled:
                cert = _certainty_signal(
                    best_outcome, best_prob, best_outcome.current_price_yes,
                    "BUY_YES", confidence, settings, portfolio, city, target_date,
                    vf=vf, observed_highs=observed_highs,
                )
                if cert:
                    # Avoid duplicate: only add if edge strategy didn't already pick this
                    if not signal:
                        signals.append(cert)
                    else:
                        logger.info("Certainty confirms edge signal for %s/%s %s",
                                    city, target_date, best_outcome.bucket.label)

        # ── BUY_NO on buckets the forecast says are wrong ──────────
        for outcome in city_outcomes:
            if outcome is best_outcome:
                continue  # skip the best bucket — we bet YES on it
            model_prob = model_probs.get(outcome.bucket.label, 0.0)
            no_price = outcome.current_price_no

            # Bet NO when forecast clearly disagrees with market.
            # NO price range 70-85c — ensures ≥15c profit margin.
            # Model must give ≤25% YES probability (i.e. ≥75% NO).
            if not (0.70 <= no_price <= settings.certainty_max_price_no):
                continue
            our_no_prob = 1.0 - model_prob
            if our_no_prob < 0.75:
                continue  # need ≥75% model confidence it's wrong

            # Edge BUY_NO disabled — too risky, main source of losses.
            # Only certainty BUY_NO (observed temps) is allowed.
            signal = None

            if settings.certainty_enabled:
                cert = _certainty_signal(
                    outcome, model_prob, no_price,
                    "BUY_NO", confidence, settings, portfolio, city, target_date,
                    vf=vf, observed_highs=observed_highs,
                )
                if cert and not signal:
                    signals.append(cert)

    # Sort: YES bets first (core picks), then NO by edge
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

    Only bets on TODAY (D+0) — never tomorrow.  Uses the real-time
    observed high temperature (not forecast) to decide.  After ~3 PM
    local time, the daily max is essentially locked in, so we can bet
    with near-certainty and collect the remaining profit margin.

    Example: observed high is 12°C, bucket 12-13°C priced at 70c → buy YES,
    collect 30c profit per share.
    """
    if not settings.certainty_enabled:
        return None

    # Only bet on today — for tomorrow we don't have observed data
    today = date.today()
    if target_date != today:
        return None

    # Check local time — BUY_YES can start earlier (observed high is already
    # useful by late morning), BUY_NO needs more certainty (later in the day).
    city_cfg = CITIES.get(city)
    if city_cfg:
        from zoneinfo import ZoneInfo
        local_now = datetime.now(ZoneInfo(city_cfg.timezone))
        min_hour = settings.certainty_min_hour_no if side == "BUY_NO" else settings.certainty_min_hour_yes
        if local_now.hour < min_hour:
            return None

    # Use observed high if available (real-time data, not forecast)
    observed_high = observed_highs.get(city) if observed_highs else None

    if observed_high is not None:
        # ── BUY_NO safety: validate observed temp is far from bucket ──
        # After 3 PM, temps can still rise 1-2° — need safety margin.
        if side == "BUY_NO":
            # Check: is observed temp dangerously close to this bucket?
            distance = _distance_from_bucket(observed_high, outcome.bucket)
            # Also check: is observed temp INSIDE the bucket? → never bet NO
            in_bucket = _temp_in_bucket(observed_high, outcome.bucket)
            if in_bucket:
                logger.debug(
                    "Certainty NO rejected: observed %.1f° is INSIDE bucket %s",
                    observed_high, outcome.bucket.label,
                )
                return None
            if distance < 3.0:
                logger.debug(
                    "Certainty NO rejected: observed %.1f° only %.1f° from bucket %s (need ≥3°)",
                    observed_high, distance, outcome.bucket.label,
                )
                return None

        # Use observed temperature with safety margin for spread.
        # Earlier in the day → temps can still rise more → wider sigma.
        if local_now.hour >= 15:
            sigma = 1.0   # after 3 PM, daily high nearly locked
        elif local_now.hour >= 13:
            sigma = 1.5   # early afternoon, some uncertainty
        else:
            sigma = 2.0   # late morning, more uncertainty
        true_prob = _gaussian_bucket_prob(observed_high, sigma, outcome.bucket)
        if side == "BUY_NO":
            true_prob = 1.0 - true_prob
        effective_price = token_price

        logger.debug(
            "Certainty using observed high %.1f° for %s/%s bucket %s → prob=%.0f%%",
            observed_high, city, target_date, outcome.bucket.label, true_prob * 100,
        )
    else:
        # Fall back to verification/model data
        if side == "BUY_YES":
            if vf is not None:
                sigma = max(vf.spread, 1.5)
                true_prob = _gaussian_bucket_prob(vf.mean_high, sigma, outcome.bucket)
            else:
                true_prob = model_prob
            effective_price = token_price
        else:  # BUY_NO
            if vf is not None:
                sigma = max(vf.spread, 1.5)
                true_prob = 1.0 - _gaussian_bucket_prob(vf.mean_high, sigma, outcome.bucket)
            else:
                true_prob = 1.0 - model_prob
            effective_price = token_price

    # Only trade if model is confident enough and price is below cap
    if true_prob < settings.certainty_min_model_prob:
        return None
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

    if position_size > portfolio.bankroll:
        return None

    edge = true_prob - effective_price
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
