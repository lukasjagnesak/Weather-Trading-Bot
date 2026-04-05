"""Trading engine: late-certainty strategy using observed temperatures.

Strategy: after 16:00 local time, use Weather Underground observed high to
BUY_YES on the correct bucket and BUY_NO on distant buckets.

SAFETY: This bot MUST NEVER withdraw, transfer, or send funds to any wallet.
The only permitted on-chain action is placing BUY orders on Polymarket.
All deposits and withdrawals must be done manually by the user.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from .config import CITIES, Settings
from .models import MarketOutcome, PortfolioState, Signal

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
    forecasts_by_key: dict,
    settings: Settings,
    portfolio: PortfolioState,
    verified: dict | None = None,
    observed_highs: dict[str, float] | None = None,
    forecast_shifts: dict[tuple[str, date], float] | None = None,
) -> list[Signal]:
    """Generate trading signals using OBSERVED temperature data only.

    Strategy:
    - BUY_YES on the bucket where the observed WU temperature IS
    - BUY_NO on buckets ≥3 buckets AND ≥5°F/3°C away from observed temp
    - Only after 16:00 local time when daily high is nearly locked in
    - Fee-adjusted edge must be ≥5%
    """
    signals: list[Signal] = []

    grouped: dict[tuple[str, date], list[MarketOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.city, outcome.target_date)
        grouped.setdefault(key, []).append(outcome)

    today = date.today()

    for (city, target_date), city_outcomes in grouped.items():
        if target_date != today or not settings.certainty_enabled:
            continue

        # Check local time ≥ 16:00
        city_cfg = CITIES.get(city)
        if not city_cfg:
            continue
        from zoneinfo import ZoneInfo
        local_now = datetime.now(ZoneInfo(city_cfg.timezone))
        local_hour = local_now.hour
        if local_hour < settings.certainty_min_hour:
            continue

        # REQUIRE observed high from WU
        observed_high = observed_highs.get(city) if observed_highs else None
        if observed_high is None:
            continue

        logger.info(
            "CERTAINTY scan %s/%s: observed=%.1f° local_time=%s",
            city, target_date, observed_high, local_now.strftime("%H:%M"),
        )

        # Sort buckets by lower bound for bucket-index distance calculation
        sorted_outcomes = sorted(city_outcomes, key=lambda o: o.bucket.lower)
        unit = city_cfg.unit  # "celsius" or "fahrenheit"

        # ── Find the observed bucket index ──
        observed_bucket_idx = None
        for i, outcome in enumerate(sorted_outcomes):
            if _temp_in_bucket(observed_high, outcome.bucket):
                observed_bucket_idx = i
                break

        # ── BUY_YES: bet on bucket where observed temp IS ──
        if observed_bucket_idx is not None:
            outcome = sorted_outcomes[observed_bucket_idx]
            bucket = outcome.bucket

            # Check margin from bucket edges
            margin = _margin_from_bucket_edge(observed_high, bucket)
            if bucket.is_lower_tail or bucket.is_upper_tail:
                min_margin = settings.certainty_tail_margin_f if unit == "fahrenheit" else settings.certainty_tail_margin_c
            else:
                min_margin = settings.certainty_bucket_margin_f if unit == "fahrenheit" else settings.certainty_bucket_margin_c

            if margin >= min_margin:
                cert = _certainty_signal(
                    outcome, outcome.best_ask,
                    "BUY_YES", settings, portfolio, city, target_date,
                    observed_high=observed_high, local_hour=local_hour,
                    margin=margin, unit=unit,
                )
                if cert:
                    signals.append(cert)
            else:
                logger.debug(
                    "YES rejected %s: margin %.1f° < %.1f° from bucket %s",
                    city, margin, min_margin, bucket.label,
                )

        # ── BUY_NO: bet on buckets far from observed temp ──
        if observed_bucket_idx is not None:
            min_dist_deg = settings.certainty_no_min_distance_f if unit == "fahrenheit" else settings.certainty_no_min_distance_c

            for i, outcome in enumerate(sorted_outcomes):
                bucket_distance = abs(i - observed_bucket_idx)
                if bucket_distance < settings.certainty_no_min_buckets:
                    continue  # too close in bucket count

                distance_deg = _distance_from_bucket(observed_high, outcome.bucket)
                if distance_deg < min_dist_deg:
                    continue  # too close in degrees

                no_price = outcome.current_price_no
                if not (settings.certainty_min_price_no <= no_price <= settings.certainty_max_price_no):
                    continue

                cert = _certainty_signal(
                    outcome, no_price,
                    "BUY_NO", settings, portfolio, city, target_date,
                    observed_high=observed_high, local_hour=local_hour,
                    margin=distance_deg, unit=unit,
                )
                if cert:
                    signals.append(cert)

    signals.sort(key=lambda s: (0 if s.side == "BUY_YES" else 1, -s.edge))
    return signals


def _margin_from_bucket_edge(temp: float, bucket) -> float:
    """How far (in degrees) the temp is from the NEAREST bucket edge (interior)."""
    if bucket.is_lower_tail:
        return bucket.upper - temp  # distance from upper edge
    elif bucket.is_upper_tail:
        return temp - bucket.lower  # distance from lower edge
    else:
        return min(temp - bucket.lower, bucket.upper - temp)



def _certainty_signal(
    outcome: MarketOutcome,
    price: float,
    side: str,
    settings: Settings,
    portfolio: PortfolioState,
    city: str,
    target_date: date,
    observed_high: float,
    local_hour: int,
    margin: float,
    unit: str,
) -> Signal | None:
    """Generate a certainty-strategy signal with fee-adjusted edge.

    All preconditions (today only, after 16:00, observed high exists,
    bucket distance/margin checks) are validated by detect_signals().

    Args:
        price: best_ask for YES, current_price_no for NO
        margin: distance from bucket edge (°) for YES, distance in degrees for NO
        unit: "fahrenheit" or "celsius"
    """
    # ── Price range checks ──
    if side == "BUY_YES":
        if not (settings.certainty_min_price_yes <= price <= settings.certainty_max_price):
            logger.debug(
                "Certainty YES rejected: price %.0fc outside [%.0f-%.0f]c for %s %s",
                price * 100, settings.certainty_min_price_yes * 100,
                settings.certainty_max_price * 100, city, outcome.bucket.label,
            )
            return None
    else:  # BUY_NO
        if not (settings.certainty_min_price_no <= price <= settings.certainty_max_price_no):
            logger.debug(
                "Certainty NO rejected: price %.0fc outside [%.0f-%.0f]c for %s %s",
                price * 100, settings.certainty_min_price_no * 100,
                settings.certainty_max_price_no * 100, city, outcome.bucket.label,
            )
            return None

    # ── Dynamic true_prob based on hour and margin ──
    if side == "BUY_YES":
        if local_hour >= 18:
            true_prob = 0.97
        elif local_hour >= 17:
            true_prob = 0.95 if margin >= 2.0 else 0.93
        else:  # 16:xx
            true_prob = 0.93 if margin >= 2.0 else 0.90
    else:  # BUY_NO
        if local_hour >= 18:
            true_prob = 0.98
        elif local_hour >= 17:
            true_prob = 0.97 if margin >= 5.0 else 0.95
        else:  # 16:xx
            true_prob = 0.95 if margin >= 5.0 else 0.93

    # ── Fee-adjusted edge ──
    cost_with_fee = price * (1.0 + settings.certainty_fee_rate)
    edge = true_prob - cost_with_fee

    if edge < settings.certainty_min_edge_after_fees:
        logger.debug(
            "Certainty %s rejected: edge %.1f%% < %.1f%% after fees for %s %s (prob=%.0f%% cost=%.1fc)",
            side, edge * 100, settings.certainty_min_edge_after_fees * 100,
            city, outcome.bucket.label, true_prob * 100, cost_with_fee * 100,
        )
        return None

    # ── Position sizing ──
    pct = settings.certainty_position_pct if side == "BUY_YES" else settings.certainty_position_pct_no
    position_size = max(pct * portfolio.bankroll, 1.0)

    # Polymarket minimum is 5 shares
    exec_price = outcome.best_ask if side == "BUY_YES" else outcome.current_price_no
    min_usd = max(5.0 * exec_price, 1.0)
    if position_size < min_usd:
        position_size = min_usd

    if position_size > portfolio.bankroll:
        return None

    tag = "CERTAINTY-YES" if side == "BUY_YES" else "CERTAINTY-NO"
    logger.info(
        "[%s] %s %s | %s %s | observed=%.1f° margin=%.1f° prob=%.0f%% "
        "price=%.0fc fee_cost=%.1fc edge=%.1f%% $%.2f",
        tag, side, outcome.bucket.label, city, target_date,
        observed_high, margin, true_prob * 100,
        price * 100, cost_with_fee * 100, edge * 100, position_size,
    )

    return Signal(
        outcome=outcome,
        model_probability=true_prob,
        market_probability=outcome.current_price_yes,
        edge=edge,
        side=side,
        kelly_fraction=pct,
        position_size_usd=round(position_size, 2),
        confidence=true_prob,
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
