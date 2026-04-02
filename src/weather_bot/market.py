"""Fetch and parse Polymarket weather temperature markets."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

import httpx

from .config import CITIES
from .models import MarketOutcome, TemperatureBucket

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"


def _parse_temperature_bucket(
    question: str,
    group_item_title: str | None,
    city_unit: str,
) -> TemperatureBucket | None:
    """Parse a temperature bucket from market question text and group item title.

    Polymarket resolves temperatures to WHOLE degrees (e.g. 13°C means
    exactly 13, which is the range [13, 14) in continuous space).
    Bucket bounds must match WU resolution precision.

    Handles formats like:
    - "13°C" -> exact degree [13, 14)
    - "58-59" -> range in °F [58, 60)
    - "≤27°F" or "27 or below" -> lower tail (-inf, 28)
    - "≥38°F" or "38 or above" -> upper tail [38, +inf)
    """
    if group_item_title:
        title = group_item_title.strip()
    else:
        title = question

    # Check for tail buckets
    lower_tail_patterns = [
        r"[≤<](\d+)",
        r"(\d+)\s*or\s*below",
        r"below\s*(\d+)",
        r"less\s*than\s*(\d+)",
    ]
    upper_tail_patterns = [
        r"[≥>](\d+)",
        r"(\d+)\s*or\s*above",
        r"above\s*(\d+)",
        r"(\d+)\s*or\s*more",
        r"(\d+)\+",
    ]

    for pattern in lower_tail_patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # "≤27" means 27 or below → includes 27 → upper bound is 28
            return TemperatureBucket(
                label=title,
                lower=float("-inf"),
                upper=val + 1,
                is_lower_tail=True,
            )

    for pattern in upper_tail_patterns:
        m = re.search(pattern, title, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # "≥38" means 38 or above → lower bound is 38
            return TemperatureBucket(
                label=title,
                lower=val,
                upper=float("inf"),
                is_upper_tail=True,
            )

    # Range bucket: "58-59" or "58-59°F"
    # WU rounds to whole degrees, so "58-59" means [58, 60)
    range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", title)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return TemperatureBucket(
            label=title,
            lower=low,
            upper=high + 1,
        )

    # Single degree: "13°C" or "13" -> [13, 14)
    single_match = re.search(r"(\d+)\s*°?[CF]?", title)
    if single_match:
        val = float(single_match.group(1))
        return TemperatureBucket(
            label=title,
            lower=val,
            upper=val + 1,
        )

    logger.warning("Could not parse temperature bucket from: %s", title)
    return None


def _extract_city_key(event_title: str) -> str | None:
    """Extract the city key from an event title like 'NYC Daily High Temperature'."""
    title_lower = event_title.lower()

    city_aliases = {
        # US cities
        "nyc": "nyc", "new york": "nyc", "laguardia": "nyc",
        "chicago": "chicago", "o'hare": "chicago",
        "dallas": "dallas", "love field": "dallas",
        "atlanta": "atlanta",
        "miami": "miami",
        "seattle": "seattle",
        "austin": "austin",
        "denver": "denver",
        "houston": "houston",
        "los angeles": "los_angeles", "lax": "los_angeles",
        "san francisco": "san_francisco", "sfo": "san_francisco",
        # Canada
        "toronto": "toronto", "pearson": "toronto",
        # Europe
        "london": "london", "london city": "london", "heathrow": "london",
        "paris": "paris", "cdg": "paris", "charles de gaulle": "paris",
        "ankara": "ankara", "esenboga": "ankara",
        "istanbul": "istanbul",
        "munich": "munich",
        "milan": "milan",
        "madrid": "madrid",
        "warsaw": "warsaw",
        "moscow": "moscow",
        # Asia
        "tokyo": "tokyo", "haneda": "tokyo", "narita": "tokyo",
        "seoul": "seoul", "incheon": "seoul",
        "shanghai": "shanghai", "pudong": "shanghai",
        "beijing": "beijing",
        "chongqing": "chongqing",
        "wuhan": "wuhan",
        "chengdu": "chengdu",
        "shenzhen": "shenzhen",
        "hong kong": "hong_kong",
        "singapore": "singapore",
        "taipei": "taipei", "taoyuan": "taipei",
        "lucknow": "lucknow",
        "tel aviv": "tel_aviv",
        # Americas
        "mexico city": "mexico_city",
        "sao paulo": "sao_paulo", "são paulo": "sao_paulo",
        "buenos aires": "buenos_aires",
        # Oceania
        "sydney": "sydney",
        "wellington": "wellington",
    }

    for alias, key in city_aliases.items():
        if alias in title_lower:
            return key

    return None


def _extract_target_date(event_title: str, market_question: str) -> date | None:
    """Extract the target date from event/market text.

    Looks for patterns like "March 20" or "Mar 20, 2026".
    """
    text = f"{event_title} {market_question}"

    # Pattern: "March 20, 2026" or "March 20"
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    for month_name, month_num in month_names.items():
        pattern = rf"{month_name}\s+(\d{{1,2}})(?:\s*,?\s*(\d{{4}}))?(?!\d)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            day = int(m.group(1))
            year = int(m.group(2)) if m.group(2) else date.today().year
            try:
                return date(year, month_num, day)
            except ValueError:
                continue

    # Pattern: "on March 20" in the question text
    on_pattern = r"on\s+(\w+)\s+(\d{1,2})"
    m = re.search(on_pattern, text, re.IGNORECASE)
    if m:
        month_str = m.group(1).lower()
        day = int(m.group(2))
        month_num = month_names.get(month_str)
        if month_num:
            year = date.today().year
            try:
                return date(year, month_num, day)
            except ValueError:
                pass

    return None


async def fetch_active_temperature_markets(
    target_cities: list[str] | None = None,
    target_dates: list[date] | None = None,
) -> list[MarketOutcome]:
    """Fetch all active temperature markets from Polymarket.

    Returns a list of MarketOutcome objects for tradeable temperature buckets.
    """
    outcomes: list[MarketOutcome] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch active temperature events
        params = {
            "tag_slug": "temperature",
            "active": "true",
            "closed": "false",
            "limit": "100",
        }

        try:
            resp = await client.get(f"{GAMMA_API_URL}/events", params=params)
            resp.raise_for_status()
            events = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Failed to fetch temperature events: %s", e)
            return []

        for event in events:
            event_title = event.get("title", "")
            event_id = str(event.get("id", ""))
            city_key = _extract_city_key(event_title)

            if not city_key:
                continue
            if target_cities and city_key not in target_cities:
                continue

            city_config = CITIES.get(city_key)
            if not city_config:
                continue

            markets = event.get("markets", [])
            for market in markets:
                if market.get("closed") or not market.get("active"):
                    continue
                if not market.get("acceptingOrders", True):
                    continue

                question = market.get("question", "")
                target_date = _extract_target_date(event_title, question)

                if not target_date:
                    continue
                if target_dates and target_date not in target_dates:
                    continue

                # Parse the temperature bucket
                group_title = market.get("groupItemTitle")
                bucket = _parse_temperature_bucket(question, group_title, city_config.unit)
                if not bucket:
                    continue

                # Extract token IDs (API returns JSON string, not list)
                raw_token_ids = market.get("clobTokenIds", [])
                if isinstance(raw_token_ids, str):
                    import json as _json
                    try:
                        clob_token_ids = _json.loads(raw_token_ids)
                    except (ValueError, TypeError):
                        continue
                else:
                    clob_token_ids = raw_token_ids
                if not clob_token_ids or len(clob_token_ids) < 2:
                    continue

                # Parse prices (API may also return JSON string)
                raw_prices = market.get("outcomePrices", ["0.5", "0.5"])
                if isinstance(raw_prices, str):
                    import json as _json
                    try:
                        outcome_prices = _json.loads(raw_prices)
                    except (ValueError, TypeError):
                        outcome_prices = ["0.5", "0.5"]
                else:
                    outcome_prices = raw_prices
                # ── CRITICAL: determine YES/NO index from outcomes field ──
                # The API may return outcomes in any order: ["Yes","No"] or
                # ["No","Yes"].  We MUST read the `outcomes` field to know
                # which index is YES and which is NO.  Getting this wrong
                # causes the bot to buy the WRONG token!
                raw_outcomes = market.get("outcomes", [])
                if isinstance(raw_outcomes, str):
                    import json as _json
                    try:
                        outcomes_list = _json.loads(raw_outcomes)
                    except (ValueError, TypeError):
                        outcomes_list = ["Yes", "No"]
                else:
                    outcomes_list = raw_outcomes
                if not outcomes_list or len(outcomes_list) < 2:
                    outcomes_list = ["Yes", "No"]

                # Find YES and NO indices
                yes_idx = 0
                no_idx = 1
                for i, o in enumerate(outcomes_list):
                    if str(o).strip().lower() == "yes":
                        yes_idx = i
                    elif str(o).strip().lower() == "no":
                        no_idx = i

                if yes_idx == no_idx:
                    logger.warning("Could not determine YES/NO indices for %s, skipping", question[:60])
                    continue

                # Log when outcomes are in non-standard order
                if yes_idx != 0:
                    logger.info(
                        "NON-STANDARD outcome order for %s: outcomes=%s → yes_idx=%d, no_idx=%d",
                        question[:60], outcomes_list, yes_idx, no_idx,
                    )

                try:
                    price_yes = float(outcome_prices[yes_idx]) if outcome_prices else 0.5
                    price_no = float(outcome_prices[no_idx]) if len(outcome_prices) > no_idx else 1 - price_yes
                except (ValueError, IndexError):
                    price_yes = 0.5
                    price_no = 0.5

                outcome = MarketOutcome(
                    market_id=str(market.get("id", "")),
                    condition_id=market.get("conditionId", ""),
                    question=question,
                    token_id_yes=clob_token_ids[yes_idx],
                    token_id_no=clob_token_ids[no_idx],
                    bucket=bucket,
                    current_price_yes=price_yes,
                    current_price_no=price_no,
                    best_bid=market.get("bestBid", price_yes),
                    best_ask=market.get("bestAsk", price_yes),
                    volume=float(market.get("volume", 0) or 0),
                    city=city_key,
                    target_date=target_date,
                    event_id=event_id,
                )
                outcomes.append(outcome)

    logger.info("Found %d active temperature market outcomes", len(outcomes))
    return outcomes


async def get_orderbook(token_id: str) -> dict:
    """Fetch the order book for a specific token from the CLOB API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                "https://clob.polymarket.com/book",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Failed to fetch orderbook for %s: %s", token_id[:20], e)
            return {}
