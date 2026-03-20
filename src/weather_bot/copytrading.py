"""Copy-trading module: track whale wallets and mirror their Polymarket positions."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Polymarket subgraph / profile API endpoints
POLYMARKET_PROFILE_API = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_API = "https://data-api.polymarket.com"

WALLETS_FILE = Path(__file__).parent.parent.parent / "wallets.json"


@dataclass
class TrackedWallet:
    """A wallet being tracked for copy-trading."""

    address: str
    label: str = ""
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    enabled: bool = True


@dataclass
class WalletPosition:
    """A position held by a tracked wallet on Polymarket."""

    wallet: str
    market_slug: str
    title: str
    outcome: str  # "Yes" or "No"
    size: float  # number of shares
    avg_price: float
    current_price: float
    condition_id: str = ""
    asset_id: str = ""
    pnl: float = 0.0
    realized_pnl: float = 0.0


def load_wallets() -> list[TrackedWallet]:
    """Load tracked wallets from persistent JSON file."""
    if not WALLETS_FILE.exists():
        return []
    try:
        data = json.loads(WALLETS_FILE.read_text())
        return [TrackedWallet(**w) for w in data]
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.error("Failed to load wallets file: %s", e)
        return []


def save_wallets(wallets: list[TrackedWallet]) -> None:
    """Persist tracked wallets to JSON file."""
    data = [asdict(w) for w in wallets]
    WALLETS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def add_wallet(address: str, label: str = "") -> TrackedWallet:
    """Add a wallet address to the tracking list.

    Returns the TrackedWallet object (new or existing).
    """
    address = address.strip().lower()
    if not address.startswith("0x") or len(address) != 42:
        raise ValueError(f"Invalid Ethereum address: {address}")

    wallets = load_wallets()

    # Check if already tracked
    for w in wallets:
        if w.address == address:
            logger.info("Wallet %s already tracked (label=%s)", address, w.label)
            if label and not w.label:
                w.label = label
                save_wallets(wallets)
            return w

    wallet = TrackedWallet(address=address, label=label)
    wallets.append(wallet)
    save_wallets(wallets)
    logger.info("Added wallet %s (label=%s)", address, label)
    return wallet


def remove_wallet(address: str) -> bool:
    """Remove a wallet address from the tracking list."""
    address = address.strip().lower()
    wallets = load_wallets()
    original_count = len(wallets)
    wallets = [w for w in wallets if w.address != address]

    if len(wallets) < original_count:
        save_wallets(wallets)
        logger.info("Removed wallet %s", address)
        return True

    logger.warning("Wallet %s not found in tracking list", address)
    return False


def toggle_wallet(address: str, enabled: bool) -> bool:
    """Enable or disable copy-trading for a wallet."""
    address = address.strip().lower()
    wallets = load_wallets()

    for w in wallets:
        if w.address == address:
            w.enabled = enabled
            save_wallets(wallets)
            return True

    return False


async def fetch_wallet_positions(address: str) -> list[WalletPosition]:
    """Fetch current Polymarket positions for a wallet address.

    Uses the Polymarket data API to retrieve open positions.
    """
    address = address.strip().lower()
    positions: list[WalletPosition] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try the gamma-api profile endpoint
        try:
            resp = await client.get(
                f"{POLYMARKET_PROFILE_API}/positions",
                params={"user": address, "sizeThreshold": "0.1"},
            )
            resp.raise_for_status()
            data = resp.json()

            for pos in data if isinstance(data, list) else []:
                market = pos.get("market", {})
                size = float(pos.get("size", 0))
                if abs(size) < 0.01:
                    continue

                avg_price = float(pos.get("avgPrice", 0))
                cur_price = float(pos.get("curPrice", avg_price))
                pnl = (cur_price - avg_price) * size if avg_price > 0 else 0.0

                positions.append(WalletPosition(
                    wallet=address,
                    market_slug=market.get("slug", ""),
                    title=market.get("question", pos.get("title", "Unknown")),
                    outcome=pos.get("outcome", "Yes"),
                    size=size,
                    avg_price=avg_price,
                    current_price=cur_price,
                    condition_id=pos.get("conditionId", market.get("conditionId", "")),
                    asset_id=pos.get("assetId", pos.get("asset", "")),
                    pnl=pnl,
                    realized_pnl=float(pos.get("realizedPnl", 0)),
                ))

            logger.info("Fetched %d positions for wallet %s", len(positions), address[:10])

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("No profile found for %s, trying data API", address[:10])
            else:
                logger.error("Profile API error for %s: %s", address[:10], e)

        # Fallback: try the data-api
        if not positions:
            try:
                resp = await client.get(
                    f"{POLYMARKET_DATA_API}/positions",
                    params={"user": address},
                )
                resp.raise_for_status()
                data = resp.json()

                for pos in data if isinstance(data, list) else []:
                    size = float(pos.get("size", 0))
                    if abs(size) < 0.01:
                        continue

                    positions.append(WalletPosition(
                        wallet=address,
                        market_slug=pos.get("slug", ""),
                        title=pos.get("title", pos.get("question", "Unknown")),
                        outcome=pos.get("outcome", "Yes"),
                        size=size,
                        avg_price=float(pos.get("avgPrice", 0)),
                        current_price=float(pos.get("curPrice", 0)),
                        condition_id=pos.get("conditionId", ""),
                        asset_id=pos.get("assetId", ""),
                    ))

                logger.info(
                    "Fetched %d positions from data-api for %s", len(positions), address[:10]
                )

            except (httpx.HTTPError, ValueError) as e:
                logger.error("Data API error for %s: %s", address[:10], e)

    return positions


async def fetch_all_tracked_positions() -> dict[str, list[WalletPosition]]:
    """Fetch positions for all enabled tracked wallets.

    Returns a dict mapping wallet address -> list of positions.
    """
    wallets = load_wallets()
    enabled = [w for w in wallets if w.enabled]

    if not enabled:
        logger.info("No enabled wallets to track")
        return {}

    results: dict[str, list[WalletPosition]] = {}

    for wallet in enabled:
        try:
            positions = await fetch_wallet_positions(wallet.address)
            results[wallet.address] = positions
        except Exception as e:
            logger.error("Failed to fetch positions for %s: %s", wallet.address[:10], e)
            results[wallet.address] = []

    total = sum(len(p) for p in results.values())
    logger.info("Fetched %d total positions across %d wallets", total, len(enabled))
    return results


def summarize_positions(
    positions_by_wallet: dict[str, list[WalletPosition]],
) -> list[dict]:
    """Aggregate positions across wallets to find consensus trades.

    Returns a list of dicts with market-level aggregation:
    - How many wallets hold the position
    - Total size across wallets
    - Average entry price
    - Current market price
    """
    wallets = {w.address: w for w in load_wallets()}
    market_agg: dict[str, dict] = {}

    for address, positions in positions_by_wallet.items():
        wallet = wallets.get(address)
        wallet_label = wallet.label if wallet else address[:10]

        for pos in positions:
            key = f"{pos.condition_id}:{pos.outcome}" if pos.condition_id else f"{pos.title}:{pos.outcome}"

            if key not in market_agg:
                market_agg[key] = {
                    "title": pos.title,
                    "outcome": pos.outcome,
                    "condition_id": pos.condition_id,
                    "slug": pos.market_slug,
                    "current_price": pos.current_price,
                    "wallets": [],
                    "total_size": 0.0,
                    "weighted_avg_price": 0.0,
                    "total_pnl": 0.0,
                }

            agg = market_agg[key]
            agg["wallets"].append({
                "address": address,
                "label": wallet_label,
                "size": pos.size,
                "avg_price": pos.avg_price,
                "pnl": pos.pnl,
            })
            agg["total_size"] += pos.size
            agg["weighted_avg_price"] += pos.avg_price * pos.size
            agg["total_pnl"] += pos.pnl

    # Finalize weighted averages
    results = []
    for agg in market_agg.values():
        if agg["total_size"] > 0:
            agg["weighted_avg_price"] /= agg["total_size"]
        agg["wallet_count"] = len(agg["wallets"])
        results.append(agg)

    # Sort by number of wallets holding, then by total size
    results.sort(key=lambda x: (x["wallet_count"], x["total_size"]), reverse=True)
    return results
