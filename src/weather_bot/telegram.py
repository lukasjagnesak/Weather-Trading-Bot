"""Telegram notifications: daily reports and trade alerts."""

from __future__ import annotations

import logging
from datetime import date, datetime

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


async def send_telegram(
    message: str,
    settings: Settings,
    parse_mode: str = "HTML",
) -> bool:
    """Send a message via Telegram Bot API."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.debug("Telegram not configured, skipping notification")
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.debug("Telegram message sent successfully")
            return True
        except httpx.HTTPError as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False


def format_trade_alert(trade: dict) -> str:
    """Format a single trade execution as a Telegram alert."""
    emoji = "\U0001f7e2" if trade.get("executed") else "\U0001f534"
    arrow = "\u2b06" if trade["side"] == "BUY_YES" else "\u2b07"

    lines = [
        f"{emoji} <b>Trade {trade['side']}</b> {arrow}",
        f"\U0001f3d9 {trade['city'].upper()} | {trade['date']}",
        f"\U0001f321 Bucket: <code>{trade['bucket']}</code>",
        f"\U0001f4ca Model: {trade['model_prob']:.1f}% vs Market: {trade['market_prob']:.1f}%",
        f"\U0001f4b0 Edge: {trade['edge']:.1f}% | Size: ${trade['size_usd']:.2f}",
    ]

    # Add verification info if available
    if trade.get("verified_temp") is not None:
        lines.append(
            f"\U0001f4cd Verified: {trade['verified_temp']:.1f}\u00b0 "
            f"({trade.get('verified_sources', '?')} sources, "
            f"agreement {trade.get('verified_agreement', 0):.0f}%)"
        )

    return "\n".join(lines)


def format_daily_report(
    trades: list[dict],
    bankroll: float,
    daily_pnl: float,
    peak_bankroll: float,
    tracked_wallets: int = 0,
    performance=None,
) -> str:
    """Format the daily summary report for Telegram.

    Args:
        performance: Optional PerformanceMetrics from evaluation module.
    """
    today = date.today().isoformat()
    drawdown = ((peak_bankroll - bankroll) / peak_bankroll * 100) if peak_bankroll > 0 else 0.0

    pnl_emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"
    pnl_sign = "+" if daily_pnl >= 0 else ""

    lines = [
        f"\U0001f4cb <b>Daily Report — {today}</b>",
        "",
        f"\U0001f4b0 Bankroll: <b>${bankroll:.2f}</b>",
        f"{pnl_emoji} Daily P&L: <b>{pnl_sign}${daily_pnl:.2f}</b>",
        f"\U0001f4c9 Drawdown: {drawdown:.1f}%",
        f"\U0001f4ca Peak: ${peak_bankroll:.2f}",
        "",
    ]

    if trades:
        executed = [t for t in trades if t.get("executed")]
        lines.append(f"\U0001f504 Trades today: {len(executed)}/{len(trades)}")
        lines.append("")

        for t in trades:
            status = "\u2705" if t.get("executed") else "\u274c"
            lines.append(
                f"  {status} {t['city'].upper()} {t['bucket']} "
                f"{t['side']} edge={t['edge']:.1f}% ${t['size_usd']:.2f}"
            )
    else:
        lines.append("\U0001f6ab No trades today")

    # Performance metrics section
    if performance and performance.resolved_trades > 0:
        pnl_sign_all = "+" if performance.total_pnl >= 0 else ""
        lines.extend([
            "",
            "\U0001f3af <b>Performance (all-time)</b>",
            f"  Win rate: <b>{performance.win_rate:.1%}</b> ({performance.wins}W / {performance.losses}L)",
            f"  Total P&L: <b>{pnl_sign_all}${performance.total_pnl:.2f}</b>",
            f"  ROI: {performance.roi_pct:+.1f}%",
            f"  Profit factor: {performance.profit_factor:.2f}",
            f"  Pending: {performance.pending_trades}",
        ])

    if tracked_wallets > 0:
        lines.append(f"\n\U0001f440 Copy-trade wallets: {tracked_wallets}")

    lines.append(f"\n\u23f0 {datetime.utcnow().strftime('%H:%M UTC')}")
    return "\n".join(lines)


def format_scan_summary(signals_count: int, markets_count: int, cities: list[str]) -> str:
    """Format a short scan completion notification."""
    if signals_count > 0:
        return (
            f"\U0001f50d <b>Scan complete</b>\n"
            f"\U0001f4b9 Signals: <b>{signals_count}</b>\n"
            f"Cities: {', '.join(c.upper() for c in cities)}\n"
            f"\u23f0 {datetime.utcnow().strftime('%H:%M UTC')}"
        )
    return (
        f"\U0001f50d <b>Scan complete</b> — no signals\n"
        f"Cities: {', '.join(c.upper() for c in cities)}\n"
        f"\u23f0 {datetime.utcnow().strftime('%H:%M UTC')}"
    )


def format_resolution_alert(resolved: list[dict], bankroll: float) -> str:
    """Format resolved trade results for Telegram."""
    total_pnl = sum(r.get("pnl", 0) for r in resolved)
    wins = sum(1 for r in resolved if r.get("outcome") == "win")
    losses = len(resolved) - wins
    pnl_sign = "+" if total_pnl >= 0 else ""
    emoji = "\U0001f3c6" if total_pnl >= 0 else "\U0001f4a5"

    lines = [
        f"{emoji} <b>Trades resolved: {len(resolved)}</b>",
        f"\U0001f3af Results: {wins}W / {losses}L",
        f"\U0001f4b0 P&L: <b>{pnl_sign}${total_pnl:.2f}</b>",
        f"\U0001f4b3 Bankroll: <b>${bankroll:.2f}</b>",
        "",
    ]

    for r in resolved:
        result_emoji = "\u2705" if r.get("outcome") == "win" else "\u274c"
        pnl = r.get("pnl", 0)
        lines.append(
            f"  {result_emoji} {r.get('city', '?').upper()} {r.get('bucket', '?')} "
            f"actual={r.get('actual_temp', '?')}\u00b0 "
            f"{'+' if pnl >= 0 else ''}${pnl:.2f}"
        )

    lines.append(f"\n\u23f0 {datetime.utcnow().strftime('%H:%M UTC')}")
    return "\n".join(lines)
