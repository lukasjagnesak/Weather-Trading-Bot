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

    return (
        f"{emoji} <b>Trade {trade['side']}</b> {arrow}\n"
        f"\U0001f3d9 {trade['city'].upper()} | {trade['date']}\n"
        f"\U0001f321 Bucket: <code>{trade['bucket']}</code>\n"
        f"\U0001f4ca Model: {trade['model_prob']:.1f}% vs Market: {trade['market_prob']:.1f}%\n"
        f"\U0001f4b0 Edge: {trade['edge']:.1f}% | Size: ${trade['size_usd']:.2f}"
    )


def format_daily_report(
    trades: list[dict],
    bankroll: float,
    daily_pnl: float,
    peak_bankroll: float,
    tracked_wallets: int = 0,
) -> str:
    """Format the daily summary report for Telegram."""
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

    if tracked_wallets > 0:
        lines.append(f"\n\U0001f440 Copy-trade wallets: {tracked_wallets}")

    lines.append(f"\n\u23f0 {datetime.utcnow().strftime('%H:%M UTC')}")
    return "\n".join(lines)


def format_scan_summary(signals_count: int, markets_count: int, cities: list[str]) -> str:
    """Format a short scan completion notification."""
    return (
        f"\U0001f50d <b>Scan complete</b>\n"
        f"Markets: {markets_count} | Signals: {signals_count}\n"
        f"Cities: {', '.join(c.upper() for c in cities)}"
    )
