"""Configuration for the weather trading bot."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class CityConfig:
    """Weather station configuration for a tradeable city.

    Coordinates must match the exact Weather Underground (WU) station that
    Polymarket uses for resolution.  Each market's rules name a specific
    ICAO station; forecasting for a different location introduces
    systematic bias that erodes edge.
    """

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        unit: str = "celsius",
        polymarket_tag: str = "temperature",
        icao: str = "",
        wunderground_url: str = "",
        timezone: str = "UTC",
    ):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.unit = unit  # "celsius" or "fahrenheit"
        self.polymarket_tag = polymarket_tag
        self.icao = icao  # ICAO station code used by Polymarket for resolution
        self.wunderground_url = wunderground_url  # WU history page for verification
        self.timezone = timezone  # IANA timezone for local time checks


# ---------------------------------------------------------------------------
# Cities with active Polymarket temperature markets.
#
# Coordinates are pinned to the Weather Underground station that each
# Polymarket market references in its resolution rules.  Do NOT use
# "city-centre" coordinates — they can differ by 20-50 km from the
# resolution station, which is enough to shift the daily-high by 1-2 °C
# and destroy the bot's edge.
#
# Station mapping verified from Polymarket market rules & Degen Doppler:
#   US cities  → Degen Doppler / Polymarket rules (Wunderground)
#   Int'l      → Polymarket individual market "Rules" section
#   Taipei     → NOAA RCTP (Taoyuan International Airport)
# ---------------------------------------------------------------------------
CITIES: dict[str, CityConfig] = {
    # --- United States (Fahrenheit) ----------------------------------------
    "nyc": CityConfig(
        "New York City", 40.7772, -73.8726, unit="fahrenheit",
        icao="KLGA", timezone="America/New_York",
        wunderground_url="https://www.wunderground.com/history/daily/us/new-york-city/KLGA",
    ),
    "chicago": CityConfig(
        "Chicago", 41.9742, -87.9073, unit="fahrenheit",
        icao="KORD", timezone="America/Chicago",
        wunderground_url="https://www.wunderground.com/history/daily/us/chicago/KORD",
    ),
    "dallas": CityConfig(
        "Dallas", 32.8471, -96.8518, unit="fahrenheit",
        icao="KDAL", timezone="America/Chicago",
        wunderground_url="https://www.wunderground.com/history/daily/us/dallas/KDAL",
    ),
    "atlanta": CityConfig(
        "Atlanta", 33.6407, -84.4277, unit="fahrenheit",
        icao="KATL", timezone="America/New_York",
        wunderground_url="https://www.wunderground.com/history/daily/us/atlanta/KATL",
    ),
    "miami": CityConfig(
        "Miami", 25.7959, -80.2870, unit="fahrenheit",
        icao="KMIA", timezone="America/New_York",
        wunderground_url="https://www.wunderground.com/history/daily/us/miami/KMIA",
    ),
    "seattle": CityConfig(
        "Seattle", 47.4502, -122.3088, unit="fahrenheit",
        icao="KSEA", timezone="America/Los_Angeles",
        wunderground_url="https://www.wunderground.com/history/daily/us/seattle/KSEA",
    ),
    # --- Canada (Fahrenheit on Polymarket) ---------------------------------
    "toronto": CityConfig(
        "Toronto", 43.6777, -79.6248, unit="fahrenheit",
        icao="CYYZ", timezone="America/Toronto",
        wunderground_url="https://www.wunderground.com/history/daily/ca/mississauga/CYYZ",
    ),
    # --- Europe (Celsius) --------------------------------------------------
    "london": CityConfig(
        "London", 51.5053, 0.0553, unit="celsius",
        icao="EGLC", timezone="Europe/London",
        wunderground_url="https://www.wunderground.com/history/daily/gb/london/EGLC",
    ),
    "paris": CityConfig(
        "Paris", 49.0097, 2.5479, unit="celsius",
        icao="LFPG", timezone="Europe/Paris",
        wunderground_url="https://www.wunderground.com/history/daily/fr/paris/LFPG",
    ),
    "ankara": CityConfig(
        "Ankara", 40.1281, 32.9951, unit="celsius",
        icao="LTAC", timezone="Europe/Istanbul",
        wunderground_url="https://www.wunderground.com/history/daily/tr/ankara/LTAC",
    ),
    # --- Asia (Celsius) ----------------------------------------------------
    "tokyo": CityConfig(
        "Tokyo", 35.5494, 139.7798, unit="celsius",
        icao="RJTT", timezone="Asia/Tokyo",
        wunderground_url="https://www.wunderground.com/history/daily/jp/tokyo/RJTT",
    ),
    "seoul": CityConfig(
        "Seoul", 37.4602, 126.4407, unit="celsius",
        icao="RKSI", timezone="Asia/Seoul",
        wunderground_url="https://www.wunderground.com/history/daily/kr/incheon/RKSI",
    ),
    "shanghai": CityConfig(
        "Shanghai", 31.1443, 121.8052, unit="celsius",
        icao="ZSPD", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/shanghai/ZSPD",
    ),
    "taipei": CityConfig(
        "Taipei", 25.0777, 121.2325, unit="celsius",
        icao="RCTP", timezone="Asia/Taipei",
        wunderground_url="https://www.wunderground.com/history/daily/tw/taoyuan-district/RCTP",
    ),
    "lucknow": CityConfig(
        "Lucknow", 26.7606, 80.8893, unit="celsius",
        icao="VILK", timezone="Asia/Kolkata",
        wunderground_url="https://www.wunderground.com/history/daily/in/lucknow/VILK",
    ),
    # --- Oceania (Celsius) -------------------------------------------------
    "sydney": CityConfig(
        "Sydney", -33.9461, 151.1772, unit="celsius",
        icao="YSSY", timezone="Australia/Sydney",
        wunderground_url="https://www.wunderground.com/history/daily/au/sydney/YSSY",
    ),
    "wellington": CityConfig(
        "Wellington", -41.3272, 174.8053, unit="celsius",
        icao="NZWN", timezone="Pacific/Auckland",
        wunderground_url="https://www.wunderground.com/history/daily/nz/wellington/NZWN",
    ),
}


class Settings(BaseSettings):
    """Bot settings loaded from environment variables."""

    # Polymarket
    polymarket_private_key: str = ""
    polymarket_funder_address: str = ""
    polymarket_signature_type: int = 0
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"

    # Trading mode
    trading_mode: str = "paper"  # "paper" or "live"

    # Risk parameters
    bankroll: float = 1000.0
    max_position_pct: float = 0.02  # max 2% of bankroll per position
    min_edge_threshold: float = 0.08  # minimum 8% edge to trade
    kelly_fraction: float = 0.25  # quarter Kelly
    daily_loss_limit_pct: float = 0.03  # stop at 3% daily loss
    max_drawdown_pct: float = 0.15  # reduce size at 15% drawdown

    # High-confidence "certainty" strategy
    # Buy YES/NO even if market is already well-priced, as long as our model
    # is confident enough and the price is below the maximum cap.
    # e.g. model says 90% → buy YES at 70c, collect 30c profit per share.
    certainty_enabled: bool = True
    certainty_min_model_prob: float = 0.70   # model must give ≥70% probability
    certainty_max_price: float = 0.95        # never pay more than 95c
    certainty_position_pct: float = 0.07     # 7% of bankroll per certainty bet (~$1.12 at $16 bankroll)

    # Ensemble models to use
    ensemble_models: list[str] = Field(
        default=["gfs_seamless", "ecmwf_ifs025"]
    )

    # Telegram notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_trade_alerts: bool = True  # send alert on each trade
    telegram_daily_report: bool = True  # send daily summary
    telegram_daily_report_hour: int = 7  # UTC hour for daily report (8:00 CET)

    # Copy-trading
    copy_trade_enabled: bool = False
    copy_trade_min_wallets: int = 2  # min wallets holding a position to trigger copy
    copy_trade_scale: float = 0.5  # scale factor for copied positions (0-1)

    # Scan interval in seconds
    scan_interval: int = 1800  # 30 minutes — markets move slowly, forecasts update every 6-12h

    # Cities to trade (keys from CITIES dict)
    active_cities: list[str] = Field(
        default=["nyc", "london", "paris", "tokyo", "seoul", "shanghai"]
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
