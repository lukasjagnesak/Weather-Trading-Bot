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
    "austin": CityConfig(
        "Austin", 30.1945, -97.6699, unit="fahrenheit",
        icao="KAUS", timezone="America/Chicago",
        wunderground_url="https://www.wunderground.com/history/daily/us/austin/KAUS",
    ),
    "denver": CityConfig(
        "Denver", 39.8561, -104.6737, unit="fahrenheit",
        icao="KDEN", timezone="America/Denver",
        wunderground_url="https://www.wunderground.com/history/daily/us/denver/KDEN",
    ),
    "houston": CityConfig(
        "Houston", 29.6454, -95.2789, unit="fahrenheit",
        icao="KIAH", timezone="America/Chicago",
        wunderground_url="https://www.wunderground.com/history/daily/us/houston/KIAH",
    ),
    "los_angeles": CityConfig(
        "Los Angeles", 33.9425, -118.4081, unit="fahrenheit",
        icao="KLAX", timezone="America/Los_Angeles",
        wunderground_url="https://www.wunderground.com/history/daily/us/los-angeles/KLAX",
    ),
    "san_francisco": CityConfig(
        "San Francisco", 37.6213, -122.3790, unit="fahrenheit",
        icao="KSFO", timezone="America/Los_Angeles",
        wunderground_url="https://www.wunderground.com/history/daily/us/san-francisco/KSFO",
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
    "istanbul": CityConfig(
        "Istanbul", 40.9769, 28.8146, unit="celsius",
        icao="LTFM", timezone="Europe/Istanbul",
        wunderground_url="https://www.wunderground.com/history/daily/tr/istanbul/LTFM",
    ),
    "munich": CityConfig(
        "Munich", 48.3537, 11.7750, unit="celsius",
        icao="EDDM", timezone="Europe/Berlin",
        wunderground_url="https://www.wunderground.com/history/daily/de/munich/EDDM",
    ),
    "milan": CityConfig(
        "Milan", 45.6301, 8.7231, unit="celsius",
        icao="LIMC", timezone="Europe/Rome",
        wunderground_url="https://www.wunderground.com/history/daily/it/milan/LIMC",
    ),
    "madrid": CityConfig(
        "Madrid", 40.4722, -3.5611, unit="celsius",
        icao="LEMD", timezone="Europe/Madrid",
        wunderground_url="https://www.wunderground.com/history/daily/es/madrid/LEMD",
    ),
    "warsaw": CityConfig(
        "Warsaw", 52.1657, 20.9671, unit="celsius",
        icao="EPWA", timezone="Europe/Warsaw",
        wunderground_url="https://www.wunderground.com/history/daily/pl/warsaw/EPWA",
    ),
    "moscow": CityConfig(
        "Moscow", 55.9726, 37.4146, unit="celsius",
        icao="UUEE", timezone="Europe/Moscow",
        wunderground_url="https://www.wunderground.com/history/daily/ru/moscow/UUEE",
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
    "beijing": CityConfig(
        "Beijing", 40.0799, 116.6031, unit="celsius",
        icao="ZBAD", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/beijing/ZBAD",
    ),
    "chongqing": CityConfig(
        "Chongqing", 29.7192, 106.6414, unit="celsius",
        icao="ZUCK", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/chongqing/ZUCK",
    ),
    "wuhan": CityConfig(
        "Wuhan", 30.7838, 114.2081, unit="celsius",
        icao="ZHHH", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/wuhan/ZHHH",
    ),
    "chengdu": CityConfig(
        "Chengdu", 30.5785, 103.9471, unit="celsius",
        icao="ZUUU", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/chengdu/ZUUU",
    ),
    "shenzhen": CityConfig(
        "Shenzhen", 22.6394, 113.8107, unit="celsius",
        icao="ZGSZ", timezone="Asia/Shanghai",
        wunderground_url="https://www.wunderground.com/history/daily/cn/shenzhen/ZGSZ",
    ),
    "hong_kong": CityConfig(
        "Hong Kong", 22.3080, 113.9185, unit="celsius",
        icao="VHHH", timezone="Asia/Hong_Kong",
        wunderground_url="https://www.wunderground.com/history/daily/hk/hong-kong/VHHH",
    ),
    "singapore": CityConfig(
        "Singapore", 1.3502, 103.9940, unit="celsius",
        icao="WSSS", timezone="Asia/Singapore",
        wunderground_url="https://www.wunderground.com/history/daily/sg/singapore/WSSS",
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
    "tel_aviv": CityConfig(
        "Tel Aviv", 32.0055, 34.8854, unit="celsius",
        icao="LLBG", timezone="Asia/Jerusalem",
        wunderground_url="https://www.wunderground.com/history/daily/il/tel-aviv/LLBG",
    ),
    # --- Americas (Celsius) ------------------------------------------------
    "mexico_city": CityConfig(
        "Mexico City", 19.4361, -99.0719, unit="celsius",
        icao="MMMX", timezone="America/Mexico_City",
        wunderground_url="https://www.wunderground.com/history/daily/mx/mexico-city/MMMX",
    ),
    "sao_paulo": CityConfig(
        "Sao Paulo", -23.6273, -46.6566, unit="celsius",
        icao="SBGR", timezone="America/Sao_Paulo",
        wunderground_url="https://www.wunderground.com/history/daily/br/sao-paulo/SBGR",
    ),
    "buenos_aires": CityConfig(
        "Buenos Aires", -34.5592, -58.4156, unit="celsius",
        icao="SAEZ", timezone="America/Argentina/Buenos_Aires",
        wunderground_url="https://www.wunderground.com/history/daily/ar/buenos-aires/SAEZ",
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
    min_edge_threshold: float = 0.15  # minimum 15% edge to trade (conservative)
    kelly_fraction: float = 0.25  # quarter Kelly
    daily_loss_limit_pct: float = 0.05  # stop at 5% daily loss
    max_drawdown_pct: float = 0.25  # reduce size at 25% drawdown

    # High-confidence "certainty" strategy
    # Uses real-time observed temperature (WU) after 16:00 local time.
    # BUY_YES on bucket where observed temp IS, BUY_NO on distant buckets.
    certainty_enabled: bool = True
    certainty_min_hour: int = 16              # bets only after 16:00 local time

    # YES pricing: buy at 50-82c (18c+ profit margin)
    certainty_max_price: float = 0.82         # never pay more than 82c for YES
    certainty_min_price_yes: float = 0.50     # don't buy YES below 50c (suspicious)

    # NO pricing: buy at 50-75c (25c+ profit margin, better risk/reward)
    certainty_max_price_no: float = 0.75      # never pay more than 75c for NO
    certainty_min_price_no: float = 0.50      # don't buy NO below 50c

    # Position sizing — higher stakes for near-certain bets
    certainty_position_pct: float = 0.15      # 15% bankroll per YES bet
    certainty_position_pct_no: float = 0.10   # 10% bankroll per NO bet

    # Fee accounting
    certainty_fee_rate: float = 0.0125        # 1.25% Polymarket taker fee
    certainty_min_edge_after_fees: float = 0.05  # 5% minimum edge after fees

    # Bucket edge margins — how far from bucket boundary the temp must be
    certainty_bucket_margin_f: float = 1.0    # °F from bucket edge for YES
    certainty_bucket_margin_c: float = 0.5    # °C from bucket edge for YES
    certainty_tail_margin_f: float = 2.0      # °F from tail bucket edge
    certainty_tail_margin_c: float = 1.0      # °C from tail bucket edge

    # BUY_NO distance requirements
    certainty_no_min_distance_f: float = 5.0  # min °F from observed to NO bucket
    certainty_no_min_distance_c: float = 3.0  # min °C from observed to NO bucket
    certainty_no_min_buckets: int = 3         # min bucket count distance for NO

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
    scan_interval: int = 1800  # 30 minutes — default between model updates
    scan_interval_model_update: int = 300  # 5 minutes — during model update windows

    # Latency arbitrage: trade immediately when forecast shifts after model update
    latency_arb_enabled: bool = True
    latency_arb_min_shift: float = 1.0  # minimum forecast shift (°C/°F) to trigger
    latency_arb_edge_threshold: float = 0.05  # lower edge threshold (5%) for latency signals

    # Cities to trade — all Polymarket temperature markets
    active_cities: list[str] = Field(
        default_factory=lambda: list(CITIES.keys())
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
