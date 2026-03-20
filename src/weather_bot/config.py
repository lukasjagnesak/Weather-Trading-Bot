"""Configuration for the weather trading bot."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class CityConfig:
    """Weather station configuration for a tradeable city."""

    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        unit: str = "celsius",
        polymarket_tag: str = "temperature",
    ):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.unit = unit  # "celsius" or "fahrenheit"
        self.polymarket_tag = polymarket_tag


# Cities with active Polymarket temperature markets
CITIES: dict[str, CityConfig] = {
    "nyc": CityConfig("New York City", 40.7789, -73.8740, unit="fahrenheit"),
    "london": CityConfig("London", 51.4706, -0.4619, unit="celsius"),
    "paris": CityConfig("Paris", 48.8566, 2.3522, unit="celsius"),
    "tokyo": CityConfig("Tokyo", 35.7646, 140.3864, unit="celsius"),
    "seoul": CityConfig("Seoul", 37.5665, 126.9780, unit="celsius"),
    "shanghai": CityConfig("Shanghai", 31.1434, 121.8052, unit="celsius"),
    "ankara": CityConfig("Ankara", 39.9334, 32.8597, unit="celsius"),
    "toronto": CityConfig("Toronto", 43.6777, -79.6248, unit="fahrenheit"),
    "chicago": CityConfig("Chicago", 41.9742, -87.9073, unit="fahrenheit"),
    "dallas": CityConfig("Dallas", 32.8998, -97.0403, unit="fahrenheit"),
    "atlanta": CityConfig("Atlanta", 33.6407, -84.4277, unit="fahrenheit"),
    "miami": CityConfig("Miami", 25.7959, -80.2870, unit="fahrenheit"),
    "wellington": CityConfig("Wellington", -41.3276, 174.8050, unit="celsius"),
    "taipei": CityConfig("Taipei", 25.0330, 121.5654, unit="celsius"),
    "lucknow": CityConfig("Lucknow", 26.8467, 80.9462, unit="celsius"),
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

    # Ensemble models to use
    ensemble_models: list[str] = Field(
        default=["gfs_seamless", "ecmwf_ifs025"]
    )

    # Scan interval in seconds
    scan_interval: int = 300  # 5 minutes

    # Cities to trade (keys from CITIES dict)
    active_cities: list[str] = Field(
        default=["nyc", "london", "paris", "tokyo", "seoul", "shanghai"]
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
