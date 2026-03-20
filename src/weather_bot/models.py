"""Data models for the weather trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class EnsembleForecast:
    """Raw ensemble forecast data for a city/date."""

    city: str
    target_date: date
    model_name: str
    members: list[float]  # temperature values from each ensemble member
    unit: str  # "celsius" or "fahrenheit"
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean(self) -> float:
        return sum(self.members) / len(self.members)

    @property
    def spread(self) -> float:
        mean = self.mean
        return (sum((m - mean) ** 2 for m in self.members) / len(self.members)) ** 0.5


@dataclass
class TemperatureBucket:
    """A temperature range bucket from a Polymarket market."""

    label: str  # e.g., "58-59" or "13°C"
    lower: float  # inclusive lower bound
    upper: float  # exclusive upper bound (or inclusive for exact match)
    is_lower_tail: bool = False  # e.g., "≤27°F"
    is_upper_tail: bool = False  # e.g., "≥38°F"


@dataclass
class MarketOutcome:
    """A single tradeable outcome on Polymarket."""

    market_id: str
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    bucket: TemperatureBucket
    current_price_yes: float  # market probability
    current_price_no: float
    best_bid: float
    best_ask: float
    volume: float
    city: str
    target_date: date
    event_id: str = ""


@dataclass
class Signal:
    """A trading signal with edge and position sizing."""

    outcome: MarketOutcome
    model_probability: float  # our estimated probability
    market_probability: float  # current market price
    edge: float  # model_prob - market_prob
    side: str  # "BUY_YES" or "BUY_NO"
    kelly_fraction: float  # raw Kelly fraction
    position_size_usd: float  # dollar amount to risk
    confidence: float  # 0-1, based on ensemble agreement


@dataclass
class PortfolioState:
    """Current state of the trading portfolio."""

    bankroll: float
    daily_pnl: float = 0.0
    peak_bankroll: float = 0.0
    open_positions: list[Signal] = field(default_factory=list)
    trades_today: int = 0

    @property
    def drawdown(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll
