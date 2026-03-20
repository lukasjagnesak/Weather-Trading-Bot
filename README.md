# Polymarket Weather Trading Bot

Automated trading bot for Polymarket temperature prediction markets. Uses ensemble weather forecasts from multiple numerical weather models (GFS, ECMWF) to identify mispriced temperature markets and execute trades.

## How It Works

1. **Fetch Markets** — Scans Polymarket for active daily high temperature markets across 15+ cities
2. **Ensemble Forecasts** — Pulls 31-member GFS and 51-member ECMWF ensemble forecasts from Open-Meteo (free, no API key)
3. **Probability Estimation** — Converts ensemble data to bucket probabilities using EMOS (Ensemble Model Output Statistics) calibration with spread inflation correction
4. **Edge Detection** — Compares model probabilities to market prices; flags opportunities where edge exceeds threshold (default 8%)
5. **Kelly Sizing** — Sizes positions using fractional Kelly criterion (default 1/4 Kelly) scaled by model confidence
6. **Risk Controls** — Enforces position limits, correlation filters, drawdown scaling, and daily loss limits
7. **Execution** — Paper trades by default; supports live trading via Polymarket's CLOB API

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# List active temperature markets
weather-bot markets

# Show ensemble forecast for a city
weather-bot forecast nyc

# Run a single scan (paper mode)
weather-bot scan

# Run continuous loop
weather-bot run --mode paper --bankroll 1000 --interval 300
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings (all configurable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `BANKROLL` | `1000.0` | Starting bankroll in USD |
| `MIN_EDGE_THRESHOLD` | `0.08` | Minimum edge to trade (8%) |
| `KELLY_FRACTION` | `0.25` | Kelly fraction (quarter Kelly) |
| `MAX_POSITION_PCT` | `0.02` | Max single position (2% of bankroll) |
| `DAILY_LOSS_LIMIT_PCT` | `0.03` | Stop trading at 3% daily loss |
| `SCAN_INTERVAL` | `300` | Seconds between scans |
| `ACTIVE_CITIES` | `nyc,london,...` | Cities to trade |

For live trading, you also need `POLYMARKET_PRIVATE_KEY` and `POLYMARKET_FUNDER_ADDRESS`.

## Architecture

```
src/weather_bot/
├── cli.py          # CLI entry point (click)
├── bot.py          # Main loop orchestration
├── config.py       # Settings and city configurations
├── models.py       # Data models (forecasts, markets, signals)
├── weather.py      # Open-Meteo ensemble API client
├── probability.py  # EMOS calibration & probability calculation
├── market.py       # Polymarket API client & market parser
├── trading.py      # Edge detection, Kelly sizing, order execution
└── risk.py         # Portfolio risk management
```

## Supported Cities

NYC, London, Paris, Tokyo, Seoul, Shanghai, Ankara, Toronto, Chicago, Dallas, Atlanta, Miami, Wellington, Taipei, Lucknow

## Trading Strategy

The bot uses a **forecast-to-odds arbitrage** approach:

- **Multi-model ensemble**: Combines GFS (31 members) and ECMWF (51 members) for robust probability estimates
- **EMOS calibration**: Corrects for known ensemble under-dispersion with 20% spread inflation
- **Weighted combination**: ECMWF weighted at 50%, GFS at 40% (reflecting ECMWF's superior skill)
- **Fractional Kelly**: Quarter Kelly sizing to manage estimation uncertainty
- **Confidence scaling**: Reduces position size when models disagree

## Tests

```bash
pytest tests/ -v
```

## Data Sources

- **Weather**: [Open-Meteo Ensemble API](https://open-meteo.com/en/docs/ensemble-api) (free, no API key)
- **Markets**: [Polymarket Gamma API](https://docs.polymarket.com) + [CLOB API](https://docs.polymarket.com)
