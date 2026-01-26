# Live Algorithmic Crypto Trading Bot

## Overview

A fully automated trading bot for cryptocurrency perpetual futures on Coinbase International Exchange. The system identifies breakouts preceded by low-volatility consolidation periods (squeezes) using a multi-timeframe Bollinger Band Squeeze strategy. 

This bot has demonstrated an average annualized return of 39.7%, Sharpe Ratio of 2.35, and max drawdown of 22.7% across BTC, DOGE, and SOL over five years of backtesting. Parameters are optimized on one-year spot price of Bitcoin. Currently deployed live 24/7 on a DigitalOcean cloud server trading BTC, SOL, XRP, and DOGE perpetual futures.

### Multi-Asset Performance (2021-2025)

![Multi-Asset Equity Curve](plots/equity_curve_DOGEUSD_BTCUSD_ETHUSD_SOLUSD_XRPUSD.png)

Trading DOGE, BTC, ETH, SOL, and XRP simultaneously:

| Metric | Value |
|--------|-------|
| Total Trades | 494 |
| Win Rate | 66.6% |
| Profit Factor | 1.97 |
| Sharpe Ratio | 1.83 |
| Max Drawdown | 27.5% |
| Total Return | +1,751.9% |

*Longs: 199 | Shorts: 295 | Wins: 329 | Losses: 165*

### Key Features

- **Multi-Asset Trading**: Trades multiple perpetual futures contracts simultaneously (BTC, SOL, XRP, DOGE)
- **Multi-Timeframe Analysis**: 4-hour signals, 1-hour ATR, minute-level execution
- **Dynamic Position Sizing**: Allocation adjusts based on squeeze quality, volume confirmation, and consecutive losses
- **Professional Monitoring**: Complete trade journal, P&L tracking, drawdown alerts, and drift detection
- **Risk Management**: ATR-based stops, daily loss limits, maximum hold periods, and position limits
- **Backtesting**: Minute-by-minute simulation with commission and slippage modeling
- **Optimization**: Bayesian hyperparameter tuning with persistent history

---

## Strategy

### Bollinger Band Squeeze

The strategy exploits the market tendency to alternate between consolidation and trending phases by detecting volatility compression followed by expansion.

**Pattern Recognition**:
1. **Squeeze Detection**: Bollinger Bands contract inside Keltner Channels
2. **Minimum Duration**: Squeeze must persist for minimum bars to filter false signals
3. **Breakout Confirmation**: Bollinger Bands expand outside Keltner Channels
4. **Volume Confirmation**: Breakout accompanied by above-average volume
5. **RSI Filter**: Rejects overbought longs and oversold shorts

### Entry Conditions

All conditions must be satisfied:
- Squeeze duration meets minimum bar requirement (2+ bars)
- Squeeze released (BB expanded outside KC)
- Volume ratio above threshold (1.02x average)
- RSI within acceptable range (25-68)
- Position limits not exceeded (max 2 positions)
- Daily loss limit not breached (3%)

### Exit Logic

Positions close when any condition is met:
- **Stop Loss**: ATR-based (3.45x ATR from entry)
- **Take Profit**: ATR-based (4.0x ATR from entry)
- **Time Exit**: Maximum hold period (7 days)
- **Daily Loss Limit**: Trading halts at 3% daily drawdown

### Position Sizing

Base allocation of 50% with adjustments:
- Longer squeeze duration â†’ increased allocation
- Higher volume ratio â†’ increased allocation
- Strong momentum â†’ increased allocation
- Consecutive losses â†’ decreased allocation

Final size constrained to 30-50% range.

---

## Backtest Performance (2021-01-01 to 2025-12-29)

### BTC/USD

| Metric | Value |
|--------|-------|
| Total Trades | 128 |
| Win Rate | 68.0% |
| Profit Factor | 2.37 |
| Sharpe Ratio | 2.57 |
| Maximum Drawdown | 11.4% |
| Total Return (after commission and slippage) | 287.2% |

![BTC Equity Curve](plots/equity_curve_BTCUSD.png)

### DOGE/USD

| Metric | Value |
|--------|-------|
| Total Trades | 96 |
| Win Rate | 76.0% |
| Profit Factor | 3.13 |
| Sharpe Ratio | 3.54 |
| Maximum Drawdown | 18.9% |
| Total Return (after commission and slippage) | 1430.6% |

![DOGE Equity Curve](plots/equity_curve_DOGEUSD.png)

### SOL/USD

| Metric | Value |
|--------|-------|
| Total Trades | 90 |
| Win Rate | 67.8% |
| Profit Factor | 1.62 |
| Sharpe Ratio | 1.37 |
| Maximum Drawdown | 35.7% |
| Total Return (after commission and slippage) | 202.9% |

![SOL Equity Curve](plots/equity_curve_SOLUSD.png)

### XRP/USD

| Metric | Value |
|--------|-------|
| Total Trades | 85 |
| Win Rate | 65.9% |
| Profit Factor | 1.89 |
| Sharpe Ratio | 1.72 |
| Maximum Drawdown | 19.2% |
| Total Return (after commission and slippage) | 198.4% |

![XRP Equity Curve](plots/equity_curve_XRPUSD.png)

**Disclaimer**: Past performance does not guarantee future results. These results are from backtesting on historical data.

---

## Live Trading Setup

### Futures Contracts Traded

| Contract | Asset | Contract Size | Leverage | Margin Rate |
|----------|-------|---------------|----------|-------------|
| BIP-20DEC30-CDE | BTC | 0.01 BTC | ~10x | ~10% intraday |
| SLP-20DEC30-CDE | SOL | 5 SOL | ~5x | ~20% intraday |
| XPP-20DEC30-CDE | XRP | 500 XRP | ~5x | ~20% intraday |
| DOP-20DEC30-CDE | DOGE | 5000 DOGE | ~4x | ~25% intraday |

### Server Deployment

- **Provider**: DigitalOcean
- **Location**: NYC3
- **Cost**: $4/month
- **OS**: Ubuntu 24.04

---

## Installation

### Requirements
- Python 3.11+
- Coinbase Advanced Trade API credentials

### Setup

```bash
# Clone repository
git clone https://github.com/jicheolha/crypto-trading-bot.git
cd crypto-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API credentials
export COINBASE_API_KEY='your_api_key'
export COINBASE_API_SECRET='your_api_secret'
```

---

## Usage

### Download Historical Data
```bash
python download_data.py
```

### Run Backtest
```bash
python run_backtest.py
```

### Parameter Optimization
```bash
# Bayesian optimization (recommended)
python optimize.py --method bayesian --trials 200

# View optimization history
python optimize.py --history
```

### Live Trading
```bash
# Start live trading
python run_live_multi_asset.py
```

---

## Project Structure

```
coinbase_live_trader.py   - Main live trading engine
run_live_multi_asset.py   - Live trading launcher
signal_generator.py       - Signal generation logic
technical.py              - Technical indicators
backtester.py             - Backtesting engine
run_backtest.py           - Backtest launcher
optimize.py               - Optimization runner
optimize_lib.py           - Optimization library
data_utils.py             - Data loading utilities
download_data.py          - Historical data downloader
utils.py                  - Shared utilities
requirements.txt          - Dependencies
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Optimization | Optuna |
| Exchange API | Coinbase Advanced Trade |
| Caching | Pickle |

---

## License

MIT License - See LICENSE file.

---

## Disclaimer

**This software is for educational and research purposes only.**

Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Never invest more than you can afford to lose. The authors are not liable for any financial losses incurred through use of this software.

---

## Contact

- GitHub: [github.com/jicheolha/crypto-trading-bot](https://github.com/jicheolha/crypto-trading-bot)

---

**Version**: 2.0.0  
**Last Updated**: January 2026