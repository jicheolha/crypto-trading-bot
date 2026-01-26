# Keltrader

Keltrader fully automated algorithmic trading system that trades cryptocurrency perpetual futures 24/7. The bot identifies low-volatility consolidation patterns and enters positions when volatility expands, capturing momentum breakouts in either direction.

**Core Capabilities:**
- Real-time signal generation using Bollinger Band / Keltner Channel squeeze detection
- Multi-asset portfolio management with dynamic position sizing
- ATR-based risk management with automated stop-loss and take-profit execution
- Bayesian hyperparameter optimization with walk-forward validation
- Complete trade journaling, P&L tracking, and performance monitoring

Currently deployed live on a cloud server trading BTC, ETH, SOL, XRP, and DOGE futures on Coinbase International Exchange.

---

## Performance (2021-2025)

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

---

## Strategy

Keltrader detects volatility compression (squeeze) followed by expansion breakouts.

**Entry Conditions:**
- Bollinger Bands contract inside Keltner Channels (squeeze)
- Squeeze releases with volume confirmation
- RSI filter rejects overbought longs / oversold shorts

**Exit Conditions:**
- ATR-based stop loss and take profit
- Maximum hold period (7 days)
- Daily loss limit (3%)

---

## Individual Asset Performance

### BTC/USD

| Metric | Value |
|--------|-------|
| Total Trades | 128 |
| Win Rate | 68.0% |
| Profit Factor | 2.37 |
| Sharpe Ratio | 2.57 |
| Max Drawdown | 11.4% |
| Total Return | 287.2% |

![BTC Equity Curve](plots/equity_curve_BTCUSD.png)

### DOGE/USD

| Metric | Value |
|--------|-------|
| Total Trades | 96 |
| Win Rate | 76.0% |
| Profit Factor | 3.13 |
| Sharpe Ratio | 3.54 |
| Max Drawdown | 18.9% |
| Total Return | 1430.6% |

![DOGE Equity Curve](plots/equity_curve_DOGEUSD.png)

### SOL/USD

| Metric | Value |
|--------|-------|
| Total Trades | 90 |
| Win Rate | 67.8% |
| Profit Factor | 1.62 |
| Sharpe Ratio | 1.37 |
| Max Drawdown | 35.7% |
| Total Return | 202.9% |

![SOL Equity Curve](plots/equity_curve_SOLUSD.png)

### XRP/USD

| Metric | Value |
|--------|-------|
| Total Trades | 85 |
| Win Rate | 65.9% |
| Profit Factor | 1.89 |
| Sharpe Ratio | 1.72 |
| Max Drawdown | 19.2% |
| Total Return | 198.4% |

![XRP Equity Curve](plots/equity_curve_XRPUSD.png)

---

## Installation

```bash
git clone https://github.com/jicheolha/crypto-trading-bot.git
cd crypto-trading-bot

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export COINBASE_API_KEY='your_key'
export COINBASE_API_SECRET='your_secret'
```

---

## Usage

```bash
# Download data
python download_data.py

# Backtest
python run_backtest.py

# Optimize
python optimize.py --trials 200

# Live trading
python run_live_multi_asset.py
```

---

## Project Structure

```
coinbase_live_trader.py   - Live trading engine
signal_generator.py       - Signal generation
technical.py              - Technical indicators
backtester.py             - Backtesting engine
optimize.py               - Parameter optimization
data_utils.py             - Data utilities
utils.py                  - Shared utilities
```

---

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk. Past performance does not guarantee future results.

---

**Version**: 2.0.0 | **Last Updated**: January 2026