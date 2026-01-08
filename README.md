# Live Algorithmic Crypto Trading Bot

## Overview

This is an algorithmic trading system designed for cryptocurrency (futures) markets. It identifies low-volatility consolidation periods (squeezes) followed by high-probability breakout opportunities. The system supports backtesting, parameter optimization, and live trading with risk management. Live trading is implemented on crypto futures (e.g. ETH futures), which allow both long and short positions. This bot has demonstrated an average return of 388.0%, Sharpe Ratio of 2.0, and max drawdown of 20.1% accross BTC, ETH, DOGE, and SOL over five years. 

### Key Features

- **Multi-Timeframe Analysis**: Signals generation on 4-hour timeframe, volatility measurement on 1-hour, trade execution on 1-minute
- **Advanced Indicators**: Bollinger Bands, Keltner Channels, RSI, ATR, and normalized momentum
- **Dynamic Position Sizing**: Allocation adjusts based on squeeze quality, volume confirmation, and drawdown state
- **Backtesting**: Minute-by-minute simulation of upto 6 years with commission and slippage customization
- **Optimization**: Hyperparameter tuning of ~20 parameters using one of Bayesian, Random, Grid Search, and Walk-Forward
- **Live Trading**: Real-time trading capability with Coinbase Advanced Trade API
- **Risk Management**: Multiple protective layers including ATR-based stops, daily loss limits, and maximum hold periods

---

## Strategy (with Optimized Parameters)

### Bollinger Band Squeeze

The Bollinger Band Squeeze is a volatility-based pattern that identifies periods of low volatility followed by expansion. The strategy exploits the market tendency to alternate between consolidation and trending phases.

**Pattern Recognition**:
1. **Squeeze Formation**: Bollinger Bands contract inside Keltner Channels, indicating volatility compression
2. **Minimum Duration**: Squeeze must persist for at least 2 bars to filter false signals
3. **Breakout Detection**: Bollinger Bands expand outside Keltner Channels on current bar
4. **Volume Confirmation**: Breakout must be accompanied by volume at least 1.02x the 45-period moving average
5. **Direction Determination**: Entry direction based on momentum sign and price position relative to bands
6. **RSI Filter**: Rejects long entries when RSI exceeds 68, short entries when RSI falls below 18

### Entry Conditions

All conditions must be satisfied simultaneously:
- Squeeze duration of at least 2 bars (8 hours on 4h timeframe)
- Bollinger Bands expanded outside Keltner Channels (squeeze released)
- Current volume ratio greater than or equal to 1.02
- RSI within acceptable range for direction (not overbought for longs, not oversold for shorts)
- Available capital and position limits not exceeded
- Daily loss limit not breached

### Exit Logic

Positions are closed when any of the following conditions are met:
- **Stop Loss**: Entry price minus/plus (ATR × 3.45), depending on direction
- **Take Profit**: Entry price plus/minus (ATR × 4.0), depending on direction  
- **Time-Based Exit**: Automatic closure after 7 days to prevent indefinite holding
- **Daily Loss Limit**: Trading halted if account experiences 3% drawdown in single day

### Position Sizing Algorithm

Base allocation starts at 60% of available capital, then adjusts based on setup quality:

```
base_allocation = 60%

Adjustments:
- If squeeze duration >= 10 bars: multiply by 1.3
- If volume ratio >= 2.0: multiply by 1.2
- If absolute momentum >= 1.5: multiply by 1.15
- If consecutive losses >= 2: multiply by 0.7

Final position size constrained to range [30%, 85%]
```

---

## Performance Results (2021-2025, 5 years)

### BTC/USD

| Metric | Value |
|--------|-------|
| Total Trades | 133 |
| Win Rate | 66.2% |
| Profit Factor | 2.22 |
| Sharpe Ratio | 2.11 |
| Maximum Drawdown | 15.3% |
| Total Return | 219.5% |
| Total Return after commission (0.05%) and slippage (0.02%) | 211.6%

### ETH/USD

| Metric | Value |
|--------|-------|
| Total Trades | 135 |
| Win Rate | 60.7% |
| Profit Factor | 1.97 |
| Sharpe Ratio | 1.58 |
| Maximum Drawdown | 20.7% |
| Total Return | 202.5% |
| Total Return after commission (0.05%) and slippage (0.02%) | 195.5%

### DOGE/USD

| Metric | Value |
|--------|-------|
| Total Trades | 127 |
| Win Rate | 69.3% |
| Profit Factor | 2.13 |
| Sharpe Ratio | 2.79 |
| Maximum Drawdown | 15.3% |
| Total Return | 931.2% |
| Total Return after commission (0.05%) and slippage (0.02%) | 909.6%

### SOL/USD

| Metric | Value |
|--------|-------|
| Total Trades | 98 |
| Win Rate | 68.0% |
| Profit Factor | 1.99 |
| Sharpe Ratio | 1.41 |
| Maximum Drawdown | 29.1% |
| Total Return | 241.0% |
| Total Return after commission (0.05%) and slippage (0.02%) | 235.4%

**Disclaimer**: Past performance does not guarantee future results. These results are for educational purposes and were generated using historical data with realistic assumptions for commission (0.05%) and slippage (0.02%).

---

## Installation and Setup

### Requirements

- Python 3.11 or higher
- API credentials from supported exchange (Coinbase Advanced Trade)

### Installation Steps

```bash
# Clone repository
git clone https://github.com/jicheolha/crypto-trading-bot.git
cd crypto-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example configuration
cp config.example.py config.py

# Edit configuration file with your settings
nano config.py
```

**Important**: Never commit `config.py` to version control. Use environment variables for API credentials in production.

```python
# Recommended: Use environment variables
import os

COINBASE_API_KEY = os.environ.get('COINBASE_API_KEY')
COINBASE_API_SECRET = os.environ.get('COINBASE_API_SECRET')
```

---

## Usage

### Running Backtests

```bash
# Execute backtest with default parameters
python run_backtest.py

# Output includes:
# - Trade log with date, direction, entry/exit prices, duration, and P&L
# - Summary statistics (Sharpe ratio, profit factor, win rate, and total return)
# - Equity curves
```

### Parameter Optimization

The optimization module supports multiple methods for hyperparameter tuning:

```bash
# Bayesian optimization (recommended)
python optimize.py --method bayesian

# Random search
python optimize.py --method random

# Walk-forward analysis
python optimize.py --method walkforward --folds 5

# Grid search
python optimize.py --method grid
```

### Live Trading

**Warning**: Always begin with paper trading to validate system behavior.

```bash
# Set environment variables
export COINBASE_API_KEY='your_api_key'
export COINBASE_API_SECRET='your_api_secret'

# Launch real trading bot
python run_eth_futures_simple.py

```

---

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Optimization | Optuna |
| Exchange APIs | CCXT, Coinbase Advanced Trade |
| Authentication | HMAC-SHA256 via cryptography library |
| Caching | Pickle |

---

## Indicators

### Bollinger Bands Calculation

```python
middle_band = simple_moving_average(close, period=19)
standard_deviation = std(close, period=19)
upper_band = middle_band + (2.47 × standard_deviation)
lower_band = middle_band - (2.47 × standard_deviation)
band_width = (upper_band - lower_band) / middle_band
```

### Keltner Channels Calculation

```python
middle_line = exponential_moving_average(close, period=17)
average_true_range = ATR(period=16)
upper_channel = middle_line + (2.38 × average_true_range)
lower_channel = middle_line - (2.38 × average_true_range)
```

### Squeeze Detection

```python
squeeze_active = (bollinger_lower > keltner_lower) AND 
                 (bollinger_upper < keltner_upper)
```

### Momentum Calculation

```python
raw_momentum = close - simple_moving_average(close, 19).shift(15)
normalized_momentum = raw_momentum / average_true_range(16)
```

### RSI Calculation

```python
price_change = close.diff()
gains = price_change.where(price_change > 0, 0).rolling(21).mean()
losses = (-price_change.where(price_change < 0, 0)).rolling(21).mean()
relative_strength = gains / losses
rsi = 100 - (100 / (1 + relative_strength))
```

---


## Testing and Validation

### Backtesting Methodology

The backtesting engine uses minute-by-minute data to accurately model real trading conditions:

- **Historical Data**: 6 years of cryptocurrency price data across multiple timeframes
- **Cost Modeling**: Commission charges of 0.05% and 0.02% slippage per trade leg
- **Data Cache**: Creates a cache of all data after the first download

### Walk-Forward Analysis

To prevent overfitting, the optimization process includes walk-forward validation:

1. Divide historical data into sequential segments
2. Optimize parameters on in-sample period
3. Test optimized parameters on subsequent out-of-sample period
4. Advance window and repeat
5. Aggregate out-of-sample results for realistic performance estimation

### Parameter Sensitivity

The optimization process evaluates robustness by:
- Testing parameter stability across different market regimes
- Measuring performance variance across parameter ranges
- Identifying parameters with largest impact on strategy performance

---

## License

This project is licensed under the MIT License. See LICENSE file for full text.

---

## Disclaimer

**This software is provided for educational and research purposes only.**

Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Never invest more than you can afford to lose. This software is not financial advice. Users are responsible for their own trading decisions and any resulting gains or losses.

The authors and contributors are not liable for any financial losses incurred through use of this software. Always thoroughly test strategies with paper trading before deploying real capital. Ensure you understand the risks involved in cryptocurrency trading.

---

## Contact

For questions, bug reports, or collaboration opportunities:

- GitHub Issues: [github.com/jicheolha/crypto-trading-bot/issues](https://github.com/jicheolha/crypto-trading-bot/issues)
- Repository: [github.com/jicheolha/crypto-trading-bot](https://github.com/jicheolha/crypto-trading-bot)

---

**Version**: 1.0.0  
**Last Updated**: January 2026
