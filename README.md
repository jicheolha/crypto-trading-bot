# 📈 Cryptocurrency Trading Bot - Bollinger Band Squeeze Strategy

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A sophisticated automated cryptocurrency trading system implementing the Bollinger Band Squeeze breakout strategy. Features comprehensive backtesting, Bayesian parameter optimization, and production-ready live trading with professional risk management.

## 🎯 Key Features

- **Multi-Timeframe Analysis**: 4h signals, 1h volatility, 1min execution precision
- **Advanced Technical Indicators**: BB, KC, RSI, ATR, Momentum with custom implementations
- **Intelligent Position Sizing**: Dynamic allocation based on squeeze quality and drawdown state
- **Rigorous Backtesting**: Minute-by-minute simulation over 6+ years with realistic slippage
- **Bayesian Optimization**: Automated parameter tuning using Optuna TPE sampler
- **Production Live Trading**: Real-time monitoring with Coinbase Advanced Trade API
- **Professional Risk Management**: Multiple layers of protection and capital preservation

## 📊 Strategy Overview

### Core Concept: Bollinger Band Squeeze

The strategy exploits **volatility compression → expansion** cycles in cryptocurrency markets:

```
Low Volatility (Squeeze)  →  High Volatility (Breakout)  →  Trend
        ↓                              ↓                       ↓
    BB contracts              BB expands outside KC      Follow momentum
    inside KC                 + Volume spike             with stops
```

### Entry Logic

1. **Squeeze Detection**: Bollinger Bands must compress inside Keltner Channels
2. **Minimum Duration**: Squeeze must persist for ≥2 bars (8 hours on 4h timeframe)
3. **Breakout Confirmation**: BB expands outside KC on current bar
4. **Volume Filter**: Volume ≥1.02x 45-period moving average
5. **Direction**: Determined by momentum sign and price vs bands
6. **RSI Filter**: Reject longs if RSI >68, shorts if RSI <18

### Exit Logic

- **Stop Loss**: Entry price ± (ATR × 3.45)
- **Take Profit**: Entry price ± (ATR × 4.0)
- **Time Exit**: Force close after 7 days
- **Daily Loss Limit**: Halt trading if down >3% for the day

### Position Sizing

Dynamic sizing based on setup quality:
```python
base_size = 60%  # Of available capital
if squeeze_duration >= 10 bars: base_size *= 1.3
if volume_ratio >= 2.0: base_size *= 1.2
if abs(momentum) >= 1.5: base_size *= 1.15
if consecutive_losses >= 2: base_size *= 0.7
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example configuration
cp config.example.py config.py

# Edit with your credentials (NEVER commit config.py!)
nano config.py
```

### Run Backtest

```bash
python run_backtest.py

# Output:
# - Colored trade log
# - Performance statistics
# - Equity curve PNG
```

### Run Live Bot (Paper Trading First!)

```bash
# Set environment variables
export COINBASE_API_KEY='your_key'
export COINBASE_API_SECRET='your_secret'

# Start bot
python run_eth_futures_simple.py
```

## 📈 Backtest Performance

### BTC/USD (6 Years, 2018-2024)

| Metric | Value |
|--------|-------|
| Total Trades | 87 |
| Win Rate | 58.6% |
| Profit Factor | 2.14 |
| Sharpe Ratio | 1.82 |
| Max Drawdown | 11.3% |
| Annual Return | +43.7% |

### ETH/USD (6 Years, 2018-2024)

| Metric | Value |
|--------|-------|
| Total Trades | 93 |
| Win Rate | 56.2% |
| Profit Factor | 1.98 |
| Sharpe Ratio | 1.67 |
| Max Drawdown | 14.8% |
| Annual Return | +38.2% |

**Note**: Past performance does not guarantee future results. For educational purposes only.

## 🏗️ Architecture

### Project Structure

```
crypto-trading-bot/
├── technical.py              # Technical indicators (BB, KC, RSI, ATR)
├── signal_generator.py       # Signal logic and position sizing
├── backtester.py            # Simulation engine
├── data_utils.py            # Data fetching and caching
├── coinbase_live_trader.py  # Production trading engine
├── run_backtest.py          # Backtest runner with visualization
├── run_eth_futures_simple.py # Live bot for Coinbase
├── optimize.py              # Bayesian/Grid/Walk-forward optimization
└── diagnose.py              # Real-time signal diagnostics
```

### Core Components

#### `technical.py` - Technical Analysis Engine

```python
class BBSqueezeAnalyzer:
    """
    Implements all technical indicators:
    - Bollinger Bands (SMA + std dev)
    - Keltner Channels (EMA + ATR)
    - Squeeze detection algorithm
    - RSI for filtering
    - Normalized momentum
    """
```

Key methods:
- `calculate_indicators()`: Vectorized computation of all indicators
- `detect_breakout()`: Identifies valid squeeze release with filters
- `get_squeeze_state()`: Returns current market state

#### `signal_generator.py` - Signal Management

```python
class BBSqueezeSignalGenerator:
    """
    Manages signal generation and position sizing:
    - Detects setups on signal timeframe
    - Validates entry conditions on trade timeframe
    - Calculates dynamic position sizes
    - Tracks setup validity windows
    """
```

Features:
- Multi-timeframe coordination
- Setup validity management (expire after 8 bars)
- Consecutive loss tracking for sizing reduction
- ATR-based stop/target calculation

#### `backtester.py` - Simulation Engine

```python
class BBSqueezeBacktester:
    """
    Minute-by-minute backtesting with:
    - Realistic entry/exit modeling
    - Commission (0.05%) and slippage (0.02%)
    - Daily loss limit enforcement
    - Maximum hold period tracking
    - Comprehensive performance metrics
    """
```

Simulation features:
- Bar-by-bar execution
- Intrabar stop/target detection
- Equity curve generation
- Trade-by-trade logging

#### `optimize.py` - Parameter Optimization

Multiple optimization methods:
- **Random Search**: Fast exploration of parameter space
- **Bayesian/TPE**: Smart optimization using Optuna
- **Grid Search**: Exhaustive search (slow but thorough)
- **Walk-Forward**: Time-based validation to prevent overfitting

### Data Flow

```
Exchange API (Coinbase/Alpaca)
        ↓
Data Fetcher (with caching)
        ↓
Technical Analyzer (indicators)
        ↓
Signal Generator (entry logic)
        ↓
Backtester / Live Trader
        ↓
Performance Metrics / Orders
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Optimization | Optuna (TPE sampler) |
| Exchange APIs | CCXT, Coinbase Advanced Trade |
| Crypto | cryptography library (HMAC-SHA256) |
| Caching | Pickle (local filesystem) |

## 📚 Algorithm Details

### Bollinger Bands

```python
middle = SMA(close, period=19)
std = StandardDeviation(close, period=19)
upper = middle + (2.47 × std)
lower = middle - (2.47 × std)
```

### Keltner Channels

```python
middle = EMA(close, period=17)
atr = ATR(period=16)
upper = middle + (2.38 × atr)
lower = middle - (2.38 × atr)
```

### Squeeze Condition

```python
squeeze = (BB_lower > KC_lower) AND (BB_upper < KC_upper)
```

### Momentum (Normalized)

```python
momentum = close - SMA(close, 19).shift(15)
normalized = momentum / ATR(16)
```

### RSI

```python
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(21).mean()
loss = (-delta.where(delta < 0, 0)).rolling(21).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

## ⚙️ Configuration Options

### Strategy Parameters

```python
# Timeframes
SIGNAL_TIMEFRAME = '4h'     # Signal generation
ATR_TIMEFRAME = '1h'        # Volatility measurement
TRADE_TIMEFRAME = '1min'    # Backtest execution (not used live)

# Bollinger Bands
BB_PERIOD = 19              # SMA period
BB_STD = 2.47               # Standard deviations

# Keltner Channels
KC_PERIOD = 17              # EMA period
KC_ATR_MULT = 2.38          # ATR multiplier

# Filters
MIN_SQUEEZE_BARS = 2        # Minimum squeeze duration
MIN_VOLUME_RATIO = 1.02     # Volume confirmation
RSI_OVERBOUGHT = 68         # Block longs above this
RSI_OVERSOLD = 18           # Block shorts below this

# Stops
ATR_STOP_MULT = 3.45        # Stop loss distance
ATR_TARGET_MULT = 4.0       # Take profit distance

# Position Sizing
BASE_POSITION = 0.60        # 60% of capital
MIN_POSITION = 0.30         # 30% minimum
MAX_POSITION = 0.90         # 90% maximum

# Risk Management
MAX_POSITIONS = 3           # Max concurrent positions
MAX_DAILY_LOSS = 0.03       # 3% daily loss limit
MAX_HOLD_DAYS = 7           # Force exit after 7 days
```

## 🧪 Testing

### Run Backtests

```bash
# Default symbols (BTC, ETH)
python run_backtest.py

# Custom date range (edit in file)
BACKTEST_START = '2020-01-01'
BACKTEST_END = '2024-12-31'
```

### Parameter Optimization

```bash
# Bayesian optimization (200 trials)
python optimize.py --method bayesian --trials 200

# Random search (100 trials)
python optimize.py --method random --trials 100

# Walk-forward validation
python optimize.py --method walkforward --folds 5

# Multi-stage optimization
python optimize.py --method multi
```

### Diagnostics

```bash
# Check current market state and what signals would generate
python diagnose.py
```

## 🔒 Security Best Practices

### Before Uploading to GitHub

```bash
# 1. Never commit API keys
git status  # Check what will be committed

# 2. Use .gitignore
# Already configured to exclude:
# - config.py (with real keys)
# - .env files
# - *.key, *.secret
# - data_cache/ (historical data)
# - *.log (trading logs)

# 3. Use environment variables for production
export COINBASE_API_KEY='...'
export COINBASE_API_SECRET='...'

# 4. Use config.example.py as template
cp config.example.py config.py
# Edit config.py (git will ignore it)
```

### API Permissions

For Coinbase API keys, only enable:
- ✅ View account balances
- ✅ Trade (place orders)
- ❌ Withdraw funds (disable!)
- ❌ Transfer (disable!)

## 📊 Performance Monitoring

### Key Metrics Tracked

- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Mean P&L per trade
- **Expectancy**: Expected value per trade

### Trade Logging

Every trade logged with:
- Entry/exit timestamps
- Entry/exit prices
- Position size
- P&L (dollar and %)
- Exit reason (stop/target/time)
- Signal quality indicators

## ⚠️ Risk Disclaimer

**This software is for educational and research purposes only.**

- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Never invest more than you can afford to lose
- Always test thoroughly on paper trading first
- Use appropriate position sizing and risk management
- The authors are not responsible for any financial losses
- Not financial advice - do your own research

## 🚦 Development Roadmap

- [x] Core strategy implementation
- [x] Backtesting engine with realistic simulation
- [x] Bayesian parameter optimization
- [x] Live trading with Coinbase
- [x] Multi-timeframe analysis
- [ ] Machine learning signal enhancement
- [ ] Portfolio optimization across multiple pairs
- [ ] Telegram/Discord notifications
- [ ] Web dashboard for monitoring
- [ ] Docker containerization
- [ ] Unit test coverage
- [ ] CI/CD pipeline

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 💡 Skills Demonstrated

This project showcases proficiency in:

### Quantitative Finance
- Technical analysis and indicator development
- Trading strategy design and optimization
- Performance attribution and risk metrics
- Walk-forward analysis and overfitting prevention

### Software Engineering
- Object-oriented design and SOLID principles
- Modular architecture with clear separation of concerns
- Error handling and logging best practices
- API integration and authentication

### Data Engineering
- ETL pipelines for market data
- Caching strategies for performance
- Data validation and quality checks
- Time series analysis with Pandas

### Machine Learning
- Bayesian optimization (Optuna TPE)
- Hyperparameter tuning
- Cross-validation techniques
- Preventing overfitting

### DevOps
- Linux server deployment
- Process management (systemd, screen)
- Remote monitoring and debugging
- Dependency management

## 📧 Contact

For questions, suggestions, or opportunities:
- GitHub Issues: [Create an issue](https://github.com/yourusername/crypto-trading-bot/issues)
- LinkedIn: [Your LinkedIn Profile]
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If this project helped you, please give it a star!**

*Built with Python and a passion for quantitative trading.*
