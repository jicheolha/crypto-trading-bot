"""
Bollinger Bands Squeeze Breakout Strategy - Backtest Runner
"""
import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from backtester import BBSqueezeBacktester


# ============================================================================
# COLORS
# ============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def colored(text, color):
    return f"{color}{text}{Colors.RESET}"


def filter_data_by_date(data, start_date=None, end_date=None):
    if not start_date and not end_date:
        return data
    filtered = {}
    for symbol in data:
        filtered[symbol] = {}
        for tf_key, df in data[symbol].items():
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= df.index >= pd.Timestamp(start_date, tz=df.index.tz)
            if end_date:
                mask &= df.index <= pd.Timestamp(end_date + ' 23:59:59', tz=df.index.tz)
            filtered[symbol][tf_key] = df[mask]
    return filtered


# ============================================================================
# OPTIMIZED PARAMETERS
# ============================================================================

SYMBOLS = ['ETH/USD']
DAYS_BACK = 365*6
INITIAL_CAPITAL = 100000
COMMISSION = 0.0005
SLIPPAGE = 0.0002

BACKTEST_START = None  # None = use all data
BACKTEST_END = None    # None = use all data

# Timeframes
TRADE_TIMEFRAME = '1min'
SIGNAL_TIMEFRAME = '4h'
ATR_TIMEFRAME = '1h'

# Bollinger Bands
BB_PERIOD = 19
BB_STD = 2.47

# Keltner Channels
KC_PERIOD = 17
KC_ATR_MULT = 2.38

# Momentum
MOMENTUM_PERIOD = 15

# RSI
RSI_PERIOD = 21
RSI_OVERBOUGHT = 68
RSI_OVERSOLD = 18

# Squeeze
MIN_SQUEEZE_BARS = 2

# Volume 
VOLUME_PERIOD = 45
MIN_VOLUME_RATIO = 1.02

# Stops
ATR_PERIOD = 16
ATR_STOP_MULT = 3.45
ATR_TARGET_MULT = 4.0

# Position Sizing
BASE_POSITION = 0.6
MIN_POSITION = 0.3
MAX_POSITION = 0.9

# Setup
SETUP_VALIDITY_BARS = 8

# Risk
MAX_POSITIONS = 3
MAX_DAILY_LOSS = 0.03
MAX_HOLD_DAYS = 7
LONG_ONLY = False

# Output
SAVE_EQUITY_PLOT = True


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'crypto_trading'))
        from data_utils import load_three_timeframe_data
        
        data = load_three_timeframe_data(
            SYMBOLS,
            trade_tf=TRADE_TIMEFRAME,
            signal_tf=SIGNAL_TIMEFRAME,
            atr_tf=ATR_TIMEFRAME,
            days_back=DAYS_BACK,
            use_cache=True
        )
        return data
    except ImportError:
        print("Error: Could not import data_utils.")
        return None


def _get_tf_minutes(tf):
    mapping = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60, '4h': 240, '1d': 1440}
    return mapping.get(tf, 60)


# ============================================================================
# COLORED OUTPUT
# ============================================================================

def print_colored_trades(trades):
    print(f"\n{Colors.BOLD}{'='*100}")
    print("TRADE LOG")
    print(f"{'='*100}{Colors.RESET}")
    print(f"{'Date':<12} {'Dir':<8} {'Entry':>12} {'Exit':>12} {'P&L':>12} {'Duration':<12} {'Reasons'}")
    print("-" * 100)
    
    for t in trades:
        date_str = t.entry_time.strftime('%Y-%m-%d')
        direction = t.direction.upper()
        
        # Duration
        if t.exit_time:
            dur = t.exit_time - t.entry_time
            days, hours = dur.days, dur.seconds // 3600
            dur_str = f"{days}d {hours}h" if days > 0 else f"{hours}h"
        else:
            dur_str = "N/A"
        
        reasons = ' '.join(t.reasons) if t.reasons else ''
        
        # Direction with proper spacing for alignment
        if direction == 'LONG':
            dir_str = "LONG"
            dir_col = colored(f"{dir_str:<8}", Colors.CYAN)
        else:
            dir_str = "SHORT"
            dir_col = colored(f"{dir_str:<8}", Colors.MAGENTA)
        
        # P&L with color
        pnl_str = f"${t.pnl:+,.0f}"
        if t.pnl >= 0:
            pnl_col = colored(f"{pnl_str:>12}", Colors.GREEN)
        else:
            pnl_col = colored(f"{pnl_str:>12}", Colors.RED)
        
        print(f"{date_str:<12} {dir_col} ${t.entry_price:>11,.0f} ${t.exit_price:>11,.0f} {pnl_col} {dur_str:<12} {reasons}")
    
    print("-" * 100)
    
    wins = [t for t in trades if t.pnl >= 0]
    losses = [t for t in trades if t.pnl < 0]
    longs = [t for t in trades if t.direction == 'long']
    shorts = [t for t in trades if t.direction == 'short']
    
    print(f"\n{colored(f'Wins: {len(wins)}', Colors.GREEN)} | {colored(f'Losses: {len(losses)}', Colors.RED)}")
    print(f"{colored(f'Longs: {len(longs)}', Colors.CYAN)} | {colored(f'Shorts: {len(shorts)}', Colors.MAGENTA)}")


def print_colored_results(stats):
    print(f"\n{Colors.BOLD}{'='*27}")
    print("BACKTEST RESULTS")
    print(f"{'='*27}{Colors.RESET}")
    
    print(f"Total Trades:    {stats['total_trades']}")
    
    wr = stats['win_rate']
    wr_color = Colors.GREEN if wr >= 55 else (Colors.YELLOW if wr >= 50 else Colors.RED)
    print(f"Win Rate:        {colored(f'{wr:.1f}%', wr_color)}")
    
    pf = stats['profit_factor']
    pf_color = Colors.GREEN if pf >= 1.5 else (Colors.YELLOW if pf >= 1.0 else Colors.RED)
    print(f"Profit Factor:   {colored(f'{pf:.2f}', pf_color)}")
    
    pnl = stats['total_pnl']
    pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
    print(f"Total P&L:       {colored(f'${pnl:+,.0f}', pnl_color)}")
    
    dd = stats['max_drawdown']
    dd_color = Colors.GREEN if dd <= 15 else (Colors.YELLOW if dd <= 25 else Colors.RED)
    print(f"Max Drawdown:    {colored(f'{dd:.1f}%', dd_color)}")
    
    sharpe = stats['sharpe']
    sh_color = Colors.GREEN if sharpe >= 1.5 else (Colors.YELLOW if sharpe >= 1.0 else Colors.RED)
    print(f"Sharpe Ratio:    {colored(f'{sharpe:.2f}', sh_color)}")
    
    print(f"Final Equity:    ${stats['final_equity']:,.0f}")
    
    ret = stats['return_pct']
    ret_color = Colors.GREEN if ret >= 0 else Colors.RED
    print(f"Return:          {colored(f'{ret:+.1f}%', ret_color)}")
    
    print("=" * 27)


def save_equity_plot(equity_series, trades, filepath):
    if equity_series is None or len(equity_series) == 0:
        print("No equity data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(equity_series.index, equity_series.values, 'b-', linewidth=1.5, label='Equity')
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Mark trades
    for t in trades:
        try:
            idx = equity_series.index.get_indexer([t.entry_time], method='nearest')[0]
            eq = equity_series.iloc[idx]
            marker = '^' if t.direction == 'long' else 'v'
            color = 'green' if t.pnl >= 0 else 'red'
            ax1.scatter(t.entry_time, eq, marker=marker, color=color, s=60, alpha=0.7, zorder=5)
        except:
            pass
    
    ax1.set_title('BB Squeeze Strategy - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    
    ax2.fill_between(equity_series.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(equity_series.index, drawdown, 'r-', linewidth=1)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Equity plot saved to {filepath}')


# ============================================================================
# MAIN
# ============================================================================

def run_backtest():
    print(f"\n{Colors.BOLD}{'='*41}")
    print("BOLLINGER BANDS SQUEEZE BREAKOUT STRATEGY")
    print(f"{'='*41}{Colors.RESET}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Period: {DAYS_BACK} days ({DAYS_BACK/365:.1f} years)")
    print(f"Timeframes: Trade={TRADE_TIMEFRAME}, Signal={SIGNAL_TIMEFRAME}, ATR={ATR_TIMEFRAME}")
    print(f"BB: {BB_PERIOD} period, {BB_STD} std")
    print(f"KC: {KC_PERIOD} period, {KC_ATR_MULT}x ATR")
    print(f"Squeeze: {MIN_SQUEEZE_BARS}+ bars")
    print(f"Stops: {ATR_STOP_MULT}x SL, {ATR_TARGET_MULT}x TP")
    print(f"Position: {BASE_POSITION:.0%} base, {MIN_POSITION:.0%}-{MAX_POSITION:.0%} range")
    
    print(f"\nLoading data...")
    data = load_data()
    
    if not data:
        print("Failed to load data")
        return
    
    data = filter_data_by_date(data, BACKTEST_START, BACKTEST_END)
    
    analyzer = BBSqueezeAnalyzer(
        bb_period=BB_PERIOD,
        bb_std=BB_STD,
        kc_period=KC_PERIOD,
        kc_atr_mult=KC_ATR_MULT,
        momentum_period=MOMENTUM_PERIOD,
        rsi_period=RSI_PERIOD,
        volume_period=VOLUME_PERIOD,
        atr_period=ATR_PERIOD
    )
    
    signal_gen = BBSqueezeSignalGenerator(
        analyzer=analyzer,
        min_squeeze_bars=MIN_SQUEEZE_BARS,
        min_volume_ratio=MIN_VOLUME_RATIO,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        atr_stop_mult=ATR_STOP_MULT,
        atr_target_mult=ATR_TARGET_MULT,
        base_position=BASE_POSITION,
        min_position=MIN_POSITION,
        max_position=MAX_POSITION,
        setup_validity_bars=SETUP_VALIDITY_BARS,
        signal_timeframe_minutes=_get_tf_minutes(SIGNAL_TIMEFRAME)
    )
    
    signal_data = {s: data[s]['signal'] for s in data}
    atr_data = {s: data[s]['atr'] for s in data}
    signal_gen.set_signal_data(signal_data)
    signal_gen.set_atr_data(atr_data)
    
    backtester = BBSqueezeBacktester(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage_pct=SLIPPAGE,
        max_positions=MAX_POSITIONS,
        max_daily_loss_pct=MAX_DAILY_LOSS,
        max_hold_days=MAX_HOLD_DAYS,
        long_only=LONG_ONLY,
        verbose=False
    )
    
    backtester.analyzer = analyzer
    backtester.signal_generator = signal_gen
    
    trade_data = {s: data[s]['trade'] for s in data}
    
    first_symbol = list(data.keys())[0]
    start = data[first_symbol]['trade'].index[0]
    end = data[first_symbol]['trade'].index[-1]
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    print(f"\nRunning backtest...")
    results = backtester.run_backtest(trade_data)
    
    if results.trades:
        print_colored_trades(results.trades)
    
    print_colored_results(results.statistics)
    
    if SAVE_EQUITY_PLOT:
        symbols_str = '_'.join([s.replace('/', '') for s in SYMBOLS])
        plot_filename = f'equity_curve_{symbols_str}.png'
        save_equity_plot(results.equity_curve, results.trades, plot_filename)
    
    return results


if __name__ == "__main__":
    run_backtest()