"""
Fast Backtest Runner - Using Cached Alpaca Data

First run: Downloads data and caches it (~2-5 min)
Future runs: Loads from cache (< 1 second)

Pro tip: Run download_data.py first to pre-download everything!
"""
import os
import sys
import logging
import warnings
from datetime import datetime
import time

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from backtester import BBSqueezeBacktester
from data_utils import load_three_timeframe_data


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


# ============================================================================
# PARAMETERS
# ============================================================================

# Symbols
SYMBOLS = ['BTC/USD', 'DOGE/USD']

# Capital & Fees
INITIAL_CAPITAL = 100000
COMMISSION = 0.0005
SLIPPAGE = 0.0002

# Date range (None = all available data)
BACKTEST_START = '2021-01-01'  # Start of your test period
BACKTEST_END = '2025-12-28'    # End of your test period

# History
DAYS_BACK = 365*6              # Always load full cache

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
BASE_POSITION = 0.4
MIN_POSITION = 0.0
MAX_POSITION = 0.5

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
# HELPER FUNCTIONS
# ============================================================================

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


def _get_tf_minutes(tf):
    mapping = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60, '4h': 240, '1d': 1440}
    return mapping.get(tf, 60)


def print_colored_trades(trades):
    print(f"\n{Colors.BOLD}{'='*155}")
    print("DETAILED TRADE LOG")
    print(f"{'='*155}{Colors.RESET}")
    print(f"{'#':<4} {'Date':<12} {'Dir':<7} {'Size%':<7} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'R':<6} {'Risk%':<7} {'Dur':<10} {'Reasons'}")
    print("-" * 155)
    
    cumulative_pnl = 0
    
    for i, t in enumerate(trades, 1):
        date_str = t.entry_time.strftime('%Y-%m-%d')
        direction = t.direction.upper()
        
        # Duration
        if t.exit_time:
            dur = t.exit_time - t.entry_time
            total_hours = dur.total_seconds() / 3600
            if dur.days > 0:
                dur_str = f"{dur.days}d {dur.seconds // 3600}h"
            else:
                dur_str = f"{int(total_hours)}h"
        else:
            dur_str = "N/A"
        
        reasons = ' '.join(t.reasons) if t.reasons else ''
        
        # Color direction
        if direction == 'LONG':
            dir_col = colored(f"â–² LONG ", Colors.CYAN)
        else:
            dir_col = colored(f"â–¼ SHORT", Colors.MAGENTA)
        
        # Color P&L
        if t.pnl >= 0:
            pnl_col = colored(f"${t.pnl:+,.0f}", Colors.GREEN)
        else:
            pnl_col = colored(f"${t.pnl:+,.0f}", Colors.RED)
        
        # Color R-multiple
        if t.r_multiple >= 1.0:
            r_col = colored(f"{t.r_multiple:+.1f}R", Colors.GREEN)
        elif t.r_multiple >= 0:
            r_col = colored(f"{t.r_multiple:+.1f}R", Colors.YELLOW)
        else:
            r_col = colored(f"{t.r_multiple:+.1f}R", Colors.RED)
        
        cumulative_pnl += t.pnl
        
        print(f"{i:<4} {date_str:<12} {dir_col} {t.position_size_pct:>5.0f}%  "
              f"${t.entry_price:>10,.0f} ${t.exit_price:>10,.0f} {pnl_col:>22} "
              f"{r_col:>13} {t.risk_pct:>5.1f}%  {dur_str:<10} {reasons}")
    
    print("-" * 155)
    
    wins = [t for t in trades if t.pnl >= 0]
    losses = [t for t in trades if t.pnl < 0]
    longs = [t for t in trades if t.direction == 'long']
    shorts = [t for t in trades if t.direction == 'short']
    
    # Calculate R-multiple stats
    r_multiples = [t.r_multiple for t in trades]
    avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
    avg_r_win = sum([t.r_multiple for t in wins]) / len(wins) if wins else 0
    avg_r_loss = sum([t.r_multiple for t in losses]) / len(losses) if losses else 0
    
    print(f"\n{colored(f'Wins: {len(wins)}', Colors.GREEN)} | {colored(f'Losses: {len(losses)}', Colors.RED)}")
    print(f"{colored(f'Longs: {len(longs)}', Colors.CYAN)} | {colored(f'Shorts: {len(shorts)}', Colors.MAGENTA)}")
    print(f"\nAvg R-multiple: {avg_r:+.2f}R | Avg Win: {avg_r_win:+.2f}R | Avg Loss: {avg_r_loss:+.2f}R")
    print(f"Avg Position Size: {sum([t.position_size_pct for t in trades])/len(trades):.1f}%")
    print(f"Avg Risk per Trade: {sum([t.risk_pct for t in trades])/len(trades):.2f}%")


def print_colored_results(stats):
    print(f"\n{Colors.BOLD}{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Core Metrics:{Colors.RESET}")
    print(f"Total Trades:       {stats['total_trades']}")
    
    wr = stats['win_rate']
    wr_color = Colors.GREEN if wr >= 55 else (Colors.YELLOW if wr >= 50 else Colors.RED)
    print(f"Win Rate:           {colored(f'{wr:.1f}%', wr_color)}")
    
    pf = stats['profit_factor']
    pf_color = Colors.GREEN if pf >= 1.5 else (Colors.YELLOW if pf >= 1.0 else Colors.RED)
    print(f"Profit Factor:      {colored(f'{pf:.2f}', pf_color)}")
    
    pnl = stats['total_pnl']
    pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
    print(f"Total P&L:          {colored(f'${pnl:+,.0f}', pnl_color)}")
    
    ret = stats['return_pct']
    ret_color = Colors.GREEN if ret >= 0 else Colors.RED
    print(f"Return:             {colored(f'{ret:+.1f}%', ret_color)}")
    
    print(f"Final Equity:       ${stats['final_equity']:,.0f}")
    
    print(f"\n{Colors.BOLD}Risk Metrics:{Colors.RESET}")
    dd = stats['max_drawdown']
    dd_color = Colors.GREEN if dd <= 15 else (Colors.YELLOW if dd <= 25 else Colors.RED)
    print(f"Max Drawdown:       {colored(f'{dd:.1f}%', dd_color)}")
    
    sharpe = stats['sharpe']
    sh_color = Colors.GREEN if sharpe >= 1.5 else (Colors.YELLOW if sharpe >= 1.0 else Colors.RED)
    print(f"Sharpe Ratio:       {colored(f'{sharpe:.2f}', sh_color)}")
    
    print(f"Max Consec Wins:    {stats['max_consec_wins']}")
    print(f"Max Consec Losses:  {stats['max_consec_losses']}")
    
    print(f"\n{Colors.BOLD}R-Multiple Analysis:{Colors.RESET}")
    avg_r = stats['avg_r_multiple']
    r_color = Colors.GREEN if avg_r >= 0.5 else (Colors.YELLOW if avg_r >= 0 else Colors.RED)
    print(f"Avg R-Multiple:     {colored(f'{avg_r:+.2f}R', r_color)}")
    avg_r_win_str = f"{stats['avg_r_win']:+.2f}R"
    print(f"Avg Win R:          {colored(avg_r_win_str, Colors.GREEN)}")
    avg_r_loss_str = f"{stats['avg_r_loss']:+.2f}R"
    print(f"Avg Loss R:         {colored(avg_r_loss_str, Colors.RED)}")
    
    print(f"\n{Colors.BOLD}Trade Quality:{Colors.RESET}")
    exp = stats['expectancy']
    exp_color = Colors.GREEN if exp >= 0 else Colors.RED
    print(f"Expectancy:         {colored(f'${exp:+,.0f} per trade', exp_color)}")
    
    kelly = stats['kelly_pct']
    kelly_color = Colors.GREEN if 10 <= kelly <= 25 else (Colors.YELLOW if kelly > 0 else Colors.RED)
    print(f"Kelly %:            {colored(f'{kelly:.1f}% suggested', kelly_color)}")
    
    avg_hold = stats['avg_hold_hours']
    if avg_hold >= 24:
        hold_str = f"{avg_hold/24:.1f} days"
    else:
        hold_str = f"{avg_hold:.1f} hours"
    print(f"Avg Hold Time:      {hold_str}")
    
    print("=" * 50)


def print_colored_results(stats):
    print(f"\n{Colors.BOLD}{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Core Metrics:{Colors.RESET}")
    print(f"Total Trades:       {stats['total_trades']}")
    
    wr = stats['win_rate']
    wr_color = Colors.GREEN if wr >= 55 else (Colors.YELLOW if wr >= 50 else Colors.RED)
    print(f"Win Rate:           {colored(f'{wr:.1f}%', wr_color)}")
    
    pf = stats['profit_factor']
    pf_color = Colors.GREEN if pf >= 1.5 else (Colors.YELLOW if pf >= 1.0 else Colors.RED)
    print(f"Profit Factor:      {colored(f'{pf:.2f}', pf_color)}")
    
    pnl = stats['total_pnl']
    pnl_color = Colors.GREEN if pnl >= 0 else Colors.RED
    print(f"Total P&L:          {colored(f'${pnl:+,.0f}', pnl_color)}")
    
    ret = stats['return_pct']
    ret_color = Colors.GREEN if ret >= 0 else Colors.RED
    print(f"Return:             {colored(f'{ret:+.1f}%', ret_color)}")
    
    print(f"Final Equity:       ${stats['final_equity']:,.0f}")
    
    print(f"\n{Colors.BOLD}Risk Metrics:{Colors.RESET}")
    dd = stats['max_drawdown']
    dd_color = Colors.GREEN if dd <= 15 else (Colors.YELLOW if dd <= 25 else Colors.RED)
    print(f"Max Drawdown:       {colored(f'{dd:.1f}%', dd_color)}")
    
    sharpe = stats['sharpe']
    sh_color = Colors.GREEN if sharpe >= 1.5 else (Colors.YELLOW if sharpe >= 1.0 else Colors.RED)
    print(f"Sharpe Ratio:       {colored(f'{sharpe:.2f}', sh_color)}")
    
    print(f"Max Consec Wins:    {stats['max_consec_wins']}")
    print(f"Max Consec Losses:  {stats['max_consec_losses']}")
    
    print(f"\n{Colors.BOLD}R-Multiple Analysis:{Colors.RESET}")
    avg_r = stats['avg_r_multiple']
    r_color = Colors.GREEN if avg_r >= 0.5 else (Colors.YELLOW if avg_r >= 0 else Colors.RED)
    print(f"Avg R-Multiple:     {colored(f'{avg_r:+.2f}R', r_color)}")
    avg_r_win_str = f"{stats['avg_r_win']:+.2f}R"
    print(f"Avg Win R:          {colored(avg_r_win_str, Colors.GREEN)}")
    avg_r_loss_str = f"{stats['avg_r_loss']:+.2f}R"
    print(f"Avg Loss R:         {colored(avg_r_loss_str, Colors.RED)}")
    
    print(f"\n{Colors.BOLD}Trade Quality:{Colors.RESET}")
    exp = stats['expectancy']
    exp_color = Colors.GREEN if exp >= 0 else Colors.RED
    print(f"Expectancy:         {colored(f'${exp:+,.0f} per trade', exp_color)}")
    
    kelly = stats['kelly_pct']
    kelly_color = Colors.GREEN if 10 <= kelly <= 25 else (Colors.YELLOW if kelly > 0 else Colors.RED)
    print(f"Kelly %:            {colored(f'{kelly:.1f}% suggested', kelly_color)}")
    
    avg_hold = stats['avg_hold_hours']
    if avg_hold >= 24:
        hold_str = f"{avg_hold/24:.1f} days"
    else:
        hold_str = f"{avg_hold:.1f} hours"
    print(f"Avg Hold Time:      {hold_str}")
    
    print("=" * 50)


def save_equity_plot(equity_series, trades, filepath):
    if equity_series is None or len(equity_series) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(equity_series.index, equity_series.values, 'b-', linewidth=1.5, label='Equity')
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
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
    
    print(f'\nEquity plot saved to {filepath}')


# ============================================================================
# MAIN
# ============================================================================

def run_backtest():
    print(f"\n{Colors.BOLD}{'='*60}")
    print("FAST BACKTEST - BB SQUEEZE STRATEGY")
    print(f"{'='*60}{Colors.RESET}")
    print(f"Data Source: Alpaca (cached)")
    print(f"Symbols: {SYMBOLS}")
    print(f"Timeframes: Trade={TRADE_TIMEFRAME}, Signal={SIGNAL_TIMEFRAME}, ATR={ATR_TIMEFRAME}")
    
    # Show cache status
    try:
        print(f"\n{colored('Cache Status:', Colors.CYAN)}")
        list_cached_files()
    except:
        pass
    
    # Load data (with caching!)
    print(f"\n{colored('Loading data...', Colors.CYAN)}")
    
    load_start = time.time()
    
    data = load_three_timeframe_data(
        SYMBOLS,
        trade_tf=TRADE_TIMEFRAME,
        signal_tf=SIGNAL_TIMEFRAME,
        atr_tf=ATR_TIMEFRAME,
        days_back=DAYS_BACK,
        use_cache=True,
        cache_max_age_hours=None  # Never expire!
    )
    
    load_time = time.time() - load_start
    
    if not data:
        print(f"\n{colored('ERROR: No data loaded!', Colors.RED)}")
        print("\nDid you download data first?")
        print("Run: python download_data.py")
        return
    
    print(f"\n{colored(f'Data loaded in {load_time:.1f} seconds', Colors.GREEN)}")
    
    # Filter by date if specified
    if BACKTEST_START or BACKTEST_END:
        data = filter_data_by_date(data, BACKTEST_START, BACKTEST_END)
    
    # Show period
    first_symbol = list(data.keys())[0]
    start = data[first_symbol]['signal'].index[0]
    end = data[first_symbol]['signal'].index[-1]
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    # Initialize strategy
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
    
    # Run backtest
    print(f"\n{colored('Running backtest...', Colors.CYAN)}")
    
    bt_start = time.time()
    results = backtester.run_backtest(trade_data)
    bt_time = time.time() - bt_start
    
    # Display results
    if results.trades:
        print_colored_trades(results.trades)
    
    print_colored_results(results.statistics)
    
    # Save plot
    if SAVE_EQUITY_PLOT:
        symbols_str = '_'.join([s.replace('/', '') for s in SYMBOLS])
        plot_filename = f'equity_curve_{symbols_str}.png'
        save_equity_plot(results.equity_curve, results.trades, plot_filename)
    
    total_time = load_time + bt_time
    print(f"\n{colored(f'Total time: {total_time:.1f}s (load: {load_time:.1f}s, backtest: {bt_time:.1f}s)', Colors.GREEN)}")
    
    return results


if __name__ == "__main__":
    run_backtest()