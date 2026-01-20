#!/usr/bin/env python3
"""
Monte Carlo Permutation Test for Trading Strategy Validation

Detects overfitting by testing if strategy performance is due to
real patterns or random chance.

Method:
1. Run strategy on REAL data Ã¢â€ â€™ get actual performance metrics
2. Shuffle returns N times (destroys temporal patterns)
3. Run strategy on each shuffled dataset
4. Calculate p-values: % of shuffled results that beat real results

Interpretation:
- p-value < 0.05: Strategy likely found REAL patterns (good!)
- p-value > 0.20: Strategy likely OVERFIT to noise (bad!)
- p-value 0.05-0.20: Inconclusive, need more testing

Usage:
    python permutation_test.py --symbol ETH/USD --permutations 100
    python permutation_test.py --symbol BTC/USD --permutations 500 --plot
"""
import argparse
import sys
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from backtester import BBSqueezeBacktester


# ============================================================================
# CONFIGURATION - Your optimized parameters
# ============================================================================

DEFAULT_PARAMS = {
    'signal_timeframe': '4h',
    'atr_timeframe': '1h',
    'bb_period': 19,
    'bb_std': 2.47,
    'kc_period': 17,
    'kc_atr_mult': 2.38,
    'momentum_period': 15,
    'rsi_period': 21,
    'rsi_overbought': 68,
    'rsi_oversold': 18,
    'min_squeeze_bars': 2,
    'volume_period': 45,
    'min_volume_ratio': 1.02,
    'atr_period': 16,
    'atr_stop_mult': 3.45,
    'atr_target_mult': 4.0,
    'setup_validity_bars': 8,
    'max_positions': 3,
    'max_daily_loss': 0.03,
}

# Position sizing
BASE_POSITION = 0.60
MIN_POSITION = 0.30
MAX_POSITION = 0.90

# Backtest settings
INITIAL_CAPITAL = 100000
LONG_ONLY = False
DAYS_BACK = 365 * 6

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')


# Import shared utilities
from utils import Colors, colored


# ============================================================================
# DATA LOADING
# ============================================================================

def _get_cache_path(symbol: str, timeframe: str, days_back: int) -> str:
    """Get cache file path."""
    symbol_clean = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{symbol_clean}_{timeframe}_{days_back}d.pkl")


def load_cached_data(symbol: str, days_back: int) -> Optional[Dict[str, pd.DataFrame]]:
    """Load all timeframes from cache."""
    timeframes = ['1min', '15min', '30min', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        cache_path = _get_cache_path(symbol, tf, days_back)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data[tf] = pickle.load(f)
            except Exception as e:
                print(f"  Warning: Failed to load {tf}: {e}")
    
    if not data:
        print(f"  No cached data found for {symbol}")
        return None
    
    return data


def filter_data_by_date(
    data: Dict[str, pd.DataFrame], 
    start_date: str = None, 
    end_date: str = None
) -> Dict[str, pd.DataFrame]:
    """Filter all timeframes by date range."""
    if not start_date and not end_date:
        return data
    
    filtered = {}
    for tf, df in data.items():
        mask = pd.Series(True, index=df.index)
        
        if start_date:
            # Handle timezone-aware indices
            start_ts = pd.Timestamp(start_date)
            if df.index.tz is not None:
                start_ts = start_ts.tz_localize(df.index.tz)
            mask &= df.index >= start_ts
        
        if end_date:
            end_ts = pd.Timestamp(end_date + ' 23:59:59')
            if df.index.tz is not None:
                end_ts = end_ts.tz_localize(df.index.tz)
            mask &= df.index <= end_ts
        
        filtered[tf] = df[mask]
    
    return filtered


# ============================================================================
# PERMUTATION METHODS
# ============================================================================

def shuffle_returns(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Create permuted price series by shuffling returns.
    
    This preserves:
    - Return distribution (mean, std, skew, kurtosis)
    - Starting price
    
    This destroys:
    - Temporal patterns (trends, squeezes, breakouts)
    - Autocorrelation
    - Volatility clustering
    """
    if seed is not None:
        np.random.seed(seed)
    
    df_perm = df.copy()
    
    # Calculate returns
    returns = df['close'].pct_change().dropna().values
    
    # Shuffle returns
    np.random.shuffle(returns)
    
    # Reconstruct prices from shuffled returns
    start_price = df['close'].iloc[0]
    new_prices = [start_price]
    
    for ret in returns:
        new_prices.append(new_prices[-1] * (1 + ret))
    
    new_prices = np.array(new_prices)
    
    # Calculate OHLC ratios from original data
    open_ratio = (df['open'] / df['close']).values
    high_ratio = (df['high'] / df['close']).values
    low_ratio = (df['low'] / df['close']).values
    
    # Shuffle the ratios (keeps relationship but randomizes when they occur)
    np.random.shuffle(open_ratio)
    np.random.shuffle(high_ratio)
    np.random.shuffle(low_ratio)
    
    # Apply ratios to new close prices
    df_perm['close'] = new_prices
    df_perm['open'] = new_prices * open_ratio
    df_perm['high'] = new_prices * np.maximum(high_ratio, 1.0)  # High must be >= close
    df_perm['low'] = new_prices * np.minimum(low_ratio, 1.0)   # Low must be <= close
    
    # Ensure OHLC consistency
    df_perm['high'] = df_perm[['open', 'high', 'close']].max(axis=1)
    df_perm['low'] = df_perm[['open', 'low', 'close']].min(axis=1)
    
    # Shuffle volume independently
    volume_shuffled = df['volume'].values.copy()
    np.random.shuffle(volume_shuffled)
    df_perm['volume'] = volume_shuffled
    
    return df_perm


def shuffle_blocks(df: pd.DataFrame, block_size: int = 20, seed: int = None) -> pd.DataFrame:
    """
    Block shuffle - preserves short-term patterns, destroys long-term.
    
    This is a less aggressive permutation that keeps local structure
    but randomizes when those patterns occur.
    """
    if seed is not None:
        np.random.seed(seed)
    
    df_perm = df.copy()
    n = len(df)
    n_blocks = n // block_size
    
    if n_blocks < 2:
        return shuffle_returns(df, seed)
    
    # Create block indices
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        blocks.append(df.iloc[start:end].copy())
    
    # Handle remainder
    if n % block_size != 0:
        blocks.append(df.iloc[n_blocks * block_size:].copy())
    
    # Shuffle blocks
    np.random.shuffle(blocks)
    
    # Reconstruct dataframe
    df_perm = pd.concat(blocks, ignore_index=False)
    
    # Re-index to maintain time order but with shuffled data
    df_perm.index = df.index[:len(df_perm)]
    
    # Stitch prices together at block boundaries
    current_price = df['close'].iloc[0]
    adjustment = current_price / df_perm['close'].iloc[0]
    
    for col in ['open', 'high', 'low', 'close']:
        df_perm[col] = df_perm[col] * adjustment
    
    return df_perm


def permute_all_timeframes(
    data: Dict[str, pd.DataFrame], 
    method: str = 'returns',
    seed: int = None
) -> Dict[str, pd.DataFrame]:
    """
    Permute all timeframes consistently.
    
    Uses the same seed so relative timing is somewhat preserved,
    but patterns are destroyed.
    """
    permuted = {}
    
    for tf, df in data.items():
        if method == 'returns':
            permuted[tf] = shuffle_returns(df, seed=seed)
        elif method == 'blocks':
            block_size = {'1min': 60, '15min': 20, '30min': 10, 
                         '1h': 10, '4h': 5, '1d': 3}.get(tf, 10)
            permuted[tf] = shuffle_blocks(df, block_size=block_size, seed=seed)
        else:
            permuted[tf] = shuffle_returns(df, seed=seed)
    
    return permuted


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def get_tf_minutes(tf: str) -> int:
    return {'1min': 1, '5min': 5, '15min': 15, '30min': 30, 
            '1h': 60, '4h': 240, '1d': 1440}.get(tf, 60)


def run_backtest_on_data(
    data: Dict[str, pd.DataFrame], 
    params: Dict,
    symbol: str
) -> Dict:
    """Run backtest and return statistics."""
    
    signal_tf = params['signal_timeframe']
    atr_tf = params['atr_timeframe']
    
    # Prepare data
    trade_data = {symbol: data.get('1min', data.get('15min', list(data.values())[0]))}
    signal_data = {symbol: data.get(signal_tf, data.get('1h', list(data.values())[0]))}
    atr_data = {symbol: data.get(atr_tf, data.get('4h', list(data.values())[0]))}
    
    # Initialize analyzer
    analyzer = BBSqueezeAnalyzer(
        bb_period=params['bb_period'],
        bb_std=params['bb_std'],
        kc_period=params['kc_period'],
        kc_atr_mult=params['kc_atr_mult'],
        momentum_period=params['momentum_period'],
        rsi_period=params['rsi_period'],
        volume_period=params['volume_period'],
        atr_period=params['atr_period']
    )
    
    # Initialize signal generator
    signal_gen = BBSqueezeSignalGenerator(
        analyzer=analyzer,
        min_squeeze_bars=params['min_squeeze_bars'],
        min_volume_ratio=params['min_volume_ratio'],
        rsi_overbought=params['rsi_overbought'],
        rsi_oversold=params.get('rsi_oversold', 18),
        atr_stop_mult=params['atr_stop_mult'],
        atr_target_mult=params['atr_target_mult'],
        base_position=BASE_POSITION,
        min_position=MIN_POSITION,
        max_position=MAX_POSITION,
        setup_validity_bars=params['setup_validity_bars'],
        signal_timeframe_minutes=get_tf_minutes(signal_tf),
    )
    
    signal_gen.set_signal_data(signal_data)
    signal_gen.set_atr_data(atr_data)
    
    # Initialize backtester
    bt = BBSqueezeBacktester(
        initial_capital=INITIAL_CAPITAL,
        commission=0.0005,
        slippage_pct=0.0002,
        max_positions=params['max_positions'],
        max_daily_loss_pct=params['max_daily_loss'],
        long_only=LONG_ONLY,
        verbose=False
    )
    
    bt.analyzer = analyzer
    bt.signal_generator = signal_gen
    
    # Run backtest
    results = bt.run_backtest(trade_data)
    
    return results.statistics


# ============================================================================
# PERMUTATION TEST
# ============================================================================

@dataclass
class PermutationTestResult:
    """Results from permutation test."""
    real_stats: Dict
    permuted_stats: List[Dict]
    p_values: Dict
    percentiles: Dict
    interpretation: str
    n_permutations: int


def run_permutation_test(
    symbol: str,
    params: Dict,
    n_permutations: int = 100,
    method: str = 'returns',
    days_back: int = DAYS_BACK,
    start_date: str = None,
    end_date: str = None,
    show_progress: bool = True
) -> Optional[PermutationTestResult]:
    """
    Run Monte Carlo permutation test.
    
    Args:
        symbol: Trading pair (e.g., 'ETH/USD')
        params: Strategy parameters
        n_permutations: Number of permutations (100-1000 recommended)
        method: 'returns' (aggressive) or 'blocks' (conservative)
        days_back: Days of historical data
        start_date: Filter start date (YYYY-MM-DD)
        end_date: Filter end date (YYYY-MM-DD)
        show_progress: Show progress bar
    
    Returns:
        PermutationTestResult with p-values and interpretation
    """
    print(f"\n{'='*70}")
    print("MONTE CARLO PERMUTATION TEST")
    print(f"{'='*70}")
    print(f"Symbol: {symbol}")
    print(f"Permutations: {n_permutations}")
    print(f"Method: {method}")
    if start_date or end_date:
        print(f"Date range: {start_date or 'start'} to {end_date or 'end'}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading cached data...")
    data = load_cached_data(symbol, days_back)
    
    if not data:
        print(f"Error: No cached data for {symbol}")
        print(f"Run download_data.py first!")
        return None
    
    # Filter by date if specified
    if start_date or end_date:
        data = filter_data_by_date(data, start_date, end_date)
        print(f"  Filtered to date range")
    
    print(f"  Loaded {len(data)} timeframes")
    for tf, df in data.items():
        print(f"    {tf}: {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    
    # Run on REAL data
    print(f"\n{Colors.BOLD}Running on REAL data...{Colors.RESET}")
    real_stats = run_backtest_on_data(data, params, symbol)
    
    print(f"  Trades: {real_stats['total_trades']}")
    print(f"  Return: {real_stats['return_pct']:+.1f}%")
    print(f"  Sharpe: {real_stats['sharpe']:.2f}")
    print(f"  Profit Factor: {real_stats['profit_factor']:.2f}")
    
    if real_stats['total_trades'] < 5:
        print(f"\n{Colors.RED}Warning: Too few trades ({real_stats['total_trades']}) for reliable test{Colors.RESET}")
    
    # Run on PERMUTED data
    print(f"\n{Colors.BOLD}Running on PERMUTED data ({n_permutations} permutations)...{Colors.RESET}")
    permuted_stats = []
    
    for i in range(n_permutations):
        if show_progress:
            progress = (i + 1) / n_permutations * 100
            bar_len = 40
            filled = int(bar_len * (i + 1) / n_permutations)
            bar = 'Ã¢â€“Ë†' * filled + 'Ã¢â€“â€˜' * (bar_len - filled)
            print(f"\r  [{bar}] {progress:5.1f}% ({i+1}/{n_permutations})", end='', flush=True)
        
        # Permute data
        permuted_data = permute_all_timeframes(data, method=method, seed=i*42)
        
        # Run backtest
        try:
            stats = run_backtest_on_data(permuted_data, params, symbol)
            permuted_stats.append(stats)
        except Exception as e:
            # Skip failed permutations
            continue
    
    print()  # Newline after progress bar
    
    if len(permuted_stats) < n_permutations * 0.9:
        print(f"\n{Colors.YELLOW}Warning: {n_permutations - len(permuted_stats)} permutations failed{Colors.RESET}")
    
    # Calculate p-values
    p_values = calculate_p_values(real_stats, permuted_stats)
    percentiles = calculate_percentiles(real_stats, permuted_stats)
    interpretation = interpret_results(p_values, real_stats)
    
    return PermutationTestResult(
        real_stats=real_stats,
        permuted_stats=permuted_stats,
        p_values=p_values,
        percentiles=percentiles,
        interpretation=interpretation,
        n_permutations=len(permuted_stats)
    )


def calculate_p_values(real_stats: Dict, permuted_stats: List[Dict]) -> Dict:
    """
    Calculate p-values for key metrics.
    
    p-value = proportion of permuted results that are >= real result
    Lower p-value = more likely the result is due to real patterns
    """
    metrics = ['return_pct', 'sharpe', 'profit_factor', 'win_rate', 'total_pnl']
    p_values = {}
    
    for metric in metrics:
        real_value = real_stats.get(metric, 0)
        permuted_values = [s.get(metric, 0) for s in permuted_stats]
        
        if not permuted_values:
            p_values[metric] = 1.0
            continue
        
        # Count how many permuted results beat or equal real result
        n_better = sum(1 for pv in permuted_values if pv >= real_value)
        p_values[metric] = n_better / len(permuted_values)
    
    return p_values


def calculate_percentiles(real_stats: Dict, permuted_stats: List[Dict]) -> Dict:
    """Calculate percentile of real result in permuted distribution."""
    metrics = ['return_pct', 'sharpe', 'profit_factor', 'win_rate', 'total_pnl']
    percentiles = {}
    
    for metric in metrics:
        real_value = real_stats.get(metric, 0)
        permuted_values = [s.get(metric, 0) for s in permuted_stats]
        
        if not permuted_values:
            percentiles[metric] = 50.0
            continue
        
        # Percentile = % of permuted values below real value
        n_below = sum(1 for pv in permuted_values if pv < real_value)
        percentiles[metric] = (n_below / len(permuted_values)) * 100
    
    return percentiles


def interpret_results(p_values: Dict, real_stats: Dict) -> str:
    """Generate interpretation of results."""
    # Use return p-value as primary indicator
    p_return = p_values.get('return_pct', 1.0)
    p_sharpe = p_values.get('sharpe', 1.0)
    p_pf = p_values.get('profit_factor', 1.0)
    
    # Combined score (average of key p-values)
    avg_p = (p_return + p_sharpe + p_pf) / 3
    
    if avg_p < 0.01:
        return "EXCELLENT - Strategy very likely found REAL patterns (p < 0.01)"
    elif avg_p < 0.05:
        return "GOOD - Strategy likely found real patterns (p < 0.05)"
    elif avg_p < 0.10:
        return "MARGINAL - Some evidence of real patterns, but inconclusive"
    elif avg_p < 0.20:
        return "WEAK - Limited evidence of real patterns"
    else:
        return "POOR - Strategy likely OVERFIT to noise (p >= 0.20)"


# ============================================================================
# OUTPUT
# ============================================================================

def print_results(result: PermutationTestResult):
    """Print detailed results."""
    print(f"\n{'='*70}")
    print("PERMUTATION TEST RESULTS")
    print(f"{'='*70}")
    
    # Real vs Average Permuted
    print(f"\n{Colors.BOLD}Performance Comparison:{Colors.RESET}")
    print(f"{'Metric':<20} {'Real':>12} {'Avg Permuted':>12} {'p-value':>10} {'Percentile':>10}")
    print("-" * 66)
    
    metrics = [
        ('return_pct', 'Return %', '{:+.1f}%'),
        ('sharpe', 'Sharpe Ratio', '{:.2f}'),
        ('profit_factor', 'Profit Factor', '{:.2f}'),
        ('win_rate', 'Win Rate %', '{:.1f}%'),
        ('total_pnl', 'Total P&L', '${:+,.0f}'),
    ]
    
    for metric, label, fmt in metrics:
        real_val = result.real_stats.get(metric, 0)
        perm_vals = [s.get(metric, 0) for s in result.permuted_stats]
        avg_perm = np.mean(perm_vals) if perm_vals else 0
        p_val = result.p_values.get(metric, 1.0)
        pctl = result.percentiles.get(metric, 50)
        
        # Color p-value
        if p_val < 0.05:
            p_color = Colors.GREEN
        elif p_val < 0.20:
            p_color = Colors.YELLOW
        else:
            p_color = Colors.RED
        
        real_str = fmt.format(real_val)
        avg_str = fmt.format(avg_perm)
        p_str = colored(f"{p_val:.3f}", p_color)
        
        print(f"{label:<20} {real_str:>12} {avg_str:>12} {p_str:>18} {pctl:>9.1f}%")
    
    # Distribution summary
    print(f"\n{Colors.BOLD}Permuted Distribution Summary:{Colors.RESET}")
    
    for metric, label, fmt in metrics[:3]:  # Just top 3
        perm_vals = [s.get(metric, 0) for s in result.permuted_stats]
        if perm_vals:
            print(f"  {label}:")
            print(f"    Min: {fmt.format(min(perm_vals))}, Max: {fmt.format(max(perm_vals))}")
            print(f"    Mean: {fmt.format(np.mean(perm_vals))}, Std: {fmt.format(np.std(perm_vals))}")
    
    # Interpretation
    print(f"\n{'='*70}")
    print(f"{Colors.BOLD}INTERPRETATION:{Colors.RESET}")
    print(f"{'='*70}")
    
    # Color the interpretation
    interp = result.interpretation
    if "EXCELLENT" in interp or "GOOD" in interp:
        print(colored(interp, Colors.GREEN))
    elif "MARGINAL" in interp or "WEAK" in interp:
        print(colored(interp, Colors.YELLOW))
    else:
        print(colored(interp, Colors.RED))
    
    print(f"\n{Colors.BOLD}What this means:{Colors.RESET}")
    avg_p = np.mean(list(result.p_values.values()))
    
    if avg_p < 0.05:
        print("  Ã¢Å“â€œ Your strategy's performance is unlikely due to random chance")
        print("  Ã¢Å“â€œ The patterns it exploits appear to be real market structure")
        print("  Ã¢Å“â€œ Proceed with cautious optimism (but always use proper risk management)")
    elif avg_p < 0.20:
        print("  Ã¢Å¡Â  Results are inconclusive")
        print("  Ã¢Å¡Â  Consider testing on different time periods")
        print("  Ã¢Å¡Â  Try reducing parameters to avoid overfitting")
    else:
        print("  Ã¢Å“â€” Your strategy performs similarly on random data")
        print("  Ã¢Å“â€” The optimized parameters likely fit noise, not signal")
        print("  Ã¢Å“â€” Consider simplifying the strategy or using different parameters")
    
    print(f"\n{'='*70}")


def save_results_csv(result: PermutationTestResult, filepath: str):
    """Save results to CSV."""
    rows = []
    
    # Real result
    row = {'type': 'REAL'}
    row.update(result.real_stats)
    rows.append(row)
    
    # Permuted results
    for i, stats in enumerate(result.permuted_stats):
        row = {'type': f'PERMUTED_{i}'}
        row.update(stats)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")


def plot_results(result: PermutationTestResult, filepath: str):
    """Create visualization of permutation test results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('return_pct', 'Return %'),
        ('sharpe', 'Sharpe Ratio'),
        ('profit_factor', 'Profit Factor'),
        ('win_rate', 'Win Rate %'),
    ]
    
    for ax, (metric, label) in zip(axes.flat, metrics):
        real_val = result.real_stats.get(metric, 0)
        perm_vals = [s.get(metric, 0) for s in result.permuted_stats]
        
        # Histogram of permuted values
        ax.hist(perm_vals, bins=30, alpha=0.7, color='steelblue', 
                edgecolor='white', label='Permuted')
        
        # Real value line
        ax.axvline(real_val, color='red', linewidth=2, linestyle='--',
                   label=f'Real: {real_val:.2f}')
        
        # p-value annotation
        p_val = result.p_values.get(metric, 1.0)
        ax.text(0.95, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label} Distribution')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo Permutation Test Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {filepath}")


# ============================================================================
# RANDOM ENTRY BENCHMARK
# ============================================================================

def run_random_entry_test(
    symbol: str,
    params: Dict,
    n_simulations: int = 100,
    days_back: int = DAYS_BACK
) -> Optional[Dict]:
    """
    Compare strategy entries vs random entries.
    
    Keeps the same exit rules (stops/targets) but randomizes entries.
    If random entries perform similarly, your entry timing has no edge.
    """
    print(f"\n{'='*70}")
    print("RANDOM ENTRY BENCHMARK TEST")
    print(f"{'='*70}")
    print("Comparing your entry timing vs random entries")
    print(f"{'='*70}\n")
    
    # Load data
    data = load_cached_data(symbol, days_back)
    if not data:
        return None
    
    # Run real strategy
    print("Running strategy with REAL entries...")
    real_stats = run_backtest_on_data(data, params, symbol)
    
    print(f"  Real Trades: {real_stats['total_trades']}")
    print(f"  Real Return: {real_stats['return_pct']:+.1f}%")
    
    # This would require modifying the signal generator to produce random entries
    # For now, we'll use the permutation test which is more comprehensive
    
    print("\nNote: For random entry testing, use the permutation test instead.")
    print("The permutation test is more rigorous and tests the full strategy.\n")
    
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo Permutation Test for Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test on 1 year of data (recommended for speed)
    python permutation_test.py --symbol ETH/USD --start-date 2024-01-01 --permutations 100

    # Full test on all data (slow!)
    python permutation_test.py --symbol BTC/USD --permutations 500 --plot
    
    # Test specific period
    python permutation_test.py --symbol ETH/USD --start-date 2023-01-01 --end-date 2024-01-01
    
    # Conservative test (preserves short-term patterns)
    python permutation_test.py --symbol ETH/USD --method blocks --start-date 2024-01-01
        """
    )
    
    parser.add_argument('--symbol', type=str, default='ETH/USD',
                        help='Trading pair (default: ETH/USD)')
    parser.add_argument('--permutations', '-n', type=int, default=100,
                        help='Number of permutations (default: 100, recommend 100-500)')
    parser.add_argument('--method', type=str, default='returns',
                        choices=['returns', 'blocks'],
                        help='Permutation method: returns (aggressive) or blocks (conservative)')
    parser.add_argument('--days', type=int, default=DAYS_BACK,
                        help=f'Days of data (default: {DAYS_BACK})')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date filter (YYYY-MM-DD) - speeds up test!')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plot')
    parser.add_argument('--csv', action='store_true',
                        help='Save results to CSV')
    
    args = parser.parse_args()
    
    # Run permutation test
    result = run_permutation_test(
        symbol=args.symbol,
        params=DEFAULT_PARAMS,
        n_permutations=args.permutations,
        method=args.method,
        days_back=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        show_progress=True
    )
    
    if result is None:
        return
    
    # Print results
    print_results(result)
    
    # Save outputs
    symbol_clean = args.symbol.replace('/', '_')
    
    if args.csv:
        save_results_csv(result, f'permutation_test_{symbol_clean}.csv')
    
    if args.plot:
        plot_results(result, f'permutation_test_{symbol_clean}.png')
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()