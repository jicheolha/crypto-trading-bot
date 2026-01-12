"""
Timeframe-Specific Optimizer

Tests EACH timeframe combination systematically and finds
the best parameters for each one.

Usage:
    python optimize_timeframes.py --trials 50 --start-date 2020-01-01 --end-date 2023-12-31
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimize_lib import (
    load_data, filter_data_by_date, run_backtest, calculate_score,
    SYMBOLS, DAYS_BACK, INITIAL_CAPITAL, MIN_TRADES, LONG_ONLY,
    FIXED_BASE_POSITION, FIXED_MIN_POSITION, FIXED_MAX_POSITION, PARAM_SPACE
)
from datetime import datetime
import pandas as pd
import random

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q", "--break-system-packages"])
    import optuna
    from optuna.samplers import TPESampler


# All timeframe combinations to test
# Includes scalping (5min/15min signal) to swing (4h signal)
SIGNAL_TIMEFRAMES = ['5min', '15min', '30min', '1h', '4h']
ATR_TIMEFRAMES = ['15min', '30min', '1h', '4h', '1d']


def optimize_for_timeframe_combo(
    data: dict,
    signal_tf: str,
    atr_tf: str,
    n_trials: int = 50
):
    """Optimize parameters for a specific timeframe combination."""
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZING: Signal={signal_tf}, ATR={atr_tf}")
    print(f"{'='*70}")
    print(f"Trials: {n_trials}")
    
    # Split train/test
    train_data, test_data = {}, {}
    train_ratio = 0.7
    
    for symbol in data:
        train_data[symbol] = {}
        test_data[symbol] = {}
        for tf, df in data[symbol].items():
            split_idx = int(len(df) * train_ratio)
            train_data[symbol][tf] = df.iloc[:split_idx]
            test_data[symbol][tf] = df.iloc[split_idx:]
    
    best_score = -float('inf')
    best_params = None
    best_stats = None
    
    def objective(trial):
        nonlocal best_score, best_params, best_stats
        
        # Fixed timeframes
        params = {
            'signal_timeframe': signal_tf,
            'atr_timeframe': atr_tf
        }
        
        # Optimize all other parameters
        for name, spec in PARAM_SPACE.items():
            if name in ['signal_timeframe', 'atr_timeframe']:
                continue  # Skip timeframes, we fixed them
            
            if spec['type'] == 'int':
                params[name] = trial.suggest_int(name, spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'])
        
        # Validate
        if params['atr_target_mult'] <= params['atr_stop_mult']:
            return -1000
        
        # Train score
        train_score, train_stats = run_backtest(train_data, params)
        
        if train_stats['total_trades'] < MIN_TRADES:
            return train_score
        
        # Test score
        test_score, test_stats = run_backtest(test_data, params)
        
        # Combined score
        if test_score < 0:
            combined = train_score * 0.3
        else:
            combined = train_score * 0.4 + test_score * 0.6
        
        # Track best
        if combined > best_score:
            best_score = combined
            best_params = params.copy()
            best_stats = {
                'train': train_stats,
                'test': test_stats,
                'train_score': train_score,
                'test_score': test_score
            }
        
        return combined
    
    # Run optimization
    sampler = TPESampler(seed=42, n_startup_trials=min(15, n_trials // 3))
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    return {
        'signal_tf': signal_tf,
        'atr_tf': atr_tf,
        'best_score': best_score,
        'best_params': best_params,
        'best_stats': best_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Timeframe-Specific Optimizer')
    parser.add_argument('--trials', type=int, default=50,
                        help='Trials per timeframe combo (default: 50)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--top', type=int, default=5,
                        help='Show top N combos (default: 5)')
    parser.add_argument('--scalping', action='store_true',
                        help='Only test scalping combos (5min, 15min signals)')
    parser.add_argument('--swing', action='store_true',
                        help='Only test swing combos (1h, 4h signals)')
    parser.add_argument('--signal-tf', type=str, default=None,
                        help='Test specific signal timeframe only (e.g., 5min)')
    parser.add_argument('--atr-tf', type=str, default=None,
                        help='Test specific ATR timeframe only (e.g., 15min)')
    
    args = parser.parse_args()
    
    # Determine which timeframes to test
    signal_tfs = SIGNAL_TIMEFRAMES
    atr_tfs = ATR_TIMEFRAMES
    
    if args.scalping:
        signal_tfs = ['5min', '15min']
        atr_tfs = ['15min', '30min', '1h']
    elif args.swing:
        signal_tfs = ['1h', '4h']
        atr_tfs = ['1h', '4h', '1d']
    
    if args.signal_tf:
        signal_tfs = [args.signal_tf]
    if args.atr_tf:
        atr_tfs = [args.atr_tf]
    
    total_combos = len(signal_tfs) * len(atr_tfs)
    total_trials = total_combos * args.trials
    
    print(f"\n{'='*70}")
    print("TIMEFRAME-SPECIFIC OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Testing {total_combos} timeframe combinations")
    print(f"Trials per combo: {args.trials}")
    print(f"Total trials: {total_trials}")
    print(f"Estimated time: {total_trials * 0.5 / 60:.0f}-{total_trials * 1.0 / 60:.0f} minutes")
    
    # Load data
    print(f"\nLoading {DAYS_BACK} days of data...")
    data = load_data(SYMBOLS, DAYS_BACK)
    
    if not data:
        print("Failed to load data")
        return
    
    # Filter by date
    if args.start_date or args.end_date:
        print(f"Filtering: {args.start_date or 'start'} to {args.end_date or 'end'}")
        data = filter_data_by_date(data, args.start_date, args.end_date)
        first_symbol = list(data.keys())[0]
        first_tf = list(data[first_symbol].keys())[0]
        df = data[first_symbol][first_tf]
        print(f"Range: {df.index[0]} to {df.index[-1]}")
    
    # Test all combinations
    results = []
    start_time = datetime.now()
    
    for i, signal_tf in enumerate(signal_tfs):
        for j, atr_tf in enumerate(atr_tfs):
            combo_num = i * len(atr_tfs) + j + 1
            print(f"\n[Combo {combo_num}/{total_combos}]")
            
            result = optimize_for_timeframe_combo(
                data, signal_tf, atr_tf, args.trials
            )
            results.append(result)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    # Sort by score
    results.sort(key=lambda x: x['best_score'], reverse=True)
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"\nTop {args.top} Timeframe Combinations:")
    print(f"{'='*70}")
    
    for i, result in enumerate(results[:args.top], 1):
        signal_tf = result['signal_tf']
        atr_tf = result['atr_tf']
        score = result['best_score']
        stats = result['best_stats']
        
        print(f"\n#{i}: {signal_tf} / {atr_tf}  |  Score: {score:.4f}")
        
        if stats and 'train' in stats:
            train = stats['train']
            test = stats['test']
            
            print(f"  Train: {train['total_trades']} trades, "
                  f"{train['win_rate']:.1f}% WR, "
                  f"PF={train['profit_factor']:.2f}, "
                  f"Return={train['return_pct']:+.1f}%")
            
            print(f"  Test:  {test['total_trades']} trades, "
                  f"{test['win_rate']:.1f}% WR, "
                  f"PF={test['profit_factor']:.2f}, "
                  f"Return={test['return_pct']:+.1f}%")
    
    # Print best parameters
    print(f"\n{'='*70}")
    print("BEST COMBO PARAMETERS")
    print(f"{'='*70}")
    
    best = results[0]
    print(f"\nSignal TF: {best['signal_tf']}")
    print(f"ATR TF: {best['atr_tf']}")
    print(f"Score: {best['best_score']:.4f}")
    print(f"\nParameters:")
    
    for key, value in sorted(best['best_params'].items()):
        if key not in ['signal_timeframe', 'atr_timeframe']:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Save results
    csv_data = []
    for result in results:
        row = {
            'signal_tf': result['signal_tf'],
            'atr_tf': result['atr_tf'],
            'score': result['best_score']
        }
        if result['best_params']:
            row.update(result['best_params'])
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    filename = 'optimize_timeframes_results.csv'
    df.to_csv(filename, index=False)
    print(f"\nâœ… Results saved to {filename}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    best = results[0]
    print(f"\nUse these parameters for {best['signal_tf']}/{best['atr_tf']}:")
    print(f"\nSIGNAL_TIMEFRAME = '{best['signal_tf']}'")
    print(f"ATR_TIMEFRAME = '{best['atr_tf']}'")
    print(f"BB_PERIOD = {best['best_params']['bb_period']}")
    print(f"BB_STD = {best['best_params']['bb_std']:.2f}")
    print(f"KC_PERIOD = {best['best_params']['kc_period']}")
    print(f"KC_ATR_MULT = {best['best_params']['kc_atr_mult']:.2f}")
    print("...")
    print("\n(Full params shown above)")


if __name__ == "__main__":
    main()