"""
Squeeze Breakout Strategy Optimizer

Supports both long and short positions.
NOW WITH PERSISTENT TRIAL HISTORY - learns from all past runs!

Methods:
1. Random Search - Fast exploration
2. Bayesian/TPE (Optuna) - Smart optimization with history
3. Grid Search - Exhaustive (for final refinement)
4. Walk-Forward - Time-based validation

Usage:
    python optimize.py --method bayesian --trials 200
    python optimize.py --method bayesian --trials 50 --study my_study  # Named study
    python optimize.py --method bayesian --trials 50 --reset  # Fresh start
    python optimize.py --history  # View history stats
"""
import argparse
import logging
import warnings
import sys
import os
import json
import itertools
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q", "--break-system-packages"])
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from backtester import BBSqueezeBacktester


# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = ['BTC/USD']
DAYS_BACK = 365*6  # Match cache (2019-2025)
INITIAL_CAPITAL = 100000
MIN_TRADES = 5

# Long only mode - set to False to allow shorts
LONG_ONLY = False

# Fixed position sizing (not optimized)
FIXED_BASE_POSITION = 0.40
FIXED_MIN_POSITION = 0.30
FIXED_MAX_POSITION = 0.90

# Persistent storage
OPTUNA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optuna_history.db')
DEFAULT_STUDY_NAME = "bb_squeeze_v1"


# ============================================================================
# PARAMETER SPACE DEFINITION
# ============================================================================

PARAM_SPACE = {
    # Timeframes - includes 5min for scalping
    'signal_timeframe': {
        'type': 'categorical',
        'values': ['5min', '15min', '30min', '1h', '4h'],
        'description': 'Signal generation timeframe'
    },
    'atr_timeframe': {
        'type': 'categorical', 
        'values': ['15min', '30min', '1h', '4h', '1d'],
        'description': 'ATR calculation timeframe'
    },
    
    # Bollinger Bands
    'bb_period': {
        'type': 'int',
        'low': 10,
        'high': 50,
        'default': 20,
        'description': 'BB SMA period'
    },
    'bb_std': {
        'type': 'float',
        'low': 1.5,
        'high': 3.0,
        'default': 2.0,
        'description': 'BB standard deviations'
    },
    
    # Keltner Channels
    'kc_period': {
        'type': 'int',
        'low': 10,
        'high': 50,
        'default': 20,
        'description': 'KC EMA period'
    },
    'kc_atr_mult': {
        'type': 'float',
        'low': 1.0,
        'high': 3.0,
        'default': 1.5,
        'description': 'KC ATR multiplier'
    },
    
    # Momentum
    'momentum_period': {
        'type': 'int',
        'low': 5,
        'high': 30,
        'default': 12,
        'description': 'Momentum lookback'
    },
    
    # RSI
    'rsi_period': {
        'type': 'int',
        'low': 7,
        'high': 28,
        'default': 14,
        'description': 'RSI period'
    },
    'rsi_overbought': {
        'type': 'int',
        'low': 60,
        'high': 85,
        'default': 70,
        'description': 'RSI overbought threshold (blocks long entries above this)'
    },
    'rsi_oversold': {
        'type': 'int',
        'low': 15,
        'high': 40,
        'default': 30,
        'description': 'RSI oversold threshold (blocks short entries below this)'
    },
    
    # Squeeze
    'min_squeeze_bars': {
        'type': 'int',
        'low': 1,
        'high': 10,
        'default': 3,
        'description': 'Minimum squeeze duration'
    },
    
    # Volume
    'volume_period': {
        'type': 'int',
        'low': 10,
        'high': 50,
        'default': 20,
        'description': 'Volume MA period'
    },
    'min_volume_ratio': {
        'type': 'float',
        'low': 0.5,
        'high': 2.5,
        'default': 1.2,
        'description': 'Minimum volume ratio'
    },
    
    # ATR & Stops
    'atr_period': {
        'type': 'int',
        'low': 7,
        'high': 21,
        'default': 14,
        'description': 'ATR period'
    },
    'atr_stop_mult': {
        'type': 'float',
        'low': 1.0,
        'high': 4.0,
        'default': 2.0,
        'description': 'Stop loss ATR multiplier'
    },
    'atr_target_mult': {
        'type': 'float',
        'low': 1.5,
        'high': 6.0,
        'default': 3.0,
        'description': 'Take profit ATR multiplier'
    },
    
    # Setup
    'setup_validity_bars': {
        'type': 'int',
        'low': 2,
        'high': 20,
        'default': 5,
        'description': 'Setup validity duration'
    },
    
    # Risk
    'max_positions': {
        'type': 'int',
        'low': 1,
        'high': 5,
        'default': 2,
        'description': 'Maximum concurrent positions'
    },
    'max_daily_loss': {
        'type': 'float',
        'low': 0.02,
        'high': 0.10,
        'default': 0.03,
        'description': 'Maximum daily loss %'
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(symbols: List[str], days_back: int) -> Optional[Dict]:
    """Load multi-timeframe data."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'crypto_trading'))
        from data_utils import prepare_data_for_backtest
    except ImportError:
        print("Error: Could not import data_utils")
        return None
    
    data = {}
    timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        data[symbol] = {}
        
        for tf in timeframes:
            df = prepare_data_for_backtest(
                symbol, 
                timeframe=tf, 
                days_back=days_back,
                use_cache=True,
                cache_max_age_hours=None  # Use cache permanently
            )
            if not df.empty:
                data[symbol][tf] = df
    
    return data if data else None


def filter_data_by_date(data: Dict, start_date: str = None, end_date: str = None) -> Dict:
    """Filter data by date range."""
    if not start_date and not end_date:
        return data
    
    filtered = {}
    for symbol in data:
        filtered[symbol] = {}
        for tf, df in data[symbol].items():
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= df.index >= pd.Timestamp(start_date, tz=df.index.tz)
            if end_date:
                mask &= df.index <= pd.Timestamp(end_date + ' 23:59:59', tz=df.index.tz)
            filtered[symbol][tf] = df[mask]
    
    return filtered


def split_data(data: Dict, train_ratio: float) -> Tuple[Dict, Dict]:
    """Split data into train/test."""
    train_data, test_data = {}, {}
    
    for symbol in data:
        train_data[symbol] = {}
        test_data[symbol] = {}
        
        for tf, df in data[symbol].items():
            split_idx = int(len(df) * train_ratio)
            train_data[symbol][tf] = df.iloc[:split_idx]
            test_data[symbol][tf] = df.iloc[split_idx:]
    
    return train_data, test_data


def split_data_walkforward(data: Dict, n_folds: int) -> List[Tuple[Dict, Dict]]:
    """Split data into walk-forward folds."""
    folds = []
    
    first_symbol = list(data.keys())[0]
    first_tf = list(data[first_symbol].keys())[0]
    total_len = len(data[first_symbol][first_tf])
    
    fold_size = total_len // (n_folds + 1)
    
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        
        train_data, test_data = {}, {}
        
        for symbol in data:
            train_data[symbol] = {}
            test_data[symbol] = {}
            
            for tf, df in data[symbol].items():
                tf_len = len(df)
                tf_fold_size = tf_len // (n_folds + 1)
                tf_train_end = tf_fold_size * (i + 1)
                tf_test_end = tf_fold_size * (i + 2)
                
                train_data[symbol][tf] = df.iloc[:tf_train_end]
                test_data[symbol][tf] = df.iloc[tf_train_end:tf_test_end]
        
        folds.append((train_data, test_data))
    
    return folds


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def get_tf_minutes(tf: str) -> int:
    return {'1min': 1, '5min': 5, '15min': 15, '30min': 30, '1h': 60, '4h': 240, '1d': 1440}.get(tf, 60)


def run_backtest(data: Dict, params: Dict[str, Any]) -> Tuple[float, Dict]:
    """Run backtest with given parameters."""
    
    signal_tf = params['signal_timeframe']
    atr_tf = params['atr_timeframe']
    
    trade_data = {s: data[s]['1min'] for s in data}
    signal_data = {s: data[s].get(signal_tf, data[s]['1h']) for s in data}
    atr_data = {s: data[s].get(atr_tf, data[s]['4h']) for s in data}
    
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
    
    signal_gen = BBSqueezeSignalGenerator(
        analyzer=analyzer,
        min_squeeze_bars=params['min_squeeze_bars'],
        min_volume_ratio=params['min_volume_ratio'],
        rsi_overbought=params['rsi_overbought'],
        rsi_oversold=params['rsi_oversold'],
        atr_stop_mult=params['atr_stop_mult'],
        atr_target_mult=params['atr_target_mult'],
        base_position=FIXED_BASE_POSITION,
        min_position=FIXED_MIN_POSITION,
        max_position=FIXED_MAX_POSITION,
        setup_validity_bars=params['setup_validity_bars'],
        signal_timeframe_minutes=get_tf_minutes(signal_tf),
    )
    
    signal_gen.set_signal_data(signal_data)
    signal_gen.set_atr_data(atr_data)
    
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
    
    results = bt.run_backtest(trade_data)
    stats = results.statistics
    score = calculate_score(stats)
    
    return score, stats


def calculate_score(stats: Dict) -> float:
    """Calculate optimization score."""
    
    if stats['total_trades'] < MIN_TRADES:
        return -1000
    
    pnl = stats['total_pnl']
    profit_factor = stats['profit_factor']
    win_rate = stats['win_rate'] / 100
    max_dd = stats['max_drawdown']
    sharpe = stats['sharpe']
    num_trades = stats['total_trades']
    
    # Composite score
    pf_capped = min(profit_factor, 3.0)
    sharpe_capped = max(min(sharpe, 3.0), -3.0)
    
    score = (
        pf_capped * 2.0 +
        sharpe_capped * 1.5 +
        win_rate * 1.0 +
        np.sqrt(num_trades) * 0.15
    )
    
    # Penalties
    score *= (1 - max_dd / 100)
    if pnl < 0:
        score *= 0.5
    
    return score


# ============================================================================
# HISTORY MANAGEMENT
# ============================================================================

def get_storage_url() -> str:
    """Get SQLite storage URL for Optuna."""
    return f"sqlite:///{OPTUNA_DB_PATH}"


def list_studies() -> List[str]:
    """List all available studies in the database."""
    if not os.path.exists(OPTUNA_DB_PATH):
        return []
    
    try:
        studies = optuna.study.get_all_study_summaries(storage=get_storage_url())
        return [s.study_name for s in studies]
    except Exception as e:
        print(f"Error listing studies: {e}")
        return []


def get_study_stats(study_name: str) -> Optional[Dict]:
    """Get statistics for a study."""
    try:
        study = optuna.load_study(study_name=study_name, storage=get_storage_url())
        trials = study.trials
        
        if not trials:
            return None
        
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        
        scores = [t.value for t in completed if t.value is not None]
        
        return {
            'study_name': study_name,
            'total_trials': len(trials),
            'completed': len(completed),
            'failed': len(failed),
            'best_score': study.best_value if completed else None,
            'best_params': study.best_params if completed else None,
            'avg_score': np.mean(scores) if scores else None,
            'score_std': np.std(scores) if scores else None,
            'first_trial': min(t.datetime_start for t in trials) if trials else None,
            'last_trial': max(t.datetime_complete or t.datetime_start for t in trials) if trials else None,
        }
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None


def delete_study(study_name: str) -> bool:
    """Delete a study from the database."""
    try:
        optuna.delete_study(study_name=study_name, storage=get_storage_url())
        print(f"Deleted study: {study_name}")
        return True
    except Exception as e:
        print(f"Error deleting study: {e}")
        return False


def show_history():
    """Display history of all optimization studies."""
    print(f"\n{'='*70}")
    print("OPTIMIZATION HISTORY")
    print(f"{'='*70}")
    print(f"Database: {OPTUNA_DB_PATH}")
    
    if not os.path.exists(OPTUNA_DB_PATH):
        print("\nNo optimization history found. Run some trials first!")
        return
    
    studies = list_studies()
    
    if not studies:
        print("\nNo studies found in database.")
        return
    
    print(f"\nFound {len(studies)} study(ies):\n")
    
    for study_name in studies:
        stats = get_study_stats(study_name)
        if not stats:
            continue
        
        print(f"  Study: {study_name}")
        print(f"  ├── Total trials: {stats['total_trials']}")
        print(f"  ├── Completed: {stats['completed']}, Failed: {stats['failed']}")
        
        if stats['best_score'] is not None:
            print(f"  ├── Best score: {stats['best_score']:.4f}")
            print(f"  ├── Avg score: {stats['avg_score']:.4f} ± {stats['score_std']:.4f}")
        
        if stats['first_trial']:
            print(f"  ├── First trial: {stats['first_trial'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  └── Last trial: {stats['last_trial'].strftime('%Y-%m-%d %H:%M')}")
        
        print()
    
    print(f"{'='*70}")


def show_best_params(study_name: str = DEFAULT_STUDY_NAME):
    """Show best parameters from a study."""
    stats = get_study_stats(study_name)
    
    if not stats or not stats['best_params']:
        print(f"No completed trials found for study: {study_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"BEST PARAMETERS - {study_name}")
    print(f"{'='*70}")
    print(f"Score: {stats['best_score']:.4f}")
    print(f"From {stats['completed']} completed trials\n")
    
    print_config(stats['best_params'])


# ============================================================================
# OPTIMIZER CLASSES
# ============================================================================

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    best_stats: Dict
    all_trials: List[Dict]
    method: str
    duration_seconds: float
    total_historical_trials: int = 0  # Includes past runs


class BaseOptimizer:
    """Base optimizer class."""
    
    def __init__(self, data: Dict, train_ratio: float = 0.7):
        self.data = data
        self.train_ratio = train_ratio
        self.train_data, self.test_data = split_data(data, train_ratio)
        self.trials = []
    
    def sample_params(self) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for name, spec in PARAM_SPACE.items():
            if spec['type'] == 'categorical':
                params[name] = random.choice(spec['values'])
            elif spec['type'] == 'int':
                params[name] = random.randint(spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[name] = random.uniform(spec['low'], spec['high'])
        return params
    
    def evaluate(self, params: Dict[str, Any], use_test: bool = True) -> Tuple[float, Dict]:
        """Evaluate parameters."""
        # Validate
        if params['atr_target_mult'] <= params['atr_stop_mult']:
            return -1000, {}
        
        train_score, train_stats = run_backtest(self.train_data, params)
        
        if not use_test or train_stats['total_trades'] < MIN_TRADES:
            return train_score, train_stats
        
        test_score, test_stats = run_backtest(self.test_data, params)
        
        # Combined score (favor test performance)
        if test_score < 0:
            combined = train_score * 0.3
        else:
            combined = train_score * 0.4 + test_score * 0.6
        
        return combined, {
            'train': train_stats,
            'test': test_stats,
            'train_score': train_score,
            'test_score': test_score
        }
    
    def optimize(self, n_trials: int) -> OptimizationResult:
        raise NotImplementedError


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer."""
    
    def optimize(self, n_trials: int) -> OptimizationResult:
        print(f"\n{'='*60}")
        print("RANDOM SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Trials: {n_trials}")
        
        start = datetime.now()
        best_score = -float('inf')
        best_params = None
        best_stats = None
        
        for i in range(n_trials):
            params = self.sample_params()
            score, stats = self.evaluate(params)
            
            self.trials.append({
                'trial': i,
                'params': params,
                'score': score,
                'stats': stats
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_stats = stats
                print(f"Trial {i+1}/{n_trials} | New best: {score:.4f}")
            elif (i + 1) % 20 == 0:
                print(f"Trial {i+1}/{n_trials} | Best so far: {best_score:.4f}")
        
        duration = (datetime.now() - start).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_stats=best_stats,
            all_trials=self.trials,
            method='random',
            duration_seconds=duration
        )


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian (Optuna TPE) optimizer with PERSISTENT HISTORY.
    
    All trials are stored in SQLite database. Each run builds on
    previous runs, making the optimizer smarter over time.
    """
    
    def __init__(
        self, 
        data: Dict, 
        train_ratio: float = 0.7, 
        n_startup: int = 30,
        study_name: str = DEFAULT_STUDY_NAME,
        reset: bool = False
    ):
        super().__init__(data, train_ratio)
        self.n_startup = n_startup
        self.study_name = study_name
        self.reset = reset
        self.storage_url = get_storage_url()
    
    def optimize(self, n_trials: int) -> OptimizationResult:
        print(f"\n{'='*60}")
        print("BAYESIAN (TPE) OPTIMIZATION WITH HISTORY")
        print(f"{'='*60}")
        
        # Delete existing study if reset requested
        if self.reset:
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.storage_url)
                print(f"Reset study: {self.study_name}")
            except:
                pass  # Study didn't exist
        
        # Load or create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=self.n_startup)
        )
        
        existing_trials = len(study.trials)
        
        if existing_trials > 0:
            print(f"Loaded {existing_trials} existing trials from history")
            print(f"Current best: {study.best_value:.4f}")
            
            # With history, TPE can start smart immediately
            # Reduce startup trials since we already have data
            effective_startup = max(5, self.n_startup - existing_trials)
            study.sampler = TPESampler(seed=42, n_startup_trials=effective_startup)
        else:
            print(f"Starting fresh study: {self.study_name}")
        
        print(f"Running {n_trials} new trials...")
        print(f"Database: {OPTUNA_DB_PATH}")
        
        start = datetime.now()
        trials_this_run = []
        
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            for name, spec in PARAM_SPACE.items():
                if spec['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, spec['values'])
                elif spec['type'] == 'int':
                    params[name] = trial.suggest_int(name, spec['low'], spec['high'])
                elif spec['type'] == 'float':
                    params[name] = trial.suggest_float(name, spec['low'], spec['high'])
            
            score, stats = self.evaluate(params)
            
            trials_this_run.append({
                'trial': len(trials_this_run),
                'params': params,
                'score': score,
                'stats': stats
            })
            
            return score
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        duration = (datetime.now() - start).total_seconds()
        
        # Get best stats
        best_params = study.best_params
        _, best_stats = self.evaluate(best_params)
        
        total_trials = len(study.trials)
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"This run: {n_trials} trials in {duration:.0f}s")
        print(f"Total historical trials: {total_trials}")
        print(f"Best score (all time): {study.best_value:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=study.best_value,
            best_stats=best_stats,
            all_trials=trials_this_run,
            method='bayesian',
            duration_seconds=duration,
            total_historical_trials=total_trials
        )


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer (reduced grid for feasibility)."""
    
    # Reduced grid for key parameters only
    GRID = {
        'signal_timeframe': ['15min', '1h', '4h'],
        'atr_timeframe': ['1h', '4h'],
        'bb_period': [15, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
        'kc_atr_mult': [1.0, 1.5, 2.0],
        'min_squeeze_bars': [2, 3, 5],
        'min_volume_ratio': [1.0, 1.2, 1.5],
        'atr_stop_mult': [1.5, 2.0, 2.5],
        'atr_target_mult': [2.5, 3.0, 4.0],
    }
    
    # Fixed values for non-grid params
    FIXED = {
        'kc_period': 20,
        'momentum_period': 12,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'volume_period': 20,
        'atr_period': 14,
        'setup_validity_bars': 5,
        'max_positions': 2,
        'max_daily_loss': 0.03,
    }
    
    def optimize(self, n_trials: int = None) -> OptimizationResult:
        # Calculate total combinations
        total = 1
        for values in self.GRID.values():
            total *= len(values)
        
        print(f"\n{'='*60}")
        print("GRID SEARCH OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Total combinations: {total:,}")
        
        if total > 10000:
            print("WARNING: This will take a long time!")
        
        start = datetime.now()
        best_score = -float('inf')
        best_params = None
        best_stats = None
        
        keys = list(self.GRID.keys())
        values = list(self.GRID.values())
        
        for i, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            params.update(self.FIXED)
            
            # Skip invalid combos
            if params['atr_target_mult'] <= params['atr_stop_mult']:
                continue
            
            score, stats = self.evaluate(params)
            
            self.trials.append({
                'trial': i,
                'params': params,
                'score': score,
                'stats': stats
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_stats = stats
                print(f"Combo {i+1}/{total} | New best: {score:.4f}")
            elif (i + 1) % 100 == 0:
                print(f"Combo {i+1}/{total} | Best: {best_score:.4f}")
        
        duration = (datetime.now() - start).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_stats=best_stats,
            all_trials=self.trials,
            method='grid',
            duration_seconds=duration
        )


class WalkForwardOptimizer(BaseOptimizer):
    """Walk-forward optimization."""
    
    def __init__(self, data: Dict, n_folds: int = 5, inner_trials: int = 50):
        self.data = data
        self.n_folds = n_folds
        self.inner_trials = inner_trials
        self.folds = split_data_walkforward(data, n_folds)
        self.trials = []
    
    def optimize(self, n_trials: int = None) -> OptimizationResult:
        print(f"\n{'='*60}")
        print("WALK-FORWARD OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Folds: {self.n_folds} | Inner trials per fold: {self.inner_trials}")
        
        start = datetime.now()
        fold_results = []
        all_params = []
        
        for fold_idx, (train_data, test_data) in enumerate(self.folds):
            print(f"\n--- Fold {fold_idx + 1}/{self.n_folds} ---")
            
            # Optimize on train
            best_score = -float('inf')
            best_params = None
            
            for i in range(self.inner_trials):
                params = self.sample_params()
                
                if params['atr_target_mult'] <= params['atr_stop_mult']:
                    continue
                
                score, stats = run_backtest(train_data, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            # Test on out-of-sample
            test_score, test_stats = run_backtest(test_data, best_params)
            
            fold_results.append({
                'fold': fold_idx,
                'train_score': best_score,
                'test_score': test_score,
                'test_stats': test_stats,
                'params': best_params
            })
            all_params.append(best_params)
            
            print(f"Train score: {best_score:.4f} | Test score: {test_score:.4f}")
            
            self.trials.append({
                'fold': fold_idx,
                'params': best_params,
                'train_score': best_score,
                'test_score': test_score
            })
        
        duration = (datetime.now() - start).total_seconds()
        
        # Aggregate: use params from best test score fold
        best_fold = max(fold_results, key=lambda x: x['test_score'])
        
        # Calculate average test performance
        avg_test_score = np.mean([f['test_score'] for f in fold_results])
        std_test_score = np.std([f['test_score'] for f in fold_results])
        
        print(f"\nAverage test score: {avg_test_score:.4f} ± {std_test_score:.4f}")
        
        return OptimizationResult(
            best_params=best_fold['params'],
            best_score=avg_test_score,
            best_stats={
                'fold_results': fold_results,
                'avg_test_score': avg_test_score,
                'std_test_score': std_test_score
            },
            all_trials=self.trials,
            method='walkforward',
            duration_seconds=duration
        )
    
    def sample_params(self) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for name, spec in PARAM_SPACE.items():
            if spec['type'] == 'categorical':
                params[name] = random.choice(spec['values'])
            elif spec['type'] == 'int':
                params[name] = random.randint(spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[name] = random.uniform(spec['low'], spec['high'])
        return params


class MultiStageOptimizer:
    """Multi-stage optimization: Random → Bayesian → Validation."""
    
    def __init__(self, data: Dict, train_ratio: float = 0.7):
        self.data = data
        self.train_ratio = train_ratio
    
    def optimize(
        self, 
        random_trials: int = 100,
        bayesian_trials: int = 200,
        walkforward_folds: int = 3
    ) -> OptimizationResult:
        
        print(f"\n{'='*70}")
        print("MULTI-STAGE OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Stage 1: Random Search ({random_trials} trials)")
        print(f"Stage 2: Bayesian/TPE ({bayesian_trials} trials)")
        print(f"Stage 3: Walk-Forward Validation ({walkforward_folds} folds)")
        
        start = datetime.now()
        all_trials = []
        
        # Stage 1: Random exploration
        print(f"\n{'='*60}")
        print("STAGE 1: RANDOM EXPLORATION")
        print(f"{'='*60}")
        
        random_opt = RandomSearchOptimizer(self.data, self.train_ratio)
        random_result = random_opt.optimize(random_trials)
        all_trials.extend(random_result.all_trials)
        
        print(f"\nRandom best score: {random_result.best_score:.4f}")
        
        # Stage 2: Bayesian refinement (with history)
        print(f"\n{'='*60}")
        print("STAGE 2: BAYESIAN REFINEMENT")
        print(f"{'='*60}")
        
        bayesian_opt = BayesianOptimizer(
            self.data, 
            self.train_ratio, 
            n_startup=20,
            study_name=f"{DEFAULT_STUDY_NAME}_multi"
        )
        bayesian_result = bayesian_opt.optimize(bayesian_trials)
        all_trials.extend(bayesian_result.all_trials)
        
        print(f"\nBayesian best score: {bayesian_result.best_score:.4f}")
        
        # Stage 3: Walk-forward validation
        print(f"\n{'='*60}")
        print("STAGE 3: WALK-FORWARD VALIDATION")
        print(f"{'='*60}")
        
        wf_opt = WalkForwardOptimizer(self.data, n_folds=walkforward_folds, inner_trials=50)
        wf_result = wf_opt.optimize()
        
        duration = (datetime.now() - start).total_seconds()
        
        # Choose best overall
        best_result = max(
            [random_result, bayesian_result, wf_result],
            key=lambda x: x.best_score
        )
        
        print(f"\n{'='*70}")
        print("MULTI-STAGE COMPLETE")
        print(f"{'='*70}")
        print(f"Best method: {best_result.method}")
        print(f"Best score: {best_result.best_score:.4f}")
        print(f"Total duration: {duration:.0f}s")
        
        return OptimizationResult(
            best_params=best_result.best_params,
            best_score=best_result.best_score,
            best_stats={
                'random_best': random_result.best_score,
                'bayesian_best': bayesian_result.best_score,
                'walkforward_avg': wf_result.best_score,
                'best_method': best_result.method
            },
            all_trials=all_trials,
            method='multi_stage',
            duration_seconds=duration
        )


# ============================================================================
# OUTPUT
# ============================================================================

def print_results(result: OptimizationResult):
    """Print optimization results."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RESULTS ({result.method.upper()})")
    print(f"{'='*70}")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Duration: {result.duration_seconds:.0f}s")
    
    if hasattr(result, 'total_historical_trials') and result.total_historical_trials > 0:
        print(f"Total Historical Trials: {result.total_historical_trials}")
    
    print(f"\nBest Parameters:")
    
    for key, value in sorted(result.best_params.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def print_config(params: Dict[str, Any]):
    """Print ready-to-copy config."""
    print(f"""
# ============================================================
# OPTIMIZED PARAMETERS
# ============================================================

# Timeframes
SIGNAL_TIMEFRAME = '{params['signal_timeframe']}'
ATR_TIMEFRAME = '{params['atr_timeframe']}'

# Bollinger Bands
BB_PERIOD = {params['bb_period']}
BB_STD = {params['bb_std']:.2f}

# Keltner Channels
KC_PERIOD = {params['kc_period']}
KC_ATR_MULT = {params['kc_atr_mult']:.2f}

# Momentum
MOMENTUM_PERIOD = {params['momentum_period']}

# RSI
RSI_PERIOD = {params['rsi_period']}
RSI_OVERBOUGHT = {params['rsi_overbought']}
RSI_OVERSOLD = {params['rsi_oversold']}

# Squeeze
MIN_SQUEEZE_BARS = {params['min_squeeze_bars']}

# Volume
VOLUME_PERIOD = {params['volume_period']}
MIN_VOLUME_RATIO = {params['min_volume_ratio']:.2f}

# ATR & Stops
ATR_PERIOD = {params['atr_period']}
ATR_STOP_MULT = {params['atr_stop_mult']:.2f}
ATR_TARGET_MULT = {params['atr_target_mult']:.2f}

# Setup
SETUP_VALIDITY_BARS = {params['setup_validity_bars']}

# Risk
MAX_POSITIONS = {params['max_positions']}
MAX_DAILY_LOSS = {params['max_daily_loss']:.2f}
LONG_ONLY = False

# Position Sizing (FIXED)
BASE_POSITION = {FIXED_BASE_POSITION:.2f}
MIN_POSITION = {FIXED_MIN_POSITION:.2f}
MAX_POSITION = {FIXED_MAX_POSITION:.2f}
""")


def save_results(result: OptimizationResult, filename: str):
    """Save results to CSV."""
    rows = []
    for trial in result.all_trials:
        row = {'score': trial.get('score', trial.get('test_score', 0))}
        row.update(trial.get('params', {}))
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(rows)} trials to {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='BB Squeeze Strategy Optimizer')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['random', 'bayesian', 'grid', 'walkforward', 'multi'],
                        help='Optimization method')
    parser.add_argument('--trials', type=int, default=200,
                        help='Number of trials (for random/bayesian)')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds (for walkforward)')
    parser.add_argument('--days', type=int, default=DAYS_BACK,
                        help='Days of historical data (default: 2190 = 6 years)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD) - filters cached data')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD) - filters cached data')
    
    # History management
    parser.add_argument('--study', type=str, default=DEFAULT_STUDY_NAME,
                        help=f'Study name for persistent history (default: {DEFAULT_STUDY_NAME})')
    parser.add_argument('--reset', action='store_true',
                        help='Reset study history and start fresh')
    parser.add_argument('--history', action='store_true',
                        help='Show optimization history and exit')
    parser.add_argument('--best', action='store_true',
                        help='Show best parameters from history and exit')
    parser.add_argument('--delete-study', type=str, default=None,
                        help='Delete a specific study from history')
    
    args = parser.parse_args()
    
    # History management commands
    if args.history:
        show_history()
        return
    
    if args.best:
        show_best_params(args.study)
        return
    
    if args.delete_study:
        delete_study(args.delete_study)
        return
    
    # Normal optimization
    print(f"\nLoading {args.days} days of data...")
    data = load_data(SYMBOLS, args.days)
    
    if not data:
        print("Failed to load data")
        return
    
    # Filter by date range if specified
    if args.start_date or args.end_date:
        print(f"Filtering data: {args.start_date or 'start'} to {args.end_date or 'end'}")
        data = filter_data_by_date(data, args.start_date, args.end_date)
        
        # Print filtered date range
        first_symbol = list(data.keys())[0]
        first_tf = list(data[first_symbol].keys())[0]
        df = data[first_symbol][first_tf]
        print(f"Filtered range: {df.index[0]} to {df.index[-1]}")
        print(f"Filtered bars: {len(df):,}")
    
    # Run optimization
    if args.method == 'random':
        optimizer = RandomSearchOptimizer(data)
        result = optimizer.optimize(args.trials)
    
    elif args.method == 'bayesian':
        optimizer = BayesianOptimizer(
            data, 
            n_startup=min(50, args.trials // 4),
            study_name=args.study,
            reset=args.reset
        )
        result = optimizer.optimize(args.trials)
    
    elif args.method == 'grid':
        optimizer = GridSearchOptimizer(data)
        result = optimizer.optimize()
    
    elif args.method == 'walkforward':
        optimizer = WalkForwardOptimizer(data, n_folds=args.folds)
        result = optimizer.optimize()
    
    elif args.method == 'multi':
        optimizer = MultiStageOptimizer(data)
        result = optimizer.optimize(
            random_trials=100,
            bayesian_trials=args.trials,
            walkforward_folds=args.folds
        )
    
    # Output
    print_results(result)
    print(f"\n{'='*70}")
    print("COPY TO run_backtest.py OR run_live.py:")
    print(f"{'='*70}")
    print_config(result.best_params)
    
    # Save
    save_results(result, f'bb_optim_{args.method}.csv')


if __name__ == "__main__":
    main()