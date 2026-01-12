#!/usr/bin/env python3
"""
Smart Data Downloader - Downloads 1min data and resamples to all timeframes

This is MUCH faster than downloading each timeframe separately!

Strategy:
1. Download 1min data ONCE from Alpaca (5-10 min per symbol)
2. Resample to create all other timeframes (instant!)
3. Cache everything

Benefits:
- Fewer API calls (1 instead of 8 per symbol!)
- Faster downloads (1/8th the time)
- Same result - all timeframes cached

Usage:
    python download_data_smart.py
"""
import os
import sys
import pickle
from datetime import datetime, timedelta
import pandas as pd

# Import Alpaca client
try:
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("ERROR: alpaca-py not installed!")
    print("Install with: pip install alpaca-py")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOLS = ['BTC/USD', 'ETH/USD', 'DOGE/USD', 'SOL/USD', 'AAVE/USD', 'XRP/USD', 'ADA/USD']
DAYS_BACK = 365 * 6  # 5 years

# All timeframes to create from 1min data
TIMEFRAMES = {
    '1min': '1T',    # Already have this
    '5min': '5T',    # Resample from 1min
    '15min': '15T',  # Resample from 1min
    '30min': '30T',  # Resample from 1min
    '1h': '1H',      # Resample from 1min
    '4h': '4H',      # Resample from 1min
    '1d': '1D',      # Resample from 1min
    '3d': '3D',      # Resample from 1min
    '1w': '1W',      # Resample from 1min
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_cache_path(symbol: str, timeframe: str, days_back: int) -> str:
    """Get cache file path."""
    symbol_clean = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{symbol_clean}_{timeframe}_{days_back}d.pkl")


def _apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bad data."""
    df = df[df['volume'] > 0]
    df = df[
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ]
    return df


def download_1min_data(symbol: str, days_back: int) -> pd.DataFrame:
    """Download 1min data from Alpaca."""
    print(f"\n{'='*70}")
    print(f"Downloading {symbol} (1min bars)")
    print(f"{'='*70}")
    
    try:
        client = CryptoHistoricalDataClient()
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        print(f"Period: {start_time.date()} to {end_time.date()}")
        print(f"Fetching from Alpaca API...")
        
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time
        )
        
        bars = client.get_crypto_bars(request_params)
        df = bars.df
        
        if df.empty:
            print(f"  âœ— No data returned")
            return pd.DataFrame()
        
        # Handle multi-index
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')
        
        # Standardize
        df.columns = df.columns.str.lower()
        df.index = pd.to_datetime(df.index)
        
        # Filter bad data
        df = _apply_quality_filters(df)
        
        print(f"  âœ“ Downloaded {len(df):,} bars")
        print(f"  âœ“ Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"  âœ— Error: {e}")
        return pd.DataFrame()


def resample_to_timeframe(df_1min: pd.DataFrame, timeframe: str, resample_rule: str) -> pd.DataFrame:
    """Resample 1min data to target timeframe."""
    if timeframe == '1min':
        return df_1min  # Already 1min
    
    print(f"  Resampling to {timeframe}...", end=" ")
    
    df = df_1min.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"{len(df):,} bars")
    
    return df


def save_to_cache(df: pd.DataFrame, symbol: str, timeframe: str, days_back: int):
    """Save dataframe to cache."""
    cache_path = _get_cache_path(symbol, timeframe, days_back)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  âœ“ Cached {timeframe}: {size_mb:.1f} MB ({len(df):,} bars)")
        
    except Exception as e:
        print(f"  âœ— Cache failed for {timeframe}: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print("SMART DATA DOWNLOADER")
    print(f"{'='*70}")
    print(f"Strategy: Download 1min data, then resample to all timeframes")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {len(TIMEFRAMES)}")
    print(f"History: {DAYS_BACK} days ({DAYS_BACK/365:.1f} years)")
    print(f"Cache: {CACHE_DIR}")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    total_downloaded = 0
    total_resampled = 0
    
    for symbol in SYMBOLS:
        # Download 1min data
        df_1min = download_1min_data(symbol, DAYS_BACK)
        
        if df_1min.empty:
            print(f"  âœ— Skipping {symbol} - no data")
            continue
        
        total_downloaded += 1
        
        # Save 1min to cache
        save_to_cache(df_1min, symbol, '1min', DAYS_BACK)
        
        # Resample to all other timeframes
        print(f"\n  Creating all timeframes from 1min data...")
        
        for timeframe, resample_rule in TIMEFRAMES.items():
            if timeframe == '1min':
                continue  # Already saved
            
            # Resample
            df_resampled = resample_to_timeframe(df_1min, timeframe, resample_rule)
            
            if not df_resampled.empty:
                save_to_cache(df_resampled, symbol, timeframe, DAYS_BACK)
                total_resampled += 1
        
        print(f"\n  âœ“ {symbol} complete - {len(TIMEFRAMES)} timeframes cached!")
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Symbols downloaded: {total_downloaded}/{len(SYMBOLS)}")
    print(f"Timeframes created: {total_resampled}")
    print(f"Total time: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"Cache location: {CACHE_DIR}")
    print(f"{'='*70}")
    
    # Show cache size
    total_size = 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for file in files:
            if file.endswith('.pkl'):
                total_size += os.path.getsize(os.path.join(root, file))
    
    print(f"\nTotal cache size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Files cached: {len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')])}")
    
    print(f"\nâœ… All data cached and ready!")
    print(f"\nNow you can run:")
    print(f"  - python run_backtest_cached.py  (instant!)")
    print(f"  - python optimize.py --all-combos  (instant!)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()