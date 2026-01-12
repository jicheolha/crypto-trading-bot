"""
Data utilities for fetching cryptocurrency data with local caching.
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')

TIMEFRAME_MAP = {
    '1min': TimeFrame.Minute,
    '5min': TimeFrame(5, 'Min'),
    '15min': TimeFrame(15, 'Min'),
    '30min': TimeFrame(30, 'Min'),
    '1h': TimeFrame.Hour,
    '4h': TimeFrame(4, 'Hour'),
    '1d': TimeFrame.Day
}

RESAMPLE_MAP = {
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1h': '1H',
    '4h': '4H',
    '1d': '1D'
}

TIMEFRAME_MINUTES = {
    '1min': 1,
    '5min': 5,
    '15min': 15,
    '30min': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}


def _get_cache_path(symbol: str, timeframe: str, days_back: int) -> str:
    """Get cache file path for given parameters."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    symbol_clean = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{symbol_clean}_{timeframe}_{days_back}d.pkl")


def _is_cache_valid(cache_path: str, max_age_hours: int = 1) -> bool:
    """Check if cache file exists and is fresh."""
    if not os.path.exists(cache_path):
        return False
    
    # None means cache never expires
    if max_age_hours is None:
        return True
    
    modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - modified_time
    return age < timedelta(hours=max_age_hours)


def prepare_data_for_backtest(
    symbol: str, 
    timeframe: str = '1h', 
    days_back: int = 30,
    use_cache: bool = True,
    cache_max_age_hours: int = None  # None = never expire
) -> pd.DataFrame:
    """Load data with optional caching."""
    
    cache_path = _get_cache_path(symbol, timeframe, days_back)
    
    # Try loading from cache
    if use_cache and _is_cache_valid(cache_path, cache_max_age_hours):
        try:
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
                print(f"  Loaded {symbol} {timeframe} from cache ({len(df)} bars)")
                return df
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
    
    # Fetch from API
    try:
        client = CryptoHistoricalDataClient()
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time
        )
        
        bars = client.get_crypto_bars(request_params)
        df = bars.df

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')

        df.columns = df.columns.str.lower()
        df.index = pd.to_datetime(df.index)

        if timeframe != '1min':
            resample_period = RESAMPLE_MAP.get(timeframe, '1H')
            df = df.resample(resample_period).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        df = _apply_quality_filters(df)
        
        # Save to cache
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
        
        print(f"  Fetched {symbol} {timeframe} from API ({len(df)} bars)")
        return df

    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()


def _apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['volume'] > 0]
    df = df[
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close'])
    ]
    return df


def load_three_timeframe_data(
    symbols: List[str],
    trade_tf: str = '1min',
    signal_tf: str = '1h',
    atr_tf: str = '4h',
    days_back: int = 60,
    use_cache: bool = True,
    cache_max_age_hours: int = None  # None = never expire, int = hours until refresh
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load data for three timeframes per symbol with caching.
    
    Args:
        symbols: List of trading pair symbols
        trade_tf: Execution timeframe (e.g., '1min')
        signal_tf: Signal generation timeframe (e.g., '1h')
        atr_tf: ATR calculation timeframe (e.g., '4h')
        days_back: Number of days of historical data
        use_cache: Whether to use local file cache
        cache_max_age_hours: How old cache can be before refresh
        
    Returns:
        Dict: {symbol: {'trade': df, 'signal': df, 'atr': df}}
    """
    data = {}
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        
        trade_df = prepare_data_for_backtest(
            symbol, timeframe=trade_tf, days_back=days_back,
            use_cache=use_cache, cache_max_age_hours=cache_max_age_hours
        )
        signal_df = prepare_data_for_backtest(
            symbol, timeframe=signal_tf, days_back=days_back,
            use_cache=use_cache, cache_max_age_hours=cache_max_age_hours
        )
        atr_days = max(days_back, 90)
        atr_df = prepare_data_for_backtest(
            symbol, timeframe=atr_tf, days_back=atr_days,
            use_cache=use_cache, cache_max_age_hours=cache_max_age_hours
        )
        
        if not trade_df.empty and not signal_df.empty and not atr_df.empty:
            data[symbol] = {
                'trade': trade_df,
                'signal': signal_df,
                'atr': atr_df
            }
        else:
            print(f"Failed to load data for {symbol}")
    
    if data:
        print(f"Symbols loaded: {len(data)}/{len(symbols)}")
    else:
        print("No data loaded")
    
    return data


def clear_cache():
    """Delete all cached data files."""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")
    else:
        print("No cache to clear")


def get_timeframe_minutes(tf: str) -> int:
    return TIMEFRAME_MINUTES.get(tf, 60)