"""
Shared utilities for Crypto Trading Bot.
"""
from datetime import datetime
from typing import Optional
import pytz


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


# Timeframe mappings
TIMEFRAME_MINUTES = {
    '1min': 1, '1m': 1,
    '5min': 5, '5m': 5,
    '15min': 15, '15m': 15,
    '30min': 30, '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}


def get_tf_minutes(tf: str) -> int:
    """Get minutes for a timeframe string."""
    return TIMEFRAME_MINUTES.get(tf, 60)


def ensure_tz_aware(dt: datetime, reference_tz=None) -> datetime:
    """Ensure datetime is timezone-aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        if reference_tz is None:
            return dt.replace(tzinfo=pytz.UTC)
        return reference_tz.localize(dt)
    return dt


def infer_timeframe_from_index(index) -> str:
    """
    Infer timeframe from a pandas DatetimeIndex.
    
    Returns timeframe string like '1min', '1h', '1d'.
    """
    if len(index) < 2:
        return '1min'
    
    # Calculate median difference between timestamps
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return '1min'
    
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60
    
    # Map to nearest standard timeframe
    if minutes <= 1.5:
        return '1min'
    elif minutes <= 7:
        return '5min'
    elif minutes <= 20:
        return '15min'
    elif minutes <= 45:
        return '30min'
    elif minutes <= 120:
        return '1h'
    elif minutes <= 360:
        return '4h'
    else:
        return '1d'