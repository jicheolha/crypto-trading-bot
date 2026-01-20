"""
Technical Analysis Module for Crypto Trading Bot.

Indicators:
- Bollinger Bands (SMA + standard deviation)
- Keltner Channels (EMA + ATR)
- Squeeze detection (BB inside KC)
- Volume confirmation
- RSI for momentum filter
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SqueezeState:
    """Current squeeze state."""
    is_squeeze: bool
    squeeze_bars: int
    bb_width: float
    momentum: float
    direction: Optional[str]  # 'bullish', 'bearish', None


class BBSqueezeAnalyzer:
    """
    Technical indicator analyzer for volatility-based trading signals.
    """
    
    def __init__(
        self,
        # Bollinger Bands
        bb_period: int = 20,
        bb_std: float = 2.0,
        # Keltner Channels
        kc_period: int = 20,
        kc_atr_mult: float = 1.5,
        # Momentum
        momentum_period: int = 12,
        # RSI
        rsi_period: int = 14,
        # Volume
        volume_period: int = 20,
        # ATR
        atr_period: int = 14,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_mult = kc_atr_mult
        self.momentum_period = momentum_period
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.atr_period = atr_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        
        # Bollinger Bands
        df['BB_Mid'] = df['close'].rolling(self.bb_period).mean()
        df['BB_Std'] = df['close'].rolling(self.bb_period).std()
        df['BB_Upper'] = df['BB_Mid'] + (self.bb_std * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (self.bb_std * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        
        # Percent B - where price is within bands (0 = lower, 1 = upper)
        df['BB_PercentB'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR for Keltner Channels
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(self.atr_period).mean()
        
        # Keltner Channels
        df['KC_Mid'] = df['close'].ewm(span=self.kc_period, adjust=False).mean()
        df['KC_Upper'] = df['KC_Mid'] + (self.kc_atr_mult * df['ATR'])
        df['KC_Lower'] = df['KC_Mid'] - (self.kc_atr_mult * df['ATR'])
        
        # Squeeze detection: BB inside KC
        df['Squeeze'] = (df['BB_Lower'] > df['KC_Lower']) & (df['BB_Upper'] < df['KC_Upper'])
        
        # Momentum (rate of change of midline)
        df['Momentum'] = df['close'] - df['BB_Mid'].shift(self.momentum_period)
        df['Momentum_Norm'] = df['Momentum'] / df['ATR']  # Normalized by ATR
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume ratio
        df['Volume_MA'] = df['volume'].rolling(self.volume_period).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
        
        # Squeeze duration (consecutive squeeze bars)
        df['Squeeze_Duration'] = df['Squeeze'].groupby(
            (~df['Squeeze']).cumsum()
        ).cumsum()
        
        return df
    
    def get_squeeze_state(self, df: pd.DataFrame) -> SqueezeState:
        """Get current squeeze state."""
        if len(df) < self.bb_period + 5:
            return SqueezeState(
                is_squeeze=False,
                squeeze_bars=0,
                bb_width=0,
                momentum=0,
                direction=None
            )
        
        latest = df.iloc[-1]
        
        # Determine momentum direction
        direction = None
        if latest['Momentum'] > 0:
            direction = 'bullish'
        elif latest['Momentum'] < 0:
            direction = 'bearish'
        
        return SqueezeState(
            is_squeeze=bool(latest['Squeeze']),
            squeeze_bars=int(latest['Squeeze_Duration']),
            bb_width=float(latest['BB_Width']),
            momentum=float(latest['Momentum_Norm']),
            direction=direction
        )
    
    def detect_breakout(
        self, 
        df: pd.DataFrame,
        min_squeeze_bars: int = 3,
        min_volume_ratio: float = 1.2
    ) -> Optional[Dict]:
        """
        Detect breakout from squeeze.
        
        Returns:
            Dict with breakout info or None
        """
        if len(df) < self.bb_period + 10:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Was in squeeze, now released
        squeeze_released = prev['Squeeze'] and not current['Squeeze']
        
        if not squeeze_released:
            return None
        
        # Check squeeze duration
        squeeze_duration = prev['Squeeze_Duration']
        if squeeze_duration < min_squeeze_bars:
            return None
        
        # Check volume confirmation
        if current['Volume_Ratio'] < min_volume_ratio:
            return None
        
        # Determine direction
        if current['close'] > current['BB_Upper']:
            direction = 'long'
        elif current['close'] < current['BB_Lower']:
            direction = 'short'
        else:
            # Check momentum direction
            if current['Momentum'] > 0:
                direction = 'long'
            elif current['Momentum'] < 0:
                direction = 'short'
            else:
                return None
        
        # NOTE: RSI filtering is done in signal_generator with configurable thresholds
        # Do not add RSI filtering here to avoid duplicate/conflicting checks
        
        return {
            'direction': direction,
            'price': current['close'],
            'squeeze_bars': squeeze_duration,
            'bb_width': current['BB_Width'],
            'momentum': current['Momentum_Norm'],
            'volume_ratio': current['Volume_Ratio'],
            'rsi': current['RSI'],
            'atr': current['ATR']
        }