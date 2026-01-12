#!/usr/bin/env python3
"""
Indicator Visualization Module for Bollinger Bands Squeeze Strategy

Visualizes all technical indicators using cached data with buy/sell signals.

Features:
- Price with Bollinger Bands and Keltner Channels
- Squeeze detection highlighting
- BUY/SELL signals from squeeze breakouts
- RSI with overbought/oversold zones
- Momentum histogram
- Volume ratio
- ATR

Usage:
    python visualize_indicators.py                          # Default BTC/USD
    python visualize_indicators.py --symbol ETH/USD         # Different symbol
    python visualize_indicators.py --start 2024-01-01       # Date range
    python visualize_indicators.py --last-n 500             # Last N bars
    python visualize_indicators.py --interactive            # Show plot
    python visualize_indicators.py --histogram              # Squeeze stats
"""
import os
import sys
import pickle
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Try importing plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from technical import BBSqueezeAnalyzer


# ============================================================================
# OPTIMAL PARAMETERS (4h signal / 1h ATR)
# ============================================================================

# Symbols & Data
DEFAULT_SYMBOLS = ['BTC/USD']
DEFAULT_DAYS_BACK = 365 * 6
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')

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
BASE_POSITION = 0.2
MIN_POSITION = 0.1
MAX_POSITION = 0.5

# Setup
SETUP_VALIDITY_BARS = 8

# Risk
MAX_POSITIONS = 3
MAX_DAILY_LOSS = 0.03
MAX_HOLD_DAYS = 7
LONG_ONLY = False


# ============================================================================
# SIGNAL DATACLASS
# ============================================================================

@dataclass
class Signal:
    """Detected trading signal."""
    timestamp: datetime
    direction: str  # 'long' or 'short'
    price: float
    stop_loss: float
    take_profit: float
    squeeze_bars: int
    volume_ratio: float
    momentum: float
    rsi: float
    atr: float


# ============================================================================
# COLORS
# ============================================================================

class Colors:
    """Chart color scheme."""
    PRICE = '#2E86AB'
    BB_FILL = '#3498DB'
    BB_LINE = '#2980B9'
    KC_LINE = '#E74C3C'
    KC_FILL = '#E74C3C'
    SQUEEZE_BG = '#F1C40F'
    VOLUME_UP = '#27AE60'
    VOLUME_DOWN = '#E74C3C'
    RSI_LINE = '#8E44AD'
    RSI_OB = '#E74C3C'
    RSI_OS = '#27AE60'
    MOM_POS = '#27AE60'
    MOM_NEG = '#E74C3C'
    ATR_LINE = '#F39C12'
    GRID = '#BDC3C7'
    BUY = '#00FF00'
    SELL = '#FF0000'


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cache_path(symbol: str, timeframe: str, days_back: int) -> str:
    """Get cache file path."""
    symbol_clean = symbol.replace('/', '_')
    return os.path.join(CACHE_DIR, f"{symbol_clean}_{timeframe}_{days_back}d.pkl")


def load_cached_data(symbol: str, timeframe: str, days_back: int = DEFAULT_DAYS_BACK) -> Optional[pd.DataFrame]:
    """Load data from cache."""
    cache_path = get_cache_path(symbol, timeframe, days_back)
    
    if not os.path.exists(cache_path):
        print(f"  âœ— Cache not found: {cache_path}")
        # Try alternative days_back values
        for alt_days in [365*6, 365*5, 365*4, 365*3, 365*2, 365, 180, 90, 60, 30]:
            alt_path = get_cache_path(symbol, timeframe, alt_days)
            if os.path.exists(alt_path):
                print(f"  âœ“ Found alternative cache: {alt_days} days")
                cache_path = alt_path
                break
        else:
            return None
    
    try:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print(f"  âœ“ Loaded {symbol} {timeframe}: {len(df)} bars")
        return df
    except Exception as e:
        print(f"  âœ— Error loading cache: {e}")
        return None


def list_available_cache() -> Dict[str, list]:
    """List all available cached data."""
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory not found: {CACHE_DIR}")
        return {}
    
    available = {}
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.pkl'):
            parts = filename.replace('.pkl', '').split('_')
            if len(parts) >= 3:
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = parts[2]
                if symbol not in available:
                    available[symbol] = []
                available[symbol].append(timeframe)
    
    return available


def filter_by_date(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Filter dataframe by date range."""
    if df.empty:
        return df
    
    if start_date:
        start_ts = pd.Timestamp(start_date)
        if df.index.tz is not None:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]
    
    if end_date:
        end_ts = pd.Timestamp(end_date + ' 23:59:59')
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]
    
    return df


# ============================================================================
# SIGNAL DETECTION
# ============================================================================

def detect_signals(df: pd.DataFrame, analyzer: BBSqueezeAnalyzer) -> List[Signal]:
    """
    Detect all buy/sell signals in the data using squeeze breakout logic.
    
    Returns list of Signal objects.
    """
    signals = []
    
    # Need enough data for indicators
    min_period = max(BB_PERIOD, KC_PERIOD, RSI_PERIOD, VOLUME_PERIOD, ATR_PERIOD) + 10
    
    if len(df) < min_period:
        return signals
    
    # Calculate indicators for full dataset
    df_ind = analyzer.calculate_indicators(df.copy())
    
    # Scan for breakouts
    for i in range(min_period, len(df_ind)):
        current = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]
        
        # Check for squeeze release (was in squeeze, now out)
        if not prev['Squeeze'] or current['Squeeze']:
            continue
        
        # Check squeeze duration
        squeeze_duration = prev['Squeeze_Duration']
        if squeeze_duration < MIN_SQUEEZE_BARS:
            continue
        
        # Check volume confirmation
        if current['Volume_Ratio'] < MIN_VOLUME_RATIO:
            continue
        
        # Determine direction
        if current['close'] > current['BB_Upper']:
            direction = 'long'
        elif current['close'] < current['BB_Lower']:
            direction = 'short'
        elif current['Momentum'] > 0:
            direction = 'long'
        elif current['Momentum'] < 0:
            direction = 'short'
        else:
            continue
        
        # RSI filter
        if direction == 'long' and current['RSI'] > RSI_OVERBOUGHT:
            continue
        if direction == 'short' and current['RSI'] < RSI_OVERSOLD:
            continue
        
        # Skip shorts if LONG_ONLY
        if LONG_ONLY and direction == 'short':
            continue
        
        # Calculate stops
        price = current['close']
        atr = current['ATR']
        
        if direction == 'long':
            stop_loss = price - (atr * ATR_STOP_MULT)
            take_profit = price + (atr * ATR_TARGET_MULT)
        else:
            stop_loss = price + (atr * ATR_STOP_MULT)
            take_profit = price - (atr * ATR_TARGET_MULT)
        
        signals.append(Signal(
            timestamp=df_ind.index[i],
            direction=direction,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            squeeze_bars=int(squeeze_duration),
            volume_ratio=current['Volume_Ratio'],
            momentum=current['Momentum_Norm'],
            rsi=current['RSI'],
            atr=atr
        ))
    
    return signals


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_indicator_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    analyzer: BBSqueezeAnalyzer,
    signals: List[Signal] = None,
    figsize: Tuple[int, int] = (18, 14)
) -> plt.Figure:
    """
    Create comprehensive indicator visualization with buy/sell signals.
    """
    # Calculate indicators
    df_ind = analyzer.calculate_indicators(df.copy())
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Grid: 5 rows with different heights
    gs = GridSpec(5, 1, height_ratios=[4, 1.2, 1.2, 1.2, 1], hspace=0.08, figure=fig)
    
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_momentum = fig.add_subplot(gs[3], sharex=ax_price)
    ax_atr = fig.add_subplot(gs[4], sharex=ax_price)
    
    # ========================================================================
    # SUBPLOT 1: Price with Bollinger Bands and Keltner Channels
    # ========================================================================
    
    # Keltner Channels (behind BB)
    ax_price.fill_between(
        df_ind.index, df_ind['KC_Upper'], df_ind['KC_Lower'],
        alpha=0.08, color=Colors.KC_FILL, label='_nolegend_'
    )
    ax_price.plot(df_ind.index, df_ind['KC_Upper'], '--', 
                  color=Colors.KC_LINE, linewidth=1, alpha=0.7, label='KC Upper')
    ax_price.plot(df_ind.index, df_ind['KC_Mid'], '-', 
                  color=Colors.KC_LINE, linewidth=0.8, alpha=0.5, label='KC Mid')
    ax_price.plot(df_ind.index, df_ind['KC_Lower'], '--', 
                  color=Colors.KC_LINE, linewidth=1, alpha=0.7, label='KC Lower')
    
    # Bollinger Bands
    ax_price.fill_between(
        df_ind.index, df_ind['BB_Upper'], df_ind['BB_Lower'],
        alpha=0.15, color=Colors.BB_FILL, label='_nolegend_'
    )
    ax_price.plot(df_ind.index, df_ind['BB_Upper'], '--', 
                  color=Colors.BB_LINE, linewidth=1.2, label='BB Upper')
    ax_price.plot(df_ind.index, df_ind['BB_Mid'], '-', 
                  color=Colors.BB_LINE, linewidth=1, alpha=0.7, label='BB Mid (SMA)')
    ax_price.plot(df_ind.index, df_ind['BB_Lower'], '--', 
                  color=Colors.BB_LINE, linewidth=1.2, label='BB Lower')
    
    # Price line
    ax_price.plot(df_ind.index, df_ind['close'], '-', 
                  color=Colors.PRICE, linewidth=1.5, label='Price')
    

    
    # Plot BUY/SELL signals
    if signals:
        buy_signals = [s for s in signals if s.direction == 'long']
        sell_signals = [s for s in signals if s.direction == 'short']
        
        if buy_signals:
            buy_times = [s.timestamp for s in buy_signals]
            buy_prices = [s.price for s in buy_signals]
            ax_price.scatter(buy_times, buy_prices, marker='^', color=Colors.BUY, 
                           s=150, zorder=10, edgecolors='black', linewidths=1, label='BUY')
            
            # Draw stop/target lines for most recent buy
            latest_buy = buy_signals[-1]
            if latest_buy.timestamp == df_ind.index[-1] or \
               (df_ind.index[-1] - latest_buy.timestamp).total_seconds() < 3600 * 24 * 3:
                ax_price.axhline(y=latest_buy.stop_loss, color='red', linestyle=':', 
                               linewidth=1, alpha=0.7)
                ax_price.axhline(y=latest_buy.take_profit, color='green', linestyle=':', 
                               linewidth=1, alpha=0.7)
        
        if sell_signals:
            sell_times = [s.timestamp for s in sell_signals]
            sell_prices = [s.price for s in sell_signals]
            ax_price.scatter(sell_times, sell_prices, marker='v', color=Colors.SELL, 
                           s=150, zorder=10, edgecolors='black', linewidths=1, label='SELL')
            
            # Draw stop/target lines for most recent sell
            latest_sell = sell_signals[-1]
            if latest_sell.timestamp == df_ind.index[-1] or \
               (df_ind.index[-1] - latest_sell.timestamp).total_seconds() < 3600 * 24 * 3:
                ax_price.axhline(y=latest_sell.stop_loss, color='red', linestyle=':', 
                               linewidth=1, alpha=0.7)
                ax_price.axhline(y=latest_sell.take_profit, color='green', linestyle=':', 
                               linewidth=1, alpha=0.7)
    
    ax_price.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax_price.set_title(
        f'{symbol} {timeframe} â€” Bollinger Bands Squeeze Indicators\n'
        f'BB({BB_PERIOD}, {BB_STD}Ïƒ) | KC({KC_PERIOD}, {KC_ATR_MULT}x ATR) | '
        f'RSI({RSI_PERIOD}) | Mom({MOMENTUM_PERIOD})',
        fontsize=13, fontweight='bold', pad=15
    )
    ax_price.legend(loc='upper left', fontsize=9, ncol=2)
    ax_price.grid(True, alpha=0.3, color=Colors.GRID)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ========================================================================
    # SUBPLOT 2: Volume with Ratio
    # ========================================================================
    
    colors = [Colors.VOLUME_UP if r >= MIN_VOLUME_RATIO else Colors.VOLUME_DOWN 
              for r in df_ind['Volume_Ratio'].fillna(0)]
    
    if len(df_ind) > 1:
        bar_width = (df_ind.index[1] - df_ind.index[0]) * 0.8
    else:
        bar_width = pd.Timedelta(hours=1)
    
    ax_volume.bar(df_ind.index, df_ind['volume'], color=colors, alpha=0.7, width=bar_width)
    ax_volume.plot(df_ind.index, df_ind['Volume_MA'], color='orange', 
                   linewidth=1.5, label=f'{VOLUME_PERIOD}-bar MA')
    
    min_vol_line = df_ind['Volume_MA'] * MIN_VOLUME_RATIO
    ax_volume.plot(df_ind.index, min_vol_line, '--', color='purple', 
                   linewidth=1, alpha=0.7, label=f'Min ({MIN_VOLUME_RATIO:.2f}x)')
    
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.legend(loc='upper left', fontsize=8)
    ax_volume.grid(True, alpha=0.3, color=Colors.GRID)
    ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'
    ))
    
    # ========================================================================
    # SUBPLOT 3: RSI
    # ========================================================================
    
    ax_rsi.plot(df_ind.index, df_ind['RSI'], color=Colors.RSI_LINE, linewidth=1.5)
    ax_rsi.axhline(y=RSI_OVERBOUGHT, color=Colors.RSI_OB, linestyle='--', 
                   linewidth=1, alpha=0.8, label=f'Overbought ({RSI_OVERBOUGHT})')
    ax_rsi.axhline(y=RSI_OVERSOLD, color=Colors.RSI_OS, linestyle='--', 
                   linewidth=1, alpha=0.8, label=f'Oversold ({RSI_OVERSOLD})')
    ax_rsi.axhline(y=50, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax_rsi.fill_between(df_ind.index, RSI_OVERBOUGHT, 100, alpha=0.1, color=Colors.RSI_OB)
    ax_rsi.fill_between(df_ind.index, 0, RSI_OVERSOLD, alpha=0.1, color=Colors.RSI_OS)
    
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc='upper left', fontsize=8)
    ax_rsi.grid(True, alpha=0.3, color=Colors.GRID)
    
    # ========================================================================
    # SUBPLOT 4: Momentum (Normalized)
    # ========================================================================
    
    mom_colors = [Colors.MOM_POS if x > 0 else Colors.MOM_NEG 
                  for x in df_ind['Momentum_Norm'].fillna(0)]
    ax_momentum.bar(df_ind.index, df_ind['Momentum_Norm'], color=mom_colors, 
                    alpha=0.7, width=bar_width)
    ax_momentum.axhline(y=0, color='black', linewidth=1)
    
    ax_momentum.set_ylabel('Momentum', fontsize=10)
    ax_momentum.grid(True, alpha=0.3, color=Colors.GRID)
    
    # ========================================================================
    # SUBPLOT 5: ATR
    # ========================================================================
    
    ax_atr.plot(df_ind.index, df_ind['ATR'], color=Colors.ATR_LINE, linewidth=1.5)
    ax_atr.fill_between(df_ind.index, 0, df_ind['ATR'], alpha=0.2, color=Colors.ATR_LINE)
    
    ax_atr.set_ylabel('ATR', fontsize=10)
    ax_atr.set_xlabel('Date', fontsize=11)
    ax_atr.grid(True, alpha=0.3, color=Colors.GRID)
    ax_atr.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ========================================================================
    # Format x-axis
    # ========================================================================
    
    for ax in [ax_price, ax_volume, ax_rsi, ax_momentum]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    ax_atr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_atr.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_atr.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig


def create_interactive_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    analyzer: BBSqueezeAnalyzer,
    signals: List[Signal] = None,
) -> str:
    """
    Create interactive Plotly chart with zoom/pan/hover.
    
    Returns HTML filename.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Install with: pip install plotly")
        return None
    
    # Calculate indicators
    df_ind = analyzer.calculate_indicators(df.copy())
    
    # Calculate equity curve from signals
    equity = calculate_equity_curve(df_ind, signals)
    
    # Create subplots (removed volume, added equity)
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.2, 0.15, 0.15, 0.15],
        subplot_titles=('', '', '', '', '')  # Remove subplot titles, cleaner
    )
    
    # ========================================================================
    # ROW 1: Price with BB and KC
    # ========================================================================
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_ind.index,
        open=df_ind['open'],
        high=df_ind['high'],
        low=df_ind['low'],
        close=df_ind['close'],
        name='Price',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['BB_Upper'],
        name='BB', line=dict(color='#2980B9', width=1, dash='dash'),
        legendgroup='bb', showlegend=True,
        hovertemplate='BB Upper: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['BB_Mid'],
        line=dict(color='#2980B9', width=1),
        legendgroup='bb', showlegend=False,
        hovertemplate='BB Mid: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['BB_Lower'],
        line=dict(color='#2980B9', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(52, 152, 219, 0.1)',
        legendgroup='bb', showlegend=False,
        hovertemplate='BB Lower: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Keltner Channels
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['KC_Upper'],
        name='KC', line=dict(color='#E74C3C', width=1, dash='dot'),
        legendgroup='kc', showlegend=True,
        hovertemplate='KC Upper: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['KC_Lower'],
        line=dict(color='#E74C3C', width=1, dash='dot'),
        legendgroup='kc', showlegend=False,
        hovertemplate='KC Lower: $%{y:,.2f}<extra></extra>'
    ), row=1, col=1)
    
    # BUY/SELL signals
    if signals:
        buy_signals = [s for s in signals if s.direction == 'long']
        sell_signals = [s for s in signals if s.direction == 'short']
        
        if buy_signals:
            fig.add_trace(go.Scatter(
                x=[s.timestamp for s in buy_signals],
                y=[s.price for s in buy_signals],
                mode='markers',
                name='BUY',
                marker=dict(symbol='triangle-up', size=10, color='#00FF00', 
                           line=dict(width=1, color='black')),
                hovertemplate='BUY @ $%{y:,.2f}<extra></extra>'
            ), row=1, col=1)
        
        if sell_signals:
            fig.add_trace(go.Scatter(
                x=[s.timestamp for s in sell_signals],
                y=[s.price for s in sell_signals],
                mode='markers',
                name='SELL',
                marker=dict(symbol='triangle-down', size=10, color='#FF0000',
                           line=dict(width=1, color='black')),
                hovertemplate='SELL @ $%{y:,.2f}<extra></extra>'
            ), row=1, col=1)
    
    # ========================================================================
    # ROW 2: Equity Curve
    # ========================================================================
    
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name='Equity',
        line=dict(color='#2ECC71', width=2),
        fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.1)',
        hovertemplate='Equity: $%{y:,.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Starting capital line
    fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                  opacity=0.5, row=2, col=1)
    
    # ========================================================================
    # ROW 3: RSI
    # ========================================================================
    
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['RSI'],
        name='RSI', line=dict(color='#8E44AD', width=1.5),
        hovertemplate='RSI: %{y:.1f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="red", 
                  opacity=0.5, row=3, col=1)
    fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="green", 
                  opacity=0.5, row=3, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="gray", 
                  opacity=0.3, row=3, col=1)
    
    # ========================================================================
    # ROW 4: Momentum
    # ========================================================================
    
    mom_colors = ['#27AE60' if x > 0 else '#E74C3C' 
                  for x in df_ind['Momentum_Norm'].fillna(0)]
    
    fig.add_trace(go.Bar(
        x=df_ind.index, y=df_ind['Momentum_Norm'],
        name='Momentum', marker_color=mom_colors, opacity=0.7,
        hovertemplate='Mom: %{y:.3f}<extra></extra>'
    ), row=4, col=1)
    
    fig.add_hline(y=0, line_color="black", row=4, col=1)
    
    # ========================================================================
    # ROW 5: ATR
    # ========================================================================
    
    fig.add_trace(go.Scatter(
        x=df_ind.index, y=df_ind['ATR'],
        name='ATR', line=dict(color='#F39C12', width=1.5),
        fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.2)',
        hovertemplate='ATR: $%{y:,.2f}<extra></extra>'
    ), row=5, col=1)
    
    # ========================================================================
    # Layout
    # ========================================================================
    
    fig.update_layout(
        title=dict(
            text=f'<b>{symbol} {timeframe}</b>',
            x=0.5,
            font=dict(size=18)
        ),
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.01, 
            xanchor="center", 
            x=0.5,
            font=dict(size=10)
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(t=80, b=40, l=60, r=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Equity", row=2, col=1, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100], gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="Mom", row=4, col=1, gridcolor='#E5E5E5')
    fig.update_yaxes(title_text="ATR", row=5, col=1, gridcolor='#E5E5E5')
    
    # X-axis
    fig.update_xaxes(gridcolor='#E5E5E5')
    
    # Save to HTML
    filename = f"indicators_{symbol.replace('/', '')}_{timeframe}_interactive.html"
    fig.write_html(filename)
    
    return filename


def calculate_equity_curve(df: pd.DataFrame, signals: List[Signal], 
                           initial_capital: float = 100000) -> pd.Series:
    """
    Calculate equity curve from signals.
    
    Simple simulation:
    - Enter at signal price with position sizing
    - Exit at next opposite signal or at stop/target
    """
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = initial_capital
    
    if not signals:
        # No signals, flat equity
        equity[:] = initial_capital
        return equity
    
    capital = initial_capital
    position = None  # {'direction': 'long'/'short', 'entry_price': x, 'size': x, 'stop': x, 'target': x}
    
    signal_dict = {s.timestamp: s for s in signals}
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        current_price = df.iloc[i]['close']
        prev_price = df.iloc[i-1]['close']
        
        # Check if we have a position
        if position is not None:
            # Check stop loss / take profit
            if position['direction'] == 'long':
                # Check stop
                if df.iloc[i]['low'] <= position['stop']:
                    # Stopped out
                    pnl = (position['stop'] - position['entry_price']) * position['shares']
                    capital += pnl
                    position = None
                # Check target
                elif df.iloc[i]['high'] >= position['target']:
                    # Target hit
                    pnl = (position['target'] - position['entry_price']) * position['shares']
                    capital += pnl
                    position = None
            else:  # short
                # Check stop
                if df.iloc[i]['high'] >= position['stop']:
                    pnl = (position['entry_price'] - position['stop']) * position['shares']
                    capital += pnl
                    position = None
                # Check target
                elif df.iloc[i]['low'] <= position['target']:
                    pnl = (position['entry_price'] - position['target']) * position['shares']
                    capital += pnl
                    position = None
        
        # Check for new signal
        if current_time in signal_dict and position is None:
            sig = signal_dict[current_time]
            
            # Position size (use BASE_POSITION of current capital)
            position_value = capital * BASE_POSITION
            shares = position_value / sig.price
            
            position = {
                'direction': sig.direction,
                'entry_price': sig.price,
                'shares': shares,
                'stop': sig.stop_loss,
                'target': sig.take_profit
            }
        
        # Calculate current equity (capital + unrealized P&L)
        if position is not None:
            if position['direction'] == 'long':
                unrealized = (current_price - position['entry_price']) * position['shares']
            else:
                unrealized = (position['entry_price'] - current_price) * position['shares']
            equity.iloc[i] = capital + unrealized
        else:
            equity.iloc[i] = capital
    
    # Forward fill any NaN values
    equity = equity.ffill().bfill()
    
    return equity


def plot_squeeze_histogram(df: pd.DataFrame, analyzer: BBSqueezeAnalyzer, 
                           symbol: str) -> plt.Figure:
    """Create histogram of squeeze durations."""
    df_ind = analyzer.calculate_indicators(df.copy())
    
    # Find squeeze periods
    squeeze_durations = []
    in_squeeze = False
    current_duration = 0
    
    for squeeze in df_ind['Squeeze']:
        if pd.isna(squeeze):
            continue
        if squeeze:
            in_squeeze = True
            current_duration += 1
        else:
            if in_squeeze and current_duration > 0:
                squeeze_durations.append(current_duration)
            in_squeeze = False
            current_duration = 0
    
    if in_squeeze and current_duration > 0:
        squeeze_durations.append(current_duration)
    
    if not squeeze_durations:
        print("No squeeze periods found!")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    bins = range(1, max(squeeze_durations) + 2)
    ax1.hist(squeeze_durations, bins=bins, color=Colors.SQUEEZE_BG, 
             edgecolor='black', alpha=0.8)
    ax1.axvline(x=MIN_SQUEEZE_BARS, color='red', linestyle='--', 
                linewidth=2, label=f'Min Squeeze ({MIN_SQUEEZE_BARS})')
    ax1.axvline(x=np.mean(squeeze_durations), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean ({np.mean(squeeze_durations):.1f})')
    
    ax1.set_xlabel('Squeeze Duration (bars)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'{symbol} Squeeze Duration Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    stats_text = (
        f"Total Squeezes: {len(squeeze_durations)}\n"
        f"Mean Duration: {np.mean(squeeze_durations):.1f} bars\n"
        f"Median: {np.median(squeeze_durations):.1f} bars\n"
        f"Max: {max(squeeze_durations)} bars\n"
        f"Min: {min(squeeze_durations)} bars\n"
        f"Std Dev: {np.std(squeeze_durations):.2f}"
    )
    
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             family='monospace')
    
    # Cumulative distribution
    sorted_durations = sorted(squeeze_durations)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    ax2.step(sorted_durations, cumulative, where='post', color='blue', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=MIN_SQUEEZE_BARS, color='red', linestyle='--', 
                linewidth=2, label=f'Min Squeeze ({MIN_SQUEEZE_BARS})')
    
    ax2.set_xlabel('Squeeze Duration (bars)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_signals(signals: List[Signal], symbol: str):
    """Print detected signals to console."""
    print(f"\n{'='*70}")
    print(f"DETECTED SIGNALS: {symbol}")
    print(f"{'='*70}")
    
    if not signals:
        print("No signals detected in this period.")
        return
    
    print(f"{'Timestamp':<20} {'Dir':<6} {'Price':>12} {'SL':>12} {'TP':>12} {'SQ':>4} {'Vol':>6} {'RSI':>5}")
    print("-" * 70)
    
    for s in signals:
        direction = "BUY" if s.direction == 'long' else "SELL"
        color = '\033[92m' if s.direction == 'long' else '\033[91m'
        reset = '\033[0m'
        print(f"{s.timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
              f"{color}{direction:<6}{reset} "
              f"${s.price:>11,.0f} "
              f"${s.stop_loss:>11,.0f} "
              f"${s.take_profit:>11,.0f} "
              f"{s.squeeze_bars:>4} "
              f"{s.volume_ratio:>5.2f}x "
              f"{s.rsi:>5.1f}")
    
    print("-" * 70)
    
    longs = len([s for s in signals if s.direction == 'long'])
    shorts = len([s for s in signals if s.direction == 'short'])
    print(f"Total: {len(signals)} signals | BUY: {longs} | SELL: {shorts}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Bollinger Bands Squeeze Indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_indicators.py                          # Default BTC/USD 4h
    python visualize_indicators.py --symbol ETH/USD         # Different symbol
    python visualize_indicators.py --timeframe 1h           # 1 hour timeframe
    python visualize_indicators.py --start 2024-06-01       # From date
    python visualize_indicators.py --last-n 500             # Last 500 bars
    python visualize_indicators.py --html                   # Interactive HTML (zoom/pan)
    python visualize_indicators.py --interactive            # Show matplotlib plot
    python visualize_indicators.py --histogram              # Squeeze duration stats
    python visualize_indicators.py --list                   # List cached data
        """
    )
    
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOLS[0],
                        help=f'Trading symbol (default: {DEFAULT_SYMBOLS[0]})')
    parser.add_argument('--timeframe', '-tf', type=str, default=SIGNAL_TIMEFRAME,
                        help=f'Timeframe (default: {SIGNAL_TIMEFRAME})')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS_BACK,
                        help=f'Days of cached data (default: {DEFAULT_DAYS_BACK})')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--last-n', type=int, default=None,
                        help='Show only last N bars')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Show interactive plot instead of saving')
    parser.add_argument('--histogram', action='store_true',
                        help='Show squeeze duration histogram')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available cached data')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--no-signals', action='store_true',
                        help='Hide buy/sell signals')
    parser.add_argument('--html', action='store_true',
                        help='Create interactive HTML chart (requires plotly)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("BOLLINGER BANDS SQUEEZE INDICATOR VISUALIZATION")
    print("=" * 70)
    
    # List available cache
    if args.list:
        print("\nAvailable cached data:")
        available = list_available_cache()
        if not available:
            print("  No cached data found. Run download_data.py first.")
        else:
            for symbol, timeframes in sorted(available.items()):
                print(f"  {symbol}: {', '.join(sorted(timeframes))}")
        print()
        return
    
    # Print configuration
    print(f"\nSymbol:    {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Cache:     {CACHE_DIR}")
    
    # Load data
    print(f"\nLoading data...")
    df = load_cached_data(args.symbol, args.timeframe, args.days)
    
    if df is None or df.empty:
        print(f"\nâœ— No data found for {args.symbol} {args.timeframe}")
        print("\nAvailable cached data:")
        available = list_available_cache()
        for symbol, timeframes in sorted(available.items()):
            print(f"  {symbol}: {', '.join(sorted(timeframes))}")
        print("\nRun download_data.py to download and cache data first.")
        return
    
    # Filter by date
    if args.start or args.end:
        print(f"Filtering: {args.start or 'start'} to {args.end or 'end'}")
        df = filter_by_date(df, args.start, args.end)
    
    # Limit to last N bars
    if args.last_n and len(df) > args.last_n:
        print(f"Showing last {args.last_n} bars")
        df = df.tail(args.last_n)
    
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    
    # Create analyzer
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
    
    # Detect signals
    signals = []
    if not args.no_signals and not args.histogram:
        print("\nDetecting signals...")
        signals = detect_signals(df, analyzer)
        print_signals(signals, args.symbol)
    
    # Generate plots
    if args.histogram:
        print("\nGenerating squeeze histogram...")
        fig = plot_squeeze_histogram(df, analyzer, args.symbol)
        if fig:
            filename = args.output or f"squeeze_histogram_{args.symbol.replace('/', '')}_{args.timeframe}.png"
        
        if fig is None:
            print("Failed to generate chart")
            return
        
        if args.interactive:
            print("\nShowing interactive plot...")
            plt.show()
        else:
            fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"\nâœ“ Chart saved: {filename}")
        
        plt.close(fig)
    
    elif args.html:
        # Interactive Plotly chart
        if not PLOTLY_AVAILABLE:
            print("\nâœ— Plotly not installed. Install with: pip install plotly --break-system-packages")
            return
        
        print("\nGenerating interactive HTML chart...")
        filename = create_interactive_chart(df, args.symbol, args.timeframe, analyzer, signals)
        
        if filename:
            print(f"\nâœ“ Interactive chart saved: {filename}")
            print("  Open in browser to zoom/pan/hover")
    
    else:
        # Static matplotlib chart
        print("\nGenerating indicator chart...")
        fig = create_indicator_chart(df, args.symbol, args.timeframe, analyzer, signals)
        filename = args.output or f"indicators_{args.symbol.replace('/', '')}_{args.timeframe}.png"
        
        if fig is None:
            print("Failed to generate chart")
            return
        
        if args.interactive:
            print("\nShowing interactive plot...")
            plt.show()
        else:
            fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"\nâœ“ Chart saved: {filename}")
        
        plt.close(fig)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()