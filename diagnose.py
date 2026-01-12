#!/usr/bin/env python3
"""
Live Trading Diagnostic Script

Tests current market conditions and shows what needs to happen for trade entry.
Uses your actual Coinbase live trading configuration.
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from coinbase.rest import RESTClient
    from technical import BBSqueezeAnalyzer
    from signal_generator import BBSqueezeSignalGenerator
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install coinbase-advanced-py")
    sys.exit(1)


# ============================================================================
# YOUR LIVE TRADING PARAMETERS (from run_eth_futures_simple.py)
# ============================================================================

# Symbols
TRADE_SYMBOL = 'ETP-20DEC30-CDE'  # Futures
SIGNAL_SYMBOL = 'ETH-USD'          # Spot (for signals)

# Timeframes
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
BASE_POSITION = 0.75
MIN_POSITION = 0.65
MAX_POSITION = 0.90


# ============================================================================
# DATA FETCHING (using Coinbase API like your live bot)
# ============================================================================

def get_candles(client: RESTClient, symbol: str, timeframe: str, bars: int = 300) -> pd.DataFrame:
    """Fetch historical candles from Coinbase."""
    try:
        gran_map = {
            '1m': 'ONE_MINUTE',
            '5m': 'FIVE_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '30m': 'THIRTY_MINUTE',
            '1h': 'ONE_HOUR',
            '4h': 'FOUR_HOUR',
            '1d': 'ONE_DAY'
        }
        
        gran = gran_map.get(timeframe, 'ONE_HOUR')
        
        # Calculate time range
        if timeframe == '1m':
            hours = 5
        elif timeframe == '5m':
            hours = 24
        elif timeframe == '15m':
            hours = 3 * 24
        elif timeframe == '30m':
            hours = 7 * 24
        elif timeframe == '1h':
            hours = 12 * 24
        elif timeframe == '4h':
            hours = 50 * 24
        elif timeframe == '1d':
            hours = 300 * 24
        else:
            hours = 7 * 24
        
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        response = client.get_candles(
            product_id=symbol,
            start=start_ts,
            end=end_ts,
            granularity=gran
        )
        
        candles = response.candles if hasattr(response, 'candles') else []
        
        data = []
        for c in candles:
            data.append({
                'timestamp': int(c.start),
                'open': float(c.open),
                'high': float(c.high),
                'low': float(c.low),
                'close': float(c.close),
                'volume': float(c.volume)
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"ERROR fetching {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def diagnose():
    """Run complete diagnostics on current market state."""
    
    print("\n" + "="*70)
    print("ETH FUTURES BOT - DIAGNOSTIC REPORT")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trade Symbol: {TRADE_SYMBOL} (futures)")
    print(f"Signal Symbol: {SIGNAL_SYMBOL} (spot)")
    print(f"Signal TF: {SIGNAL_TIMEFRAME} | ATR TF: {ATR_TIMEFRAME}")
    print("="*70)
    
    # Initialize Coinbase client
    api_key = os.environ.get('COINBASE_API_KEY')
    api_secret = os.environ.get('COINBASE_API_SECRET')
    
    if not api_key or not api_secret:
        print("\nERROR: Coinbase API credentials not found!")
        print("Set them with:")
        print("  export COINBASE_API_KEY='your_key'")
        print("  export COINBASE_API_SECRET='your_secret'")
        return
    
    client = RESTClient(api_key=api_key, api_secret=api_secret)
    
    # Get current price
    print(f"\n--- CURRENT MARKET DATA ---")
    try:
        product = client.get_product(product_id=SIGNAL_SYMBOL)
        current_price = float(product.price) if hasattr(product, 'price') else 0
        print(f"ETH Price (spot): ${current_price:,.2f}")
    except Exception as e:
        print(f"Could not get current price: {e}")
        current_price = 0
    
    # Fetch data
    print(f"\n--- FETCHING DATA ---")
    print(f"Downloading {SIGNAL_TIMEFRAME} candles for {SIGNAL_SYMBOL}...")
    signal_df = get_candles(client, SIGNAL_SYMBOL, SIGNAL_TIMEFRAME)
    
    print(f"Downloading {ATR_TIMEFRAME} candles for {SIGNAL_SYMBOL}...")
    atr_df = get_candles(client, SIGNAL_SYMBOL, ATR_TIMEFRAME)
    
    if signal_df.empty:
        print("ERROR: No signal data fetched!")
        return
    
    print(f"Got {len(signal_df)} signal bars, {len(atr_df)} ATR bars")
    print(f"Signal data range: {signal_df.index[0]} to {signal_df.index[-1]}")
    
    # Initialize analyzer
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
    
    # Calculate indicators
    print(f"\n--- CALCULATING INDICATORS ---")
    df = analyzer.calculate_indicators(signal_df.tail(100))
    
    if len(df) < 2:
        print("ERROR: Not enough data for indicators")
        return
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # ========================================================================
    # VOLUME ANALYSIS - THE CRITICAL ISSUE
    # ========================================================================
    print(f"\n" + "="*70)
    print("VOLUME ANALYSIS (CRITICAL)")
    print("="*70)
    
    print(f"Volume Period: {VOLUME_PERIOD} bars")
    print(f"Latest bar volume: {current['volume']:,.0f}")
    
    # Check if volume data exists
    if 'Volume_MA' in df.columns:
        print(f"Volume MA: {current['Volume_MA']:,.0f}")
        
        if pd.isna(current['Volume_MA']) or current['Volume_MA'] == 0:
            print("\nWARNING: Volume_MA is invalid!")
            print(f"  Value: {current['Volume_MA']}")
            print("  Possible causes:")
            print(f"    1. Not enough bars (need {VOLUME_PERIOD}, have {len(df)})")
            print("    2. All volume values are zero")
            
            # Show recent volumes
            print(f"\n  Last 10 volume values:")
            for i in range(min(10, len(df))):
                idx = -(i+1)
                vol = df['volume'].iloc[idx]
                ts = df.index[idx]
                print(f"    {ts}: {vol:,.0f}")
    else:
        print("ERROR: Volume_MA column not found!")
    
    # Volume Ratio
    if 'Volume_Ratio' in df.columns:
        vol_ratio = current['Volume_Ratio']
        print(f"\nVolume Ratio: {vol_ratio:.2f}")
        print(f"Minimum required: {MIN_VOLUME_RATIO:.2f}")
        
        if vol_ratio == 0 or pd.isna(vol_ratio):
            print("\n*** BLOCKING ISSUE: Volume_Ratio = 0.00! ***")
            print("This will BLOCK ALL TRADES even when squeeze releases!")
            print("\nTo fix:")
            print("  1. Check if Coinbase provides volume data for ETH-USD")
            print("  2. Try lowering MIN_VOLUME_RATIO to 0.01 for testing")
            print("  3. Or disable volume filter temporarily")
        elif vol_ratio < MIN_VOLUME_RATIO:
            print(f"\nVolume too low: {vol_ratio:.2f} < {MIN_VOLUME_RATIO:.2f}")
            print("Trade will be blocked by volume filter")
        else:
            print(f"\nVolume OK: {vol_ratio:.2f} >= {MIN_VOLUME_RATIO:.2f}")
    else:
        print("ERROR: Volume_Ratio column not found!")
    
    # ========================================================================
    # SQUEEZE STATE
    # ========================================================================
    print(f"\n" + "="*70)
    print("SQUEEZE STATE")
    print("="*70)
    
    squeeze_active = current['Squeeze']
    squeeze_duration = current['Squeeze_Duration']
    
    print(f"State: {'IN SQUEEZE' if squeeze_active else 'NOT IN SQUEEZE'}")
    
    if squeeze_active:
        print(f"Duration: {squeeze_duration:.0f} bars ({squeeze_duration * 4:.0f} hours)")
        print(f"Minimum required: {MIN_SQUEEZE_BARS} bars")
        
        if squeeze_duration >= MIN_SQUEEZE_BARS:
            print(f"Duration OK: {squeeze_duration:.0f} >= {MIN_SQUEEZE_BARS}")
        else:
            print(f"Duration too short: {squeeze_duration:.0f} < {MIN_SQUEEZE_BARS}")
    
    print(f"\nBB Width: {current['BB_Width']:.4f}")
    print(f"Momentum: {current['Momentum_Norm']:+.2f}")
    print(f"RSI: {current['RSI']:.1f}")
    
    # Check if squeeze released
    squeeze_released = prev['Squeeze'] and not current['Squeeze']
    
    if squeeze_released:
        print(f"\n*** SQUEEZE RELEASED ON CURRENT BAR! ***")
        print(f"  Previous bar: IN squeeze")
        print(f"  Current bar: NOT in squeeze")
        print(f"  This is a BREAKOUT candidate!")
    else:
        if squeeze_active:
            print(f"\nWaiting for squeeze release...")
            print(f"  Squeeze has lasted {squeeze_duration:.0f} bars")
            print(f"  Will trigger when BB expands outside KC")
        else:
            print(f"\nNo active squeeze")
            print(f"  Need new squeeze to form")
    
    # ========================================================================
    # BREAKOUT DETECTION
    # ========================================================================
    print(f"\n" + "="*70)
    print("BREAKOUT DETECTION")
    print("="*70)
    
    breakout = analyzer.detect_breakout(
        df,
        min_squeeze_bars=MIN_SQUEEZE_BARS,
        min_volume_ratio=MIN_VOLUME_RATIO
    )
    
    if breakout:
        print("*** BREAKOUT DETECTED! ***")
        print(f"  Direction: {breakout['direction'].upper()}")
        print(f"  Price: ${breakout['price']:,.2f}")
        print(f"  Squeeze bars: {breakout['squeeze_bars']:.0f}")
        print(f"  Volume ratio: {breakout['volume_ratio']:.2f}")
        print(f"  Momentum: {breakout['momentum']:.2f}")
        print(f"  RSI: {breakout['rsi']:.1f}")
        print(f"  ATR: ${breakout['atr']:.2f}")
    else:
        print("No breakout detected")
        
        # Explain why
        reasons = []
        
        if not squeeze_released:
            reasons.append("Squeeze has not released")
        
        if squeeze_released and prev['Squeeze_Duration'] < MIN_SQUEEZE_BARS:
            reasons.append(f"Squeeze too short ({prev['Squeeze_Duration']:.0f} < {MIN_SQUEEZE_BARS})")
        
        if current['Volume_Ratio'] < MIN_VOLUME_RATIO:
            reasons.append(f"Volume too low ({current['Volume_Ratio']:.2f} < {MIN_VOLUME_RATIO})")
        
        if reasons:
            print("\nBlocked by:")
            for reason in reasons:
                print(f"  - {reason}")
    
    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================
    print(f"\n" + "="*70)
    print("SIGNAL GENERATION")
    print("="*70)
    
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
        signal_timeframe_minutes=240  # 4h
    )
    
    signal_gen.set_signal_data({SIGNAL_SYMBOL: signal_df})
    signal_gen.set_atr_data({SIGNAL_SYMBOL: atr_df})
    
    current_time = datetime.now(pytz.UTC)
    signal = signal_gen.generate_signal(signal_df, SIGNAL_SYMBOL, current_time)
    
    print(f"Direction: {signal.direction.upper()}")
    
    if signal.direction != 'neutral':
        print(f"\n*** SIGNAL GENERATED! ***")
        print(f"  Entry: ${signal.entry_price:,.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${signal.take_profit:,.2f}")
        print(f"  Position Size: {signal.position_size:.0%}")
        print(f"  Reasons: {' '.join(signal.reasons)}")
    else:
        print("No signal generated")
    
    # ========================================================================
    # WHAT NEEDS TO HAPPEN
    # ========================================================================
    print(f"\n" + "="*70)
    print("WHAT NEEDS TO HAPPEN FOR TRADE ENTRY")
    print("="*70)
    
    checklist = []
    
    # 1. Squeeze release
    if squeeze_active:
        checklist.append(("Squeeze must RELEASE", False, 
                         f"Currently in squeeze ({squeeze_duration:.0f} bars). "
                         f"Will release when BB expands outside KC."))
    else:
        checklist.append(("Squeeze released", True, 
                         "Not currently in squeeze. Need new squeeze to form."))
    
    # 2. Duration
    if squeeze_active and squeeze_duration >= MIN_SQUEEZE_BARS:
        checklist.append(("Duration >= 2 bars", True, 
                         f"Duration: {squeeze_duration:.0f} bars"))
    elif squeeze_active:
        checklist.append(("Duration >= 2 bars", False, 
                         f"Currently {squeeze_duration:.0f} bars, need {MIN_SQUEEZE_BARS}"))
    else:
        checklist.append(("Duration >= 2 bars", None, 
                         "No active squeeze"))
    
    # 3. Volume
    vol_ratio = current['Volume_Ratio']
    if vol_ratio >= MIN_VOLUME_RATIO:
        checklist.append(("Volume >= 1.02", True, 
                         f"Current: {vol_ratio:.2f}"))
    elif vol_ratio == 0:
        checklist.append(("Volume >= 1.02", False, 
                         f"PROBLEM: Volume ratio = 0.00 (will block all trades!)"))
    else:
        checklist.append(("Volume >= 1.02", False, 
                         f"Current: {vol_ratio:.2f}, need {MIN_VOLUME_RATIO:.2f}"))
    
    # 4. RSI
    rsi = current['RSI']
    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        checklist.append(("RSI in range", True, 
                         f"RSI: {rsi:.1f} (not overbought/oversold)"))
    elif rsi >= RSI_OVERBOUGHT:
        checklist.append(("RSI in range", False, 
                         f"RSI: {rsi:.1f} >= {RSI_OVERBOUGHT} (blocks LONG)"))
    else:
        checklist.append(("RSI in range", False, 
                         f"RSI: {rsi:.1f} <= {RSI_OVERSOLD} (blocks SHORT)"))
    
    # Print checklist
    for item, status, detail in checklist:
        if status is True:
            mark = "[OK]"
        elif status is False:
            mark = "[X]"
        else:
            mark = "[-]"
        
        print(f"\n{mark} {item}")
        print(f"     {detail}")
    
    # Timeline estimate
    print(f"\n" + "="*70)
    print("TIMELINE ESTIMATE")
    print("="*70)
    
    if squeeze_active:
        print(f"\nCurrent squeeze: {squeeze_duration:.0f} bars = {squeeze_duration * 4:.0f} hours")
        print(f"Average squeeze: 3-7 bars = 12-28 hours")
        print(f"Next candle: {(df.index[-1] + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"\nEstimate: Trade entry in 4 hours to several days")
        print("(When squeeze releases + volume confirms)")
    else:
        print("\nNo active squeeze")
        print("Waiting for new squeeze to form (unpredictable timing)")
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        diagnose()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()