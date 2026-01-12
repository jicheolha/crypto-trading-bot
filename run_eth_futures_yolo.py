#!/usr/bin/env python3
"""
ETH Futures YOLO Trader - TESTING ONLY
 Trades FAST with TIGHT stops - For functionality testing only!
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_live_trader import CoinbaseLiveTrader

# API credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# ETH FUTURES YOLO SETTINGS
TRADE_SYMBOLS = ['ETP-20DEC30-CDE']  # Trade on FUTURES
SIGNAL_SYMBOLS = ['ETH-USD']         # Get signals from SPOT (your backtested params!)

# ULTRA-FAST TIMEFRAMES (1min for rapid testing)
SIGNAL_TIMEFRAME = '1m'
ATR_TIMEFRAME = '1m'

# LOOSE SQUEEZE (generate signals quickly)
BB_PERIOD = 10
BB_STD = 1.5
KC_PERIOD = 10
KC_ATR_MULT = 1.0
MOMENTUM_PERIOD = 5

# NO RSI FILTER (let everything through)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 95  # Almost never blocks
RSI_OVERSOLD = 5     # Almost never blocks

# LOOSE SQUEEZE REQUIREMENTS
MIN_SQUEEZE_BARS = 1  # Accept any squeeze

# NO VOLUME FILTER
VOLUME_PERIOD = 10
MIN_VOLUME_RATIO = 0.1  # Accept almost any volume

# TIGHT STOPS (3x ATR for both)
ATR_PERIOD = 5
ATR_STOP_MULT = 5.0   # 3x ATR stop
ATR_TARGET_MULT = 6.0  # 3x ATR target (same as stop)

# ALL IN POSITION SIZING
BASE_POSITION = 0.95
MIN_POSITION = 0.90
MAX_POSITION = 1.00

# SETUP
SETUP_VALIDITY_BARS = 20  # Longer validity for 1m bars

# RISK
MAX_POSITIONS = 1
MAX_DAILY_LOSS = 0.50  # 50% daily loss (testing only!)
MAX_HOLD_DAYS = 0.01   # ~15 minutes max hold

# BOT
CHECK_INTERVAL = 10  # Check every 10 seconds
LOOKBACK_BARS = 100


def main():
    if not all([API_KEY, API_SECRET]):
        print("Missing Coinbase API credentials!")
        return
    
    print("\n" + "="*60)
    print("ETH FUTURES YOLO TRADER - TESTING ONLY")
    print("="*60)
    print("WARNING: TRADES VERY FAST")
    print(f"Signals: {SIGNAL_SYMBOLS[0]} SPOT")
    print(f"Trading:    {TRADE_SYMBOLS[0]} FUTURES 4x leverage")
    print(f"Timeframe: {SIGNAL_TIMEFRAME}")
    print(f"Position: {BASE_POSITION:.0%}-{MAX_POSITION:.0%}")
    print(f"Stops: {ATR_STOP_MULT}x ATR stop, {ATR_TARGET_MULT}x ATR target")
    print(f"Max hold: {MAX_HOLD_DAYS * 24 * 60:.0f} minutes")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    print("="*60)
    
    confirm = input("Type 'YOLO' to start testing: ")
    if confirm != 'YOLO':
        print("Aborted.")
        return
    
    bot = CoinbaseLiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbols=TRADE_SYMBOLS,           # Trade on futures
        signal_symbols=SIGNAL_SYMBOLS,   # Signals from spot
        signal_timeframe=SIGNAL_TIMEFRAME,
        atr_timeframe=ATR_TIMEFRAME,
        bb_period=BB_PERIOD,
        bb_std=BB_STD,
        kc_period=KC_PERIOD,
        kc_atr_mult=KC_ATR_MULT,
        momentum_period=MOMENTUM_PERIOD,
        rsi_period=RSI_PERIOD,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        min_squeeze_bars=MIN_SQUEEZE_BARS,
        volume_period=VOLUME_PERIOD,
        min_volume_ratio=MIN_VOLUME_RATIO,
        atr_period=ATR_PERIOD,
        atr_stop_mult=ATR_STOP_MULT,
        atr_target_mult=ATR_TARGET_MULT,
        base_position=BASE_POSITION,
        min_position=MIN_POSITION,
        max_position=MAX_POSITION,
        max_positions=MAX_POSITIONS,
        max_daily_loss=MAX_DAILY_LOSS,
        max_hold_days=MAX_HOLD_DAYS,
        check_interval_seconds=CHECK_INTERVAL,
        lookback_bars=LOOKBACK_BARS,
    )
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nYOLO stopped")
        bot.stop()


if __name__ == "__main__":
    main()