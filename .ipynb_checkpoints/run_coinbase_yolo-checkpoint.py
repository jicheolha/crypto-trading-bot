#!/usr/bin/env python3
"""
Coinbase YOLO TRADER - Ultra Aggressive for Testing

⚠️ WARNING: Trades FREQUENTLY with LARGE positions
Only use with small amounts!
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_live_trader import CoinbaseLiveTrader

# API credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# YOLO Settings
SYMBOLS = ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SOL-USD']

# Ultra-fast timeframes
SIGNAL_TIMEFRAME = '1m'
ATR_TIMEFRAME = '1m'

# Looser squeeze
BB_PERIOD = 10
BB_STD = 1.5
KC_PERIOD = 10
KC_ATR_MULT = 1.0
MOMENTUM_PERIOD = 5

# No RSI filter
RSI_PERIOD = 14
RSI_OVERBOUGHT = 95
RSI_OVERSOLD = 5

MIN_SQUEEZE_BARS = 1

# No volume filter
VOLUME_PERIOD = 10
MIN_VOLUME_RATIO = 0.1

# Tight stops
ATR_PERIOD = 3
ATR_STOP_MULT = 20
ATR_TARGET_MULT = 25

# HUGE positions
BASE_POSITION = 0.2
MIN_POSITION = 0.1
MAX_POSITION = 0.5

# Max risk
MAX_POSITIONS = 4
MAX_DAILY_LOSS = 0.2
MAX_HOLD_DAYS = 0.02  # ~30 minutes

# Fast checking
CHECK_INTERVAL = 30
LOOKBACK_BARS = 50


def main():
    if not all([API_KEY, API_SECRET]):
        print("❌ Missing Coinbase API credentials!")
        return
    
    print("\n" + "="*60)
    print("🎰 COINBASE YOLO TRADER 🎰")
    print("="*60)
    print("⚠️  ULTRA AGGRESSIVE SETTINGS")
    print(f"Symbols: {SYMBOLS}")
    print(f"Timeframe: {SIGNAL_TIMEFRAME} (SCALPING)")
    print(f"Position Size: {BASE_POSITION:.0%} (HUGE)")
    print(f"Max Positions: {MAX_POSITIONS} (ALL AT ONCE)")
    print("="*60)
    print("\n🔥 WARNING: This WILL burn through your balance!\n")
    
    confirm = input("Type 'YOLO' to confirm: ")
    if confirm != 'YOLO':
        print("Aborted.")
        return
    
    bot = CoinbaseLiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbols=SYMBOLS,
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
        print("\n👋 YOLO stopped")
        bot.stop()


if __name__ == "__main__":
    main()