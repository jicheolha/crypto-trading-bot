#!/usr/bin/env python3
"""
Coinbase Advanced Trade - Conservative Settings
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_live_trader import CoinbaseLiveTrader

# API credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# Symbols (use -USD on Coinbase)
SYMBOLS = ['BTC-USD', 'ETH-USD']

# Timeframes
SIGNAL_TIMEFRAME = '4h'
ATR_TIMEFRAME = '1h'

# Optimized params
BB_PERIOD = 19
BB_STD = 2.47
KC_PERIOD = 17
KC_ATR_MULT = 2.38
MOMENTUM_PERIOD = 15

RSI_PERIOD = 21
RSI_OVERBOUGHT = 68
RSI_OVERSOLD = 18

MIN_SQUEEZE_BARS = 2

VOLUME_PERIOD = 45
MIN_VOLUME_RATIO = 1.02

ATR_PERIOD = 16
ATR_STOP_MULT = 3.45
ATR_TARGET_MULT = 4.0

BASE_POSITION = 0.40
MIN_POSITION = 0.30
MAX_POSITION = 0.85

MAX_POSITIONS = 3
MAX_DAILY_LOSS = 0.03
MAX_HOLD_DAYS = 7

CHECK_INTERVAL = 60
LOOKBACK_BARS = 100


def main():
    if not all([API_KEY, API_SECRET]):
        print("❌ Missing Coinbase API credentials!")
        print("\nSet them with:")
        print("  export COINBASE_API_KEY='organizations/{org_id}/apiKeys/{key_id}'")
        print("  export COINBASE_API_SECRET='-----BEGIN EC PRIVATE KEY-----...")
        return
    
    print(f"\n{'='*60}")
    print("COINBASE LIVE TRADER")
    print(f"{'='*60}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Timeframes: {SIGNAL_TIMEFRAME} signal, {ATR_TIMEFRAME} ATR")
    print(f"Position: {BASE_POSITION:.0%} base, {MIN_POSITION:.0%}-{MAX_POSITION:.0%} range")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"{'='*60}\n")
    
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
        print("\n👋 Stopped by user")
        bot.stop()


if __name__ == "__main__":
    main()