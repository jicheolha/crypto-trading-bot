#!/usr/bin/env python3
"""
ETH Perpetual Futures Trader - OPTIMAL PARAMETERS

Uses your backtested optimal parameters with adjusted position sizing
for current $99.50 balance.

Position Sizing Strategy:
- Current: 75% base (1 contract)
- When balance reaches $150+: Lower to 50%
- When balance reaches $300+: Lower to 30%
- When balance reaches $750+: Lower to 20% (optimal)

Risk Management:
- MAX_POSITIONS starts at 1
- Increase to 2 when balance > $150
- Increase to 3 when balance > $225
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_live_trader import CoinbaseLiveTrader

# API credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# ETH FUTURES SETTINGS
SYMBOLS = ['ETP-20DEC30-CDE']  # Trade on FUTURES
SIGNAL_SYMBOLS = ['ETH-USD']    # Get signals from SPOT (backtested params)  # ETH Futures (0.1 ETH per contract)

# YOUR OPTIMIZED PARAMETERS
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

# Position Sizing - ADJUSTED FOR CURRENT BALANCE
# (Will use 20% when balance grows to $375+)
BASE_POSITION = 0.75  # 75% = $74.62 â†’ Can afford 1 contract
MIN_POSITION = 0.65   # 65% minimum
MAX_POSITION = 0.90   # 90% maximum

# Setup
SETUP_VALIDITY_BARS = 8

# Risk - OPTIMAL (From backtesting)
MAX_POSITIONS = 1         # 1 position for now (increase as balance grows)
MAX_DAILY_LOSS = 0.03     # 3% max daily loss
MAX_HOLD_DAYS = 7         # Max 7 days hold time
LONG_ONLY = False         # Can short on futures!

# Bot
CHECK_INTERVAL = 60
LOOKBACK_BARS = 300


def main():
    if not all([API_KEY, API_SECRET]):
        print("Missing Coinbase API credentials!")
        return
    
    print("\n" + "="*60)
    print("ETH PERPETUAL FUTURES TRADER")
    print("="*60)
    print(f"Signals: {SIGNAL_SYMBOLS[0]} SPOT")
    print(f"Trading:    {SYMBOLS[0]} FUTURES")
    print(f"Timeframes: {SIGNAL_TIMEFRAME} signal | {ATR_TIMEFRAME} ATR")
    print(f"Position: {BASE_POSITION:.0%} base, {MIN_POSITION:.0%}-{MAX_POSITION:.0%} range")
    print(f"Max positions: {MAX_POSITIONS}")
    print(f"Max daily loss: {MAX_DAILY_LOSS:.0%}")
    print(f"Can SHORT: Yes")
    print(f"Max hold: {MAX_HOLD_DAYS} days")
    print("="*60)
    
    confirm = input("Type 'START' to begin: ")
    if confirm != 'START':
        print("Aborted.")
        return
    
    bot = CoinbaseLiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbols=SYMBOLS,                  # Trade on futures
        signal_symbols=SIGNAL_SYMBOLS,    # Signals from spot
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
        print("\nBot stopped")
        bot.stop()


if __name__ == "__main__":
    main()