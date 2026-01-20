#!/usr/bin/env python3
"""
Crypto Trading Bot - Multi-Asset Live Trader

Trades multiple futures contracts simultaneously:
- BIP (BTC futures) - signals from BTC-USD
- SLP (SOL futures) - signals from SOL-USD
- XPP (XRP futures) - signals from XRP-USD
- DOP (DOGE futures) - signals from DOGE-USD

Risk Management:
- max_positions=2 (shared across all assets)
- max_position=50% per trade
- Can go LONG and SHORT on futures
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase_live_trader import CoinbaseLiveTrader

# =============================================================================
# API CREDENTIALS (from environment variables)
# =============================================================================
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# =============================================================================
# MULTI-ASSET CONFIGURATION
# =============================================================================

# Perpetual futures contracts to trade (from Coinbase International)
SYMBOLS = [
    'BIP-20DEC30-CDE',   # BTC futures (0.01 BTC, 10x leverage, ~$95 margin)
    'SLP-20DEC30-CDE',   # SOL futures (5 SOL, 5x leverage, ~$134 margin)
    'XPP-20DEC30-CDE',   # XRP futures (500 XRP, 5x leverage, ~$200 margin)
    'DOP-20DEC30-CDE',   # DOGE futures (5000 DOGE, 4x leverage, ~$162 margin)
]

# Spot markets for signals (backtested parameters work on spot data)
SIGNAL_SYMBOLS = [
    'BTC-USD',   # Signals for BIP
    'SOL-USD',   # Signals for SLP
    'XRP-USD',   # Signals for XPP
    'DOGE-USD',  # Signals for DOP
]

# =============================================================================
# OPTIMIZED STRATEGY PARAMETERS
# =============================================================================

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

# Stops (ATR multiples)
ATR_PERIOD = 16
ATR_STOP_MULT = 3.45
ATR_TARGET_MULT = 4.0

# Setup validity
SETUP_VALIDITY_BARS = 8

# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================

# Position sizing
BASE_POSITION = 0.50    # 50% base position size
MIN_POSITION = 0.30     # 30% minimum
MAX_POSITION = 0.50     # 50% maximum (as requested)

# Risk limits
MAX_POSITIONS = 2       # Max 2 positions across ALL assets
MAX_DAILY_LOSS = 0.03   # 3% max daily loss
MAX_HOLD_DAYS = 7       # Max 7 days hold time

# Direction
LONG_ONLY = False       # Can SHORT on futures!

# Bot settings
CHECK_INTERVAL = 60     # Check every 60 seconds
LOOKBACK_BARS = 300     # Historical bars to load


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not API_KEY or not API_SECRET:
        print("ERROR: Missing Coinbase API credentials!")
        print("")
        print("Set environment variables:")
        print("  export COINBASE_API_KEY='your_key'")
        print("  export COINBASE_API_SECRET='your_secret'")
        return
    
    print("")
    print("=" * 70)
    print("CRYPTO TRADING BOT - MULTI-ASSET LIVE TRADER")
    print("=" * 70)
    print("")
    
    # Check for --list-futures flag
    if len(sys.argv) > 1 and sys.argv[1] == '--list-futures':
        print("Listing available futures products...")
        print("")
        from coinbase.rest import RESTClient
        client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
        
        try:
            products = client.get_products()
            futures = []
            
            for product in products.products:
                product_type = getattr(product, 'product_type', '')
                if 'FUTURE' in product_type.upper():
                    product_id = getattr(product, 'product_id', '')
                    base = getattr(product, 'base_currency_id', '')
                    quote = getattr(product, 'quote_currency_id', '')
                    status = getattr(product, 'status', '')
                    futures.append((product_id, base, quote, status, product_type))
            
            print(f"Found {len(futures)} futures products:")
            print("-" * 70)
            print(f"{'PRODUCT_ID':<25} {'BASE':<8} {'QUOTE':<8} {'STATUS':<12} {'TYPE'}")
            print("-" * 70)
            for f in sorted(futures):
                print(f"{f[0]:<25} {f[1]:<8} {f[2]:<8} {f[3]:<12} {f[4]}")
            print("-" * 70)
            
        except Exception as e:
            print(f"Error: {e}")
        return
    
    print("TRADING CONFIGURATION:")
    print("-" * 70)
    print(f"  Futures:        {', '.join(SYMBOLS)}")
    print(f"  Signal Sources: {', '.join(SIGNAL_SYMBOLS)}")
    print(f"  Timeframes:     {SIGNAL_TIMEFRAME} signal | {ATR_TIMEFRAME} ATR")
    print("")
    print("RISK MANAGEMENT:")
    print("-" * 70)
    print(f"  Max Positions:  {MAX_POSITIONS} (shared across all assets)")
    print(f"  Position Size:  {BASE_POSITION:.0%} base, {MIN_POSITION:.0%}-{MAX_POSITION:.0%} range")
    print(f"  Max Daily Loss: {MAX_DAILY_LOSS:.0%}")
    print(f"  Max Hold Time:  {MAX_HOLD_DAYS} days")
    print(f"  Direction:      {'LONG only' if LONG_ONLY else 'LONG and SHORT'}")
    print("")
    print("=" * 70)
    print("")
    print("[!] WARNING: This will trade with REAL MONEY on your account!")
    print("")
    
    confirm = input("Type 'LIVE' to start live trading: ")
    if confirm != 'LIVE':
        print("Aborted. You must type 'LIVE' to confirm.")
        return
    
    print("")
    print("Starting live trader...")
    print("")
    
    bot = CoinbaseLiveTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        
        # Multi-asset configuration
        symbols=SYMBOLS,
        signal_symbols=SIGNAL_SYMBOLS,
        
        # Timeframes
        signal_timeframe=SIGNAL_TIMEFRAME,
        atr_timeframe=ATR_TIMEFRAME,
        
        # Strategy parameters
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
        
        # Position sizing
        base_position=BASE_POSITION,
        min_position=MIN_POSITION,
        max_position=MAX_POSITION,
        
        # Risk management
        max_positions=MAX_POSITIONS,
        max_daily_loss=MAX_DAILY_LOSS,
        max_hold_days=MAX_HOLD_DAYS,
        long_only=LONG_ONLY,
        
        # Setup
        setup_validity_bars=SETUP_VALIDITY_BARS,
        
        # Bot settings
        check_interval_seconds=CHECK_INTERVAL,
        lookback_bars=LOOKBACK_BARS,
    )
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("")
        print("Shutting down...")
        bot.stop()


if __name__ == "__main__":
    main()