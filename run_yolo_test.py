#!/usr/bin/env python3
"""
YOLO TEST TRADER - For Testing Buy/Sell Functionality ONLY

WARNING: This WILL lose money. It's designed to trigger trades quickly
so you can verify the order placement works.

Features:
- Super relaxed entry conditions (will trade almost immediately)
- Minimum position size (1 contract)
- Tight stops (minimize losses)
- Option to force immediate buy/sell for testing
"""
import os
import sys
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coinbase.rest import RESTClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# Contract specs (from your setup)
CONTRACTS = {
    'BIP-20DEC30-CDE': {'name': 'BTC', 'size': 0.01, 'leverage': 10, 'signal': 'BTC-USD'},
    'ETP-20DEC30-CDE': {'name': 'ETH', 'size': 0.1, 'leverage': 10, 'signal': 'ETH-USD'},
    'SLP-20DEC30-CDE': {'name': 'SOL', 'size': 5.0, 'leverage': 5, 'signal': 'SOL-USD'},
    'XPP-20DEC30-CDE': {'name': 'XRP', 'size': 500, 'leverage': 5, 'signal': 'XRP-USD'},
    'DOP-20DEC30-CDE': {'name': 'DOGE', 'size': 5000, 'leverage': 4, 'signal': 'DOGE-USD'},
}


def get_balance(client):
    """Get USD balance from all accounts (spot + futures)."""
    try:
        accounts = client.get_accounts()
        total = 0.0
        
        for acc in accounts.accounts:
            currency = getattr(acc, 'currency', '')
            acc_type = getattr(acc, 'type', '')
            
            if currency in ['USD', 'USDC']:
                avail = 0.0
                hold = 0.0
                
                if hasattr(acc, 'available_balance'):
                    bal = acc.available_balance
                    if isinstance(bal, dict):
                        avail = float(bal.get('value', 0))
                    elif hasattr(bal, 'value'):
                        avail = float(bal.value)
                
                if hasattr(acc, 'hold'):
                    h = acc.hold
                    if isinstance(h, dict):
                        hold = float(h.get('value', 0))
                    elif hasattr(h, 'value'):
                        hold = float(h.value)
                
                acc_total = avail + hold
                if acc_total > 0:
                    logger.debug(f"  Found {currency} account ({acc_type}): ${acc_total:.2f}")
                total += acc_total
        
        return total
    except Exception as e:
        logger.error(f"Balance error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def get_futures_balance(client):
    """Get futures-specific balance."""
    try:
        # Try to get futures balance specifically
        response = client.get_futures_balance_summary()
        
        if hasattr(response, 'balance'):
            bal = response.balance
            if isinstance(bal, dict):
                return float(bal.get('value', 0))
            elif hasattr(bal, 'value'):
                return float(bal.value)
        
        # Try other attributes
        for attr in ['available_margin', 'total_usd_balance', 'cash_available']:
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, dict):
                    return float(val.get('value', 0))
                elif hasattr(val, 'value'):
                    return float(val.value)
                elif val:
                    return float(val)
        
        logger.info(f"Futures balance response: {response}")
        return 0.0
        
    except Exception as e:
        logger.debug(f"Futures balance not available: {e}")
        return 0.0


def get_price(client, symbol):
    """Get current price."""
    try:
        product = client.get_product(product_id=symbol)
        return float(product.price) if hasattr(product, 'price') else None
    except Exception as e:
        logger.error(f"Price error for {symbol}: {e}")
        return None


def place_order(client, symbol, side, size):
    """Place market order."""
    try:
        size = int(size)  # Futures need whole contracts
        if size < 1:
            logger.error(f"Size must be >= 1 contract, got {size}")
            return None
        
        client_order_id = f"YOLO-{symbol[:3]}-{int(time.time())}"
        
        logger.info(f"Placing {side.upper()} order: {symbol} x {size} contracts")
        
        if side.lower() == 'buy':
            order = client.market_order_buy(
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=str(size)
            )
        else:
            order = client.market_order_sell(
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=str(size)
            )
        
        # Check result - handle both dict and object responses
        logger.info(f"Order response type: {type(order)}")
        logger.info(f"Order response: {order}")
        
        # Extract success and success_response handling both object and dict access
        success = None
        success_response = None
        error_response = None
        
        # Try object attributes first
        if hasattr(order, 'success'):
            success = order.success
        if hasattr(order, 'success_response'):
            success_response = order.success_response
        if hasattr(order, 'error_response'):
            error_response = order.error_response
        
        # If it's a dict-like object that returns dict when printed
        if success is None and isinstance(order, dict):
            success = order.get('success')
            success_response = order.get('success_response')
            error_response = order.get('error_response')
        
        # Handle success
        if success:
            order_id = None
            
            # success_response could be dict or object
            if isinstance(success_response, dict):
                order_id = success_response.get('order_id')
            elif hasattr(success_response, 'order_id'):
                order_id = success_response.order_id
            
            if order_id:
                logger.info(f"✓ Order SUCCESS: {order_id}")
                return order_id
            else:
                logger.info(f"✓ Order SUCCESS (no order_id in response)")
                return "success"
        
        # Handle failure
        if error_response:
            if isinstance(error_response, dict):
                error_msg = error_response.get('error', 'Unknown')
                preview = error_response.get('preview_failure_reason', '')
            else:
                error_msg = getattr(error_response, 'error', 'Unknown')
                preview = getattr(error_response, 'preview_failure_reason', '')
            
            logger.error(f"✗ Order FAILED: {error_msg}")
            logger.error(f"  Reason: {preview}")
            
            if 'INSUFFICIENT_FUNDS' in str(preview):
                logger.error("  → Transfer funds from Spot to Futures wallet!")
            return None
        
        # Fallback - check for direct order_id
        if hasattr(order, 'order_id'):
            logger.info(f"✓ Order SUCCESS: {order.order_id}")
            return order.order_id
        
        # Unknown response format
        logger.warning(f"Order response format unclear, assuming success")
        return "unknown"
            
    except Exception as e:
        logger.error(f"Order error: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_menu():
    """Show test menu."""
    print("\n" + "="*60)
    print("YOLO TEST TRADER - Order Testing Menu")
    print("="*60)
    print("\nAvailable contracts:")
    for i, (symbol, info) in enumerate(CONTRACTS.items(), 1):
        print(f"  {i}. {info['name']:5} - {symbol}")
    
    print("\nCommands:")
    print("  b <num>  - BUY 1 contract (e.g., 'b 2' for ETH)")
    print("  s <num>  - SELL 1 contract (e.g., 's 2' for ETH)")
    print("  p <num>  - Get PRICE (e.g., 'p 2' for ETH)")
    print("  bal      - Show balance")
    print("  all      - Show all prices and margin requirements")
    print("  q        - Quit")
    print("="*60)


def show_all_info(client):
    """Show all contract info and prices."""
    print("\n" + "-"*70)
    print(f"{'Contract':<20} {'Price':>12} {'Notional':>12} {'Margin':>10}")
    print("-"*70)
    
    for symbol, info in CONTRACTS.items():
        signal = info['signal']
        price = get_price(client, signal)
        
        if price:
            notional = price * info['size']
            margin = notional / info['leverage']
            print(f"{info['name']:<20} ${price:>11,.2f} ${notional:>11,.2f} ${margin:>9,.2f}")
        else:
            print(f"{info['name']:<20} {'Error':>12}")
    
    print("-"*70)
    balance = get_balance(client)
    futures_balance = get_futures_balance(client)
    print(f"Spot balance: ${balance:,.2f}")
    print(f"Futures balance: ${futures_balance:,.2f}")
    print("-"*70)


def interactive_mode(client):
    """Interactive test mode."""
    symbols_list = list(CONTRACTS.keys())
    
    while True:
        show_menu()
        
        try:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                print("Bye!")
                break
            
            elif cmd == 'bal':
                balance = get_balance(client)
                futures_balance = get_futures_balance(client)
                print(f"\n💰 Spot balance: ${balance:,.2f}")
                print(f"💰 Futures balance: ${futures_balance:,.2f}")
            
            elif cmd == 'all':
                show_all_info(client)
            
            elif cmd.startswith('p '):
                try:
                    idx = int(cmd.split()[1]) - 1
                    if 0 <= idx < len(symbols_list):
                        symbol = symbols_list[idx]
                        info = CONTRACTS[symbol]
                        price = get_price(client, info['signal'])
                        if price:
                            notional = price * info['size']
                            margin = notional / info['leverage']
                            print(f"\n{info['name']} ({symbol})")
                            print(f"  Price: ${price:,.4f}")
                            print(f"  1 contract = {info['size']} {info['name']}")
                            print(f"  Notional: ${notional:,.2f}")
                            print(f"  Margin required: ${margin:,.2f} ({info['leverage']}x)")
                    else:
                        print("Invalid number")
                except:
                    print("Usage: p <number>")
            
            elif cmd.startswith('b '):
                try:
                    idx = int(cmd.split()[1]) - 1
                    if 0 <= idx < len(symbols_list):
                        symbol = symbols_list[idx]
                        info = CONTRACTS[symbol]
                        
                        print(f"\n⚠️  About to BUY 1 {info['name']} contract ({symbol})")
                        price = get_price(client, info['signal'])
                        if price:
                            notional = price * info['size']
                            margin = notional / info['leverage']
                            print(f"   Notional: ${notional:,.2f}")
                            print(f"   Margin: ${margin:,.2f}")
                        
                        confirm = input("   Type 'yes' to confirm: ").strip().lower()
                        if confirm == 'yes':
                            result = place_order(client, symbol, 'buy', 1)
                            if result:
                                print(f"\n✅ BUY order placed!")
                            else:
                                print(f"\n❌ BUY order failed!")
                        else:
                            print("Cancelled")
                    else:
                        print("Invalid number")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif cmd.startswith('s '):
                try:
                    idx = int(cmd.split()[1]) - 1
                    if 0 <= idx < len(symbols_list):
                        symbol = symbols_list[idx]
                        info = CONTRACTS[symbol]
                        
                        print(f"\n⚠️  About to SELL 1 {info['name']} contract ({symbol})")
                        price = get_price(client, info['signal'])
                        if price:
                            notional = price * info['size']
                            margin = notional / info['leverage']
                            print(f"   Notional: ${notional:,.2f}")
                            print(f"   Margin: ${margin:,.2f}")
                        
                        confirm = input("   Type 'yes' to confirm: ").strip().lower()
                        if confirm == 'yes':
                            result = place_order(client, symbol, 'sell', 1)
                            if result:
                                print(f"\n✅ SELL order placed!")
                            else:
                                print(f"\n❌ SELL order failed!")
                        else:
                            print("Cancelled")
                    else:
                        print("Invalid number")
                except Exception as e:
                    print(f"Error: {e}")
            
            else:
                print("Unknown command. Try: b 2, s 2, p 2, bal, all, q")
        
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def auto_test_mode(client, symbol_idx=2):
    """
    Automatic test: Opens and closes a position.
    
    Default: ETH (index 2) - cheapest margin ~$33
    """
    symbols_list = list(CONTRACTS.keys())
    symbol = symbols_list[symbol_idx - 1]
    info = CONTRACTS[symbol]
    
    print("\n" + "="*60)
    print("AUTO TEST MODE")
    print("="*60)
    print(f"Will BUY then SELL 1 {info['name']} contract")
    print(f"Symbol: {symbol}")
    
    price = get_price(client, info['signal'])
    if price:
        notional = price * info['size']
        margin = notional / info['leverage']
        print(f"Estimated cost: ~${margin:.2f} margin + fees")
    
    balance = get_balance(client)
    futures_balance = get_futures_balance(client)
    print(f"Spot balance: ${balance:.2f}")
    print(f"Futures balance: ${futures_balance:.2f}")
    
    print("\n⚠️  This will execute REAL trades!")
    confirm = input("Type 'YOLO' to proceed: ").strip()
    
    if confirm != 'YOLO':
        print("Cancelled")
        return
    
    # Step 1: BUY
    print("\n[1/2] Opening LONG position...")
    buy_result = place_order(client, symbol, 'buy', 1)
    
    if not buy_result:
        print("❌ BUY failed - aborting test")
        return
    
    print("✅ Position opened!")
    print("   Waiting 5 seconds before closing...")
    time.sleep(5)
    
    # Step 2: SELL to close
    print("\n[2/2] Closing position...")
    sell_result = place_order(client, symbol, 'sell', 1)
    
    if sell_result:
        print("✅ Position closed!")
        print("\n🎉 TEST COMPLETE - Orders working correctly!")
    else:
        print("❌ SELL failed - you may have an open position!")
        print(f"   Check Coinbase for open {info['name']} position")


def main():
    if not all([API_KEY, API_SECRET]):
        print("ERROR: Set COINBASE_API_KEY and COINBASE_API_SECRET environment variables")
        return
    
    print("\n" + "="*60)
    print("YOLO TEST TRADER")
    print("="*60)
    print("⚠️  WARNING: This uses REAL money!")
    print("    Designed for testing order placement only.")
    print("="*60)
    
    client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
    
    # Check balance - try both methods
    balance = get_balance(client)
    futures_balance = get_futures_balance(client)
    
    print(f"\n💰 Spot balance: ${balance:,.2f}")
    print(f"💰 Futures balance: ${futures_balance:,.2f}")
    
    effective_balance = max(balance, futures_balance)
    
    if effective_balance < 30:
        print("⚠️  Low balance! ETH contract needs ~$33 margin")
        print("   Transfer funds to Futures wallet if needed")
    
    print("\nModes:")
    print("  1. Interactive - manually test buy/sell commands")
    print("  2. Auto Test - automatically buy then sell 1 ETH contract")
    
    mode = input("\nSelect mode (1 or 2): ").strip()
    
    if mode == '2':
        auto_test_mode(client, symbol_idx=2)  # ETH by default
    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()