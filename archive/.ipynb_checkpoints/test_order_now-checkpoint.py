#!/usr/bin/env python3
"""
MINIMAL ORDER TEST - Find out why orders are failing

Run this locally:
    export COINBASE_API_KEY='your_key'
    export COINBASE_API_SECRET=$'-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----'
    python test_order_now.py
"""
import os
import sys
import time

# Get credentials
API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

if not API_KEY or not API_SECRET:
    print("ERROR: Set COINBASE_API_KEY and COINBASE_API_SECRET")
    print("")
    print("Example:")
    print("  export COINBASE_API_KEY='your_key'")
    print("  export COINBASE_API_SECRET=$'-----BEGIN EC PRIVATE KEY-----\\nMHcC...\\n-----END EC PRIVATE KEY-----'")
    sys.exit(1)

from coinbase.rest import RESTClient

client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

# Test with BTC futures - smallest margin requirement
SYMBOL = 'BIP-20DEC30-CDE'  # BTC perp, 0.01 BTC/contract, ~$87 margin at 10x

print("="*60)
print("COINBASE ORDER TEST")
print("="*60)

# 1. Check balance
print("\n[1] Checking futures balance...")
try:
    response = client.get_futures_balance_summary()
    print(f"    Raw response: {response}")
    
    # Try to extract balance
    if hasattr(response, 'balance_summary'):
        summary = response.balance_summary
        print(f"    Balance summary: {summary}")
        
        # Try different fields
        for field in ['futures_buying_power', 'available_margin', 'total_usd_balance', 'cash_available_to_withdraw']:
            if hasattr(summary, field):
                val = getattr(summary, field)
                print(f"    {field}: {val}")
except Exception as e:
    print(f"    ERROR: {e}")

# 2. Check product info
print(f"\n[2] Checking product {SYMBOL}...")
try:
    product = client.get_product(product_id=SYMBOL)
    print(f"    Product type: {getattr(product, 'product_type', 'unknown')}")
    print(f"    Price: ${float(product.price):,.2f}")
    print(f"    Status: {getattr(product, 'status', 'unknown')}")
    print(f"    Trading disabled: {getattr(product, 'trading_disabled', 'unknown')}")
except Exception as e:
    print(f"    ERROR: {e}")

# 3. Try to place order
print(f"\n[3] Attempting to place BUY order for 1 contract...")
print("    (This will open a real position if successful!)")

input("    Press ENTER to continue or Ctrl+C to cancel...")

try:
    client_order_id = f"TEST-{int(time.time())}"
    
    print(f"\n    Calling market_order_buy...")
    print(f"    client_order_id: {client_order_id}")
    print(f"    product_id: {SYMBOL}")
    print(f"    base_size: '1'")
    
    order = client.market_order_buy(
        client_order_id=client_order_id,
        product_id=SYMBOL,
        base_size='1'
    )
    
    print(f"\n    === FULL RESPONSE ===")
    print(f"    Type: {type(order)}")
    print(f"    Value: {order}")
    
    if hasattr(order, '__dict__'):
        print(f"    __dict__: {order.__dict__}")
    
    # Check success
    if hasattr(order, 'success'):
        print(f"\n    success: {order.success}")
        
        if order.success:
            print(f"    SUCCESS! Order placed!")
            if hasattr(order, 'success_response'):
                sr = order.success_response
                print(f"    success_response: {sr}")
                if hasattr(sr, 'order_id'):
                    print(f"    order_id: {sr.order_id}")
        else:
            print(f"    FAILED!")
            if hasattr(order, 'error_response'):
                er = order.error_response
                print(f"    error_response: {er}")
                if hasattr(er, '__dict__'):
                    print(f"    error __dict__: {er.__dict__}")
                
                # Try all possible error fields
                for field in ['error', 'message', 'error_details', 'preview_failure_reason', 
                              'new_order_failure_reason', 'failure_reason']:
                    if hasattr(er, field):
                        print(f"    {field}: {getattr(er, field)}")
    
    # Also check for direct order_id (different response format)
    if hasattr(order, 'order_id'):
        print(f"\n    Direct order_id found: {order.order_id}")
        print(f"    This means SUCCESS!")

except Exception as e:
    print(f"\n    EXCEPTION: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 4. Check if we have an open position
print(f"\n[4] Checking for open positions...")
try:
    # Try to list positions
    positions = client.list_futures_positions()
    print(f"    Positions response: {positions}")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nIf order succeeded, close it manually in Coinbase app!")
print("Or run this script again with 'sell' to close.")