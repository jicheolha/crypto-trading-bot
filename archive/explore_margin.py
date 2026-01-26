#!/usr/bin/env python3
"""
Explore Coinbase API to find where leverage/margin requirements are stored.
"""
import os
import sys
import traceback

print("Starting script...")

API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

print(f"API_KEY set: {bool(API_KEY)}")
print(f"API_SECRET set: {bool(API_SECRET)}")

if not API_KEY or not API_SECRET:
    print("ERROR: Set COINBASE_API_KEY and COINBASE_API_SECRET")
    sys.exit(1)

print("Importing coinbase...")
from coinbase.rest import RESTClient

print("Creating client...")
client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
print("Client created")

symbol = 'BIP-20DEC30-CDE'

# 1. Explore get_product
print(f"\n{'='*70}")
print(f"get_product({symbol})")
print(f"{'='*70}")
try:
    product = client.get_product(product_id=symbol)
    for key, value in vars(product).items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# 2. Explore futures balance
print(f"\n{'='*70}")
print("get_futures_balance_summary()")
print(f"{'='*70}")
try:
    response = client.get_futures_balance_summary()
    for key, value in vars(response).items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# 3. Explore futures positions
print(f"\n{'='*70}")
print("list_futures_positions()")
print(f"{'='*70}")
try:
    response = client.list_futures_positions()
    print(f"  Raw: {response}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# 4. List all futures-related methods
print(f"\n{'='*70}")
print("All futures/margin methods available:")
print(f"{'='*70}")
for method in sorted(dir(client)):
    if not method.startswith('_'):
        if 'future' in method.lower() or 'margin' in method.lower() or 'leverage' in method.lower():
            print(f"  {method}")

# 5. Try to get contract details if method exists
print(f"\n{'='*70}")
print("Trying other methods...")
print(f"{'='*70}")

for method_name in ['get_futures_contract', 'get_contract_details', 'get_margin_requirements']:
    if hasattr(client, method_name):
        print(f"Found: {method_name}")
        try:
            method = getattr(client, method_name)
            result = method(symbol)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")

print("\nDone!")