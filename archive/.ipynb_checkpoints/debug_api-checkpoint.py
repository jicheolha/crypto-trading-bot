#!/usr/bin/env python3
"""Dump raw API response to see actual structure."""
import os
from coinbase.rest import RESTClient

API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

print("="*60)
print("RAW get_futures_balance_summary() RESPONSE")
print("="*60)

response = client.get_futures_balance_summary()

print(f"\nType: {type(response)}")
print(f"\nRaw: {response}")

print("\n" + "="*60)
print("Trying to access attributes:")
print("="*60)

# Try different access patterns
for attr in ['balance_summary', 'futures_buying_power', 'total_usd_balance', 'available_margin']:
    if hasattr(response, attr):
        val = getattr(response, attr)
        print(f"\nresponse.{attr} = {val}")

# If it has balance_summary, dig into it
if hasattr(response, 'balance_summary'):
    bs = response.balance_summary
    print(f"\nbalance_summary type: {type(bs)}")
    if isinstance(bs, dict):
        for k, v in bs.items():
            print(f"  {k}: {v}")
    elif hasattr(bs, '__dict__'):
        for k, v in vars(bs).items():
            print(f"  {k}: {v}")