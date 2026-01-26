#!/usr/bin/env python3
"""
Test script to verify dynamic margin calculations.
"""
import os
from coinbase.rest import RESTClient

API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

def test_margin_calculations():
    client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
    
    # Get balance - FIXED parsing
    response = client.get_futures_balance_summary()
    summary = response.balance_summary
    
    # Access as object attributes, then get 'value' from the dict
    balance = float(summary.futures_buying_power['value'])
    total_usd = float(summary.total_usd_balance['value'])
    
    print(f"\n{'='*70}")
    print("MARGIN CALCULATION TEST")
    print(f"{'='*70}")
    print(f"Total USD Balance: ${total_usd:.2f}")
    print(f"Futures Buying Power: ${balance:.2f}")
    
    # Test BTC perpetual
    symbol = 'BIP-20DEC30-CDE'
    product = client.get_product(product_id=symbol)
    
    # Extract margin rates
    future_details = getattr(product, 'future_product_details', None)
    price = float(getattr(product, 'price', 0))
    contract_size = 0.01  # BTC
    
    print(f"\n{symbol} (BTC Perpetual)")
    print(f"  Price: ${price:,.2f}")
    print(f"  Contract size: {contract_size} BTC")
    print(f"  Notional per contract: ${price * contract_size:,.2f}")
    
    if future_details:
        if isinstance(future_details, dict):
            intraday = future_details.get('intraday_margin_rate', {})
            overnight = future_details.get('overnight_margin_rate', {})
        else:
            intraday = getattr(future_details, 'intraday_margin_rate', {}) or {}
            overnight = getattr(future_details, 'overnight_margin_rate', {}) or {}
        
        margin_rate_long = float(intraday.get('long_margin_rate', 0.25))
        margin_rate_short = float(intraday.get('short_margin_rate', 0.25))
        overnight_long = float(overnight.get('long_margin_rate', 0.30))
        overnight_short = float(overnight.get('short_margin_rate', 0.30))
        
        notional = price * contract_size
        
        print(f"\n  INTRADAY MARGIN:")
        print(f"    Long:  {margin_rate_long:.2%} = ${notional * margin_rate_long:.2f}/contract ({1/margin_rate_long:.1f}x leverage)")
        print(f"    Short: {margin_rate_short:.2%} = ${notional * margin_rate_short:.2f}/contract ({1/margin_rate_short:.1f}x leverage)")
        
        print(f"\n  OVERNIGHT MARGIN:")
        print(f"    Long:  {overnight_long:.2%} = ${notional * overnight_long:.2f}/contract ({1/overnight_long:.1f}x leverage)")
        print(f"    Short: {overnight_short:.2%} = ${notional * overnight_short:.2f}/contract ({1/overnight_short:.1f}x leverage)")
        
        # Calculate max contracts with 50% position size
        position_pct = 0.50
        budget = balance * position_pct
        
        print(f"\n  POSITION SIZING (50% of buying power = ${budget:.2f}):")
        
        intraday_margin_long = notional * margin_rate_long
        max_contracts_long = int(budget / intraday_margin_long)
        print(f"    LONG:  {max_contracts_long} contracts (${max_contracts_long * intraday_margin_long:.2f} margin)")
        
        intraday_margin_short = notional * margin_rate_short
        max_contracts_short = int(budget / intraday_margin_short)
        print(f"    SHORT: {max_contracts_short} contracts (${max_contracts_short * intraday_margin_short:.2f} margin)")
        
        print(f"\n{'='*70}")
        if max_contracts_long >= 1:
            print(f"✓ Bot CAN enter LONG - {max_contracts_long} contract(s) with 50% sizing")
        else:
            print(f"✗ Bot CANNOT enter LONG - need ${intraday_margin_long:.2f} margin per contract")
            
        if max_contracts_short >= 1:
            print(f"✓ Bot CAN enter SHORT - {max_contracts_short} contract(s) with 50% sizing")
        else:
            print(f"✗ Bot CANNOT enter SHORT - need ${intraday_margin_short:.2f} margin per contract")
    else:
        print("  WARNING: Could not fetch margin rates from API!")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_margin_calculations()