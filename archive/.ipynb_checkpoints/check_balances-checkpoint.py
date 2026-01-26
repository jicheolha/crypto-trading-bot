#!/usr/bin/env python3
"""
Check both spot and futures balances.
Shows where your funds are and how to transfer.
"""
import os
from coinbase.rest import RESTClient

API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

def check_balances():
    client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
    
    print("\n" + "="*60)
    print("WALLET BALANCES")
    print("="*60)
    
    # Check spot accounts
    print("\n📍 SPOT WALLET:")
    try:
        accounts = client.get_accounts()
        spot_total = 0.0
        for acc in accounts.accounts:
            currency = getattr(acc, 'currency', '')
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
                
                total = avail + hold
                if total > 0:
                    print(f"   {currency}: ${total:.2f} (available: ${avail:.2f}, hold: ${hold:.2f})")
                    spot_total += total
        
        if spot_total == 0:
            print("   (empty)")
        else:
            print(f"   TOTAL: ${spot_total:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        spot_total = 0
    
    # Check futures balance
    print("\n📍 FUTURES WALLET:")
    try:
        response = client.get_futures_balance_summary()
        summary = response.balance_summary if hasattr(response, 'balance_summary') else {}
        
        if isinstance(summary, dict):
            futures_buying_power = float(summary.get('futures_buying_power', {}).get('value', 0))
            total_usd = float(summary.get('total_usd_balance', {}).get('value', 0))
            unrealized_pnl = float(summary.get('unrealized_pnl', {}).get('value', 0))
            initial_margin = float(summary.get('initial_margin', {}).get('value', 0))
            available_margin = float(summary.get('available_margin', {}).get('value', 0))
        else:
            futures_buying_power = total_usd = unrealized_pnl = initial_margin = available_margin = 0
        
        print(f"   Total USD: ${total_usd:.2f}")
        print(f"   Buying Power: ${futures_buying_power:.2f}")
        print(f"   Initial Margin (in use): ${initial_margin:.2f}")
        print(f"   Available Margin: ${available_margin:.2f}")
        print(f"   Unrealized P&L: ${unrealized_pnl:+.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        total_usd = 0
    
    # Check pending transfers
    print("\n📍 PENDING TRANSFERS:")
    try:
        sweeps = client.list_futures_sweeps()
        if hasattr(sweeps, 'sweeps') and sweeps.sweeps:
            for sweep in sweeps.sweeps:
                status = getattr(sweep, 'status', 'UNKNOWN')
                amount = getattr(sweep, 'requested_amount', {})
                if isinstance(amount, dict):
                    val = amount.get('value', '0')
                else:
                    val = getattr(amount, 'value', '0')
                print(f"   {status}: ${float(val):.2f}")
        else:
            print("   (none)")
    except Exception as e:
        print(f"   Error checking sweeps: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if spot_total > 0 and total_usd == 0:
        print(f"\n⚠️  Your funds (${spot_total:.2f}) are in SPOT wallet!")
        print("   To trade futures, transfer to futures wallet:")
        print("")
        print("   Option 1 - Coinbase App/Web:")
        print("   1. Go to Coinbase → Assets → USD")
        print("   2. Click 'Transfer'")
        print("   3. Select 'Spot' → 'Futures'")
        print("   4. Enter amount and confirm")
        print("")
        print("   Option 2 - Schedule automatic sweep:")
        print("   Run: python schedule_sweep.py")
    elif total_usd > 0:
        print(f"\n✓ Futures wallet has ${total_usd:.2f}")
        print(f"  Buying power: ${futures_buying_power:.2f}")
        print("\n  Bot can trade with this balance!")
    else:
        print("\n⚠️  Both wallets appear empty!")
        print("   Funds may be settling (takes a few minutes)")
        print("   Or check Coinbase app for actual balance")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    check_balances()