#!/usr/bin/env python3
"""
Keltrade Diagnostic Script

Tests all components required for live trading:
1. API connectivity
2. Account balances
3. Product/contract info
4. Margin rates
5. Price feeds
6. Position sizing calculations
7. Order placement (dry run)
8. Signal generation

Run this before deploying to ensure everything works.
"""
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from coinbase.rest import RESTClient
except ImportError:
    print("ERROR: coinbase-advanced-py not installed")
    print("Run: pip install coinbase-advanced-py")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_KEY = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

# Futures contracts to test
FUTURES_SYMBOLS = [
    'BIP-20DEC30-CDE',  # BTC
    'ETP-20DEC30-CDE',  # ETH
    'SLP-20DEC30-CDE',  # SOL
    'XPP-20DEC30-CDE',  # XRP
    'DOP-20DEC30-CDE',  # DOGE
]

# Spot symbols for signals
SPOT_SYMBOLS = [
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
    'XRP-USD',
    'DOGE-USD',
]

# Contract specs (for validation)
EXPECTED_CONTRACT_SPECS = {
    'BIP-20DEC30-CDE': {'contract_size': 0.01, 'base': 'BTC'},
    'ETP-20DEC30-CDE': {'contract_size': 0.1, 'base': 'ETH'},
    'SLP-20DEC30-CDE': {'contract_size': 5.0, 'base': 'SOL'},
    'XPP-20DEC30-CDE': {'contract_size': 500.0, 'base': 'XRP'},
    'DOP-20DEC30-CDE': {'contract_size': 5000.0, 'base': 'DOGE'},
}


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def ok(msg: str) -> str:
    return f"{Colors.GREEN}âœ“ {msg}{Colors.RESET}"


def fail(msg: str) -> str:
    return f"{Colors.RED}âœ— {msg}{Colors.RESET}"


def warn(msg: str) -> str:
    return f"{Colors.YELLOW}âš  {msg}{Colors.RESET}"


def info(msg: str) -> str:
    return f"{Colors.CYAN}â„¹ {msg}{Colors.RESET}"


def header(msg: str) -> str:
    return f"\n{Colors.BOLD}{'='*60}\n{msg}\n{'='*60}{Colors.RESET}"


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

class KeltradeDiagnostics:
    """Comprehensive diagnostics for Keltrade."""
    
    def __init__(self):
        self.client: Optional[RESTClient] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.results: Dict = {}
    
    def run_all(self):
        """Run all diagnostic tests."""
        print(header("KELTRADE DIAGNOSTICS"))
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            ("API Credentials", self.test_credentials),
            ("API Connection", self.test_connection),
            ("Spot Balances", self.test_spot_balances),
            ("Futures Balances", self.test_futures_balances),
            ("Futures Products", self.test_futures_products),
            ("Margin Rates", self.test_margin_rates),
            ("Spot Prices", self.test_spot_prices),
            ("Futures Prices", self.test_futures_prices),
            ("Historical Candles", self.test_candles),
            ("Position Sizing", self.test_position_sizing),
            ("Order Validation", self.test_order_validation),
            ("Signal Generation", self.test_signal_generation),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            print(header(name))
            try:
                success = test_func()
                if success:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(fail(f"Exception: {e}"))
                import traceback
                traceback.print_exc()
                failed += 1
                self.errors.append(f"{name}: {e}")
        
        # Summary
        print(header("SUMMARY"))
        print(f"Tests passed: {passed}/{passed + failed}")
        print(f"Tests failed: {failed}/{passed + failed}")
        
        if self.errors:
            print(f"\n{Colors.RED}ERRORS:{Colors.RESET}")
            for err in self.errors:
                print(f"  â€¢ {err}")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}WARNINGS:{Colors.RESET}")
            for warn_msg in self.warnings:
                print(f"  â€¢ {warn_msg}")
        
        if failed == 0 and not self.errors:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ALL SYSTEMS GO - Ready for live trading{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}FIX ERRORS BEFORE LIVE TRADING{Colors.RESET}")
        
        return failed == 0
    
    def test_credentials(self) -> bool:
        """Test API credentials are set."""
        if not API_KEY:
            print(fail("COINBASE_API_KEY not set"))
            self.errors.append("Missing COINBASE_API_KEY environment variable")
            return False
        
        if not API_SECRET:
            print(fail("COINBASE_API_SECRET not set"))
            self.errors.append("Missing COINBASE_API_SECRET environment variable")
            return False
        
        print(ok(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}"))
        print(ok(f"API Secret: {API_SECRET[:8]}..."))
        return True
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            self.client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
            
            # Simple API call to test connection
            accounts = self.client.get_accounts()
            
            if hasattr(accounts, 'accounts'):
                count = len(accounts.accounts)
            else:
                count = 0
            
            print(ok(f"Connected to Coinbase API"))
            print(ok(f"Found {count} accounts"))
            return True
            
        except Exception as e:
            print(fail(f"Connection failed: {e}"))
            self.errors.append(f"API connection failed: {e}")
            return False
    
    def test_spot_balances(self) -> bool:
        """Test spot account balances."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        try:
            accounts = self.client.get_accounts()
            account_list = accounts.accounts if hasattr(accounts, 'accounts') else []
            
            total_usd = 0.0
            found_currencies = []
            
            for account in account_list:
                currency = getattr(account, 'currency', '')
                
                # Get available balance
                avail = 0.0
                if hasattr(account, 'available_balance'):
                    avail_obj = account.available_balance
                    if isinstance(avail_obj, dict):
                        avail = float(avail_obj.get('value', 0))
                    elif hasattr(avail_obj, 'value'):
                        avail = float(avail_obj.value)
                
                if avail > 0:
                    found_currencies.append(f"{currency}: {avail:.4f}")
                    if currency in ['USD', 'USDC']:
                        total_usd += avail
            
            if found_currencies:
                for curr in found_currencies[:10]:  # Limit output
                    print(info(curr))
            
            print(ok(f"Total USD/USDC: ${total_usd:,.2f}"))
            
            self.results['spot_balance'] = total_usd
            
            if total_usd < 10:
                print(warn("Low spot balance - may not be able to trade"))
                self.warnings.append("Spot balance below $10")
            
            return True
            
        except Exception as e:
            print(fail(f"Error fetching spot balances: {e}"))
            self.errors.append(f"Spot balance fetch failed: {e}")
            return False
    
    def test_futures_balances(self) -> bool:
        """Test futures account balances."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        try:
            response = self.client.get_futures_balance_summary()
            
            # Extract balance from response
            summary = None
            if hasattr(response, 'balance_summary'):
                summary = response.balance_summary
            
            if summary is None:
                print(warn("Could not parse futures balance response"))
                print(info(f"Raw response: {response}"))
                self.warnings.append("Futures balance parsing issue")
                return True  # Not a failure, just no futures account
            
            # Try different fields
            balance = 0.0
            for field in ['futures_buying_power', 'available_margin', 'total_usd_balance']:
                val = None
                if isinstance(summary, dict):
                    val = summary.get(field, {})
                elif hasattr(summary, field):
                    val = getattr(summary, field)
                
                if val:
                    if isinstance(val, dict) and 'value' in val:
                        balance = float(val['value'])
                        print(ok(f"{field}: ${balance:,.2f}"))
                        break
                    elif hasattr(val, 'value'):
                        balance = float(val.value)
                        print(ok(f"{field}: ${balance:,.2f}"))
                        break
            
            self.results['futures_balance'] = balance
            
            if balance < 10:
                print(warn("Low futures balance - transfer funds from spot"))
                self.warnings.append("Futures balance below $10")
            
            # Also print other useful info
            if isinstance(summary, dict):
                for key in ['unrealized_pnl', 'total_open_orders_hold_amount']:
                    if key in summary:
                        val = summary[key]
                        if isinstance(val, dict) and 'value' in val:
                            print(info(f"{key}: {val['value']}"))
            
            return True
            
        except Exception as e:
            print(warn(f"Futures balance not available: {e}"))
            self.warnings.append(f"No futures account or error: {e}")
            return True  # Not a hard failure
    
    def test_futures_products(self) -> bool:
        """Test futures product info."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        success = True
        
        for symbol in FUTURES_SYMBOLS:
            try:
                product = self.client.get_product(product_id=symbol)
                
                product_type = getattr(product, 'product_type', 'UNKNOWN')
                status = getattr(product, 'status', 'UNKNOWN')
                
                if 'FUTURE' not in product_type.upper():
                    print(fail(f"{symbol}: Not a futures product (type={product_type})"))
                    self.errors.append(f"{symbol} is not a futures product")
                    success = False
                    continue
                
                if status not in ['online', '']:
                    print(warn(f"{symbol}: Status is {status}"))
                    self.warnings.append(f"{symbol} status is {status}")
                
                # Get contract details
                future_details = getattr(product, 'future_product_details', None)
                
                if future_details:
                    # Contract size - future_details is a dict
                    contract_size = future_details.get('contract_size') if isinstance(future_details, dict) else getattr(future_details, 'contract_size', None)
                    if contract_size:
                        print(ok(f"{symbol}: contract_size={contract_size}"))
                        
                        # Validate against expected
                        expected = EXPECTED_CONTRACT_SPECS.get(symbol, {}).get('contract_size')
                        if expected and float(contract_size) != expected:
                            print(warn(f"  Expected {expected}, got {contract_size}"))
                            self.warnings.append(f"{symbol} contract size mismatch")
                    else:
                        print(warn(f"{symbol}: No contract_size in response"))
                else:
                    print(warn(f"{symbol}: No future_product_details"))
                
            except Exception as e:
                print(fail(f"{symbol}: Error - {e}"))
                self.errors.append(f"{symbol} product fetch failed: {e}")
                success = False
        
        return success
    
    def test_margin_rates(self) -> bool:
        """Test margin rate fetching."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        success = True
        self.results['margin_rates'] = {}
        
        for symbol in FUTURES_SYMBOLS:
            try:
                product = self.client.get_product(product_id=symbol)
                future_details = getattr(product, 'future_product_details', None)
                
                if not future_details:
                    print(warn(f"{symbol}: No future_product_details"))
                    continue
                
                # Get margin rates - future_details is a dict
                if isinstance(future_details, dict):
                    intraday = future_details.get('intraday_margin_rate')
                    overnight = future_details.get('overnight_margin_rate')
                else:
                    intraday = getattr(future_details, 'intraday_margin_rate', None)
                    overnight = getattr(future_details, 'overnight_margin_rate', None)
                
                rates = {}
                
                if intraday:
                    if isinstance(intraday, dict):
                        rates['intraday_long'] = float(intraday.get('long_margin_rate', 0))
                        rates['intraday_short'] = float(intraday.get('short_margin_rate', 0))
                    else:
                        rates['intraday_long'] = float(getattr(intraday, 'long_margin_rate', 0))
                        rates['intraday_short'] = float(getattr(intraday, 'short_margin_rate', 0))
                
                if overnight:
                    if isinstance(overnight, dict):
                        rates['overnight_long'] = float(overnight.get('long_margin_rate', 0))
                        rates['overnight_short'] = float(overnight.get('short_margin_rate', 0))
                    else:
                        rates['overnight_long'] = float(getattr(overnight, 'long_margin_rate', 0))
                        rates['overnight_short'] = float(getattr(overnight, 'short_margin_rate', 0))
                
                if rates:
                    intra_lev = 1.0 / rates.get('intraday_long', 0.25) if rates.get('intraday_long', 0) > 0 else 0
                    overnight_lev = 1.0 / rates.get('overnight_long', 0.25) if rates.get('overnight_long', 0) > 0 else 0
                    
                    print(ok(f"{symbol}:"))
                    print(f"    Intraday:  L={rates.get('intraday_long', 0):.2%} S={rates.get('intraday_short', 0):.2%} (~{intra_lev:.0f}x)")
                    print(f"    Overnight: L={rates.get('overnight_long', 0):.2%} S={rates.get('overnight_short', 0):.2%} (~{overnight_lev:.0f}x)")
                    
                    self.results['margin_rates'][symbol] = rates
                else:
                    print(warn(f"{symbol}: No margin rates found"))
                    self.warnings.append(f"{symbol} missing margin rates")
                
            except Exception as e:
                print(fail(f"{symbol}: Error - {e}"))
                self.errors.append(f"{symbol} margin rate fetch failed: {e}")
                success = False
        
        return success
    
    def test_spot_prices(self) -> bool:
        """Test spot price feeds."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        success = True
        self.results['spot_prices'] = {}
        
        for symbol in SPOT_SYMBOLS:
            try:
                product = self.client.get_product(product_id=symbol)
                price = float(product.price) if hasattr(product, 'price') else None
                
                if price:
                    print(ok(f"{symbol}: ${price:,.4f}"))
                    self.results['spot_prices'][symbol] = price
                else:
                    print(fail(f"{symbol}: No price"))
                    self.errors.append(f"{symbol} no price available")
                    success = False
                    
            except Exception as e:
                print(fail(f"{symbol}: Error - {e}"))
                self.errors.append(f"{symbol} price fetch failed: {e}")
                success = False
        
        return success
    
    def test_futures_prices(self) -> bool:
        """Test futures price feeds."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        success = True
        self.results['futures_prices'] = {}
        
        for symbol in FUTURES_SYMBOLS:
            try:
                product = self.client.get_product(product_id=symbol)
                price = float(product.price) if hasattr(product, 'price') else None
                
                if price:
                    print(ok(f"{symbol}: ${price:,.4f}"))
                    self.results['futures_prices'][symbol] = price
                else:
                    print(warn(f"{symbol}: No price (market may be closed)"))
                    self.warnings.append(f"{symbol} no futures price")
                    
            except Exception as e:
                print(fail(f"{symbol}: Error - {e}"))
                self.errors.append(f"{symbol} futures price fetch failed: {e}")
                success = False
        
        return success
    
    def test_candles(self) -> bool:
        """Test historical candle fetching."""
        if not self.client:
            print(fail("No API connection"))
            return False
        
        success = True
        
        # Test one spot symbol with different timeframes
        symbol = 'BTC-USD'
        timeframes = [
            ('ONE_HOUR', '1h'),
            ('FOUR_HOUR', '4h'),
        ]
        
        for gran, tf_name in timeframes:
            try:
                end_ts = int(datetime.now().timestamp())
                start_ts = int((datetime.now() - timedelta(days=7)).timestamp())
                
                response = self.client.get_candles(
                    product_id=symbol,
                    start=start_ts,
                    end=end_ts,
                    granularity=gran
                )
                
                candles = response.candles if hasattr(response, 'candles') else []
                
                if candles:
                    print(ok(f"{symbol} {tf_name}: {len(candles)} candles"))
                    
                    # Validate candle structure
                    c = candles[0]
                    required = ['start', 'open', 'high', 'low', 'close', 'volume']
                    missing = [f for f in required if not hasattr(c, f)]
                    
                    if missing:
                        print(warn(f"  Missing fields: {missing}"))
                        self.warnings.append(f"Candle missing fields: {missing}")
                else:
                    print(fail(f"{symbol} {tf_name}: No candles returned"))
                    self.errors.append(f"No candles for {symbol} {tf_name}")
                    success = False
                    
            except Exception as e:
                print(fail(f"{symbol} {tf_name}: Error - {e}"))
                self.errors.append(f"Candle fetch failed: {e}")
                success = False
        
        return success
    
    def test_position_sizing(self) -> bool:
        """Test position sizing calculations."""
        print(info("Testing position sizing logic..."))
        
        # Get balances and prices
        balance = self.results.get('futures_balance', 0) or self.results.get('spot_balance', 0)
        
        if balance < 10:
            print(warn(f"Balance too low for meaningful test: ${balance:.2f}"))
            return True
        
        print(info(f"Using balance: ${balance:,.2f}"))
        
        success = True
        
        for symbol in FUTURES_SYMBOLS:
            price = self.results.get('futures_prices', {}).get(symbol)
            if not price:
                price = self.results.get('spot_prices', {}).get(symbol.split('-')[0] + '-USD')
            
            if not price:
                print(warn(f"{symbol}: No price available for sizing test"))
                continue
            
            # Get margin rate - always use conservative overnight rates
            margin_rates = self.results.get('margin_rates', {}).get(symbol, {})
            
            # Always use overnight (conservative) rates - no intraday high leverage
            margin_rate = margin_rates.get('overnight_long', 0.25)
            margin_type = "overnight"
            
            # Get contract size
            contract_size = EXPECTED_CONTRACT_SPECS.get(symbol, {}).get('contract_size', 0.1)
            
            # Calculate position sizing
            position_pct = 0.50  # 50% position
            position_value = balance * position_pct
            
            notional_per_contract = price * contract_size
            margin_per_contract = notional_per_contract * margin_rate
            
            max_contracts = position_value / margin_per_contract
            contracts = int(max_contracts)
            
            actual_margin = contracts * margin_per_contract
            actual_notional = contracts * notional_per_contract
            
            print(ok(f"{symbol} ({margin_type}):"))
            print(f"    Price: ${price:,.2f}")
            print(f"    Contract size: {contract_size}")
            print(f"    Margin rate: {margin_rate:.2%} ({margin_type})")
            print(f"    Notional/contract: ${notional_per_contract:,.2f}")
            print(f"    Margin/contract: ${margin_per_contract:,.2f}")
            print(f"    Target position: ${position_value:,.2f} ({position_pct:.0%})")
            print(f"    Contracts: {contracts}")
            print(f"    Actual margin: ${actual_margin:,.2f}")
            print(f"    Actual notional: ${actual_notional:,.2f}")
            
            if contracts < 1:
                print(warn(f"    Cannot afford 1 contract with current balance"))
                self.warnings.append(f"{symbol}: Cannot afford 1 contract")
        
        return success
    
    def test_order_validation(self) -> bool:
        """Test order validation (dry run - no actual orders)."""
        print(info("Validating order parameters (DRY RUN - no orders placed)"))
        
        # We're NOT placing orders, just validating the parameters would work
        
        for symbol in FUTURES_SYMBOLS[:1]:  # Just test one
            price = self.results.get('futures_prices', {}).get(symbol)
            if not price:
                print(warn(f"{symbol}: Skipping - no price"))
                continue
            
            contract_size = EXPECTED_CONTRACT_SPECS.get(symbol, {}).get('contract_size', 0.1)
            
            # Test order parameters
            test_size = 1  # 1 contract
            
            print(ok(f"{symbol} order validation:"))
            print(f"    Side: BUY")
            print(f"    Size: {test_size} contract(s)")
            print(f"    Type: MARKET")
            print(f"    Estimated notional: ${price * contract_size * test_size:,.2f}")
            
            # Validate client_order_id format
            client_order_id = f"{symbol.replace('/', '').replace('-', '')}-{int(time.time())}"
            if len(client_order_id) > 36:
                print(warn(f"    client_order_id too long: {len(client_order_id)} chars"))
                self.warnings.append("client_order_id may be too long")
            else:
                print(ok(f"    client_order_id: {client_order_id}"))
        
        print(info("To test actual order placement, use --place-test-order flag"))
        return True
    
    def test_signal_generation(self) -> bool:
        """Test signal generation components."""
        print(info("Testing signal generation imports..."))
        
        try:
            from technical import BBSqueezeAnalyzer
            print(ok("Imported BBSqueezeAnalyzer"))
        except ImportError as e:
            print(fail(f"Cannot import BBSqueezeAnalyzer: {e}"))
            self.errors.append(f"Import error: {e}")
            return False
        
        try:
            from signal_generator import BBSqueezeSignalGenerator
            print(ok("Imported BBSqueezeSignalGenerator"))
        except ImportError as e:
            print(fail(f"Cannot import BBSqueezeSignalGenerator: {e}"))
            self.errors.append(f"Import error: {e}")
            return False
        
        # Test with real data
        print(info("Testing signal generation with live data..."))
        
        try:
            import pandas as pd
            
            # Fetch candles
            symbol = 'BTC-USD'
            end_ts = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(days=30)).timestamp())
            
            response = self.client.get_candles(
                product_id=symbol,
                start=start_ts,
                end=end_ts,
                granularity='FOUR_HOUR'
            )
            
            candles = response.candles if hasattr(response, 'candles') else []
            
            if not candles:
                print(warn("No candles for signal test"))
                return True
            
            # Convert to DataFrame
            data = []
            for c in candles:
                data.append({
                    'timestamp': int(c.start),
                    'open': float(c.open),
                    'high': float(c.high),
                    'low': float(c.low),
                    'close': float(c.close),
                    'volume': float(c.volume)
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(ok(f"Created DataFrame: {len(df)} rows"))
            
            # Initialize analyzer
            analyzer = BBSqueezeAnalyzer()
            df_with_indicators = analyzer.calculate_indicators(df)
            
            print(ok(f"Calculated indicators"))
            
            # Check squeeze state
            squeeze_state = analyzer.get_squeeze_state(df_with_indicators)
            print(ok(f"Squeeze state: {squeeze_state.is_squeeze}, duration: {squeeze_state.squeeze_bars} bars"))
            
            # Initialize signal generator
            signal_gen = BBSqueezeSignalGenerator(analyzer=analyzer)
            signal_gen.set_signal_data({symbol: df_with_indicators})
            signal_gen.set_atr_data({symbol: df_with_indicators})
            
            # Generate signal
            signal = signal_gen.generate_signal(df_with_indicators, symbol, datetime.now())
            
            print(ok(f"Signal generated: direction={signal.direction}"))
            if signal.direction != 'neutral':
                print(f"    Entry: ${signal.entry_price:,.2f}")
                print(f"    Stop: ${signal.stop_loss:,.2f}")
                print(f"    Target: ${signal.take_profit:,.2f}")
                print(f"    Size: {signal.position_size:.0%}")
            
            return True
            
        except Exception as e:
            print(fail(f"Signal generation test failed: {e}"))
            import traceback
            traceback.print_exc()
            self.errors.append(f"Signal generation error: {e}")
            return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Keltrade Diagnostics')
    parser.add_argument('--place-test-order', action='store_true',
                        help='Actually place a small test order (USE WITH CAUTION)')
    
    args = parser.parse_args()
    
    if args.place_test_order:
        print(f"\n{Colors.RED}{Colors.BOLD}WARNING: --place-test-order will place a REAL order!{Colors.RESET}")
        print("This is not implemented for safety. Edit the code if you really want this.")
        return
    
    diag = KeltradeDiagnostics()
    success = diag.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()