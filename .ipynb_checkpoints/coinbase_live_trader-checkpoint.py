"""
Coinbase Live Trader - Supports SPOT and FUTURES

Handles:
- Spot trading (fractional amounts, e.g. 0.032 ETH)
- Futures trading (whole contracts only, e.g. 1, 2, 3 contracts)
- Automatic leverage detection
- Margin calculation
- Bracket orders with server-side stops
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from coinbase.rest import RESTClient

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CoinbaseLiveTrader:
    """
    Live trader for Coinbase spot and futures markets.
    """
    
    # Hardcoded contract specs for Coinbase International perpetual futures
    # Based on actual contract specifications from the exchange
    PERP_CONTRACT_SPECS = {
        # BTC (BIP): 0.01 BTC per contract, 10x leverage
        'BIP-20DEC30-CDE': {'contract_size': 0.01, 'leverage': 10.0, 'base': 'BTC'},
        
        # ETH (ETP): 0.1 ETH per contract, 10x leverage
        'ETP-20DEC30-CDE': {'contract_size': 0.1, 'leverage': 10.0, 'base': 'ETH'},
        
        # SOL (SLP): 5 SOL per contract, 5x leverage
        'SLP-20DEC30-CDE': {'contract_size': 5.0, 'leverage': 5.0, 'base': 'SOL'},
        
        # XRP (XPP): 500 XRP per contract, 5x leverage
        'XPP-20DEC30-CDE': {'contract_size': 500.0, 'leverage': 5.0, 'base': 'XRP'},
        
        # DOGE (DOP): 5000 DOGE per contract, 4x leverage
        'DOP-20DEC30-CDE': {'contract_size': 5000.0, 'leverage': 4.0, 'base': 'DOGE'},
    }
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        
        # NEW: Separate signal vs trade symbols
        signal_symbols: List[str] = None,  # Get signals from these (e.g., ETH-USD spot)
        
        # Timeframes
        signal_timeframe: str = '1h',
        atr_timeframe: str = '4h',
        
        # Strategy parameters
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_mult: float = 1.5,
        momentum_period: int = 12,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        min_squeeze_bars: int = 3,
        volume_period: int = 20,
        min_volume_ratio: float = 1.2,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        
        # Position sizing
        base_position: float = 0.10,
        min_position: float = 0.05,
        max_position: float = 0.30,
        
        # Risk management
        max_positions: int = 2,
        max_daily_loss: float = 0.03,
        max_hold_days: float = None,
        long_only: bool = False,
        
        # Setup
        setup_validity_bars: int = 5,
        
        # Bot settings
        check_interval_seconds: int = 60,
        lookback_bars: int = 100,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        
        self.symbols = symbols  # Trading symbols (e.g., ETP-20DEC30-CDE)
        
        # Signal symbols (e.g., ETH-USD for spot data)
        # If not provided, use trading symbols for signals
        if signal_symbols is None:
            self.signal_symbols = symbols
            self.symbol_mapping = {s: s for s in symbols}  # Same for both
        else:
            self.signal_symbols = signal_symbols
            # Create mapping: trade_symbol -> signal_symbol
            if len(signal_symbols) == len(symbols):
                self.symbol_mapping = dict(zip(symbols, signal_symbols))
            else:
                # If counts don't match, use first signal symbol for all
                self.symbol_mapping = {s: signal_symbols[0] for s in symbols}
        
        logger.info(f"Trading symbols: {self.symbols}")
        logger.info(f"Signal symbols: {self.signal_symbols}")
        logger.info(f"Mapping: {self.symbol_mapping}")
        
        self.signal_timeframe = signal_timeframe
        self.atr_timeframe = atr_timeframe
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.max_hold_days = max_hold_days
        self.long_only = long_only
        self.check_interval = check_interval_seconds
        self.lookback_bars = lookback_bars
        
        # Detect product types and contract info
        self.product_info = {}
        self._detect_product_types()
        
        # Initialize strategy
        self.analyzer = BBSqueezeAnalyzer(
            bb_period=bb_period,
            bb_std=bb_std,
            kc_period=kc_period,
            kc_atr_mult=kc_atr_mult,
            momentum_period=momentum_period,
            rsi_period=rsi_period,
            volume_period=volume_period,
            atr_period=atr_period
        )
        
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        self.signal_generator = BBSqueezeSignalGenerator(
            analyzer=self.analyzer,
            min_squeeze_bars=min_squeeze_bars,
            min_volume_ratio=min_volume_ratio,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            atr_stop_mult=atr_stop_mult,
            atr_target_mult=atr_target_mult,
            base_position=base_position,
            min_position=min_position,
            max_position=max_position,
            setup_validity_bars=setup_validity_bars,
            signal_timeframe_minutes=tf_minutes.get(signal_timeframe, 60)
        )
        
        # State
        self.positions: Dict[str, dict] = {}
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.now().date()
        self.last_candles: Dict[str, datetime] = {}
        self.running = False
        
        logger.info(f"Initialized trader for {len(symbols)} symbols")
    
    def _detect_product_types(self):
        """Detect if symbols are spot or futures and get contract info."""
        for symbol in self.symbols:
            try:
                product = self.client.get_product(product_id=symbol)
                
                product_type = getattr(product, 'product_type', 'UNKNOWN')
                is_futures = 'FUTURE' in product_type.upper()
                
                # Get contract size for futures (default to 1.0 for spot)
                contract_size = 1.0
                leverage = 1.0
                
                if is_futures:
                    # Use hardcoded specs - API returns incorrect values
                    if symbol in self.PERP_CONTRACT_SPECS:
                        spec = self.PERP_CONTRACT_SPECS[symbol]
                        contract_size = spec['contract_size']
                        leverage = spec['leverage']
                        logger.info(f"  Using hardcoded spec for {symbol}: {contract_size} {spec['base']}/contract, {leverage}x")
                    else:
                        # Fallback defaults for unknown futures
                        logger.warning(f"  No hardcoded spec for {symbol}, using defaults")
                        contract_size = 0.1
                        leverage = 5.0
                
                self.product_info[symbol] = {
                    'is_futures': is_futures,
                    'contract_size': contract_size,
                    'leverage': leverage,
                    'product_type': product_type
                }
                
                logger.info(
                    f"{symbol}: "
                    f"{'FUTURES' if is_futures else 'SPOT'} | "
                    f"Contract: {contract_size} | "
                    f"Leverage: {leverage}x"
                )
                
            except Exception as e:
                logger.error(f"Error detecting {symbol}: {e}")
                # Default to spot
                self.product_info[symbol] = {
                    'is_futures': False,
                    'contract_size': 1.0,
                    'leverage': 1.0,
                    'product_type': 'SPOT'
                }
    
    def get_balance(self) -> float:
        """Get available balance (spot + futures)."""
        try:
            accounts_response = self.client.get_accounts()
            accounts = accounts_response.accounts if hasattr(accounts_response, 'accounts') else []
            
            total = 0.0
            
            for account in accounts:
                currency = getattr(account, 'currency', '')
                
                if currency in ['USD', 'USDC']:
                    # Get available balance
                    if hasattr(account, 'available_balance'):
                        avail_obj = account.available_balance
                        
                        # Handle different response formats
                        if isinstance(avail_obj, dict):
                            avail = float(avail_obj.get('value', 0))
                        elif hasattr(avail_obj, 'value'):
                            avail = float(avail_obj.value)
                        elif avail_obj:
                            avail = float(avail_obj)
                        else:
                            avail = 0.0
                    else:
                        avail = 0.0
                    
                    # Get hold balance (locked in orders)
                    if hasattr(account, 'hold'):
                        hold_obj = account.hold
                        
                        # Handle different response formats
                        if isinstance(hold_obj, dict):
                            hold = float(hold_obj.get('value', 0))
                        elif hasattr(hold_obj, 'value'):
                            hold = float(hold_obj.value)
                        elif hold_obj:
                            hold = float(hold_obj)
                        else:
                            hold = 0.0
                    else:
                        hold = 0.0
                    
                    total += avail + hold
            
            return total
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def get_futures_balance(self) -> float:
        """Get futures trading balance (buying power)."""
        try:
            response = self.client.get_futures_balance_summary()
            
            # Response is a custom object - try multiple access methods
            summary = None
            
            # Method 1: Direct attribute access
            if hasattr(response, 'balance_summary'):
                summary = response.balance_summary
            
            # Method 2: Dict-like access (some SDK objects support this)
            if summary is None:
                try:
                    summary = response['balance_summary']
                except (TypeError, KeyError):
                    pass
            
            # Method 3: Convert to dict if possible
            if summary is None:
                try:
                    if hasattr(response, 'to_dict'):
                        resp_dict = response.to_dict()
                    elif hasattr(response, '__dict__'):
                        resp_dict = vars(response)
                    else:
                        # Try eval if it prints as dict
                        resp_dict = eval(str(response))
                    summary = resp_dict.get('balance_summary', {})
                except:
                    pass
            
            if summary is None:
                logger.warning(f"Could not extract balance_summary from: {type(response)}")
                return 0.0
            
            # Now extract futures_buying_power from summary
            # Summary could be dict or object
            for key in ['futures_buying_power', 'available_margin', 'total_usd_balance']:
                val = None
                
                # Try dict access
                if isinstance(summary, dict):
                    val = summary.get(key, {})
                else:
                    # Try attribute access
                    if hasattr(summary, key):
                        val = getattr(summary, key)
                    # Try dict-like access
                    elif hasattr(summary, '__getitem__'):
                        try:
                            val = summary[key]
                        except (KeyError, TypeError):
                            pass
                
                if val:
                    # Extract numeric value
                    if isinstance(val, dict):
                        if 'value' in val:
                            return float(val['value'])
                    elif hasattr(val, 'value'):
                        return float(val.value)
                    elif isinstance(val, (int, float, str)):
                        try:
                            return float(val)
                        except:
                            pass
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Futures balance not available: {e}")
            return 0.0
    
    def get_trading_balance(self) -> float:
        """Get the appropriate balance for trading (futures if trading futures, else spot)."""
        # Check if we're trading futures
        is_futures = any(
            self.product_info.get(s, {}).get('is_futures', False) 
            for s in self.symbols
        )
        
        if is_futures:
            futures_bal = self.get_futures_balance()
            if futures_bal > 0:
                return futures_bal
        
        # Fallback to spot balance
        return self.get_balance()
    
    def show_account_info(self):
        """Display comprehensive account information."""
        logger.info("="*60)
        logger.info("ACCOUNT OVERVIEW")
        logger.info("="*60)
        
        try:
            accounts_response = self.client.get_accounts()
            accounts = accounts_response.accounts if hasattr(accounts_response, 'accounts') else []
            
            total_usd = 0.0
            total_crypto_value = 0.0
            spot_positions = []
            
            # Process accounts
            for account in accounts:
                currency = getattr(account, 'currency', '')
                
                # Get balance
                if hasattr(account, 'available_balance'):
                    avail_obj = account.available_balance
                    if isinstance(avail_obj, dict):
                        avail = float(avail_obj.get('value', 0))
                    elif hasattr(avail_obj, 'value'):
                        avail = float(avail_obj.value)
                    else:
                        avail = 0.0
                else:
                    avail = 0.0
                
                if hasattr(account, 'hold'):
                    hold_obj = account.hold
                    if isinstance(hold_obj, dict):
                        hold = float(hold_obj.get('value', 0))
                    elif hasattr(hold_obj, 'value'):
                        hold = float(hold_obj.value)
                    else:
                        hold = 0.0
                else:
                    hold = 0.0
                
                total_balance = avail + hold
                
                if total_balance > 0:
                    if currency in ['USD', 'USDC']:
                        total_usd += total_balance
                        logger.info(f"  Ã°Å¸â€™Âµ {currency}: ${total_balance:,.2f} (available: ${avail:,.2f})")
                    else:
                        # Try to get crypto value
                        try:
                            product_id = f"{currency}-USD"
                            product = self.client.get_product(product_id=product_id)
                            price = float(product.price) if hasattr(product, 'price') else 0
                            usd_value = total_balance * price
                            total_crypto_value += usd_value
                            spot_positions.append((currency, total_balance, price, usd_value))
                        except:
                            pass
            
            # Show crypto positions if any
            if spot_positions:
                logger.info("")
                logger.info("Spot Crypto Holdings:")
                for currency, amount, price, value in spot_positions:
                    logger.info(f"  Ã°Å¸Âªâ„¢ {currency}: {amount:.8f} @ ${price:,.2f} = ${value:,.2f}")
            
            # Check for futures positions
            logger.info("")
            try:
                positions_response = self.client.list_perps_positions()
                if hasattr(positions_response, 'positions') and positions_response.positions:
                    logger.info("Futures Positions:")
                    for pos in positions_response.positions:
                        product_id = getattr(pos, 'product_id', 'N/A')
                        side = getattr(pos, 'side', 'N/A')
                        size = getattr(pos, 'net_size', 0)
                        unrealized_pnl = getattr(pos, 'unrealized_pnl', 0)
                        logger.info(f"  {product_id} {side} {size} contracts | P&L: ${float(unrealized_pnl):+,.2f}")
                else:
                    logger.info("Futures Positions: None")
            except:
                logger.info("Futures Positions: None (not enabled)")
            
            # Summary
            logger.info("")
            logger.info("="*60)
            futures_bal = self.get_futures_balance()
            logger.info(f"Spot Balance (USD/USDC): ${total_usd:,.2f}")
            logger.info(f"Futures Buying Power: ${futures_bal:,.2f}")
            if total_crypto_value > 0:
                logger.info(f"Crypto Holdings (not trading): ${total_crypto_value:,.2f}")
            trading_bal = self.get_trading_balance()
            logger.info(f"Effective Trading Balance: ${trading_bal:,.2f}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error displaying account info: {e}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        try:
            product = self.client.get_product(product_id=symbol)
            price = float(product.price) if hasattr(product, 'price') else None
            return price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get historical candles."""
        try:
            gran_map = {
                '1m': 'ONE_MINUTE',
                '5m': 'FIVE_MINUTE', 
                '15m': 'FIFTEEN_MINUTE',
                '30m': 'THIRTY_MINUTE',
                '1h': 'ONE_HOUR',
                '4h': 'FOUR_HOUR',
                '1d': 'ONE_DAY'
            }
            
            gran = gran_map.get(timeframe, 'ONE_HOUR')
            
            # Calculate bars needed (stay under 350 limit)
            bars_needed = min(self.lookback_bars, 300)
            
            # Calculate time range - CRITICAL: Stay under 350 candles!
            if timeframe == '1m':
                # 1min: 350 candles = 5.8 hours, use 5 hours to be safe
                hours = 5
            elif timeframe == '5m':
                # 5min: 350 candles = 29 hours, use 24 hours
                hours = 24
            elif timeframe == '15m':
                # 15min: 350 candles = 87 hours = 3.6 days, use 3 days
                hours = 3 * 24
            elif timeframe == '30m':
                # 30min: 350 candles = 175 hours = 7.3 days, use 7 days
                hours = 7 * 24
            elif timeframe == '1h':
                # 1h: 350 candles = 14.5 days, use 12 days
                hours = 12 * 24
            elif timeframe == '4h':
                # 4h: 350 candles = 58 days, use 50 days
                hours = 50 * 24
            elif timeframe == '1d':
                # 1d: 350 candles = 350 days, use 300 days
                hours = 300 * 24
            else:
                # Default: 7 days
                hours = 7 * 24
            
            end_ts = int(datetime.now().timestamp())
            start_ts = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            response = self.client.get_candles(
                product_id=symbol,
                start=start_ts,
                end=end_ts,
                granularity=gran
            )
            
            candles = response.candles if hasattr(response, 'candles') else []
            
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
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, symbol: str, signal_position_pct: float, balance: float, price: float) -> float:
        """
        Calculate position size in proper units (fractional for spot, whole contracts for futures).
        
        Returns:
            For spot: fractional amount (e.g., 0.032 ETH)
            For futures: whole number of contracts (e.g., 1, 2, 3)
        """
        info = self.product_info.get(symbol, {})
        is_futures = info.get('is_futures', False)
        
        # Calculate dollar value to allocate
        position_value = balance * signal_position_pct
        
        if is_futures:
            # FUTURES: Calculate whole contracts based on margin requirement
            contract_size = info.get('contract_size', 0.1)
            leverage = info.get('leverage', 4.0)
            
            # Notional value per contract
            notional_per_contract = price * contract_size
            
            # Margin required per contract
            margin_per_contract = notional_per_contract / leverage
            
            # How many contracts can we afford?
            max_contracts = position_value / margin_per_contract
            
            # CRITICAL: Round DOWN to whole number (Coinbase requirement)
            contracts = int(max_contracts)
            
            # Ensure at least 1 contract if we have enough
            if contracts < 1 and position_value >= margin_per_contract:
                contracts = 1
            
            logger.info(
                f"{symbol} sizing: ${position_value:.2f} / "
                f"${margin_per_contract:.2f} per contract = {contracts} contracts"
            )
            
            return float(contracts)
        
        else:
            # SPOT: Calculate fractional amount
            quantity = position_value / price
            
            logger.info(
                f"{symbol} sizing: ${position_value:.2f} / "
                f"${price:.2f} = {quantity:.6f}"
            )
            
            return quantity
    
    def place_order(self, symbol: str, side: str, size: float) -> Optional[str]:
        """
        Place market order.
        
        Args:
            symbol: Product ID
            side: 'buy' or 'sell'
            size: For spot: fractional amount, For futures: whole contracts
        """
        try:
            info = self.product_info.get(symbol, {})
            is_futures = info.get('is_futures', False)
            
            # For futures, ensure whole number
            if is_futures:
                size = int(size)
                if size < 1:
                    logger.warning(f"Cannot place order: size={size} < 1 contract")
                    return None
            
            logger.info(f"Placing {side.upper()} order: {symbol} size={size}")
            
            client_order_id = f"{symbol.replace('/', '').replace('-', '')}-{int(time.time())}"
            
            if side.lower() == 'buy':
                order = self.client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=symbol,
                    base_size=str(size)
                )
            else:
                order = self.client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=symbol,
                    base_size=str(size)
                )
            
            # Handle response - can be object with dict-like attributes
            # Extract success and success_response handling both formats
            success = None
            success_response = None
            error_response = None
            
            if hasattr(order, 'success'):
                success = order.success
            if hasattr(order, 'success_response'):
                success_response = order.success_response
            if hasattr(order, 'error_response'):
                error_response = order.error_response
            
            # Handle success
            if success:
                order_id = None
                
                # success_response could be dict or object
                if isinstance(success_response, dict):
                    order_id = success_response.get('order_id')
                elif hasattr(success_response, 'order_id'):
                    order_id = success_response.order_id
                
                if order_id:
                    logger.info(f"Order placed: {order_id}")
                    return order_id
                else:
                    logger.info(f"Order placed (no order_id in response)")
                    return "success"
            
            # Handle failure
            if success is False:
                error_msg = "Unknown error"
                preview_reason = ""
                
                if error_response:
                    if isinstance(error_response, dict):
                        error_msg = error_response.get('error', 'Unknown')
                        preview_reason = error_response.get('preview_failure_reason', '')
                    else:
                        error_msg = getattr(error_response, 'error', 'Unknown')
                        preview_reason = getattr(error_response, 'preview_failure_reason', '')
                
                if 'INSUFFICIENT_FUNDS' in str(preview_reason):
                    logger.error(f"INSUFFICIENT FUNDS FOR FUTURES!")
                    logger.error(f"   Transfer funds: Coinbase -> Transfer -> Spot to Futures")
                else:
                    logger.error(f"Order failed: {error_msg} - {preview_reason}")
                
                return None
            
            # Fallback: try to get order_id directly
            if hasattr(order, 'order_id'):
                order_id = order.order_id
                logger.info(f"Order placed: {order_id}")
                return order_id
            
            # If we get here, unclear if it succeeded
            logger.warning(f"Order response unclear: {order}")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def check_position(self, symbol: str) -> Optional[dict]:
        """Check if we have an open position (via open orders)."""
        try:
            # Check for open orders (bracket orders)
            orders = self.client.get_orders(symbol=symbol)
            
            if hasattr(orders, 'orders'):
                open_orders = [o for o in orders.orders if hasattr(o, 'status') and o.status == 'OPEN']
                
                if open_orders:
                    # We have bracket orders = position is open
                    return self.positions.get(symbol)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking position for {symbol}: {e}")
            return None
    
    def enter_position(self, symbol: str, signal):
        """Enter new position."""
        balance = self.get_trading_balance()
        price = self.get_price(symbol)
        
        if not price or balance < 10:
            logger.warning(f"Cannot enter {symbol}: price={price}, balance={balance}")
            return
        
        # Calculate size (handles spot vs futures)
        size = self.calculate_position_size(symbol, signal.position_size, balance, price)
        
        if size <= 0:
            logger.warning(f"Cannot enter {symbol}: calculated size={size}")
            return
        
        # Place order
        side = 'buy' if signal.direction == 'long' else 'sell'
        order_id = self.place_order(symbol, side, size)
        
        if not order_id:
            logger.error(f"Failed to enter {signal.direction.upper()} {symbol}")
            return
        
        # Only create position if order succeeded
        self.positions[symbol] = {
            'side': signal.direction.upper(),
            'entry_price': price,
            'size': size,
            'stop': signal.stop_loss,
            'target': signal.take_profit,
            'entry_time': datetime.now(),
            'order_id': order_id
        }
        
        info = self.product_info.get(symbol, {})
        is_futures = info.get('is_futures', False)
        size_str = f"{int(size)} contracts" if is_futures else f"{size:.6f}"
        
        logger.info(f"ENTERED {self.positions[symbol]['side']} {symbol}")
        logger.info(f"   Size: {size_str}")
        logger.info(f"   Entry: ${price:.2f}")
        logger.info(f"   Stop: ${signal.stop_loss:.2f}")
        logger.info(f"   Target: ${signal.take_profit:.2f}")
        
        if is_futures:
            logger.warning(f"Futures don't support bracket orders on Coinbase")
            logger.warning(f"Bot will MANUALLY monitor price and exit at stop/target")
            logger.warning(f"Checking every {self.check_interval} seconds")
    
    def exit_position(self, symbol: str, reason: str):
        """Exit position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        price = self.get_price(symbol)
        
        if not price:
            return
        
        # Place opposite order
        side = 'sell' if pos['side'] == 'LONG' else 'buy'
        order_id = self.place_order(symbol, side, pos['size'])
        
        if order_id:
            # Calculate P&L
            info = self.product_info.get(symbol, {})
            is_futures = info.get('is_futures', False)
            
            if is_futures:
                contract_size = info.get('contract_size', 0.1)
                if pos['side'] == 'LONG':
                    pnl_per_contract = (price - pos['entry_price']) * contract_size
                else:
                    pnl_per_contract = (pos['entry_price'] - price) * contract_size
                total_pnl = pnl_per_contract * pos['size']
            else:
                if pos['side'] == 'LONG':
                    total_pnl = (price - pos['entry_price']) * pos['size']
                else:
                    total_pnl = (pos['entry_price'] - price) * pos['size']
            
            self.daily_pnl += total_pnl
            self.signal_generator.record_trade_result(total_pnl)
            
            logger.info(f"EXITED {pos['side']} {symbol}")
            logger.info(f"   Entry: ${pos['entry_price']:.2f}")
            logger.info(f"   Exit: ${price:.2f}")
            logger.info(f"   P&L: ${total_pnl:+.2f}")
            logger.info(f"   Reason: {reason}")
            
            del self.positions[symbol]
    
    def check_exits(self):
        """Check exit conditions for all positions."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = self.get_price(symbol)
            
            if not price:
                continue
            
            # Show current status vs targets
            if pos['side'] == 'LONG':
                distance_to_stop = ((price - pos['stop']) / price) * 100
                distance_to_target = ((pos['target'] - price) / price) * 100
                logger.debug(
                    f"{symbol}: ${price:.2f} | "
                    f"Stop ${pos['stop']:.2f} ({distance_to_stop:+.2f}%) | "
                    f"Target ${pos['target']:.2f} ({distance_to_target:+.2f}%)"
                )
            else:
                distance_to_stop = ((pos['stop'] - price) / price) * 100
                distance_to_target = ((price - pos['target']) / price) * 100
                logger.debug(
                    f"{symbol}: ${price:.2f} | "
                    f"Stop ${pos['stop']:.2f} ({distance_to_stop:+.2f}%) | "
                    f"Target ${pos['target']:.2f} ({distance_to_target:+.2f}%)"
                )
            
            # Time exit
            if self.max_hold_days is not None:
                hold_time = datetime.now() - pos['entry_time']
                if hold_time.total_seconds() >= self.max_hold_days * 24 * 3600:
                    self.exit_position(symbol, "Time exit")
                    continue
            
            # Stop loss
            if pos['side'] == 'LONG' and price <= pos['stop']:
                logger.warning(f"STOP LOSS HIT: ${price:.2f} <= ${pos['stop']:.2f}")
                self.exit_position(symbol, "Stop loss")
                continue
            elif pos['side'] == 'SHORT' and price >= pos['stop']:
                logger.warning(f"STOP LOSS HIT: ${price:.2f} >= ${pos['stop']:.2f}")
                self.exit_position(symbol, "Stop loss")
                continue
            
            # Take profit
            if pos['side'] == 'LONG' and price >= pos['target']:
                logger.info(f"TAKE PROFIT HIT: ${price:.2f} >= ${pos['target']:.2f}")
                self.exit_position(symbol, "Take profit")
                continue
            elif pos['side'] == 'SHORT' and price <= pos['target']:
                logger.info(f"TAKE PROFIT HIT: ${price:.2f} <= ${pos['target']:.2f}")
                self.exit_position(symbol, "Take profit")
                continue
    
    def _is_candle_complete(self, candle_time: datetime, timeframe: str) -> bool:
        """
        Check if a candle is complete (closed).
        
        A candle is complete if the current time is past its close time.
        We add a small buffer (1 minute) to account for API delays.
        """
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        duration = timedelta(minutes=tf_minutes.get(timeframe, 60))
        candle_close_time = candle_time + duration
        
        # Add 1 minute buffer for API delays
        buffer = timedelta(minutes=1)
        
        now = datetime.now(pytz.UTC)
        return now >= (candle_close_time + buffer)
    
    def check_entries(self):
        """
        Check for entry signals.
        
        Two-phase approach (matches backtest):
        1. Detect new setups at 4h candle close (using complete data)
        2. Check entry on active setups every 60 seconds
        """
        if len(self.positions) >= self.max_positions:
            return
        
        # Daily loss limit
        balance = self.get_trading_balance()
        if self.daily_pnl < -self.max_daily_loss * balance:
            logger.warning(f"Daily loss limit: ${self.daily_pnl:.2f}")
            return
        
        for trade_symbol in self.symbols:
            if trade_symbol in self.positions:
                continue
            
            # Get the signal symbol (might be different from trade symbol)
            signal_symbol = self.symbol_mapping.get(trade_symbol, trade_symbol)
            
            # Get candles from SIGNAL symbol (e.g., ETH-USD spot)
            signal_df = self.get_candles(signal_symbol, self.signal_timeframe)
            atr_df = self.get_candles(signal_symbol, self.atr_timeframe)
            
            if signal_df.empty or atr_df.empty:
                continue
            
            # PHASE 1: Check for NEW squeeze breakouts (only at candle close)
            # Use completed candles for squeeze detection
            latest = signal_df.index[-1]
            
            if not self._is_candle_complete(latest, self.signal_timeframe):
                # Current candle still forming - skip NEW setup detection
                if len(signal_df) < 2:
                    continue
                # Use the previous (completed) candle for NEW setup detection
                signal_df_for_detection = signal_df.iloc[:-1]
                latest_completed = signal_df_for_detection.index[-1]
            else:
                signal_df_for_detection = signal_df
                latest_completed = latest
            
            # Also check ATR data
            if not atr_df.empty:
                latest_atr = atr_df.index[-1]
                if not self._is_candle_complete(latest_atr, self.atr_timeframe):
                    if len(atr_df) >= 2:
                        atr_df = atr_df.iloc[:-1]
            
            # Check if this is a NEW candle (for detecting new setups)
            candle_key = f"{signal_symbol}_{self.signal_timeframe}"
            is_new_candle = self.last_candles.get(candle_key) != latest_completed
            
            if is_new_candle:
                self.last_candles[candle_key] = latest_completed
                
                # Log new candle
                logger.info(f"New {self.signal_timeframe} candle CLOSED at {latest_completed}")
                
                # Calculate indicators for diagnostics
                df = self.analyzer.calculate_indicators(signal_df_for_detection.tail(100))
                
                if len(df) >= 2:
                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    # Show current state
                    squeeze_state = "IN SQUEEZE" if current['Squeeze'] else "NOT SQUEEZED"
                    logger.info(f"   State: {squeeze_state} | Duration: {current['Squeeze_Duration']:.0f} bars")
                    logger.info(f"   BB Width: {current['BB_Width']:.4f} | Momentum: {current['Momentum_Norm']:+.2f}")
                    logger.info(f"   RSI: {current['RSI']:.1f} | Volume Ratio: {current['Volume_Ratio']:.2f}")
                    
                    # Check if squeeze released
                    if prev['Squeeze'] and not current['Squeeze']:
                        logger.info(f"   Ã°Å¸Å½Â¯ SQUEEZE RELEASED after {prev['Squeeze_Duration']:.0f} bars!")
                    elif current['Squeeze']:
                        logger.info(f"   Ã¢ÂÂ³ Waiting for squeeze release...")
            
            # PHASE 2: Check entry on ACTIVE setups (every cycle, using current data)
            # This allows entry at any time within 32 hours of setup (like backtest)
            self.signal_generator.set_signal_data({signal_symbol: signal_df_for_detection})
            self.signal_generator.set_atr_data({signal_symbol: atr_df})
            
            signal = self.signal_generator.generate_signal(
                signal_df,  # Use ALL data (including forming candle) for price check
                signal_symbol,
                datetime.now(pytz.UTC)
            )
            
            # Check signal and execute trade if valid
            if signal.direction != 'neutral':
                # Skip shorts if long_only mode
                if self.long_only and signal.direction == 'short':
                    logger.info(f"   Skipping SHORT signal (long_only mode)")
                    continue
                    
                logger.info(f"   ✓ SIGNAL: {signal.direction.upper()}")
                logger.info(f"   Entry: ${signal.entry_price:,.2f} | Stop: ${signal.stop_loss:,.2f} | Target: ${signal.take_profit:,.2f}")
                logger.info(f"   Size: {signal.position_size:.0%} | Score: {signal.score:.2f}")
                logger.info(f"   Reasons: {' '.join(signal.reasons)}")
                
                # Execute on TRADE symbol (e.g., ETP-20DEC30-CDE futures)
                signal.symbol = trade_symbol
                self.enter_position(trade_symbol, signal)
            else:
                # Only explain why no signal on NEW candles (avoid spam)
                if is_new_candle and len(df) >= 2:
                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    if current['Squeeze']:
                        logger.info(f"   No signal: Still in squeeze")
                    elif not prev['Squeeze']:
                        logger.info(f"   No signal: No squeeze to break out from")
                    elif current['Volume_Ratio'] < self.signal_generator.min_volume_ratio:
                        logger.info(f"   No signal: Volume too low ({current['Volume_Ratio']:.2f} < {self.signal_generator.min_volume_ratio})")
                    elif current['RSI'] > self.signal_generator.rsi_overbought:
                        logger.info(f"   No signal: RSI overbought ({current['RSI']:.1f} > {self.signal_generator.rsi_overbought})")
                    elif current['RSI'] < self.signal_generator.rsi_oversold:
                        logger.info(f"   No signal: RSI oversold ({current['RSI']:.1f} < {self.signal_generator.rsi_oversold})")
                    else:
                        logger.info(f"   No signal: Conditions not met")
    
    def reset_daily(self):
        """Reset daily tracking."""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            logger.info(f"Daily reset | Yesterday P&L: ${self.daily_pnl:+.2f}")
            self.daily_pnl = 0.0
            self.last_daily_reset = today
    
    def run(self):
        """Start trading bot."""
        logger.info("="*60)
        logger.info("COINBASE LIVE TRADER")
        logger.info("="*60)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Signal TF: {self.signal_timeframe}")
        logger.info(f"Max positions: {self.max_positions}")
        logger.info("")
        
        # Show comprehensive account info
        self.show_account_info()
        logger.info("")
        
        self.running = True
        
        try:
            while self.running:
                try:
                    self.reset_daily()
                    self.check_exits()
                    self.check_entries()
                    
                    logger.info(
                        f"Tick | Positions: {len(self.positions)}/{self.max_positions} "
                        f"({'MONITORING' if self.positions else 'Waiting'}) | "
                        f"Daily P&L: ${self.daily_pnl:+.2f}"
                    )
                    
                    time.sleep(self.check_interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
    
    def stop(self):
        """Stop the bot."""
        self.running = False