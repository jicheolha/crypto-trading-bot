"""
Unified Data Manager for Crypto Trading Bot

Backtesting: CryptoDataDownload (download once, cache forever)
Live Trading: ccxt (real-time data from any exchange)
"""
import os
import pickle
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import ccxt
import logging

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================================
# CRYPTODATADOWNLOAD - FOR BACKTESTING
# ============================================================================

class CryptoDataDownloader:
    """
    Download historical data from CryptoDataDownload.com
    
    Pros:
    - FREE
    - 7+ years of data for major pairs
    - Multiple timeframes
    - Clean, reliable data
    """
    
    # Symbol mapping: Our format -> CDD format
    SYMBOL_MAP = {
        'BTC/USD': 'BTCUSD',
        'BTC/USDT': 'BTCUSDT',
        'ETH/USD': 'ETHUSD',
        'ETH/USDT': 'ETHUSDT',
        'DOGE/USD': 'DOGEUSD',
        'DOGE/USDT': 'DOGEUSDT',
        'AAVE/USD': 'AAVEUSD',
        'AAVE/USDT': 'AAVEUSDT',
        'SOL/USD': 'SOLUSD',
        'SOL/USDT': 'SOLUSDT',
    }
    
    # Timeframe mapping
    TIMEFRAME_MAP = {
        '1min': 'minute',
        '1m': 'minute',
        '1h': 'hourly',
        '1d': 'daily',
    }
    
    # Available exchanges
    EXCHANGES = {
        'binance': 'Binance',
        'kraken': 'Kraken',
        'coinbase': 'Coinbase',
        'gemini': 'Gemini',
    }
    
    @staticmethod
    def download(
        symbol: str,
        timeframe: str = '1h',
        exchange: str = 'binance',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download historical data from CryptoDataDownload
        
        Args:
            symbol: e.g., 'BTC/USD', 'ETH/USDT'
            timeframe: '1min', '1h', '1d'
            exchange: 'binance', 'kraken', 'coinbase'
            use_cache: If True, save to disk and reuse
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_path = os.path.join(
            CACHE_DIR,
            f"{exchange}_{symbol.replace('/', '')}_{timeframe}_cdd.pkl"
        )
        
        if use_cache and os.path.exists(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                print(f"  Loaded {symbol} {timeframe} from cache ({len(df)} bars, "
                      f"{df.index[0].date()} to {df.index[-1].date()})")
                return df
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Download from CryptoDataDownload
        print(f"  Downloading {symbol} {timeframe} from CryptoDataDownload...")
        
        try:
            # Map symbols
            cdd_symbol = CryptoDataDownloader.SYMBOL_MAP.get(symbol)
            if not cdd_symbol:
                raise ValueError(f"Symbol {symbol} not mapped. Add to SYMBOL_MAP.")
            
            # Map timeframe
            cdd_tf = CryptoDataDownloader.TIMEFRAME_MAP.get(timeframe, 'hourly')
            
            # Map exchange
            cdd_exchange = CryptoDataDownloader.EXCHANGES.get(exchange, 'Binance')
            
            # Build URL
            # Format: https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv
            # Note: They use different naming, need to check their actual files
            
            # Try multiple URL patterns (CDD has inconsistent naming)
            urls_to_try = [
                f"https://www.cryptodatadownload.com/cdd/{cdd_exchange}_{cdd_symbol}_{timeframe}.csv",
                f"https://www.cryptodatadownload.com/cdd/{cdd_exchange}_{cdd_symbol}_{cdd_tf}.csv",
                f"https://www.cryptodatadownload.com/cdd/{exchange}_{cdd_symbol}_{timeframe}.csv",
            ]
            
            df = None
            for url in urls_to_try:
                try:
                    print(f"    Trying: {url}")
                    df = CryptoDataDownloader._fetch_csv(url)
                    if df is not None and len(df) > 0:
                        break
                except:
                    continue
            
            if df is None or len(df) == 0:
                raise Exception(f"Could not download data from any URL. "
                              f"Visit https://www.cryptodatadownload.com/data/{exchange}/ "
                              f"to check available files manually.")
            
            # Save to cache
            if use_cache:
                df.to_pickle(cache_path)
                print(f"    Cached to {cache_path}")
            
            print(f"  Downloaded {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            print(f"  âœ— Download failed: {e}")
            print(f"\n  MANUAL DOWNLOAD INSTRUCTIONS:")
            print(f"  1. Go to: https://www.cryptodatadownload.com/data/{exchange}/")
            print(f"  2. Find and download: {symbol} {timeframe} CSV")
            print(f"  3. Save to: {cache_path}")
            print(f"  4. Run script again\n")
            return pd.DataFrame()
    
    @staticmethod
    def _fetch_csv(url: str) -> pd.DataFrame:
        """Fetch and parse CSV from URL"""
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        # CryptoDataDownload CSVs have a header row to skip
        from io import StringIO
        
        # Try reading with different formats
        try:
            # Format 1: Standard CSV with timestamp column
            df = pd.read_csv(StringIO(response.text), skiprows=1)
            
            # Identify timestamp column
            time_col = None
            for col in ['unix', 'timestamp', 'date', 'time']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col is None:
                # Try first column
                time_col = df.columns[0]
            
            # Parse timestamp
            if 'unix' in time_col.lower():
                df['timestamp'] = pd.to_datetime(df[time_col], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df[time_col])
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df = df.rename(columns={
                'vol': 'volume',
                'vol_base': 'volume',
                'volume_base': 'volume',
            })
            
            # Select relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Remove any duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            return df
            
        except Exception as e:
            raise Exception(f"CSV parse error: {e}")


# ============================================================================
# CCXT - FOR LIVE TRADING
# ============================================================================

class CCXTDataFetcher:
    """
    Fetch real-time data via ccxt (for live trading)
    
    Supports: Kraken, Binance, Coinbase, and 100+ other exchanges
    """
    
    def __init__(
        self,
        exchange_name: str = 'kraken',
        api_key: str = None,
        secret: str = None,
        testnet: bool = True
    ):
        """
        Initialize ccxt exchange
        
        Args:
            exchange_name: 'kraken', 'binance', 'coinbase', etc.
            api_key: API key (optional for fetching data)
            secret: API secret (optional for fetching data)
            testnet: Use testnet/sandbox (always start with True!)
        """
        self.exchange_name = exchange_name
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        
        config = {
            'enableRateLimit': True,
        }
        
        if api_key and secret:
            config['apiKey'] = api_key
            config['secret'] = secret
        
        self.exchange = exchange_class(config)
        
        # Enable sandbox/testnet if supported
        if testnet and hasattr(self.exchange, 'set_sandbox_mode'):
            try:
                self.exchange.set_sandbox_mode(True)
                print(f"  âœ“ {exchange_name} TESTNET mode enabled")
            except:
                print(f"  âš  {exchange_name} doesn't support testnet")
        elif testnet:
            print(f"  âš  {exchange_name} doesn't support set_sandbox_mode")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV data
        
        Args:
            symbol: e.g., 'BTC/USDT', 'ETH/USD'
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from {self.exchange_name}: {e}")
            return pd.DataFrame()
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker (price, volume, etc)"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def fetch_balance(self) -> Dict:
        """Get account balance"""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float = None,
        params: dict = None
    ):
        """
        Create an order
        
        Args:
            symbol: e.g., 'BTC/USDT'
            order_type: 'market', 'limit'
            side: 'buy', 'sell'
            amount: Order size
            price: Limit price (for limit orders)
            params: Exchange-specific parameters
        """
        try:
            return self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    def fetch_open_orders(self, symbol: str = None) -> List:
        """Get open orders"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str, symbol: str):
        """Cancel an order"""
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise


# ============================================================================
# UNIFIED DATA MANAGER
# ============================================================================

class UnifiedDataManager:
    """
    Unified interface for both backtesting and live trading data
    """
    
    @staticmethod
    def load_backtest_data(
        symbols: List[str],
        timeframe: str = '1h',
        exchange: str = 'binance',
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load backtesting data (from CryptoDataDownload)
        
        Args:
            symbols: List of symbols, e.g., ['BTC/USD', 'ETH/USD']
            timeframe: '1min', '1h', '1d'
            exchange: 'binance', 'kraken', etc.
            use_cache: Cache downloaded data
            
        Returns:
            Dict of {symbol: DataFrame}
        """
        print(f"\nLoading backtest data from CryptoDataDownload...")
        print(f"Exchange: {exchange}, Timeframe: {timeframe}")
        
        data = {}
        
        for symbol in symbols:
            df = CryptoDataDownloader.download(
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
                use_cache=use_cache
            )
            
            if not df.empty:
                data[symbol] = df
            else:
                print(f"  âœ— Failed to load {symbol}")
        
        print(f"\nLoaded {len(data)}/{len(symbols)} symbols")
        return data
    
    @staticmethod
    def load_live_data(
        exchange_name: str,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 100,
        api_key: str = None,
        secret: str = None,
        testnet: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load live data (from ccxt)
        
        Args:
            exchange_name: 'kraken', 'binance', etc.
            symbols: List of symbols
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Number of candles
            api_key: API key (optional for data)
            secret: API secret (optional for data)
            testnet: Use testnet
            
        Returns:
            Dict of {symbol: DataFrame}
        """
        print(f"\nFetching live data from {exchange_name}...")
        
        fetcher = CCXTDataFetcher(
            exchange_name=exchange_name,
            api_key=api_key,
            secret=secret,
            testnet=testnet
        )
        
        data = {}
        
        for symbol in symbols:
            df = fetcher.fetch_ohlcv(symbol, timeframe, limit)
            
            if not df.empty:
                data[symbol] = df
                print(f"  âœ“ {symbol}: {len(df)} bars")
            else:
                print(f"  âœ— Failed to fetch {symbol}")
        
        return data


# ============================================================================
# SYMBOL MAPPING BETWEEN EXCHANGES
# ============================================================================

class SymbolMapper:
    """
    Map symbols between exchanges and data sources
    """
    
    # Map: Alpaca -> Kraken
    ALPACA_TO_KRAKEN = {
        'BTC/USD': 'BTC/USDT',
        'ETH/USD': 'ETH/USDT',
        'DOGE/USD': 'DOGE/USDT',
        'AAVE/USD': 'AAVE/USDT',
        'SOL/USD': 'SOL/USDT',
    }
    
    # Map: Alpaca -> Binance
    ALPACA_TO_BINANCE = {
        'BTC/USD': 'BTC/USDT',
        'ETH/USD': 'ETH/USDT',
        'DOGE/USD': 'DOGE/USDT',
        'AAVE/USD': 'AAVE/USDT',
        'SOL/USD': 'SOL/USDT',
    }
    
    @staticmethod
    def map_symbols(symbols: List[str], from_exchange: str, to_exchange: str) -> List[str]:
        """Map symbols between exchanges"""
        
        mapping = None
        
        if from_exchange == 'alpaca' and to_exchange == 'kraken':
            mapping = SymbolMapper.ALPACA_TO_KRAKEN
        elif from_exchange == 'alpaca' and to_exchange == 'binance':
            mapping = SymbolMapper.ALPACA_TO_BINANCE
        
        if mapping:
            return [mapping.get(s, s) for s in symbols]
        
        return symbols


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def clear_cache():
    """Delete all cached data"""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")
    else:
        print("No cache to clear")


def list_cached_files():
    """List all cached data files"""
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    if not files:
        print("No cached files")
        return
    
    print(f"\nCached files in {CACHE_DIR}:")
    for f in sorted(files):
        path = os.path.join(CACHE_DIR, f)
        size = os.path.getsize(path) / 1024 / 1024  # MB
        modified = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"  {f:<50} {size:>6.1f} MB  {modified.strftime('%Y-%m-%d %H:%M')}")