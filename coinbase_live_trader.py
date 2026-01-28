"""
Coinbase Live Trader - Supports SPOT and FUTURES

Features:
- Complete trade journal with persistence
- P&L tracking, drawdown monitoring
- Win rate and profit factor tracking
- Rolling performance metrics
- Drift detection vs backtest

Handles:
- Spot trading (fractional amounts)
- Futures trading (whole contracts only)
- Automatic leverage detection
- Margin calculation
"""
import os
import csv
import json
import time
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from coinbase.rest import RESTClient

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator
from utils import Colors, colored

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TRADE RECORD & STATISTICS (Professional Monitoring)
# =============================================================================

@dataclass
class TradeRecord:
    """Complete record of a closed trade."""
    trade_id: int
    symbol: str
    direction: str
    
    # Timing
    entry_time: datetime
    exit_time: datetime
    duration_hours: float
    
    # Prices
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    
    # Size
    size: float
    notional: float
    position_pct: float
    
    # P&L
    gross_pnl: float
    pnl_pct: float
    
    # Exit
    exit_reason: str
    
    # Signal info
    signal_score: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)


class TradingStats:
    """
    Professional trading statistics tracker.
    Tracks everything a hedge fund risk manager would want.
    """
    
    # Alert thresholds
    THRESHOLDS = {
        'daily_loss_warning': -0.02,
        'daily_loss_critical': -0.03,
        'drawdown_warning': -0.05,
        'drawdown_critical': -0.10,
        'consecutive_losses_warning': 4,
        'slippage_warning': 0.002,
    }
    
    def __init__(
        self,
        initial_balance: float,
        data_dir: str = None,
        backtest_stats: Dict = None,
    ):
        # Balance tracking
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Trade history
        self.completed_trades: List[TradeRecord] = []
        self.trade_counter = 0
        
        # P&L tracking
        self.cumulative_pnl = 0.0
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.now().date()
        
        # Performance counters
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Streak tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Drawdown tracking
        self.max_drawdown_pct = 0.0
        self.current_drawdown_pct = 0.0
        
        # Rolling window (last N trades)
        self.rolling_window = 20
        self.rolling_pnls: deque = deque(maxlen=self.rolling_window)
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        
        # Slippage tracking
        self.total_slippage = 0.0
        self.slippage_entries: List[float] = []
        
        # System health
        self.start_time = datetime.now()
        self.tick_count = 0
        
        # Persistence
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'trading_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Backtest comparison
        self.backtest_stats = backtest_stats or {}
        
        logger.info(f"TradingStats initialized: ${initial_balance:,.2f}")
    
    def check_daily_reset(self):
        """Reset daily stats at midnight."""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            self._log_daily_summary()
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_daily_reset = today
            logger.info(f"Daily reset: {today}")
    
    def _log_daily_summary(self):
        """Log and save yesterday's performance."""
        if self.daily_trades == 0:
            return
        
        summary = {
            'date': self.last_daily_reset.isoformat(),
            'trades': self.daily_trades,
            'wins': self.daily_wins,
            'losses': self.daily_losses,
            'win_rate': self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0,
            'pnl': round(self.daily_pnl, 2),
            'ending_balance': round(self.current_balance, 2),
        }
        
        filepath = os.path.join(self.data_dir, 'daily_summary.csv')
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
        
        logger.info(
            f"DAILY SUMMARY [{self.last_daily_reset}]: "
            f"{self.daily_trades} trades | "
            f"Win Rate: {summary['win_rate']:.1%} | "
            f"P&L: ${self.daily_pnl:+,.2f}"
        )
    
    def record_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        signal_price: float,
        size: float,
        notional: float,
        stop_loss: float,
        take_profit: float,
        signal_score: float = 0.0,
        signal_reasons: List[str] = None,
    ) -> Dict:
        """Record trade entry. Returns entry_data dict to store in position."""
        # Calculate slippage
        if direction.lower() == 'long':
            slippage = (entry_price - signal_price) / signal_price if signal_price > 0 else 0
        else:
            slippage = (signal_price - entry_price) / signal_price if signal_price > 0 else 0
        
        slippage_cost = abs(slippage * notional)
        self.total_slippage += slippage_cost
        self.slippage_entries.append(slippage)
        
        if abs(slippage) > self.THRESHOLDS['slippage_warning']:
            logger.warning(f"HIGH SLIPPAGE: {slippage:.2%} on {symbol}")
        
        position_pct = notional / self.current_balance if self.current_balance > 0 else 0
        
        return {
            'entry_time': datetime.now(),
            'signal_price': signal_price,
            'entry_slippage': slippage,
            'notional': notional,
            'position_pct': position_pct,
            'signal_score': signal_score,
            'signal_reasons': signal_reasons or [],
            'equity_at_entry': self.current_balance,
        }
    
    def record_exit(
        self,
        symbol: str,
        direction: str,
        entry_data: Dict,
        exit_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        exit_reason: str,
        gross_pnl: float,
    ):
        """Record trade exit."""
        self.trade_counter += 1
        
        entry_time = entry_data.get('entry_time', datetime.now())
        exit_time = datetime.now()
        duration = (exit_time - entry_time).total_seconds() / 3600
        
        notional = entry_data.get('notional', 0)
        pnl_pct = gross_pnl / notional if notional > 0 else 0
        
        trade = TradeRecord(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction.lower(),
            entry_time=entry_time,
            exit_time=exit_time,
            duration_hours=duration,
            entry_price=entry_data.get('signal_price', 0),
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            notional=notional,
            position_pct=entry_data.get('position_pct', 0),
            gross_pnl=gross_pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            signal_score=entry_data.get('signal_score', 0),
            signal_reasons=entry_data.get('signal_reasons', []),
        )
        
        self.completed_trades.append(trade)
        
        # Update counters
        self.total_trades += 1
        self.daily_trades += 1
        
        if gross_pnl > 0:
            self.winning_trades += 1
            self.daily_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        elif gross_pnl < 0:
            self.losing_trades += 1
            self.daily_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Update P&L
        self.daily_pnl += gross_pnl
        self.cumulative_pnl += gross_pnl
        self.current_balance += gross_pnl
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.current_drawdown_pct = (self.current_balance - self.peak_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if self.current_drawdown_pct < self.max_drawdown_pct:
            self.max_drawdown_pct = self.current_drawdown_pct
        
        # Rolling P&L
        self.rolling_pnls.append(gross_pnl)
        
        # Check alerts
        self._check_alerts()
        
        # Log detailed exit
        self._log_trade_exit(trade)
        
        # Save trade to journal
        self._save_trade(trade)
    
    def _check_alerts(self):
        """Check for warning/critical thresholds."""
        daily_return = self.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0
        
        if daily_return < self.THRESHOLDS['daily_loss_critical']:
            logger.critical(f"CRITICAL: Daily loss limit breached: {daily_return:.2%}")
            logger.critical(f"RECOMMENDATION: HALT TRADING")
        elif daily_return < self.THRESHOLDS['daily_loss_warning']:
            logger.warning(f"WARNING: Approaching daily loss limit: {daily_return:.2%}")
        
        if self.current_drawdown_pct < self.THRESHOLDS['drawdown_critical']:
            logger.critical(f"CRITICAL: Drawdown critical: {self.current_drawdown_pct:.2%}")
        elif self.current_drawdown_pct < self.THRESHOLDS['drawdown_warning']:
            logger.warning(f"WARNING: Elevated drawdown: {self.current_drawdown_pct:.2%}")
        
        if self.consecutive_losses >= self.THRESHOLDS['consecutive_losses_warning']:
            logger.warning(f"WARNING: {self.consecutive_losses} consecutive losses")
    
    def _log_trade_exit(self, trade: TradeRecord):
        """Log detailed trade exit information."""
        # Color direction
        if trade.direction.lower() == 'long':
            dir_colored = colored(trade.direction.upper(), Colors.GREEN)
        else:
            dir_colored = colored(trade.direction.upper(), Colors.MAGENTA)
        
        # Color P&L
        if trade.gross_pnl >= 0:
            pnl_colored = colored(f"${trade.gross_pnl:+,.2f}", Colors.GREEN)
        else:
            pnl_colored = colored(f"${trade.gross_pnl:+,.2f}", Colors.RED)
        
        logger.info("=" * 70)
        logger.info(f"TRADE #{trade.trade_id} | EXIT | {dir_colored}")
        logger.info("=" * 70)
        logger.info(f"Symbol:      {trade.symbol}")
        logger.info(f"Duration:    {trade.duration_hours:.1f} hours")
        logger.info(f"Entry:       ${trade.entry_price:,.2f}")
        logger.info(f"Exit:        ${trade.exit_price:,.2f}")
        logger.info(f"Reason:      {trade.exit_reason}")
        logger.info(f"")
        logger.info(f"P&L:         {pnl_colored} ({trade.pnl_pct:+.2%})")
        logger.info(f"")
        logger.info(f"--- Running Totals ---")
        # Color running P&L
        daily_col = colored(f"${self.daily_pnl:+,.2f}", Colors.GREEN if self.daily_pnl >= 0 else Colors.RED)
        cum_col = colored(f"${self.cumulative_pnl:+,.2f}", Colors.GREEN if self.cumulative_pnl >= 0 else Colors.RED)
        logger.info(f"Daily P&L:   {daily_col}")
        logger.info(f"Cumulative:  {cum_col}")
        logger.info(f"Balance:     ${self.current_balance:,.2f}")
        logger.info(f"Drawdown:    {self.current_drawdown_pct:.2%}")
        logger.info(f"Win Rate:    {self.win_rate:.1%} ({self.winning_trades}W / {self.losing_trades}L)")
        streak = f"+{self.consecutive_wins}W" if self.consecutive_wins else f"-{self.consecutive_losses}L"
        logger.info(f"Streak:      {streak}")
        logger.info("=" * 70)
    
    def _save_trade(self, trade: TradeRecord):
        """Append trade to CSV journal."""
        filepath = os.path.join(self.data_dir, 'trade_journal.csv')
        file_exists = os.path.exists(filepath)
        
        trade_dict = {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_time': trade.entry_time.isoformat(),
            'exit_time': trade.exit_time.isoformat(),
            'duration_hours': round(trade.duration_hours, 2),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'size': trade.size,
            'notional': round(trade.notional, 2),
            'position_pct': round(trade.position_pct, 4),
            'gross_pnl': round(trade.gross_pnl, 2),
            'pnl_pct': round(trade.pnl_pct, 4),
            'exit_reason': trade.exit_reason,
            'signal_score': trade.signal_score,
            'cumulative_pnl': round(self.cumulative_pnl, 2),
            'balance': round(self.current_balance, 2),
        }
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_dict)
    
    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.gross_pnl for t in self.completed_trades if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in self.completed_trades if t.gross_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def avg_win(self) -> float:
        wins = [t.gross_pnl for t in self.completed_trades if t.gross_pnl > 0]
        return statistics.mean(wins) if wins else 0.0
    
    @property
    def avg_loss(self) -> float:
        losses = [abs(t.gross_pnl) for t in self.completed_trades if t.gross_pnl < 0]
        return statistics.mean(losses) if losses else 0.0
    
    @property
    def expectancy(self) -> float:
        return self.cumulative_pnl / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def rolling_win_rate(self) -> float:
        if len(self.rolling_pnls) < 5:
            return 0.0
        wins = sum(1 for p in self.rolling_pnls if p > 0)
        return wins / len(self.rolling_pnls)
    
    @property
    def rolling_pf(self) -> float:
        if len(self.rolling_pnls) < 5:
            return 0.0
        profit = sum(p for p in self.rolling_pnls if p > 0)
        loss = abs(sum(p for p in self.rolling_pnls if p < 0))
        return profit / loss if loss > 0 else float('inf')
    
    def print_status_report(self, open_positions: Dict = None):
        """Print comprehensive status report."""
        uptime = datetime.now() - self.start_time
        uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
        
        print()
        print("=" * 70)
        print(f"{'LIVE TRADING STATUS':^70}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^70}")
        print("=" * 70)
        
        # Account
        print(f"\n{'ACCOUNT':-^70}")
        print(f"  Starting Balance:  ${self.initial_balance:>12,.2f}")
        ret_pct = (self.current_balance / self.initial_balance - 1) if self.initial_balance > 0 else 0
        print(f"  Current Balance:   ${self.current_balance:>12,.2f}  ({ret_pct:+.1%})")
        print(f"  Peak Balance:      ${self.peak_balance:>12,.2f}")
        print(f"  Cumulative P&L:    ${self.cumulative_pnl:>12,.2f}")
        print(f"  Current Drawdown:  {self.current_drawdown_pct:>12.2%}  (max: {self.max_drawdown_pct:.2%})")
        
        # Today
        print(f"\n{'TODAY':-^70}")
        daily_wr = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0
        print(f"  Trades:            {self.daily_trades:>12}  ({self.daily_wins}W / {self.daily_losses}L)")
        print(f"  Win Rate:          {daily_wr:>12.1%}")
        print(f"  P&L:               ${self.daily_pnl:>12,.2f}")
        
        # All Time
        print(f"\n{'ALL TIME':-^70}")
        print(f"  Total Trades:      {self.total_trades:>12}")
        print(f"  Win Rate:          {self.win_rate:>12.1%}")
        pf = self.profit_factor
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"  Profit Factor:     {pf_str:>12}")
        print(f"  Avg Win:           ${self.avg_win:>12,.2f}")
        print(f"  Avg Loss:          ${self.avg_loss:>12,.2f}")
        print(f"  Expectancy:        ${self.expectancy:>12,.2f}/trade")
        
        # Rolling
        if self.total_trades >= 5:
            print(f"\n{'ROLLING (Last ' + str(len(self.rolling_pnls)) + ' trades)':-^70}")
            print(f"  Win Rate:          {self.rolling_win_rate:>12.1%}")
            rpf = self.rolling_pf
            rpf_str = f"{rpf:.2f}" if rpf < 100 else "inf"
            print(f"  Profit Factor:     {rpf_str:>12}")
            streak = f"+{self.consecutive_wins}W" if self.consecutive_wins else f"-{self.consecutive_losses}L"
            print(f"  Current Streak:    {streak:>12}")
        
        # Open Positions
        if open_positions:
            print(f"\n{'OPEN POSITIONS':-^70}")
            for symbol, pos in open_positions.items():
                direction = pos.get('side', 'UNKNOWN')
                entry = pos.get('entry_price', 0)
                stop = pos.get('stop', 0)
                target = pos.get('target', 0)
                print(f"  [{direction}] {symbol}")
                print(f"    Entry: ${entry:,.2f}  Stop: ${stop:,.2f}  Target: ${target:,.2f}")
        
        # Backtest comparison
        if self.backtest_stats and self.total_trades >= 10:
            print(f"\n{'DRIFT VS BACKTEST':-^70}")
            bt_wr = self.backtest_stats.get('win_rate', 0) / 100
            bt_pf = self.backtest_stats.get('profit_factor', 0)
            wr_drift = self.win_rate - bt_wr
            pf_drift = self.profit_factor - bt_pf
            
            wr_status = "OK" if abs(wr_drift) < 0.10 else "DRIFT"
            pf_status = "OK" if abs(pf_drift) < 0.5 else "DRIFT"
            
            print(f"  Win Rate:   Live {self.win_rate:.1%} vs BT {bt_wr:.1%}  ({wr_drift:+.1%}) [{wr_status}]")
            print(f"  Profit Factor: Live {self.profit_factor:.2f} vs BT {bt_pf:.2f}  ({pf_drift:+.2f}) [{pf_status}]")
        
        # System
        print(f"\n{'SYSTEM':-^70}")
        print(f"  Uptime:            {uptime_str:>12}")
        print(f"  Ticks:             {self.tick_count:>12}")
        avg_slip = statistics.mean(self.slippage_entries) if self.slippage_entries else 0
        print(f"  Avg Slippage:      {avg_slip:>12.3%}")
        print(f"  Total Slippage:    ${self.total_slippage:>12,.2f}")
        
        print()
        print("=" * 70)
        print()
    
    def save_state(self):
        """Save current state to disk for bot restarts."""
        state = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'cumulative_pnl': self.cumulative_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trade_counter': self.trade_counter,
            'total_slippage': self.total_slippage,
            'start_time': self.start_time.isoformat(),
            'saved_at': datetime.now().isoformat(),
        }
        
        filepath = os.path.join(self.data_dir, 'trading_state.json')
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved: {self.total_trades} trades, ${self.cumulative_pnl:+,.2f} P&L")
    
    def load_state(self):
        """Load state from disk (for bot restarts)."""
        filepath = os.path.join(self.data_dir, 'trading_state.json')
        
        if not os.path.exists(filepath):
            logger.info("No saved state found, starting fresh")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_balance = state.get('current_balance', self.initial_balance)
            self.peak_balance = state.get('peak_balance', self.initial_balance)
            self.cumulative_pnl = state.get('cumulative_pnl', 0)
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.losing_trades = state.get('losing_trades', 0)
            self.max_drawdown_pct = state.get('max_drawdown_pct', 0)
            self.max_consecutive_wins = state.get('max_consecutive_wins', 0)
            self.max_consecutive_losses = state.get('max_consecutive_losses', 0)
            self.trade_counter = state.get('trade_counter', 0)
            self.total_slippage = state.get('total_slippage', 0)
            
            logger.info(f"Loaded state: {self.total_trades} trades, ${self.cumulative_pnl:+,.2f} P&L")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# =============================================================================
# COINBASE LIVE TRADER
# =============================================================================

class CoinbaseLiveTrader:
    """
    Live trader for Coinbase spot and futures markets.
    Enhanced with professional monitoring.
    """
    
    # Contract specs - contract_size is static, margin rates fetched dynamically from API
    PERP_CONTRACT_SPECS = {
        'BIP-20DEC30-CDE': {'contract_size': 0.01, 'base': 'BTC'},
        'ETP-20DEC30-CDE': {'contract_size': 0.1, 'base': 'ETH'},
        'SLP-20DEC30-CDE': {'contract_size': 5.0, 'base': 'SOL'},
        'XPP-20DEC30-CDE': {'contract_size': 500.0, 'base': 'XRP'},
        'DOP-20DEC30-CDE': {'contract_size': 5000.0, 'base': 'DOGE'},
    }
    
    # Always use conservative overnight margin rates (lower leverage)
    # Intraday higher leverage is too risky
    USE_CONSERVATIVE_MARGIN = True
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        signal_symbols: List[str] = None,
        signal_timeframe: str = '1h',
        atr_timeframe: str = '4h',
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
        base_position: float = 0.10,
        min_position: float = 0.05,
        max_position: float = 0.30,
        max_positions: int = 2,
        max_daily_loss: float = 0.03,
        max_hold_days: float = None,
        long_only: bool = False,
        setup_validity_bars: int = 5,
        check_interval_seconds: int = 60,
        lookback_bars: int = 100,
        backtest_stats: Dict = None,  # For drift comparison
        data_dir: str = None,  # For persistence
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        
        self.symbols = symbols
        
        if signal_symbols is None:
            self.signal_symbols = symbols
            self.symbol_mapping = {s: s for s in symbols}
        else:
            self.signal_symbols = signal_symbols
            if len(signal_symbols) == len(symbols):
                self.symbol_mapping = dict(zip(symbols, signal_symbols))
            else:
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
        
        # Detect product types
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
        self.last_candles: Dict[str, datetime] = {}
        self.running = False
        
        # Initialize professional monitoring
        initial_balance = self.get_trading_balance()
        if initial_balance <= 0:
            initial_balance = self.get_balance()
        if initial_balance <= 0:
            initial_balance = 100  # Fallback
        
        self.stats = TradingStats(
            initial_balance=initial_balance,
            data_dir=data_dir,
            backtest_stats=backtest_stats,
        )
        self.stats.load_state()
        
        # For status report timing
        self.last_status_report = datetime.now()
        self.status_report_interval = 3600  # 1 hour
        
        logger.info(f"Initialized trader for {len(symbols)} symbols")
    
    def _detect_product_types(self):
        """Detect if symbols are spot or futures and get contract info with dynamic margin rates."""
        for symbol in self.symbols:
            try:
                product = self.client.get_product(product_id=symbol)
                
                product_type = getattr(product, 'product_type', 'UNKNOWN')
                is_futures = 'FUTURE' in product_type.upper()
                
                contract_size = 1.0
                margin_rate_long = 0.25  # Default 25% = 4x leverage
                margin_rate_short = 0.25
                overnight_long = 0.30    # Default 30% overnight margin
                overnight_short = 0.30
                
                if is_futures:
                    # Fetch contract details from API
                    future_details = getattr(product, 'future_product_details', None)
                    
                    # Get contract size - prefer API, fallback to hardcoded specs
                    if future_details and isinstance(future_details, dict):
                        api_contract_size = future_details.get('contract_size')
                        if api_contract_size:
                            contract_size = float(api_contract_size)
                            logger.info(f"  {symbol} contract_size from API: {contract_size}")
                        elif symbol in self.PERP_CONTRACT_SPECS:
                            contract_size = self.PERP_CONTRACT_SPECS[symbol]['contract_size']
                            logger.warning(f"  {symbol} contract_size from hardcoded specs: {contract_size}")
                        else:
                            contract_size = 0.1
                            logger.warning(f"  {symbol} using default contract_size: {contract_size}")
                    elif symbol in self.PERP_CONTRACT_SPECS:
                        contract_size = self.PERP_CONTRACT_SPECS[symbol]['contract_size']
                        logger.warning(f"  {symbol} no API details, using hardcoded: {contract_size}")
                    else:
                        logger.warning(f"  No contract spec for {symbol}, using default 0.1")
                        contract_size = 0.1
                    
                    # Fetch dynamic margin rates from API - we only use overnight (conservative)
                    if future_details and isinstance(future_details, dict):
                        overnight = future_details.get('overnight_margin_rate', {})
                        
                        # Get overnight rates (conservative - always used)
                        if isinstance(overnight, dict):
                            overnight_long = float(overnight.get('long_margin_rate', 0.30))
                            overnight_short = float(overnight.get('short_margin_rate', 0.30))
                        
                        lev_long = 1.0 / overnight_long if overnight_long > 0 else 1.0
                        logger.info(f"  {symbol} margin: {overnight_long:.1%} ({lev_long:.1f}x leverage)")
                
                # Store overnight rates only - we always use conservative rates
                self.product_info[symbol] = {
                    'is_futures': is_futures,
                    'contract_size': contract_size,
                    'margin_rate_long': overnight_long if is_futures else 1.0,
                    'margin_rate_short': overnight_short if is_futures else 1.0,
                    'product_type': product_type
                }
                
                # Log summary
                if is_futures:
                    base = self.PERP_CONTRACT_SPECS.get(symbol, {}).get('base', '?')
                    lev = 1.0 / overnight_long if overnight_long > 0 else 1.0
                    logger.info(f"{symbol}: {colored('FUTURES', Colors.CYAN)} | {contract_size} {base}/contract | {lev:.1f}x leverage")
                else:
                    logger.info(f"{symbol}: SPOT")
                
            except Exception as e:
                logger.error(f"Error detecting {symbol}: {e}")
                import traceback
                traceback.print_exc()
                self.product_info[symbol] = {
                    'is_futures': False,
                    'contract_size': 1.0,
                    'margin_rate_long_intraday': 1.0,
                    'margin_rate_short_intraday': 1.0,
                    'margin_rate_long_overnight': 1.0,
                    'margin_rate_short_overnight': 1.0,
                    'product_type': 'SPOT'
                }
    
    def is_intraday_hours(self) -> bool:
        """
        Always return False to use conservative overnight margin rates.
        
        We disable intraday higher leverage (10x) because it's too risky.
        Always use overnight rates (~4x for BTC/ETH, ~2.5x for alts).
        """
        return False
    
    def get_current_margin_rate(self, symbol: str, direction: str) -> float:
        """Get margin rate - fetches live from API before each trade."""
        info = self.product_info.get(symbol, {})
        
        if not info.get('is_futures', False):
            return 1.0
        
        # Fetch current margin rates from API
        try:
            product = self.client.get_product(product_id=symbol)
            future_details = getattr(product, 'future_product_details', None)
            
            if future_details:
                # Always use overnight (conservative) rates
                if isinstance(future_details, dict):
                    rates = future_details.get('overnight_margin_rate', {})
                else:
                    rates = getattr(future_details, 'overnight_margin_rate', {}) or {}
                
                if isinstance(rates, dict):
                    rate = float(rates.get('long_margin_rate' if direction == 'long' else 'short_margin_rate', 0.25))
                    logger.debug(f"{symbol} margin rate: {rate:.2%} ({1/rate:.1f}x leverage)")
                    return rate
                    
        except Exception as e:
            logger.warning(f"Failed to fetch margin rate for {symbol}, using cached: {e}")
        
        # Fallback to cached rates
        if direction == 'long':
            key = 'margin_rate_long'
        else:
            key = 'margin_rate_short'
        
        return info.get(key, 0.25)
    
    def get_balance(self) -> float:
        """Get available balance (spot)."""
        try:
            accounts_response = self.client.get_accounts()
            accounts = accounts_response.accounts if hasattr(accounts_response, 'accounts') else []
            
            total = 0.0
            
            for account in accounts:
                currency = getattr(account, 'currency', '')
                
                if currency in ['USD', 'USDC']:
                    if hasattr(account, 'available_balance'):
                        avail_obj = account.available_balance
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
                    
                    if hasattr(account, 'hold'):
                        hold_obj = account.hold
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
            return 0.0
    
    def get_futures_balance(self) -> float:
        """Get futures trading balance."""
        try:
            response = self.client.get_futures_balance_summary()
            
            summary = None
            if hasattr(response, 'balance_summary'):
                summary = response.balance_summary
            
            if summary is None:
                try:
                    summary = response['balance_summary']
                except (TypeError, KeyError):
                    pass
            
            if summary is None:
                try:
                    if hasattr(response, 'to_dict'):
                        resp_dict = response.to_dict()
                    elif hasattr(response, '__dict__'):
                        resp_dict = vars(response)
                    else:
                        resp_dict = eval(str(response))
                    summary = resp_dict.get('balance_summary', {})
                except:
                    pass
            
            if summary is None:
                return 0.0
            
            for key in ['futures_buying_power', 'available_margin', 'total_usd_balance']:
                val = None
                
                if isinstance(summary, dict):
                    val = summary.get(key, {})
                else:
                    if hasattr(summary, key):
                        val = getattr(summary, key)
                    elif hasattr(summary, '__getitem__'):
                        try:
                            val = summary[key]
                        except (KeyError, TypeError):
                            pass
                
                if val:
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
        """Get appropriate balance for trading."""
        is_futures = any(
            self.product_info.get(s, {}).get('is_futures', False) 
            for s in self.symbols
        )
        
        if is_futures:
            futures_bal = self.get_futures_balance()
            if futures_bal > 0:
                return futures_bal
        
        return self.get_balance()
    
    def show_account_info(self):
        """Display account information."""
        logger.info("=" * 60)
        logger.info("ACCOUNT OVERVIEW")
        logger.info("=" * 60)
        
        try:
            accounts_response = self.client.get_accounts()
            accounts = accounts_response.accounts if hasattr(accounts_response, 'accounts') else []
            
            total_usd = 0.0
            
            for account in accounts:
                currency = getattr(account, 'currency', '')
                
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
                
                if total_balance > 0 and currency in ['USD', 'USDC']:
                    total_usd += total_balance
                    logger.info(f"  {currency}: ${total_balance:,.2f} (available: ${avail:,.2f})")
            
            # Futures balance
            futures_bal = self.get_futures_balance()
            if futures_bal > 0:
                logger.info(f"  Futures Buying Power: ${futures_bal:,.2f}")
            
            logger.info("")
            logger.info(f"Total Trading Capital: ${max(total_usd, futures_bal):,.2f}")
            logger.info("=" * 60)
            
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
            
            # Calculate time range
            if timeframe == '1m':
                hours = 5
            elif timeframe == '5m':
                hours = 24
            elif timeframe == '15m':
                hours = 3 * 24
            elif timeframe == '30m':
                hours = 7 * 24
            elif timeframe == '1h':
                hours = 12 * 24
            elif timeframe == '4h':
                hours = 50 * 24
            elif timeframe == '1d':
                hours = 300 * 24
            else:
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
    
    def calculate_position_size(self, symbol: str, signal_position_pct: float, balance: float, price: float, direction: str = 'long') -> float:
        """Calculate position size in proper units using dynamic margin rates based on time of day."""
        info = self.product_info.get(symbol, {})
        is_futures = info.get('is_futures', False)
        
        position_value = balance * signal_position_pct
        
        if is_futures:
            contract_size = info.get('contract_size', 0.1)
            
            # Get dynamic margin rate based on time of day and direction
            margin_rate = self.get_current_margin_rate(symbol, direction)
            
            notional_per_contract = price * contract_size
            margin_per_contract = notional_per_contract * margin_rate
            
            max_contracts = position_value / margin_per_contract
            contracts = int(max_contracts)
            
            if contracts < 1 and position_value >= margin_per_contract:
                contracts = 1
            
            leverage = 1.0 / margin_rate if margin_rate > 0 else 1.0
            period = "intraday" if self.is_intraday_hours() else "overnight"
            logger.info(
                f"{symbol} sizing ({direction}, {period}): ${position_value:.2f} budget / "
                f"${margin_per_contract:.2f} margin per contract ({margin_rate:.1%} rate, ~{leverage:.1f}x) = {contracts} contracts"
            )
            
            return float(contracts)
        
        else:
            quantity = position_value / price
            logger.info(
                f"{symbol} sizing: ${position_value:.2f} / "
                f"${price:.2f} = {quantity:.6f}"
            )
            return quantity
    
    def place_order(self, symbol: str, side: str, size: float) -> Optional[str]:
        """Place market order."""
        try:
            info = self.product_info.get(symbol, {})
            is_futures = info.get('is_futures', False)
            
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
            
            # Log response for debugging
            logger.debug(f"Order response type: {type(order)}")
            logger.debug(f"Order response: {order}")
            if hasattr(order, '__dict__'):
                logger.debug(f"Order __dict__: {order.__dict__}")
            
            if hasattr(order, 'success') and not order.success:
                error_msg = "Unknown error"
                if hasattr(order, 'error_response'):
                    err = order.error_response
                    logger.error(f"Error response object: {err}")
                    logger.error(f"Error response type: {type(err)}")
                    if hasattr(err, '__dict__'):
                        logger.error(f"Error __dict__: {err.__dict__}")
                    
                    error_msg = getattr(err, 'error', 'Unknown')
                    preview_reason = getattr(err, 'preview_failure_reason', '')
                    message = getattr(err, 'message', '')
                    
                    logger.error(f"Order FAILED:")
                    logger.error(f"  error: {error_msg}")
                    logger.error(f"  preview_failure_reason: {preview_reason}")
                    logger.error(f"  message: {message}")
                    
                    if preview_reason == 'PREVIEW_INSUFFICIENT_FUNDS_FOR_FUTURES':
                        logger.error(f"  -> Transfer funds: Coinbase -> Transfer -> Spot to Futures")
                
                return None
            
            if hasattr(order, 'success') and order.success:
                order_id = order.success_response.order_id if hasattr(order.success_response, 'order_id') else str(order)
                logger.info(f"Order placed: {order_id}")
                return order_id
            
            if hasattr(order, 'order_id'):
                order_id = order.order_id
                logger.info(f"Order placed: {order_id}")
                return order_id
            
            logger.warning(f"Order response unclear: {order}")
            return None
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def enter_position(self, symbol: str, signal):
        """Enter new position with monitoring."""
        balance = self.get_trading_balance()
        price = self.get_price(symbol)
        
        if not price or balance < 10:
            logger.warning(f"Cannot enter {symbol}: price={price}, balance={balance}")
            return
        
        size = self.calculate_position_size(symbol, signal.position_size, balance, price, signal.direction)
        
        if size <= 0:
            logger.warning(f"Cannot enter {symbol}: calculated size={size}")
            return
        
        side = 'buy' if signal.direction == 'long' else 'sell'
        order_id = self.place_order(symbol, side, size)
        
        if not order_id:
            logger.error(f"Failed to enter {signal.direction.upper()} {symbol}")
            return
        
        # Calculate notional value
        info = self.product_info.get(symbol, {})
        is_futures = info.get('is_futures', False)
        
        if is_futures:
            contract_size = info.get('contract_size', 0.1)
            notional = price * contract_size * size
        else:
            notional = price * size
        
        # Record entry with monitoring
        entry_data = self.stats.record_entry(
            symbol=symbol,
            direction=signal.direction,
            entry_price=price,
            signal_price=signal.entry_price,
            size=size,
            notional=notional,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_score=signal.score,
            signal_reasons=signal.reasons,
        )
        
        # Create position with monitoring data
        self.positions[symbol] = {
            'side': signal.direction.upper(),
            'entry_price': price,
            'size': size,
            'stop': signal.stop_loss,
            'target': signal.take_profit,
            'entry_time': datetime.now(),
            'order_id': order_id,
            **entry_data  # Include monitoring data
        }
        
        size_str = f"{int(size)} contracts" if is_futures else f"{size:.6f}"
        
        # Color direction
        if signal.direction == 'long':
            dir_colored = colored("LONG", Colors.GREEN)
        else:
            dir_colored = colored("SHORT", Colors.MAGENTA)
        
        logger.info("=" * 70)
        logger.info(f"TRADE #{self.stats.trade_counter + 1} | ENTRY | {dir_colored}")
        logger.info("=" * 70)
        logger.info(f"Symbol:      {symbol}")
        logger.info(f"Size:        {size_str} (${notional:,.2f})")
        logger.info(f"Entry:       ${price:,.2f}")
        logger.info(f"Stop:        ${signal.stop_loss:,.2f}")
        logger.info(f"Target:      ${signal.take_profit:,.2f}")
        logger.info(f"Score:       {signal.score:.2f}")
        logger.info(f"Reasons:     {' '.join(signal.reasons)}")
        logger.info("=" * 70)
        
        if is_futures:
            logger.warning(f"Bot will MANUALLY monitor price and exit at stop/target")
    
    def exit_position(self, symbol: str, reason: str):
        """Exit position with monitoring."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        price = self.get_price(symbol)
        
        if not price:
            return
        
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
            
            # Record exit with monitoring
            self.stats.record_exit(
                symbol=symbol,
                direction=pos['side'],
                entry_data=pos,
                exit_price=price,
                size=pos['size'],
                stop_loss=pos['stop'],
                take_profit=pos['target'],
                exit_reason=reason,
                gross_pnl=total_pnl,
            )
            
            self.signal_generator.record_trade_result(total_pnl)
            
            del self.positions[symbol]
    
    def check_exits(self):
        """Check exit conditions for all positions."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = self.get_price(symbol)
            
            if not price:
                continue
            
            # Log position status
            if pos['side'] == 'LONG':
                distance_to_stop = ((price - pos['stop']) / price) * 100
                distance_to_target = ((pos['target'] - price) / price) * 100
            else:
                distance_to_stop = ((pos['stop'] - price) / price) * 100
                distance_to_target = ((price - pos['target']) / price) * 100
            
            logger.debug(
                f"{symbol}: ${price:.2f} | "
                f"Stop: {distance_to_stop:+.1f}% | "
                f"Target: {distance_to_target:+.1f}%"
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
        """Check if a candle is complete (closed)."""
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        duration = timedelta(minutes=tf_minutes.get(timeframe, 60))
        candle_close_time = candle_time + duration
        buffer = timedelta(minutes=1)
        
        now = datetime.now(pytz.UTC)
        return now >= (candle_close_time + buffer)
    
    def check_entries(self):
        """Check for entry signals."""
        if len(self.positions) >= self.max_positions:
            return
        
        # Daily loss limit
        if self.stats.daily_pnl < -self.max_daily_loss * self.stats.initial_balance:
            logger.warning(f"Daily loss limit reached: ${self.stats.daily_pnl:.2f}")
            return
        
        for trade_symbol in self.symbols:
            if trade_symbol in self.positions:
                continue
            
            signal_symbol = self.symbol_mapping.get(trade_symbol, trade_symbol)
            
            signal_df = self.get_candles(signal_symbol, self.signal_timeframe)
            atr_df = self.get_candles(signal_symbol, self.atr_timeframe)
            
            if signal_df.empty or atr_df.empty:
                continue
            
            latest = signal_df.index[-1]
            
            if not self._is_candle_complete(latest, self.signal_timeframe):
                if len(signal_df) < 2:
                    continue
                signal_df_for_detection = signal_df.iloc[:-1]
                latest_completed = signal_df_for_detection.index[-1]
            else:
                signal_df_for_detection = signal_df
                latest_completed = latest
            
            if not atr_df.empty:
                latest_atr = atr_df.index[-1]
                if not self._is_candle_complete(latest_atr, self.atr_timeframe):
                    if len(atr_df) >= 2:
                        atr_df = atr_df.iloc[:-1]
            
            candle_key = f"{signal_symbol}_{self.signal_timeframe}"
            is_new_candle = self.last_candles.get(candle_key) != latest_completed
            
            if is_new_candle:
                self.last_candles[candle_key] = latest_completed
                logger.info(f"[{signal_symbol}] New {self.signal_timeframe} candle CLOSED at {latest_completed}")
                
                df = self.analyzer.calculate_indicators(signal_df_for_detection.tail(100))
                
                if len(df) >= 2:
                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    # Color squeeze state
                    if current['Squeeze']:
                        squeeze_state = colored("IN SQUEEZE", Colors.CYAN)
                    else:
                        squeeze_state = colored("NOT SQUEEZED", Colors.YELLOW)
                    logger.info(f"[{signal_symbol}] State: {squeeze_state} | Duration: {current['Squeeze_Duration']:.0f} bars")
                    logger.info(f"[{signal_symbol}] BB Width: {current['BB_Width']:.4f} | Momentum: {current['Momentum_Norm']:+.2f}")
                    logger.info(f"[{signal_symbol}] RSI: {current['RSI']:.1f} | Volume Ratio: {current['Volume_Ratio']:.2f}")
                    
                    if prev['Squeeze'] and not current['Squeeze']:
                        logger.info(f"[{signal_symbol}] {colored('SQUEEZE RELEASED', Colors.YELLOW)} after {prev['Squeeze_Duration']:.0f} bars!")
            
            self.signal_generator.set_signal_data({signal_symbol: signal_df_for_detection})
            self.signal_generator.set_atr_data({signal_symbol: atr_df})
            
            # CRITICAL: Pass signal_df_for_detection (complete candles only)
            # to match backtest behavior. Entry price comes from df.iloc[-1]
            # so we must exclude incomplete candles for backtest-live parity.
            signal = self.signal_generator.generate_signal(
                signal_df_for_detection,
                signal_symbol,
                datetime.now(pytz.UTC)
            )
            
            if signal.direction != 'neutral':
                # Check long_only filter
                if self.long_only and signal.direction == 'short':
                    logger.info(f"   Skipping SHORT signal (long_only=True)")
                    continue
                
                # Color signal direction
                if signal.direction == 'long':
                    dir_str = colored("LONG", Colors.GREEN)
                else:
                    dir_str = colored("SHORT", Colors.MAGENTA)
                logger.info(f"[{signal_symbol}] SIGNAL: {dir_str}")
                logger.info(f"[{signal_symbol}] Entry: ${signal.entry_price:,.2f} | Stop: ${signal.stop_loss:,.2f} | Target: ${signal.take_profit:,.2f}")
                logger.info(f"[{signal_symbol}] Size: {signal.position_size:.0%} | Score: {signal.score:.2f}")
                logger.info(f"[{signal_symbol}] Reasons: {' '.join(signal.reasons)}")
                
                signal.symbol = trade_symbol
                self.enter_position(trade_symbol, signal)
            else:
                if is_new_candle and len(df) >= 2:
                    current = df.iloc[-1]
                    prev = df.iloc[-2]
                    
                    if current['Squeeze']:
                        logger.info(f"[{signal_symbol}] No signal: Still in squeeze")
                    elif not prev['Squeeze']:
                        logger.info(f"[{signal_symbol}] No signal: No squeeze to break out from")
                    elif current['Volume_Ratio'] < self.signal_generator.min_volume_ratio:
                        logger.info(f"[{signal_symbol}] No signal: Volume too low ({current['Volume_Ratio']:.2f})")
                    elif current['RSI'] > self.signal_generator.rsi_overbought:
                        logger.info(f"[{signal_symbol}] No signal: RSI overbought ({current['RSI']:.1f})")
                    elif current['RSI'] < self.signal_generator.rsi_oversold:
                        logger.info(f"[{signal_symbol}] No signal: RSI oversold ({current['RSI']:.1f})")
                    else:
                        logger.info(f"[{signal_symbol}] No signal: Conditions not met")
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.stats.check_daily_reset()
    
    def run(self):
        """Start trading bot."""
        logger.info("=" * 60)
        logger.info("COINBASE LIVE TRADER")
        logger.info("=" * 60)
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Signal TF: {self.signal_timeframe}")
        logger.info(f"Max positions: {self.max_positions}")
        logger.info(f"Long only: {self.long_only}")
        logger.info("")
        
        self.show_account_info()
        logger.info("")
        
        # Print initial status
        self.stats.print_status_report(self.positions)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    self.stats.tick_count += 1
                    
                    self.reset_daily()
                    self.check_exits()
                    self.check_entries()
                    
                    if self.stats.tick_count % 10 == 0:
                        logger.info(
                            f"Positions: {len(self.positions)}/{self.max_positions} | "
                            f"Daily: ${self.stats.daily_pnl:+.2f} | "
                            f"Cumulative: ${self.stats.cumulative_pnl:+.2f}"
                        )
                    
                    # Save state every hour (no verbose print)
                    if (datetime.now() - self.last_status_report).total_seconds() >= self.status_report_interval:
                        self.stats.save_state()
                        self.last_status_report = datetime.now()
                    
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
            self.stats.save_state()
            self.running = False
    
    def stop(self):
        """Stop the bot."""
        self.stats.save_state()
        self.running = False