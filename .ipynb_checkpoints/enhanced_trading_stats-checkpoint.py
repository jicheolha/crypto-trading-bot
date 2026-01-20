"""
Enhanced State Tracking & Monitoring for CoinbaseLiveTrader

This module adds professional hedge-fund grade monitoring to the live trader.
To use: import and initialize at the start of CoinbaseLiveTrader.__init__

Add these enhancements to your coinbase_live_trader.py
"""
import os
import json
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import deque
import statistics
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENHANCED STATE CLASS - Add to CoinbaseLiveTrader.__init__
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
    size: float  # contracts or amount
    notional: float  # dollar value
    position_pct: float  # % of equity at entry
    
    # P&L
    gross_pnl: float
    net_pnl: float  # after costs
    pnl_pct: float  # return %
    
    # Exit
    exit_reason: str
    
    # Signal info
    signal_score: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)


class TradingStats:
    """
    Professional trading statistics tracker.
    
    Add this to CoinbaseLiveTrader:
        self.stats = TradingStats(initial_balance, data_dir='trading_data')
    """
    
    def __init__(
        self,
        initial_balance: float,
        data_dir: str = None,
        backtest_stats: Dict = None,  # For drift comparison
    ):
        # Balance tracking
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Trade history (CRITICAL - this is what's missing!)
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
        self.breakeven_trades = 0
        
        # Streak tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # Drawdown tracking
        self.max_drawdown_pct = 0.0
        self.max_drawdown_dollar = 0.0
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
        self.api_errors = 0
        self.api_calls = 0
        self.start_time = datetime.now()
        
        # Persistence
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'trading_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Backtest comparison
        self.backtest_stats = backtest_stats or {}
        
        # Alert thresholds
        self.THRESHOLDS = {
            'daily_loss_warning': -0.02,
            'daily_loss_critical': -0.03,
            'drawdown_warning': -0.05,
            'drawdown_critical': -0.10,
            'consecutive_losses_warning': 4,
            'slippage_warning': 0.002,
        }
        
        logger.info(f"TradingStats initialized: ${initial_balance:,.2f}")
    
    # =========================================================================
    # DAILY RESET
    # =========================================================================
    
    def check_daily_reset(self):
        """Reset daily stats at midnight."""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            # Log yesterday's summary
            self._log_daily_summary()
            
            # Reset daily counters
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
            'pnl': self.daily_pnl,
            'ending_balance': self.current_balance,
        }
        
        # Append to daily log
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
    
    # =========================================================================
    # TRADE RECORDING (Call from enter_position and exit_position)
    # =========================================================================
    
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
        """
        Record trade entry. Call this from enter_position().
        Returns entry_data dict to store in positions[symbol].
        """
        # Calculate slippage
        if direction == 'long':
            slippage = (entry_price - signal_price) / signal_price
        else:
            slippage = (signal_price - entry_price) / signal_price
        
        slippage_cost = abs(slippage * notional)
        self.total_slippage += slippage_cost
        self.slippage_entries.append(slippage)
        
        # Check slippage alert
        if abs(slippage) > self.THRESHOLDS['slippage_warning']:
            logger.warning(f"⚠️ HIGH SLIPPAGE: {slippage:.2%} on {symbol}")
        
        position_pct = notional / self.current_balance if self.current_balance > 0 else 0
        
        entry_data = {
            'entry_time': datetime.now(),
            'signal_price': signal_price,
            'entry_price': entry_price,
            'entry_slippage': slippage,
            'notional': notional,
            'position_pct': position_pct,
            'signal_score': signal_score,
            'signal_reasons': signal_reasons or [],
            'equity_at_entry': self.current_balance,
        }
        
        return entry_data
    
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
        """
        Record trade exit. Call this from exit_position().
        """
        self.trade_counter += 1
        
        entry_time = entry_data.get('entry_time', datetime.now())
        exit_time = datetime.now()
        duration = (exit_time - entry_time).total_seconds() / 3600  # hours
        
        # Create trade record
        trade = TradeRecord(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            duration_hours=duration,
            entry_price=entry_data.get('entry_price', 0),
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            notional=entry_data.get('notional', 0),
            position_pct=entry_data.get('position_pct', 0),
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,  # Add commission tracking if needed
            pnl_pct=gross_pnl / entry_data.get('notional', 1) if entry_data.get('notional') else 0,
            exit_reason=exit_reason,
            signal_score=entry_data.get('signal_score', 0),
            signal_reasons=entry_data.get('signal_reasons', []),
        )
        
        # Store trade
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
        else:
            self.breakeven_trades += 1
            # Breakeven doesn't reset streaks
        
        # Update P&L
        self.daily_pnl += gross_pnl
        self.cumulative_pnl += gross_pnl
        self.current_balance += gross_pnl
        
        # Update peak and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        self.current_drawdown_pct = (self.current_balance - self.peak_balance) / self.peak_balance
        if self.current_drawdown_pct < self.max_drawdown_pct:
            self.max_drawdown_pct = self.current_drawdown_pct
            self.max_drawdown_dollar = self.peak_balance - self.current_balance
        
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
        # Daily loss
        daily_return = self.daily_pnl / self.initial_balance
        if daily_return < self.THRESHOLDS['daily_loss_critical']:
            logger.critical(f"🚨 CRITICAL: Daily loss limit breached: {daily_return:.2%}")
            logger.critical(f"🚨 RECOMMENDATION: HALT TRADING")
        elif daily_return < self.THRESHOLDS['daily_loss_warning']:
            logger.warning(f"⚠️ WARNING: Approaching daily loss limit: {daily_return:.2%}")
        
        # Drawdown
        if self.current_drawdown_pct < self.THRESHOLDS['drawdown_critical']:
            logger.critical(f"🚨 CRITICAL: Drawdown critical: {self.current_drawdown_pct:.2%}")
        elif self.current_drawdown_pct < self.THRESHOLDS['drawdown_warning']:
            logger.warning(f"⚠️ WARNING: Elevated drawdown: {self.current_drawdown_pct:.2%}")
        
        # Consecutive losses
        if self.consecutive_losses >= self.THRESHOLDS['consecutive_losses_warning']:
            logger.warning(f"⚠️ WARNING: {self.consecutive_losses} consecutive losses")
    
    def _log_trade_exit(self, trade: TradeRecord):
        """Log detailed trade exit information."""
        logger.info("=" * 70)
        logger.info(f"TRADE #{trade.trade_id} | EXIT | {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        logger.info(f"Symbol:      {trade.symbol}")
        logger.info(f"Direction:   {trade.direction.upper()}")
        logger.info(f"Duration:    {trade.duration_hours:.1f} hours")
        logger.info(f"")
        logger.info(f"Entry:       ${trade.entry_price:,.2f}")
        logger.info(f"Exit:        ${trade.exit_price:,.2f}")
        logger.info(f"Reason:      {trade.exit_reason}")
        logger.info(f"")
        logger.info(f"Gross P&L:   ${trade.gross_pnl:+,.2f} ({trade.pnl_pct:+.2%})")
        logger.info(f"")
        logger.info(f"--- Running Totals ---")
        logger.info(f"Daily P&L:   ${self.daily_pnl:+,.2f}")
        logger.info(f"Cumulative:  ${self.cumulative_pnl:+,.2f}")
        logger.info(f"Balance:     ${self.current_balance:,.2f}")
        logger.info(f"Drawdown:    {self.current_drawdown_pct:.2%}")
        logger.info(f"Win Rate:    {self.win_rate:.1%} ({self.winning_trades}W / {self.losing_trades}L)")
        logger.info(f"Streak:      {'+' + str(self.consecutive_wins) + 'W' if self.consecutive_wins else '-' + str(self.consecutive_losses) + 'L'}")
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
            'notional': trade.notional,
            'position_pct': round(trade.position_pct, 4),
            'gross_pnl': round(trade.gross_pnl, 2),
            'pnl_pct': round(trade.pnl_pct, 4),
            'exit_reason': trade.exit_reason,
            'signal_score': trade.signal_score,
        }
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_dict)
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    @property
    def win_rate(self) -> float:
        """Current win rate."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.gross_pnl for t in self.completed_trades if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in self.completed_trades if t.gross_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def avg_win(self) -> float:
        """Average winning trade."""
        wins = [t.gross_pnl for t in self.completed_trades if t.gross_pnl > 0]
        return statistics.mean(wins) if wins else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade."""
        losses = [abs(t.gross_pnl) for t in self.completed_trades if t.gross_pnl < 0]
        return statistics.mean(losses) if losses else 0.0
    
    @property
    def expectancy(self) -> float:
        """Expected $ per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.cumulative_pnl / self.total_trades
    
    @property
    def rolling_win_rate(self) -> float:
        """Win rate over last N trades."""
        if len(self.rolling_pnls) < 5:
            return 0.0
        wins = sum(1 for p in self.rolling_pnls if p > 0)
        return wins / len(self.rolling_pnls)
    
    @property
    def rolling_pf(self) -> float:
        """Profit factor over last N trades."""
        if len(self.rolling_pnls) < 5:
            return 0.0
        profit = sum(p for p in self.rolling_pnls if p > 0)
        loss = abs(sum(p for p in self.rolling_pnls if p < 0))
        return profit / loss if loss > 0 else float('inf')
    
    def get_performance_dict(self) -> Dict:
        """Get all metrics as dictionary."""
        return {
            # Balance
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'return_pct': (self.current_balance - self.initial_balance) / self.initial_balance,
            
            # P&L
            'cumulative_pnl': self.cumulative_pnl,
            'daily_pnl': self.daily_pnl,
            
            # Trades
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
            
            # Drawdown
            'current_drawdown_pct': self.current_drawdown_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            
            # Streaks
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            
            # Rolling
            'rolling_win_rate': self.rolling_win_rate,
            'rolling_pf': self.rolling_pf,
            
            # Costs
            'total_slippage': self.total_slippage,
            'avg_slippage': statistics.mean(self.slippage_entries) if self.slippage_entries else 0,
        }
    
    # =========================================================================
    # DRIFT ANALYSIS
    # =========================================================================
    
    def get_drift_analysis(self) -> Dict:
        """Compare live performance to backtest."""
        if not self.backtest_stats or self.total_trades < 10:
            return {}
        
        bt = self.backtest_stats
        
        return {
            'win_rate_live': self.win_rate,
            'win_rate_backtest': bt.get('win_rate', 0) / 100,
            'win_rate_drift': self.win_rate - bt.get('win_rate', 0) / 100,
            
            'pf_live': self.profit_factor,
            'pf_backtest': bt.get('profit_factor', 0),
            'pf_drift': self.profit_factor - bt.get('profit_factor', 0),
            
            'avg_win_live': self.avg_win,
            'avg_win_backtest': bt.get('avg_win', 0),
            
            'avg_loss_live': self.avg_loss,
            'avg_loss_backtest': bt.get('avg_loss', 0),
        }
    
    # =========================================================================
    # STATUS REPORT
    # =========================================================================
    
    def print_status_report(self, open_positions: Dict = None):
        """Print comprehensive status report."""
        uptime = datetime.now() - self.start_time
        uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
        
        print()
        print("╔" + "═" * 68 + "╗")
        print(f"║{'LIVE TRADING STATUS':^68}║")
        print(f"║{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^68}║")
        print("╠" + "═" * 68 + "╣")
        
        # Account
        print(f"║ {'ACCOUNT':<66} ║")
        print(f"║   Starting Balance:  ${self.initial_balance:>12,.2f}{' ' * 32}║")
        print(f"║   Current Balance:   ${self.current_balance:>12,.2f}  ({(self.current_balance/self.initial_balance - 1):+.1%}){' ' * 18}║")
        print(f"║   Peak Balance:      ${self.peak_balance:>12,.2f}{' ' * 32}║")
        print(f"║   Cumulative P&L:    ${self.cumulative_pnl:>12,.2f}{' ' * 32}║")
        print(f"║   Current Drawdown:  {self.current_drawdown_pct:>12.2%}  (max: {self.max_drawdown_pct:.2%}){' ' * 13}║")
        
        print("╠" + "═" * 68 + "╣")
        
        # Today
        print(f"║ {'TODAY':<66} ║")
        daily_wr = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0
        print(f"║   Trades:            {self.daily_trades:>12}  ({self.daily_wins}W / {self.daily_losses}L){' ' * 18}║")
        print(f"║   Win Rate:          {daily_wr:>12.1%}{' ' * 32}║")
        print(f"║   P&L:               ${self.daily_pnl:>12,.2f}{' ' * 32}║")
        
        print("╠" + "═" * 68 + "╣")
        
        # All Time
        print(f"║ {'ALL TIME':<66} ║")
        print(f"║   Total Trades:      {self.total_trades:>12}{' ' * 32}║")
        print(f"║   Win Rate:          {self.win_rate:>12.1%}{' ' * 32}║")
        print(f"║   Profit Factor:     {self.profit_factor:>12.2f}{' ' * 32}║")
        print(f"║   Avg Win:           ${self.avg_win:>12,.2f}{' ' * 32}║")
        print(f"║   Avg Loss:          ${self.avg_loss:>12,.2f}{' ' * 32}║")
        print(f"║   Expectancy:        ${self.expectancy:>12,.2f}/trade{' ' * 26}║")
        
        # Rolling
        if self.total_trades >= 5:
            print("╠" + "═" * 68 + "╣")
            print(f"║ {'ROLLING (Last ' + str(len(self.rolling_pnls)) + ' trades)':<66} ║")
            print(f"║   Win Rate:          {self.rolling_win_rate:>12.1%}{' ' * 32}║")
            print(f"║   Profit Factor:     {self.rolling_pf:>12.2f}{' ' * 32}║")
            streak = f"+{self.consecutive_wins}W" if self.consecutive_wins else f"-{self.consecutive_losses}L"
            print(f"║   Current Streak:    {streak:>12}{' ' * 32}║")
        
        # Open Positions
        if open_positions:
            print("╠" + "═" * 68 + "╣")
            print(f"║ {'OPEN POSITIONS':<66} ║")
            for symbol, pos in open_positions.items():
                direction = pos.get('side', 'UNKNOWN')
                entry = pos.get('entry_price', 0)
                print(f"║   [{direction}] {symbol:<20} Entry: ${entry:,.2f}{' ' * 18}║")
        
        # System
        print("╠" + "═" * 68 + "╣")
        print(f"║ {'SYSTEM':<66} ║")
        print(f"║   Uptime:            {uptime_str:>12}{' ' * 32}║")
        print(f"║   Slippage (total):  ${self.total_slippage:>12,.2f}{' ' * 32}║")
        
        print("╚" + "═" * 68 + "╝")
        print()
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_state(self):
        """Save current state to disk."""
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
            'start_time': self.start_time.isoformat(),
            'saved_at': datetime.now().isoformat(),
        }
        
        filepath = os.path.join(self.data_dir, 'trading_state.json')
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self):
        """Load state from disk (for bot restarts)."""
        filepath = os.path.join(self.data_dir, 'trading_state.json')
        
        if not os.path.exists(filepath):
            logger.info("No saved state found, starting fresh")
            return
        
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
            
            logger.info(f"Loaded state: {self.total_trades} trades, ${self.cumulative_pnl:+,.2f} P&L")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
