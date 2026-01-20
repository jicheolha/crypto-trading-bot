"""
Live Trading Monitor - Hedge Fund Grade Monitoring

What a professional trader monitors in real-time:
1. P&L Management (daily, cumulative, per-trade)
2. Risk Metrics (drawdown, exposure, VaR)
3. Performance Drift (live vs backtest)
4. System Health (API errors, latency)
5. Position Health (distance to stops/targets)
6. Slippage Tracking (expected vs actual)
7. Trade Journal (detailed audit trail)
8. Alerts & Thresholds
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """Complete record of a single trade."""
    trade_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    
    # Timing
    signal_time: datetime
    entry_time: datetime
    exit_time: Optional[datetime] = None
    
    # Prices
    signal_price: float = 0.0
    expected_entry: float = 0.0
    actual_entry: float = 0.0
    expected_exit: float = 0.0
    actual_exit: float = 0.0
    
    # Levels
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Size
    position_size_pct: float = 0.0
    contracts: float = 0.0
    notional_value: float = 0.0
    
    # P&L
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    
    # Exit info
    exit_reason: str = ""
    
    # Metadata
    signal_score: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)


@dataclass
class DailyStats:
    """Daily performance statistics."""
    date: str
    starting_balance: float
    ending_balance: float
    
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L
    gross_pnl: float = 0.0
    commissions: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_exposure_pct: float = 0.0
    peak_equity: float = 0.0
    
    # Derived
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # System health
    api_errors: int = 0
    missed_signals: int = 0
    avg_latency_ms: float = 0.0


@dataclass 
class Alert:
    """Trading alert."""
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'CRITICAL'
    category: str  # 'RISK', 'PERFORMANCE', 'SYSTEM', 'TRADE'
    message: str
    data: Dict = field(default_factory=dict)


# =============================================================================
# LIVE TRADING MONITOR
# =============================================================================

class LiveTradingMonitor:
    """
    Comprehensive monitoring for live trading operations.
    
    Tracks everything a hedge fund risk manager would want to see.
    """
    
    # Alert thresholds
    THRESHOLDS = {
        # P&L thresholds
        'daily_loss_warning': -0.02,      # -2% daily loss warning
        'daily_loss_critical': -0.03,     # -3% daily loss critical (halt)
        'drawdown_warning': -0.05,        # -5% drawdown warning
        'drawdown_critical': -0.10,       # -10% drawdown critical
        
        # Performance thresholds
        'win_rate_warning': 0.35,         # Below 35% win rate
        'profit_factor_warning': 0.8,     # Below 0.8 profit factor
        'consecutive_losses': 4,          # 4+ consecutive losses
        
        # Drift thresholds (live vs backtest)
        'sharpe_drift': 0.5,              # Sharpe differs by 0.5+
        'win_rate_drift': 0.10,           # Win rate differs by 10%+
        
        # System thresholds
        'api_latency_warning': 2000,      # 2 second API response
        'api_error_rate': 0.05,           # 5% error rate
        
        # Slippage thresholds
        'slippage_warning': 0.002,        # 0.2% slippage warning
        'slippage_critical': 0.005,       # 0.5% slippage critical
    }
    
    def __init__(
        self,
        initial_balance: float,
        data_dir: str = None,
        backtest_stats: Dict = None,  # For drift comparison
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Storage
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'monitor_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Trade history
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        
        # Daily tracking
        self.daily_stats: Dict[str, DailyStats] = {}
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self._init_daily_stats()
        
        # Rolling metrics (last N trades)
        self.rolling_window = 20
        self.rolling_pnls: deque = deque(maxlen=self.rolling_window)
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_handlers: List[callable] = []
        
        # System health
        self.api_latencies: deque = deque(maxlen=100)
        self.api_errors: int = 0
        self.api_calls: int = 0
        
        # Backtest comparison
        self.backtest_stats = backtest_stats or {}
        
        # Consecutive tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        logger.info(f"LiveTradingMonitor initialized with ${initial_balance:,.2f}")
    
    def _init_daily_stats(self):
        """Initialize daily stats for current date."""
        if self.current_date not in self.daily_stats:
            self.daily_stats[self.current_date] = DailyStats(
                date=self.current_date,
                starting_balance=self.current_balance,
                ending_balance=self.current_balance,
                peak_equity=self.current_balance,
            )
    
    def _check_date_rollover(self):
        """Check if we've rolled to a new day."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.current_date:
            # Finalize yesterday's stats
            self._finalize_daily_stats()
            
            # Start new day
            self.current_date = today
            self._init_daily_stats()
            
            logger.info(f"Daily rollover to {today}")
    
    def _finalize_daily_stats(self):
        """Calculate final stats for the day."""
        stats = self.daily_stats.get(self.current_date)
        if not stats:
            return
        
        stats.ending_balance = self.current_balance
        
        # Calculate derived metrics
        if stats.total_trades > 0:
            stats.win_rate = stats.winning_trades / stats.total_trades
        
        wins = [t.net_pnl for t in self.trades 
                if t.exit_time and t.exit_time.strftime('%Y-%m-%d') == self.current_date and t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in self.trades 
                  if t.exit_time and t.exit_time.strftime('%Y-%m-%d') == self.current_date and t.net_pnl < 0]
        
        if wins:
            stats.avg_win = statistics.mean(wins)
        if losses:
            stats.avg_loss = statistics.mean(losses)
        
        gross_loss = sum(losses)
        if gross_loss > 0:
            stats.profit_factor = sum(wins) / gross_loss
        
        # Save to disk
        self._save_daily_stats(stats)
    
    # =========================================================================
    # TRADE TRACKING
    # =========================================================================
    
    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        signal_price: float,
        actual_entry: float,
        stop_loss: float,
        take_profit: float,
        position_size_pct: float,
        contracts: float,
        notional_value: float,
        signal_score: float = 0.0,
        signal_reasons: List[str] = None,
    ) -> TradeRecord:
        """Record a new trade entry."""
        self._check_date_rollover()
        
        now = datetime.now()
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            signal_time=now,
            entry_time=now,
            signal_price=signal_price,
            expected_entry=signal_price,
            actual_entry=actual_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            contracts=contracts,
            notional_value=notional_value,
            signal_score=signal_score,
            signal_reasons=signal_reasons or [],
        )
        
        # Calculate entry slippage
        if direction == 'long':
            trade.slippage_cost = (actual_entry - signal_price) * contracts
        else:
            trade.slippage_cost = (signal_price - actual_entry) * contracts
        
        self.open_positions[symbol] = trade
        
        # Update daily stats
        stats = self.daily_stats[self.current_date]
        stats.total_trades += 1
        
        # Check slippage alerts
        slippage_pct = abs(actual_entry - signal_price) / signal_price
        if slippage_pct > self.THRESHOLDS['slippage_critical']:
            self._raise_alert(
                'CRITICAL', 'TRADE',
                f"High slippage on {symbol}: {slippage_pct:.2%}",
                {'symbol': symbol, 'expected': signal_price, 'actual': actual_entry}
            )
        elif slippage_pct > self.THRESHOLDS['slippage_warning']:
            self._raise_alert(
                'WARNING', 'TRADE',
                f"Elevated slippage on {symbol}: {slippage_pct:.2%}",
                {'symbol': symbol, 'expected': signal_price, 'actual': actual_entry}
            )
        
        logger.info(f"ENTRY RECORDED: {direction.upper()} {symbol} @ ${actual_entry:,.2f}")
        
        return trade
    
    def record_exit(
        self,
        symbol: str,
        actual_exit: float,
        exit_reason: str,
        commission: float = 0.0,
    ) -> Optional[TradeRecord]:
        """Record a trade exit."""
        self._check_date_rollover()
        
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return None
        
        trade = self.open_positions.pop(symbol)
        trade.exit_time = datetime.now()
        trade.actual_exit = actual_exit
        trade.exit_reason = exit_reason
        trade.commission = commission
        
        # Calculate P&L
        if trade.direction == 'long':
            trade.realized_pnl = (actual_exit - trade.actual_entry) * trade.contracts
            exit_slippage = (trade.take_profit - actual_exit) if exit_reason == "Take profit" else 0
        else:
            trade.realized_pnl = (trade.actual_entry - actual_exit) * trade.contracts
            exit_slippage = (actual_exit - trade.take_profit) if exit_reason == "Take profit" else 0
        
        trade.slippage_cost += exit_slippage * trade.contracts
        trade.net_pnl = trade.realized_pnl - trade.commission - abs(trade.slippage_cost)
        
        # Update balance
        self.current_balance += trade.net_pnl
        
        # Update peak (for drawdown)
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Track rolling P&L
        self.rolling_pnls.append(trade.net_pnl)
        
        # Update daily stats
        stats = self.daily_stats[self.current_date]
        stats.net_pnl += trade.net_pnl
        stats.commissions += trade.commission
        stats.slippage += abs(trade.slippage_cost)
        stats.ending_balance = self.current_balance
        
        if trade.net_pnl > 0:
            stats.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            stats.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update max drawdown
        current_dd = (self.current_balance - self.peak_balance) / self.peak_balance
        if current_dd < stats.max_drawdown_pct:
            stats.max_drawdown_pct = current_dd
        
        # Store completed trade
        self.trades.append(trade)
        
        # Check alerts
        self._check_post_trade_alerts(trade)
        
        logger.info(
            f"EXIT RECORDED: {trade.direction.upper()} {symbol} @ ${actual_exit:,.2f} | "
            f"P&L: ${trade.net_pnl:+,.2f} | Reason: {exit_reason}"
        )
        
        return trade
    
    # =========================================================================
    # REAL-TIME POSITION MONITORING
    # =========================================================================
    
    def get_position_health(self, symbol: str, current_price: float) -> Dict:
        """Get health metrics for an open position."""
        if symbol not in self.open_positions:
            return {}
        
        trade = self.open_positions[symbol]
        
        # Calculate unrealized P&L
        if trade.direction == 'long':
            unrealized_pnl = (current_price - trade.actual_entry) * trade.contracts
            distance_to_stop = (current_price - trade.stop_loss) / current_price
            distance_to_target = (trade.take_profit - current_price) / current_price
            risk_reward_current = distance_to_target / distance_to_stop if distance_to_stop > 0 else 0
        else:
            unrealized_pnl = (trade.actual_entry - current_price) * trade.contracts
            distance_to_stop = (trade.stop_loss - current_price) / current_price
            distance_to_target = (current_price - trade.take_profit) / current_price
            risk_reward_current = distance_to_target / distance_to_stop if distance_to_stop > 0 else 0
        
        unrealized_pnl_pct = unrealized_pnl / trade.notional_value if trade.notional_value > 0 else 0
        
        # Time in trade
        time_in_trade = datetime.now() - trade.entry_time
        hours_in_trade = time_in_trade.total_seconds() / 3600
        
        return {
            'symbol': symbol,
            'direction': trade.direction,
            'entry_price': trade.actual_entry,
            'current_price': current_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'distance_to_stop_pct': distance_to_stop,
            'distance_to_target_pct': distance_to_target,
            'risk_reward_current': risk_reward_current,
            'hours_in_trade': hours_in_trade,
            'contracts': trade.contracts,
            'notional_value': trade.notional_value,
        }
    
    def get_portfolio_exposure(self) -> Dict:
        """Get current portfolio exposure metrics."""
        total_exposure = sum(t.notional_value for t in self.open_positions.values())
        exposure_pct = total_exposure / self.current_balance if self.current_balance > 0 else 0
        
        long_exposure = sum(t.notional_value for t in self.open_positions.values() if t.direction == 'long')
        short_exposure = sum(t.notional_value for t in self.open_positions.values() if t.direction == 'short')
        
        return {
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'num_positions': len(self.open_positions),
            'positions': list(self.open_positions.keys()),
        }
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    def get_live_performance(self) -> Dict:
        """Get comprehensive live performance metrics."""
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if not completed_trades:
            return {
                'total_trades': 0,
                'message': 'No completed trades yet'
            }
        
        wins = [t for t in completed_trades if t.net_pnl > 0]
        losses = [t for t in completed_trades if t.net_pnl < 0]
        
        total_pnl = sum(t.net_pnl for t in completed_trades)
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = statistics.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = statistics.mean([abs(t.net_pnl) for t in losses]) if losses else 0
        
        # Drawdown
        current_dd = (self.current_balance - self.peak_balance) / self.peak_balance
        
        # Expectancy
        expectancy = total_pnl / len(completed_trades)
        
        # Rolling metrics
        rolling_win_rate = 0
        rolling_pf = 0
        if len(self.rolling_pnls) >= 5:
            recent_wins = sum(1 for p in self.rolling_pnls if p > 0)
            rolling_win_rate = recent_wins / len(self.rolling_pnls)
            
            recent_profit = sum(p for p in self.rolling_pnls if p > 0)
            recent_loss = abs(sum(p for p in self.rolling_pnls if p < 0))
            rolling_pf = recent_profit / recent_loss if recent_loss > 0 else float('inf')
        
        # Total slippage and commissions
        total_slippage = sum(abs(t.slippage_cost) for t in completed_trades)
        total_commission = sum(t.commission for t in completed_trades)
        
        return {
            # Overall
            'total_trades': len(completed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            
            # P&L
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            
            # Balance
            'starting_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'return_pct': (self.current_balance - self.initial_balance) / self.initial_balance,
            'current_drawdown_pct': current_dd,
            
            # Rolling
            'rolling_win_rate': rolling_win_rate,
            'rolling_profit_factor': rolling_pf,
            
            # Costs
            'total_slippage': total_slippage,
            'total_commission': total_commission,
            'slippage_pct_of_pnl': total_slippage / abs(total_pnl) if total_pnl != 0 else 0,
            
            # Streaks
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
        }
    
    def get_drift_analysis(self) -> Dict:
        """Compare live performance to backtest."""
        if not self.backtest_stats:
            return {'message': 'No backtest stats provided for comparison'}
        
        live = self.get_live_performance()
        bt = self.backtest_stats
        
        if live.get('total_trades', 0) < 10:
            return {
                'message': 'Need at least 10 trades for drift analysis',
                'current_trades': live.get('total_trades', 0)
            }
        
        drift = {
            'win_rate_drift': live['win_rate'] - bt.get('win_rate', 0) / 100,
            'profit_factor_drift': live['profit_factor'] - bt.get('profit_factor', 0),
            'avg_win_drift_pct': (live['avg_win'] - bt.get('avg_win', 0)) / bt.get('avg_win', 1) if bt.get('avg_win') else 0,
            'avg_loss_drift_pct': (live['avg_loss'] - bt.get('avg_loss', 0)) / bt.get('avg_loss', 1) if bt.get('avg_loss') else 0,
        }
        
        # Check for concerning drift
        alerts = []
        if abs(drift['win_rate_drift']) > self.THRESHOLDS['win_rate_drift']:
            alerts.append(f"Win rate drifted by {drift['win_rate_drift']:.1%}")
        
        drift['alerts'] = alerts
        drift['is_concerning'] = len(alerts) > 0
        
        return drift
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def _raise_alert(self, level: str, category: str, message: str, data: Dict = None):
        """Raise an alert."""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            data=data or {}
        )
        
        self.alerts.append(alert)
        
        # Log based on level
        if level == 'CRITICAL':
            logger.critical(f"🚨 {category}: {message}")
        elif level == 'WARNING':
            logger.warning(f"⚠️ {category}: {message}")
        else:
            logger.info(f"ℹ️ {category}: {message}")
        
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _check_post_trade_alerts(self, trade: TradeRecord):
        """Check for alerts after a trade closes."""
        stats = self.daily_stats[self.current_date]
        
        # Daily loss check
        daily_return = stats.net_pnl / stats.starting_balance
        if daily_return < self.THRESHOLDS['daily_loss_critical']:
            self._raise_alert(
                'CRITICAL', 'RISK',
                f"Daily loss limit breached: {daily_return:.2%}",
                {'daily_pnl': stats.net_pnl, 'threshold': self.THRESHOLDS['daily_loss_critical']}
            )
        elif daily_return < self.THRESHOLDS['daily_loss_warning']:
            self._raise_alert(
                'WARNING', 'RISK',
                f"Approaching daily loss limit: {daily_return:.2%}",
                {'daily_pnl': stats.net_pnl}
            )
        
        # Drawdown check
        current_dd = (self.current_balance - self.peak_balance) / self.peak_balance
        if current_dd < self.THRESHOLDS['drawdown_critical']:
            self._raise_alert(
                'CRITICAL', 'RISK',
                f"Critical drawdown: {current_dd:.2%}",
                {'current_balance': self.current_balance, 'peak': self.peak_balance}
            )
        elif current_dd < self.THRESHOLDS['drawdown_warning']:
            self._raise_alert(
                'WARNING', 'RISK',
                f"Elevated drawdown: {current_dd:.2%}",
                {'current_balance': self.current_balance, 'peak': self.peak_balance}
            )
        
        # Consecutive losses
        if self.consecutive_losses >= self.THRESHOLDS['consecutive_losses']:
            self._raise_alert(
                'WARNING', 'PERFORMANCE',
                f"Consecutive losses: {self.consecutive_losses}",
                {'streak': self.consecutive_losses}
            )
        
        # Rolling performance degradation
        if len(self.rolling_pnls) >= 10:
            recent_wins = sum(1 for p in self.rolling_pnls if p > 0)
            rolling_wr = recent_wins / len(self.rolling_pnls)
            
            if rolling_wr < self.THRESHOLDS['win_rate_warning']:
                self._raise_alert(
                    'WARNING', 'PERFORMANCE',
                    f"Rolling win rate declining: {rolling_wr:.1%}",
                    {'rolling_trades': len(self.rolling_pnls), 'rolling_win_rate': rolling_wr}
                )
    
    def add_alert_handler(self, handler: callable):
        """Add a custom alert handler (e.g., for Slack/email notifications)."""
        self.alert_handlers.append(handler)
    
    # =========================================================================
    # SYSTEM HEALTH
    # =========================================================================
    
    def record_api_call(self, latency_ms: float, success: bool):
        """Record an API call for health monitoring."""
        self.api_calls += 1
        self.api_latencies.append(latency_ms)
        
        if not success:
            self.api_errors += 1
            
            error_rate = self.api_errors / self.api_calls
            if error_rate > self.THRESHOLDS['api_error_rate']:
                self._raise_alert(
                    'WARNING', 'SYSTEM',
                    f"High API error rate: {error_rate:.1%}",
                    {'errors': self.api_errors, 'total_calls': self.api_calls}
                )
        
        if latency_ms > self.THRESHOLDS['api_latency_warning']:
            self._raise_alert(
                'WARNING', 'SYSTEM',
                f"High API latency: {latency_ms:.0f}ms",
                {'latency_ms': latency_ms}
            )
    
    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        avg_latency = statistics.mean(self.api_latencies) if self.api_latencies else 0
        max_latency = max(self.api_latencies) if self.api_latencies else 0
        error_rate = self.api_errors / self.api_calls if self.api_calls > 0 else 0
        
        return {
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'error_rate': error_rate,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'is_healthy': error_rate < 0.05 and avg_latency < 1000,
        }
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_status_report(self):
        """Print comprehensive status report."""
        perf = self.get_live_performance()
        exposure = self.get_portfolio_exposure()
        health = self.get_system_health()
        
        print("\n" + "="*70)
        print("LIVE TRADING STATUS REPORT")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Balance
        print(f"\n{'ACCOUNT':=^70}")
        print(f"  Starting Balance:   ${self.initial_balance:>12,.2f}")
        print(f"  Current Balance:    ${self.current_balance:>12,.2f}")
        print(f"  Peak Balance:       ${self.peak_balance:>12,.2f}")
        print(f"  Total Return:       {perf.get('return_pct', 0):>12.2%}")
        print(f"  Current Drawdown:   {perf.get('current_drawdown_pct', 0):>12.2%}")
        
        # Performance
        print(f"\n{'PERFORMANCE':=^70}")
        print(f"  Total Trades:       {perf.get('total_trades', 0):>12}")
        print(f"  Win Rate:           {perf.get('win_rate', 0):>12.1%}")
        print(f"  Profit Factor:      {perf.get('profit_factor', 0):>12.2f}")
        print(f"  Total P&L:          ${perf.get('total_pnl', 0):>12,.2f}")
        print(f"  Avg Win:            ${perf.get('avg_win', 0):>12,.2f}")
        print(f"  Avg Loss:           ${perf.get('avg_loss', 0):>12,.2f}")
        print(f"  Expectancy:         ${perf.get('expectancy', 0):>12,.2f}")
        
        # Rolling
        if perf.get('total_trades', 0) >= 5:
            print(f"\n{'ROLLING ({self.rolling_window} trades)':=^70}")
            print(f"  Rolling Win Rate:   {perf.get('rolling_win_rate', 0):>12.1%}")
            print(f"  Rolling PF:         {perf.get('rolling_profit_factor', 0):>12.2f}")
            print(f"  Win Streak:         {self.consecutive_wins:>12}")
            print(f"  Loss Streak:        {self.consecutive_losses:>12}")
        
        # Exposure
        print(f"\n{'EXPOSURE':=^70}")
        print(f"  Open Positions:     {exposure['num_positions']:>12}")
        print(f"  Total Exposure:     ${exposure['total_exposure']:>12,.2f}")
        print(f"  Exposure %:         {exposure['exposure_pct']:>12.1%}")
        print(f"  Net Exposure:       ${exposure['net_exposure']:>12,.2f}")
        
        # Costs
        print(f"\n{'COSTS':=^70}")
        print(f"  Total Slippage:     ${perf.get('total_slippage', 0):>12,.2f}")
        print(f"  Total Commission:   ${perf.get('total_commission', 0):>12,.2f}")
        print(f"  Slippage % of P&L:  {perf.get('slippage_pct_of_pnl', 0):>12.1%}")
        
        # System
        print(f"\n{'SYSTEM HEALTH':=^70}")
        print(f"  API Calls:          {health['api_calls']:>12}")
        print(f"  API Errors:         {health['api_errors']:>12}")
        print(f"  Error Rate:         {health['error_rate']:>12.2%}")
        print(f"  Avg Latency:        {health['avg_latency_ms']:>12.0f} ms")
        print(f"  Status:             {'✓ HEALTHY' if health['is_healthy'] else '✗ DEGRADED':>12}")
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]
        if recent_alerts:
            print(f"\n{'RECENT ALERTS (24h)':=^70}")
            for alert in recent_alerts[-5:]:
                print(f"  [{alert.level}] {alert.category}: {alert.message}")
        
        print("\n" + "="*70)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_daily_stats(self, stats: DailyStats):
        """Save daily stats to disk."""
        filepath = os.path.join(self.data_dir, f"daily_{stats.date}.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(stats), f, indent=2, default=str)
    
    def save_trade_journal(self):
        """Save all trades to disk."""
        filepath = os.path.join(self.data_dir, "trade_journal.json")
        trades_data = [asdict(t) for t in self.trades]
        with open(filepath, 'w') as f:
            json.dump(trades_data, f, indent=2, default=str)
        logger.info(f"Trade journal saved: {len(self.trades)} trades")
    
    def load_trade_journal(self):
        """Load trade journal from disk."""
        filepath = os.path.join(self.data_dir, "trade_journal.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                trades_data = json.load(f)
            # Convert back to TradeRecord objects
            for t in trades_data:
                if t.get('entry_time'):
                    t['entry_time'] = datetime.fromisoformat(t['entry_time'])
                if t.get('exit_time'):
                    t['exit_time'] = datetime.fromisoformat(t['exit_time'])
                if t.get('signal_time'):
                    t['signal_time'] = datetime.fromisoformat(t['signal_time'])
            self.trades = [TradeRecord(**t) for t in trades_data]
            logger.info(f"Loaded {len(self.trades)} trades from journal")


# =============================================================================
# EXAMPLE ALERT HANDLERS
# =============================================================================

def console_alert_handler(alert: Alert):
    """Print alerts to console with formatting."""
    colors = {
        'CRITICAL': '\033[91m',  # Red
        'WARNING': '\033[93m',   # Yellow
        'INFO': '\033[94m',      # Blue
    }
    reset = '\033[0m'
    color = colors.get(alert.level, '')
    print(f"{color}[{alert.timestamp.strftime('%H:%M:%S')}] {alert.level} - {alert.category}: {alert.message}{reset}")


def file_alert_handler(alert: Alert, filepath: str = "alerts.log"):
    """Write alerts to a file."""
    with open(filepath, 'a') as f:
        f.write(f"{alert.timestamp.isoformat()} | {alert.level} | {alert.category} | {alert.message}\n")


# Placeholder for Slack/Discord/Email handlers
# def slack_alert_handler(alert: Alert):
#     """Send alert to Slack webhook."""
#     webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
#     if webhook_url:
#         requests.post(webhook_url, json={'text': f"[{alert.level}] {alert.message}"})