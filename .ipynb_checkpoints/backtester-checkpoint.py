"""
Backtester for Crypto Trading Bot.
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: float
    capital_used: float
    stop_loss: float
    take_profit: float
    atr: float
    reasons: List[str]
    entry_equity: float = 0.0
    position_size_pct: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    r_multiple: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    is_open: bool = True


@dataclass
class BacktestResults:
    """Backtest results container."""
    trades: List[Position]
    equity_curve: pd.Series
    statistics: Dict


def _infer_timeframe_from_index(index) -> str:
    """Infer timeframe from pandas DatetimeIndex."""
    if len(index) < 2:
        return '1min'
    
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return '1min'
    
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60
    
    if minutes <= 1.5:
        return '1min'
    elif minutes <= 7:
        return '5min'
    elif minutes <= 20:
        return '15min'
    elif minutes <= 45:
        return '30min'
    elif minutes <= 120:
        return '1h'
    elif minutes <= 360:
        return '4h'
    else:
        return '1d'



class BBSqueezeBacktester:
    """
    Backtester for crypto trading strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.0005,
        slippage_pct: float = 0.0002,
        max_positions: int = 2,
        max_daily_loss_pct: float = 0.03,
        max_hold_days: float = None,
        long_only: bool = False,
        verbose: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_hold_days = max_hold_days
        self.long_only = long_only
        self.verbose = verbose
        
        # Set by run_backtest
        self.analyzer: BBSqueezeAnalyzer = None
        self.signal_generator: BBSqueezeSignalGenerator = None
        
        # State
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Position] = []
        self.equity_history: List[tuple] = []
        self.daily_pnl = 0.0
        self.last_daily_reset = None
        
        # Inferred from data
        self._trade_timeframe = '1min'
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        Run backtest on trade timeframe data.
        
        Args:
            data: {symbol: DataFrame} with trade timeframe data
        """
        self._reset()
        
        # Infer timeframe from data for Sharpe calculation
        first_df = next(iter(data.values()))
        self._trade_timeframe = _infer_timeframe_from_index(first_df.index)
        
        # Get all timestamps
        all_times = set()
        for symbol, df in data.items():
            all_times.update(df.index.tolist())
        all_times = sorted(all_times)
        
        if self.verbose:
            print(f"\nBacktesting {len(all_times):,} bars")
            print(f"Period: {all_times[0].strftime('%Y-%m-%d')} to {all_times[-1].strftime('%Y-%m-%d')}\n")
        
        for i, ts in enumerate(all_times):
            # Reset daily P&L
            self._check_daily_reset(ts)
            
            # Check daily loss limit (use current equity, not initial capital)
            current_equity = self._calc_equity(data, ts)
            if self.daily_pnl < -self.max_daily_loss_pct * current_equity:
                continue
            
            # Process each symbol
            for symbol, df in data.items():
                if ts not in df.index:
                    continue
                
                idx = df.index.get_loc(ts)
                if idx < 50:
                    continue
                
                bar_df = df.iloc[:idx+1]
                price = float(bar_df.iloc[-1]['close'])
                equity = self._calc_equity(data, ts)
                
                # Check exits first
                if symbol in self.positions:
                    self._check_exit(symbol, price, ts, bar_df, equity)
                
                # Check entries
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    self._check_entry(symbol, bar_df, ts, equity)
            
            # Record equity
            equity = self._calc_equity(data, ts)
            self.equity_history.append((ts, equity))
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in data:
                final_price = float(data[symbol].iloc[-1]['close'])
                final_equity = self._calc_equity(data, all_times[-1])
                self._close(symbol, final_price, all_times[-1], "End of backtest", final_equity)
        
        return self._compile_results()
    
    def _reset(self):
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.daily_pnl = 0.0
        self.last_daily_reset = None
        if self.signal_generator:
            self.signal_generator.active_setups = {}
            self.signal_generator.consecutive_losses = 0
    
    def _check_daily_reset(self, ts: datetime):
        """Reset daily P&L at midnight."""
        current_date = ts.date()
        if self.last_daily_reset != current_date:
            self.daily_pnl = 0.0
            self.last_daily_reset = current_date
    
    def _check_entry(self, symbol: str, df: pd.DataFrame, ts: datetime, equity: float):
        """Check for entry signal."""
        signal = self.signal_generator.generate_signal(df, symbol, ts)
        
        if signal.direction == 'neutral':
            return
        
        # Skip shorts if long_only mode
        if self.long_only and signal.direction == 'short':
            return
        
        # Calculate position size
        position_value = equity * signal.position_size
        price = signal.entry_price
        
        # Apply slippage
        if signal.direction == 'long':
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)
        
        quantity = position_value / entry_price
        commission_cost = quantity * entry_price * self.commission
        
        if position_value + commission_cost > self.capital:
            return
        
        # Open position
        self.capital -= position_value + commission_cost
        
        # Calculate risk metrics
        if signal.direction == 'long':
            risk_per_share = entry_price - signal.stop_loss
        else:
            risk_per_share = signal.stop_loss - entry_price
        
        risk_amount = risk_per_share * quantity
        risk_pct = (risk_amount / equity) * 100
        
        pos = Position(
            symbol=symbol,
            direction=signal.direction,
            entry_time=ts,
            entry_price=entry_price,
            quantity=quantity,
            capital_used=position_value,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            atr=signal.atr,
            reasons=signal.reasons,
            entry_equity=equity,
            position_size_pct=signal.position_size * 100,
            risk_amount=risk_amount,
            risk_pct=risk_pct
        )
        
        self.positions[symbol] = pos
        
        if self.verbose:
            dir_str = "LONG " if signal.direction == 'long' else "SHORT"
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  {dir_str}  ${entry_price:>10,.2f}  "
                  f"{signal.position_size:>5.0%}  ${equity:>12,.0f}  {' '.join(signal.reasons)}")
    
    def _check_exit(self, symbol: str, price: float, ts: datetime, df: pd.DataFrame, equity: float):
        """Check exit conditions."""
        pos = self.positions[symbol]
        
        # Time-based exit (max holding period)
        if self.max_hold_days is not None:
            hold_time = ts - pos.entry_time
            if hold_time.total_seconds() >= self.max_hold_days * 24 * 3600:
                self._close(symbol, price, ts, "Time exit", equity)
                return
        
        # Stop loss
        if pos.direction == 'long' and price <= pos.stop_loss:
            self._close(symbol, price, ts, "Stop loss hit", equity)
            return
        elif pos.direction == 'short' and price >= pos.stop_loss:
            self._close(symbol, price, ts, "Stop loss hit", equity)
            return
        
        # Take profit
        if pos.direction == 'long' and price >= pos.take_profit:
            self._close(symbol, price, ts, "Take profit hit", equity)
            return
        elif pos.direction == 'short' and price <= pos.take_profit:
            self._close(symbol, price, ts, "Take profit hit", equity)
            return
    
    def _close(self, symbol: str, price: float, ts: datetime, reason: str, equity: float):
        """Close position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos.exit_time = ts
        
        # Apply slippage
        if pos.direction == 'long':
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)
        
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.is_open = False
        
        # Calculate P&L
        if pos.direction == 'long':
            gross = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity
        
        commission_cost = pos.quantity * exit_price * self.commission
        net = gross - commission_cost
        pos.pnl = net
        pos.pnl_pct = (net / pos.capital_used) * 100
        
        # Calculate R-multiple (reward:risk ratio)
        if pos.risk_amount > 0:
            pos.r_multiple = net / pos.risk_amount
        else:
            pos.r_multiple = 0.0
        
        # Update capital
        self.capital += pos.capital_used + net
        self.daily_pnl += net
        
        # Record result
        self.signal_generator.record_trade_result(net)
        
        self.trades.append(pos)
        del self.positions[symbol]
        
        if self.verbose:
            new_equity = self._calc_equity_now()
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  EXIT   ${exit_price:>10,.2f}  "
                  f"${net:>+9,.2f}  ${new_equity:>12,.0f}  {reason}")
    
    def _calc_equity(self, data: Dict[str, pd.DataFrame], ts: datetime) -> float:
        """Calculate current equity."""
        equity = self.capital
        
        for symbol, pos in self.positions.items():
            if symbol not in data:
                # Symbol not in data at all - just add capital_used back
                equity += pos.capital_used
                continue
            
            df = data[symbol]
            
            # Find price: use exact timestamp or most recent before it
            if ts in df.index:
                price = float(df.loc[ts, 'close'])
            else:
                # Use last known price before this timestamp
                prior = df.index[df.index <= ts]
                if len(prior) > 0:
                    price = float(df.loc[prior[-1], 'close'])
                else:
                    # No data yet - use entry price
                    price = pos.entry_price
            
            if pos.direction == 'long':
                unrealized = (price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - price) * pos.quantity
            
            equity += pos.capital_used + unrealized
        
        return equity
    
    def _calc_equity_now(self) -> float:
        """Calculate equity without data lookup."""
        return self.capital + sum(p.capital_used for p in self.positions.values())
    
    def _compile_results(self) -> BacktestResults:
        """Compile backtest results."""
        if not self.trades:
            return BacktestResults(
                trades=[],
                equity_curve=pd.Series([self.initial_capital]),
                statistics=self._empty_stats()
            )
        
        # Equity curve
        eq_df = pd.DataFrame(self.equity_history, columns=['time', 'equity'])
        eq_df.set_index('time', inplace=True)
        
        # Statistics
        stats = self._calculate_stats(eq_df['equity'])
        
        if self.verbose:
            self._print_stats(stats)
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=eq_df['equity'],
            statistics=stats
        )
    
    def _calculate_stats(self, equity: pd.Series) -> Dict:
        """Calculate performance statistics."""
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]  # Fixed: exclude breakeven
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        max_dd = abs(drawdown.min())
        
        # Sharpe Ratio - INDUSTRY STANDARD: Calculate on DAILY returns
        # Resampling equity to daily prevents artificial inflation from
        # near-zero intraday returns when signals are on higher timeframes (4h)
        try:
            daily_equity = equity.resample('D').last().dropna()
            daily_returns = daily_equity.pct_change().dropna()
            
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                # Annualize: sqrt(252 trading days)
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
        except Exception:
            # Fallback if resampling fails
            sharpe = 0
        
        # R-multiple stats
        r_multiples = [t.r_multiple for t in self.trades]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        avg_r_win = sum([t.r_multiple for t in wins]) / len(wins) if wins else 0
        avg_r_loss = sum([t.r_multiple for t in losses]) / len(losses) if losses else 0
        
        # Expectancy (average $ per trade)
        expectancy = total_pnl / len(self.trades) if self.trades else 0
        
        # Kelly Criterion - FIXED formula: K = W - (1-W)/R
        # where W = win rate, R = avg_win / avg_loss
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        
        if avg_loss > 0 and avg_win > 0:
            win_loss_ratio = avg_win / avg_loss  # R
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        else:
            kelly = 0
        kelly_pct = max(0, min(kelly * 100, 100))  # Cap at 0-100%
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_consec = 0
        last_was_win = None
        
        for t in self.trades:
            is_win = t.pnl > 0
            if is_win == last_was_win:
                current_consec += 1
            else:
                current_consec = 1
                last_was_win = is_win
            
            if is_win:
                max_consec_wins = max(max_consec_wins, current_consec)
            else:
                max_consec_losses = max(max_consec_losses, current_consec)
        
        # Average hold time
        hold_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades if t.exit_time]
        avg_hold_hours = sum(hold_times) / len(hold_times) if hold_times else 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'final_equity': float(equity.iloc[-1]),
            'return_pct': (float(equity.iloc[-1]) - self.initial_capital) / self.initial_capital * 100,
            'avg_r_multiple': avg_r,
            'avg_r_win': avg_r_win,
            'avg_r_loss': avg_r_loss,
            'expectancy': expectancy,
            'kelly_pct': kelly_pct,
            'max_consec_wins': max_consec_wins,
            'max_consec_losses': max_consec_losses,
            'avg_hold_hours': avg_hold_hours
        }
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe': 0,
            'final_equity': self.initial_capital,
            'return_pct': 0,
            'avg_r_multiple': 0,
            'avg_r_win': 0,
            'avg_r_loss': 0,
            'expectancy': 0,
            'kelly_pct': 0,
            'max_consec_wins': 0,
            'max_consec_losses': 0,
            'avg_hold_hours': 0
        }
    
    def _print_stats(self, stats: Dict):
        """Print statistics summary."""
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:    {stats['total_trades']}")
        print(f"Win Rate:        {stats['win_rate']:.1f}%")
        print(f"Profit Factor:   {stats['profit_factor']:.2f}")
        print(f"Total P&L:       ${stats['total_pnl']:+,.0f}")
        print(f"Max Drawdown:    {stats['max_drawdown']:.1f}%")
        print(f"Sharpe Ratio:    {stats['sharpe']:.2f}")
        print(f"Final Equity:    ${stats['final_equity']:,.0f}")
        print(f"Return:          {stats['return_pct']:+.1f}%")
        print(f"{'='*60}")