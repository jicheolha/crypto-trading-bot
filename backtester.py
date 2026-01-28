"""
Backtester with optional leverage support.

Leverage mode simulates futures trading with:
- Conservative overnight margin rates ONLY (no 10x intraday)
- Asset-specific margin rates (matching Coinbase International overnight)
- Liquidation simulation using bar high/low
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from technical import BBSqueezeAnalyzer
from signal_generator import BBSqueezeSignalGenerator, TradeSignal
from utils import infer_timeframe_from_index

logger = logging.getLogger(__name__)


# =============================================================================
# LEVERAGE CONFIGURATION - CONSERVATIVE OVERNIGHT RATES ONLY
# =============================================================================

# Always use overnight (conservative) rates - NO intraday 10x leverage
LEVERAGE_RATES = {
    # Format: 'BASE': margin_rate
    # Leverage = 1 / margin_rate
    'BTC': 0.25,   # 4x leverage
    'ETH': 0.25,   # 4x leverage
    'SOL': 0.37,   # 2.7x leverage
    'XRP': 0.39,   # 2.6x leverage
    'DOGE': 0.50,  # 2x leverage
}

# Default for unknown assets
DEFAULT_MARGIN_RATE = 0.50  # 2x leverage


# =============================================================================
# DATA CLASSES
# =============================================================================

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
    # Leverage fields
    leverage: float = 1.0
    margin_used: float = 0.0
    notional_value: float = 0.0
    liquidation_price: Optional[float] = None


@dataclass
class BacktestResults:
    """Backtest results container"""
    trades: List[Position]
    equity_curve: pd.Series
    statistics: Dict


# =============================================================================
# BACKTESTER
# =============================================================================

class BBSqueezeBacktester:
    """
    Backtester for crypto trading strategies.
    
    Supports optional leverage mode that simulates futures trading
    with realistic margin requirements and liquidation.
    
    ALWAYS uses conservative overnight margin rates (no 10x intraday).
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
        verbose: bool = True,
        # Leverage settings
        leverage: bool = False,
        maintenance_margin_pct: float = 0.5,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_hold_days = max_hold_days
        self.long_only = long_only
        self.verbose = verbose
        
        # Leverage settings
        self.leverage_enabled = leverage
        self.maintenance_margin_pct = maintenance_margin_pct
        
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
        
        # Stats
        self.liquidation_count = 0
        
        # Inferred from data
        self._trade_timeframe = '1min'
    
    def _extract_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol (e.g., 'BTC/USD' -> 'BTC')."""
        if '/' in symbol:
            return symbol.split('/')[0]
        elif '-' in symbol:
            return symbol.split('-')[0]
        return symbol
    
    def _get_margin_rate(self, symbol: str) -> float:
        """Get margin rate for symbol - always uses conservative overnight rates."""
        base = self._extract_base_currency(symbol)
        return LEVERAGE_RATES.get(base, DEFAULT_MARGIN_RATE)
    
    def _get_leverage(self, symbol: str) -> float:
        """Get leverage multiplier for symbol."""
        margin_rate = self._get_margin_rate(symbol)
        return 1.0 / margin_rate if margin_rate > 0 else 1.0
    
    def _calculate_liquidation_price(
        self, 
        entry_price: float, 
        direction: str, 
        margin_rate: float,
        maintenance_pct: float
    ) -> float:
        """
        Calculate liquidation price.
        
        Liquidation occurs when unrealized loss equals (1 - maintenance_pct) of margin.
        
        For long: liq_price = entry * (1 - margin_rate * (1 - maintenance_pct))
        For short: liq_price = entry * (1 + margin_rate * (1 - maintenance_pct))
        """
        # How much of margin can be lost before liquidation
        max_loss_pct = margin_rate * (1 - maintenance_pct)
        
        if direction == 'long':
            return entry_price * (1 - max_loss_pct)
        else:
            return entry_price * (1 + max_loss_pct)
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        Run backtest on trade timeframe data.
        
        Args:
            data: {symbol: DataFrame} with trade timeframe data
        """
        self._reset()
        
        # Infer timeframe from data for Sharpe calculation
        first_df = next(iter(data.values()))
        self._trade_timeframe = infer_timeframe_from_index(first_df.index)
        
        # Get all timestamps
        all_times = set()
        for symbol, df in data.items():
            all_times.update(df.index.tolist())
        all_times = sorted(all_times)
        
        if self.verbose:
            print(f"\nBacktesting {len(all_times):,} bars")
            print(f"Period: {all_times[0].strftime('%Y-%m-%d')} to {all_times[-1].strftime('%Y-%m-%d')}")
            if self.leverage_enabled:
                print(f"Leverage: ENABLED (conservative overnight rates, maintenance margin: {self.maintenance_margin_pct:.0%})")
            else:
                print(f"Leverage: DISABLED (spot mode)")
            print()
        
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
                current_bar = bar_df.iloc[-1]
                price = float(current_bar['close'])
                high = float(current_bar['high'])
                low = float(current_bar['low'])
                equity = self._calc_equity(data, ts)
                
                # Check exits first (pass high/low for liquidation check)
                if symbol in self.positions:
                    self._check_exit(symbol, price, high, low, ts)
                
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
                self._close(symbol, final_price, all_times[-1], "End of backtest")
        
        return self._compile_results()
    
    def _reset(self):
        """Reset state for new backtest."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.daily_pnl = 0.0
        self.last_daily_reset = None
        self.liquidation_count = 0
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
        
        # Leverage calculations
        if self.leverage_enabled:
            margin_rate = self._get_margin_rate(symbol)
            leverage = 1.0 / margin_rate
            
            # position_value is the margin we're using
            # notional_value is the actual position size (leveraged)
            margin_used = position_value
            notional_value = position_value * leverage
            quantity = notional_value / entry_price
            
            # Calculate liquidation price
            liquidation_price = self._calculate_liquidation_price(
                entry_price, signal.direction, margin_rate, self.maintenance_margin_pct
            )
        else:
            leverage = 1.0
            margin_used = position_value
            notional_value = position_value
            quantity = position_value / entry_price
            liquidation_price = None
        
        commission_cost = quantity * entry_price * self.commission
        
        if margin_used + commission_cost > self.capital:
            return
        
        # Open position - deduct margin (not full notional)
        self.capital -= margin_used + commission_cost
        
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
            capital_used=margin_used,  # This is margin, not notional
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            atr=signal.atr,
            reasons=signal.reasons,
            entry_equity=equity,
            position_size_pct=signal.position_size * 100,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            leverage=leverage,
            margin_used=margin_used,
            notional_value=notional_value,
            liquidation_price=liquidation_price,
        )
        
        self.positions[symbol] = pos
        
        if self.verbose:
            dir_str = "LONG " if signal.direction == 'long' else "SHORT"
            if self.leverage_enabled:
                lev_str = f" ({leverage:.1f}x)"
                liq_str = f" Liq:${liquidation_price:,.2f}" if liquidation_price else ""
                print(f"{ts.strftime('%Y-%m-%d %H:%M')}  {dir_str}{lev_str}  ${entry_price:>10,.2f}  "
                      f"{signal.position_size:>5.0%}  ${equity:>12,.0f}  {' '.join(signal.reasons)}{liq_str}")
            else:
                print(f"{ts.strftime('%Y-%m-%d %H:%M')}  {dir_str}  ${entry_price:>10,.2f}  "
                      f"{signal.position_size:>5.0%}  ${equity:>12,.0f}  {' '.join(signal.reasons)}")
    
    def _check_exit(self, symbol: str, price: float, high: float, low: float, ts: datetime):
        """Check exit conditions including liquidation."""
        pos = self.positions[symbol]
        
        # Time-based exit (max holding period)
        if self.max_hold_days is not None:
            hold_time = ts - pos.entry_time
            if hold_time.total_seconds() >= self.max_hold_days * 24 * 3600:
                self._close(symbol, price, ts, "Time exit")
                return
        
        # Check liquidation first (using high/low for realism)
        if self.leverage_enabled and pos.liquidation_price is not None:
            if pos.direction == 'long' and low <= pos.liquidation_price:
                self._close(symbol, pos.liquidation_price, ts, "LIQUIDATED")
                self.liquidation_count += 1
                return
            elif pos.direction == 'short' and high >= pos.liquidation_price:
                self._close(symbol, pos.liquidation_price, ts, "LIQUIDATED")
                self.liquidation_count += 1
                return
        
        # Stop loss (check with high/low for realism)
        if pos.direction == 'long' and low <= pos.stop_loss:
            # Use stop price, not low (assuming stop order fills at stop)
            self._close(symbol, pos.stop_loss, ts, "Stop loss hit")
            return
        elif pos.direction == 'short' and high >= pos.stop_loss:
            self._close(symbol, pos.stop_loss, ts, "Stop loss hit")
            return
        
        # Take profit (check with high/low)
        if pos.direction == 'long' and high >= pos.take_profit:
            self._close(symbol, pos.take_profit, ts, "Take profit hit")
            return
        elif pos.direction == 'short' and low <= pos.take_profit:
            self._close(symbol, pos.take_profit, ts, "Take profit hit")
            return
    
    def _close(self, symbol: str, price: float, ts: datetime, reason: str):
        """Close position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos.exit_time = ts
        
        # Apply slippage (except for liquidation which uses exact price)
        if reason == "LIQUIDATED":
            exit_price = price
        elif pos.direction == 'long':
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)
        
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.is_open = False
        
        # Calculate P&L (on full notional, not just margin)
        if pos.direction == 'long':
            gross = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity
        
        commission_cost = pos.quantity * exit_price * self.commission
        net = gross - commission_cost
        pos.pnl = net
        
        # P&L percentage is relative to margin used (not notional)
        pos.pnl_pct = (net / pos.margin_used) * 100 if pos.margin_used > 0 else 0
        
        # Calculate R-multiple (reward:risk ratio)
        if pos.risk_amount > 0:
            pos.r_multiple = net / pos.risk_amount
        else:
            pos.r_multiple = 0.0
        
        # Update capital - return margin plus P&L
        self.capital += pos.margin_used + net
        self.daily_pnl += net
        
        # Record result
        self.signal_generator.record_trade_result(net)
        
        self.trades.append(pos)
        del self.positions[symbol]
        
        if self.verbose:
            new_equity = self._calc_equity_now()
            lev_str = f" ({pos.leverage:.1f}x)" if self.leverage_enabled else ""
            print(f"{ts.strftime('%Y-%m-%d %H:%M')}  EXIT{lev_str}   ${exit_price:>10,.2f}  "
                  f"${net:>+9,.2f}  ${new_equity:>12,.0f}  {reason}")
    
    def _calc_equity(self, data: Dict[str, pd.DataFrame], ts: datetime) -> float:
        """Calculate current equity."""
        equity = self.capital
        
        for symbol, pos in self.positions.items():
            if symbol not in data:
                equity += pos.margin_used
                continue
            
            df = data[symbol]
            
            # Find price
            if ts in df.index:
                price = float(df.loc[ts, 'close'])
            else:
                prior = df.index[df.index <= ts]
                if len(prior) > 0:
                    price = float(df.loc[prior[-1], 'close'])
                else:
                    price = pos.entry_price
            
            # Calculate unrealized P&L on full position
            if pos.direction == 'long':
                unrealized = (price - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - price) * pos.quantity
            
            # Equity = margin + unrealized P&L
            equity += pos.margin_used + unrealized
        
        return equity
    
    def _calc_equity_now(self) -> float:
        """Calculate equity without data lookup."""
        return self.capital + sum(p.margin_used for p in self.positions.values())
    
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
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        # Drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        max_dd = abs(drawdown.min())
        
        try:
            daily_equity = equity.resample('D').last().dropna()
            daily_returns = daily_equity.pct_change().dropna()
            
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
        except Exception:
            sharpe = 0
        
        # R-multiple stats
        r_multiples = [t.r_multiple for t in self.trades]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        avg_r_win = sum([t.r_multiple for t in wins]) / len(wins) if wins else 0
        avg_r_loss = sum([t.r_multiple for t in losses]) / len(losses) if losses else 0
        
        # Expectancy
        expectancy = total_pnl / len(self.trades) if self.trades else 0
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        
        if avg_loss > 0 and avg_win > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        else:
            kelly = 0
        kelly_pct = max(0, min(kelly * 100, 100))
        
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
        
        # Leverage-specific stats
        if self.leverage_enabled:
            avg_leverage = sum(t.leverage for t in self.trades) / len(self.trades) if self.trades else 1.0
            liquidations = sum(1 for t in self.trades if t.exit_reason == "LIQUIDATED")
        else:
            avg_leverage = 1.0
            liquidations = 0
        
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
            'avg_hold_hours': avg_hold_hours,
            # Leverage stats
            'leverage_enabled': self.leverage_enabled,
            'avg_leverage': avg_leverage,
            'liquidations': liquidations,
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
            'avg_hold_hours': 0,
            'leverage_enabled': self.leverage_enabled,
            'avg_leverage': 1.0,
            'liquidations': 0,
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
        
        if stats.get('leverage_enabled'):
            print(f"{'='*60}")
            print(f"LEVERAGE STATS (Conservative Overnight Rates)")
            print(f"{'='*60}")
            print(f"Avg Leverage:    {stats['avg_leverage']:.1f}x")
            print(f"Liquidations:    {stats['liquidations']}")
        
        print(f"{'='*60}")