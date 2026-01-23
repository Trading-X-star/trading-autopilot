"""Backtesting system for ML ensemble"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_orchestrator import EnsembleOrchestrator, Signal

@dataclass
class Trade:
    entry_date: datetime
    entry_price: float
    direction: int  # 1 = long, -1 = short
    size: float
    signal_confidence: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_holding_period: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    daily_returns: pd.Series = None

class Backtester:
    """Backtests ML ensemble signals on historical data"""
    
    def __init__(self, orchestrator: EnsembleOrchestrator,
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 0.1%
                 slippage: float = 0.0005):  # 0.05%
        self.orchestrator = orchestrator
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Trading rules
        self.min_confidence = 0.55
        self.max_position_pct = 0.2  # Max 20% per position
        self.stop_loss_pct = 0.03   # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        # Time filters
        self.skip_monday_morning = True
        self.skip_friday_afternoon = True
    
    def run(self, df: pd.DataFrame, window: int = 50) -> BacktestResult:
        """Run backtest on historical data"""
        trades = []
        equity = [self.initial_capital]
        daily_returns = []
        current_position = None
        
        for i in range(window, len(df) - 1):
            date = df.index[i] if hasattr(df.index, 'date') else df.iloc[i].get('date', i)
            current_price = float(df.iloc[i]['close'])
            next_price = float(df.iloc[i + 1]['close'])
            
            # Time filter
            if self._should_skip_time(date):
                daily_returns.append(0)
                equity.append(equity[-1])
                continue
            
            # Check stop loss / take profit
            if current_position:
                pnl_pct = (current_price - current_position.entry_price) / current_position.entry_price
                pnl_pct *= current_position.direction
                
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    # Close position
                    current_position = self._close_position(
                        current_position, date, current_price
                    )
                    trades.append(current_position)
                    current_position = None
            
            # Get signal
            window_df = df.iloc[i-window:i+1].copy()
            signal = self.orchestrator.get_signal(window_df)
            
            # Check if should trade
            if self._should_trade(signal, current_position):
                if current_position:
                    # Close existing position
                    current_position = self._close_position(
                        current_position, date, current_price
                    )
                    trades.append(current_position)
                
                # Open new position
                if signal.signal in [Signal.BUY, Signal.STRONG_BUY]:
                    direction = 1
                elif signal.signal in [Signal.SELL, Signal.STRONG_SELL]:
                    direction = -1
                else:
                    direction = 0
                
                if direction != 0:
                    size = self._calculate_position_size(
                        equity[-1], signal.position_size_multiplier
                    )
                    current_position = Trade(
                        entry_date=date,
                        entry_price=current_price * (1 + self.slippage * direction),
                        direction=direction,
                        size=size,
                        signal_confidence=signal.confidence
                    )
            
            # Calculate daily P&L
            if current_position:
                daily_pnl = (next_price - current_price) * current_position.direction
                daily_pnl_pct = daily_pnl / current_price * current_position.size / equity[-1]
                daily_returns.append(daily_pnl_pct)
                equity.append(equity[-1] * (1 + daily_pnl_pct))
            else:
                daily_returns.append(0)
                equity.append(equity[-1])
        
        # Close final position
        if current_position:
            current_position = self._close_position(
                current_position, 
                df.index[-1] if hasattr(df.index, 'date') else len(df)-1,
                float(df.iloc[-1]['close'])
            )
            trades.append(current_position)
        
        return self._calculate_metrics(trades, equity, daily_returns)
    
    def _should_skip_time(self, date) -> bool:
        """Check if should skip based on time filters"""
        if not hasattr(date, 'weekday'):
            return False
        if self.skip_monday_morning and date.weekday() == 0:
            return True
        if self.skip_friday_afternoon and date.weekday() == 4:
            return True
        return False
    
    def _should_trade(self, signal, current_position) -> bool:
        """Determine if should execute trade"""
        # Minimum confidence
        if signal.confidence < self.min_confidence:
            return False
        
        # Validation must pass
        if not signal.validation_passed:
            return False
        
        # HOLD means no action
        if signal.signal == Signal.HOLD:
            return False
        
        # Don't open same direction
        if current_position:
            if signal.signal in [Signal.BUY, Signal.STRONG_BUY] and current_position.direction == 1:
                return False
            if signal.signal in [Signal.SELL, Signal.STRONG_SELL] and current_position.direction == -1:
                return False
        
        return True
    
    def _calculate_position_size(self, equity: float, size_mult: float) -> float:
        """Calculate position size"""
        base_size = equity * self.max_position_pct
        return base_size * size_mult
    
    def _close_position(self, trade: Trade, date, price: float) -> Trade:
        """Close a position"""
        exit_price = price * (1 - self.slippage * trade.direction)
        pnl = (exit_price - trade.entry_price) * trade.direction * trade.size
        pnl -= trade.size * self.commission * 2  # Entry + exit commission
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * trade.direction
        
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        return trade
    
    def _calculate_metrics(self, trades: List[Trade], equity: List[float], 
                          daily_returns: List[float]) -> BacktestResult:
        """Calculate backtest metrics"""
        equity_series = pd.Series(equity)
        returns_series = pd.Series(daily_returns)
        
        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe (annualized)
        if returns_series.std() > 0:
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = returns_series.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = sharpe
        
        # Max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_pnl = np.mean([t.pnl for t in trades if t.pnl]) if trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=avg_pnl,
            avg_holding_period=0,  # TODO
            trades=trades,
            equity_curve=equity_series,
            daily_returns=returns_series
        )
    
    def print_report(self, result: BacktestResult):
        """Print backtest report"""
        print("=" * 50)
        print("BACKTEST REPORT")
        print("=" * 50)
        print(f"Total Return:     {result.total_return:>10.2%}")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:    {result.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:     {result.max_drawdown:>10.2%}")
        print(f"Win Rate:         {result.win_rate:>10.2%}")
        print(f"Profit Factor:    {result.profit_factor:>10.2f}")
        print(f"Total Trades:     {result.total_trades:>10}")
        print(f"Avg Trade P&L:    {result.avg_trade_pnl:>10.2f}")
        print("=" * 50)
