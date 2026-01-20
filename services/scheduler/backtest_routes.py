from fastapi import APIRouter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

router = APIRouter()

# ============================================================
# BACKTEST ENGINE
# ============================================================
@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    initial_capital: float = 1000000
    final_capital: float = 1000000


class BacktestEngine:
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]
    
    async def run(self, db_pool, start_date: str, end_date: str, tickers: List[str] = None) -> BacktestResult:
        capital = self.initial_capital
        self.trades = []
        tickers = tickers or ['SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK']
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT date, ticker, open, close FROM features
                WHERE date BETWEEN $1 AND $2 AND ticker = ANY($3)
                ORDER BY date, ticker
            """, start_date, end_date, tickers)
        
        if not rows:
            return BacktestResult()
        
        positions = {}
        for row in rows:
            ticker = row['ticker']
            close = float(row['close'] or 0)
            open_p = float(row['open'] or 0)
            
            if open_p == 0 or close == 0:
                continue
            
            change = (close - open_p) / open_p
            
            # Simple contrarian strategy
            if ticker not in positions and change < -0.03:
                qty = int(capital * 0.1 / close)
                if qty > 0:
                    positions[ticker] = {'entry': close, 'qty': qty, 'date': str(row['date'])}
                    capital -= close * qty
            elif ticker in positions and change > 0.02:
                pos = positions.pop(ticker)
                pnl = (close - pos['entry']) * pos['qty']
                capital += pos['entry'] * pos['qty'] + pnl
                self.trades.append({
                    'ticker': ticker,
                    'entry': pos['entry'],
                    'exit': close,
                    'qty': pos['qty'],
                    'pnl': round(pnl, 2),
                    'entry_date': pos['date'],
                    'exit_date': str(row['date'])
                })
                self.equity_curve.append(capital)
        
        # Close remaining positions
        for ticker, pos in list(positions.items()):
            capital += pos['entry'] * pos['qty']
        
        pnls = [t['pnl'] for t in self.trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]
        
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        
        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(len(winning)/len(pnls), 4) if pnls else 0,
            total_pnl=round(sum(pnls), 2),
            profit_factor=round(gross_profit/gross_loss, 2) if gross_loss > 0 else 0,
            initial_capital=self.initial_capital,
            final_capital=round(capital, 2)
        )


# ============================================================
# AUTO-EXECUTE ENGINE  
# ============================================================
class AutoExecuteEngine:
    def __init__(self):
        self.mode = "paper"
        self.enabled = False
        self.executed_orders = []
        self.daily_count = 0
        self.min_confidence = 0.25
        self.max_daily = 20
    
    def get_status(self):
        return {
            'mode': self.mode,
            'enabled': self.enabled,
            'daily_executions': self.daily_count,
            'max_daily': self.max_daily,
            'min_confidence': self.min_confidence,
            'total_orders': len(self.executed_orders)
        }
    
    async def execute(self, signals, risk_manager, prices):
        if not self.enabled:
            return []
        
        results = []
        for ticker, signal in signals.items():
            if self.daily_count >= self.max_daily:
                break
            
            if signal.signal.value in ['strong_buy', 'buy'] and signal.confidence >= self.min_confidence:
                price = prices.get(ticker, signal.entry_price)
                order = {
                    'ticker': ticker,
                    'side': 'BUY',
                    'signal': signal.signal.value,
                    'price': price,
                    'confidence': round(signal.confidence, 3),
                    'target': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'status': 'paper_filled' if self.mode == 'paper' else 'pending',
                    'timestamp': datetime.now().isoformat()
                }
                self.executed_orders.append(order)
                self.daily_count += 1
                results.append(order)
        
        return results
    
    def reset_daily(self):
        self.daily_count = 0


# Global instances
backtest_engine = BacktestEngine()
auto_execute_engine = AutoExecuteEngine()


def get_engines():
    return backtest_engine, auto_execute_engine
