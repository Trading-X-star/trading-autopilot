import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    account_id: str

@dataclass
class RiskDecision:
    allowed: bool
    reason: Optional[str] = None
    adjusted_quantity: Optional[float] = None

class AdvancedRiskManager:
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        
        # Конфигурируемые параметры
        self.max_drawdown_pct = float(self.r.get("RISK:MAX_DRAWDOWN") or 5.0)
        self.max_position_pct = float(self.r.get("RISK:MAX_POSITION") or 10.0)
        self.daily_loss_limit_pct = float(self.r.get("RISK:DAILY_LOSS") or 2.0)
        self.max_correlation = float(self.r.get("RISK:MAX_CORRELATION") or 0.7)
        self.volatility_threshold = float(self.r.get("RISK:VOL_THRESHOLD") or 30.0)
    
    async def check_trade(self, trade: Trade) -> RiskDecision:
        """Комплексная проверка сделки"""
        
        # Проверка Kill Switch
        if self.r.get("TRADING_ENABLED") != "true":
            return RiskDecision(False, "Trading is disabled by kill switch")
        
        checks = await asyncio.gather(
            self.check_position_size(trade),
            self.check_drawdown(trade.account_id),
            self.check_daily_loss(trade.account_id),
            self.check_volatility_regime(trade.symbol),
            self.check_concentration_risk(trade),
            return_exceptions=True
        )
        
        for i, result in enumerate(checks):
            if isinstance(result, Exception):
                logger.error(f"Risk check {i} failed: {result}")
                return RiskDecision(False, f"Risk check error: {result}")
            if not result[0]:
                return RiskDecision(False, result[1])
        
        return RiskDecision(True)
    
    async def check_position_size(self, trade: Trade) -> tuple:
        portfolio_value = float(self.r.get(f"PORTFOLIO:{trade.account_id}:VALUE") or 0)
        if portfolio_value == 0:
            return (False, "Portfolio value unknown")
        
        trade_value = trade.quantity * trade.price
        position_pct = (trade_value / portfolio_value) * 100
        
        if position_pct > self.max_position_pct:
            return (False, f"Position size {position_pct:.1f}% exceeds limit {self.max_position_pct}%")
        return (True, None)
    
    async def check_drawdown(self, account_id: str) -> tuple:
        peak = float(self.r.get(f"PORTFOLIO:{account_id}:PEAK") or 0)
        current = float(self.r.get(f"PORTFOLIO:{account_id}:VALUE") or 0)
        
        if peak == 0:
            return (True, None)
        
        drawdown = ((peak - current) / peak) * 100
        if drawdown > self.max_drawdown_pct:
            return (False, f"Drawdown {drawdown:.1f}% exceeds limit {self.max_drawdown_pct}%")
        return (True, None)
    
    async def check_daily_loss(self, account_id: str) -> tuple:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_pnl = float(self.r.get(f"PNL:{account_id}:{today}") or 0)
        portfolio_value = float(self.r.get(f"PORTFOLIO:{account_id}:VALUE") or 1)
        
        daily_loss_pct = abs(min(0, daily_pnl)) / portfolio_value * 100
        if daily_loss_pct > self.daily_loss_limit_pct:
            return (False, f"Daily loss {daily_loss_pct:.1f}% exceeds limit {self.daily_loss_limit_pct}%")
        return (True, None)
    
    async def check_volatility_regime(self, symbol: str) -> tuple:
        rvi = float(self.r.get("MARKET:RVI") or 20)
        if rvi > self.volatility_threshold:
            return (False, f"High volatility regime (RVI={rvi:.1f}), trading paused")
        return (True, None)
    
    async def check_concentration_risk(self, trade: Trade) -> tuple:
        """Проверка концентрации по секторам"""
        sector = self.r.get(f"SYMBOL:{trade.symbol}:SECTOR") or "unknown"
        sector_exposure = float(self.r.get(f"EXPOSURE:{trade.account_id}:{sector}") or 0)
        
        if sector_exposure > 30:  # 30% max на сектор
            return (False, f"Sector {sector} exposure {sector_exposure:.1f}% too high")
        return (True, None)
