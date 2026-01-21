"""
Volatility-Based Position Sizing
ATR-based и Kelly Criterion sizing
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    ATR_BASED = "atr_based"
    KELLY = "kelly"
    VOLATILITY_PARITY = "volatility_parity"

@dataclass
class SizingConfig:
    # Risk parameters
    max_risk_per_trade: float = 0.02  # 2% риска на сделку
    max_position_size: float = 0.10   # 10% портфеля макс на позицию
    min_position_size: float = 0.01   # 1% портфеля мин
    
    # ATR parameters
    atr_period: int = 14
    atr_multiplier: float = 2.0  # Stop-loss = ATR * multiplier
    
    # Kelly parameters
    kelly_fraction: float = 0.25  # Используем 25% от Kelly (fractional Kelly)
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% годовой волатильности

@dataclass
class PositionSize:
    shares: int
    position_value: float
    position_pct: float
    risk_amount: float
    stop_loss_price: float
    method_used: SizingMethod
    details: dict

class PositionSizer:
    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Average True Range"""
        if len(close) < self.config.atr_period + 1:
            return (high[-1] - low[-1]) if len(high) > 0 else 0
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return np.mean(tr[-self.config.atr_period:])
    
    def calculate_volatility(self, close: np.ndarray, period: int = 20) -> float:
        """Annualized volatility"""
        if len(close) < period + 1:
            return 0.2  # Default 20%
        
        returns = np.diff(np.log(close[-period-1:]))
        daily_vol = np.std(returns)
        return daily_vol * np.sqrt(252)  # Annualized
    
    def atr_based_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        side: str = "long"
    ) -> PositionSize:
        """Position sizing на основе ATR"""
        
        atr = self.calculate_atr(high, low, close)
        stop_distance = atr * self.config.atr_multiplier
        
        if side == "long":
            stop_loss_price = entry_price - stop_distance
        else:
            stop_loss_price = entry_price + stop_distance
        
        # Риск на сделку
        risk_amount = portfolio_value * self.config.max_risk_per_trade
        
        # Размер позиции исходя из риска
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
        else:
            shares = 0
        
        position_value = shares * entry_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Ограничения
        if position_pct > self.config.max_position_size:
            position_pct = self.config.max_position_size
            position_value = portfolio_value * position_pct
            shares = int(position_value / entry_price)
        
        if position_pct < self.config.min_position_size:
            shares = 0
            position_value = 0
            position_pct = 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=risk_amount,
            stop_loss_price=stop_loss_price,
            method_used=SizingMethod.ATR_BASED,
            details={
                "atr": atr,
                "stop_distance": stop_distance,
                "atr_multiplier": self.config.atr_multiplier
            }
        )
    
    def kelly_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> PositionSize:
        """Kelly Criterion sizing"""
        
        if avg_loss == 0:
            kelly_pct = 0
        else:
            # Kelly formula: f = (bp - q) / b
            # b = avg_win / avg_loss (win/loss ratio)
            # p = win_rate
            # q = 1 - p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_pct = max(0, (b * p - q) / b)  # Защита от negative edge
        
        # Fractional Kelly
        position_pct = kelly_pct * self.config.kelly_fraction
        
        # Ограничения
        position_pct = max(0, min(position_pct, self.config.max_position_size))
        
        position_value = portfolio_value * position_pct
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=portfolio_value * self.config.max_risk_per_trade,
            stop_loss_price=0,  # Не определён для Kelly
            method_used=SizingMethod.KELLY,
            details={
                "full_kelly": kelly_pct,
                "fractional_kelly": self.config.kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0
            }
        )
    
    def volatility_parity_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        close: np.ndarray,
        regime_multiplier: float = 1.0
    ) -> PositionSize:
        """Volatility targeting / Risk parity sizing"""
        
        current_vol = self.calculate_volatility(close)
        
        if current_vol == 0:
            vol_adjustment = 1.0
        else:
            vol_adjustment = self.config.target_volatility / current_vol
        
        # Base position size
        base_pct = self.config.max_risk_per_trade * 5  # ~10% base
        
        # Adjust for volatility and regime
        position_pct = base_pct * vol_adjustment * regime_multiplier
        
        # Ограничения
        position_pct = max(
            self.config.min_position_size,
            min(position_pct, self.config.max_position_size)
        )
        
        position_value = portfolio_value * position_pct
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        
        return PositionSize(
            shares=shares,
            position_value=position_value,
            position_pct=position_pct,
            risk_amount=portfolio_value * self.config.max_risk_per_trade,
            stop_loss_price=0,
            method_used=SizingMethod.VOLATILITY_PARITY,
            details={
                "current_volatility": current_vol,
                "target_volatility": self.config.target_volatility,
                "vol_adjustment": vol_adjustment,
                "regime_multiplier": regime_multiplier
            }
        )
    
    def calculate_size(
        self,
        portfolio_value: float,
        entry_price: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        method: SizingMethod = SizingMethod.ATR_BASED,
        side: str = "long",
        regime_multiplier: float = 1.0,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0
    ) -> PositionSize:
        """Универсальный метод расчёта размера позиции"""
        
        if method == SizingMethod.ATR_BASED:
            size = self.atr_based_sizing(portfolio_value, entry_price, high, low, close, side)
        elif method == SizingMethod.KELLY:
            size = self.kelly_sizing(portfolio_value, entry_price, win_rate, avg_win, avg_loss)
        elif method == SizingMethod.VOLATILITY_PARITY:
            size = self.volatility_parity_sizing(portfolio_value, entry_price, close, regime_multiplier)
        else:
            # Fixed fractional
            position_pct = self.config.max_risk_per_trade * 5
            position_value = portfolio_value * position_pct
            shares = int(position_value / entry_price) if entry_price > 0 else 0
            size = PositionSize(
                shares=shares,
                position_value=position_value,
                position_pct=position_pct,
                risk_amount=portfolio_value * self.config.max_risk_per_trade,
                stop_loss_price=0,
                method_used=SizingMethod.FIXED_FRACTIONAL,
                details={}
            )
        
        # Apply regime multiplier
        if regime_multiplier != 1.0 and method != SizingMethod.VOLATILITY_PARITY:
            size.shares = int(size.shares * regime_multiplier)
            size.position_value *= regime_multiplier
            size.position_pct *= regime_multiplier
            size.details["regime_multiplier"] = regime_multiplier
        
        logger.info(f"Position Size: {size.shares} shares ({size.position_pct:.1%}), method: {method.value}")
        
        return size

# Singleton
position_sizer = PositionSizer()
