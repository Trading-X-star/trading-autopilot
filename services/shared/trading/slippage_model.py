"""
Realistic Slippage Model for Backtesting
Учитывает: спред, market impact, время суток, волатильность
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    OPENING = "opening"  # Первые 30 мин торгов
    CLOSING = "closing"  # Последние 30 мин торгов

@dataclass
class SlippageConfig:
    # Base slippage (в процентах)
    base_slippage_pct: float = 0.05  # 0.05% базовый slippage
    
    # Spread
    avg_spread_pct: float = 0.1  # 0.1% средний спред
    
    # Market impact parameters
    market_impact_coefficient: float = 0.1  # Impact = coef * sqrt(order_size / avg_volume)
    
    # Time-of-day multipliers
    opening_multiplier: float = 2.0  # Открытие рынка
    closing_multiplier: float = 1.5  # Закрытие рынка
    
    # Volatility multipliers
    high_volatility_multiplier: float = 2.5
    normal_volatility_multiplier: float = 1.0
    
    # Order type adjustments
    limit_order_fill_rate: float = 0.85  # 85% лимитных ордеров исполняются
    stop_order_slippage_multiplier: float = 1.5  # Стопы проскальзывают больше

@dataclass
class SlippageResult:
    expected_price: float
    executed_price: float
    slippage_amount: float
    slippage_pct: float
    total_cost: float
    components: dict

class SlippageModel:
    def __init__(self, config: Optional[SlippageConfig] = None):
        self.config = config or SlippageConfig()
    
    def get_market_condition(
        self,
        timestamp: datetime,
        volatility_ratio: float = 1.0
    ) -> MarketCondition:
        """Определить рыночные условия"""
        
        trade_time = timestamp.time()
        
        # MOEX: 10:00 - 18:50 основная сессия
        opening_start = time(10, 0)
        opening_end = time(10, 30)
        closing_start = time(18, 20)
        closing_end = time(18, 50)
        
        if opening_start <= trade_time <= opening_end:
            return MarketCondition.OPENING
        elif closing_start <= trade_time <= closing_end:
            return MarketCondition.CLOSING
        elif volatility_ratio > 1.5:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.NORMAL
    
    def calculate_spread_cost(
        self,
        price: float,
        side: str,
        condition: MarketCondition
    ) -> float:
        """Рассчитать стоимость спреда"""
        
        spread_pct = self.config.avg_spread_pct
        
        # Adjust for market condition
        if condition == MarketCondition.OPENING:
            spread_pct *= 1.5
        elif condition == MarketCondition.CLOSING:
            spread_pct *= 1.2
        elif condition == MarketCondition.VOLATILE:
            spread_pct *= 2.0
        elif condition == MarketCondition.ILLIQUID:
            spread_pct *= 3.0
        
        # Half spread for execution
        half_spread = (spread_pct / 100) * price / 2
        
        return half_spread if side == "buy" else -half_spread
    
    def calculate_market_impact(
        self,
        price: float,
        order_size: float,
        avg_daily_volume: float,
        side: str
    ) -> float:
        """Рассчитать market impact"""
        
        if avg_daily_volume == 0:
            participation_rate = 0.01
        else:
            participation_rate = order_size / avg_daily_volume
        
        # Square root market impact model
        impact_pct = self.config.market_impact_coefficient * np.sqrt(participation_rate)
        
        impact_amount = (impact_pct / 100) * price
        
        return impact_amount if side == "buy" else -impact_amount
    
    def calculate_volatility_slippage(
        self,
        price: float,
        volatility_ratio: float,
        side: str
    ) -> float:
        """Slippage из-за волатильности"""
        
        if volatility_ratio > 1.5:
            multiplier = self.config.high_volatility_multiplier
        else:
            multiplier = self.config.normal_volatility_multiplier
        
        base_slip = (self.config.base_slippage_pct / 100) * price * multiplier
        
        # Random component (symmetric around expected slippage)
        random_factor = np.random.normal(1.0, 0.3)
        random_factor = max(0.5, min(1.5, random_factor))
        
        slippage = base_slip * random_factor
        
        return slippage if side == "buy" else -slippage
    
    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,  # "buy" or "sell"
        order_type: OrderType = OrderType.MARKET,
        timestamp: Optional[datetime] = None,
        avg_daily_volume: float = 1_000_000,
        volatility_ratio: float = 1.0,
        include_random: bool = True
    ) -> SlippageResult:
        """
        Рассчитать полный slippage для ордера
        
        Returns executed price with all slippage components
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        order_value = price * quantity
        condition = self.get_market_condition(timestamp, volatility_ratio)
        
        # Components
        spread_cost = self.calculate_spread_cost(price, side, condition)
        market_impact = self.calculate_market_impact(price, order_value, avg_daily_volume, side)
        vol_slippage = self.calculate_volatility_slippage(price, volatility_ratio, side) if include_random else 0
        
        # Time-of-day adjustment
        if condition == MarketCondition.OPENING:
            time_multiplier = self.config.opening_multiplier
        elif condition == MarketCondition.CLOSING:
            time_multiplier = self.config.closing_multiplier
        else:
            time_multiplier = 1.0
        
        # Order type adjustment
        if order_type == OrderType.STOP:
            type_multiplier = self.config.stop_order_slippage_multiplier
        else:
            type_multiplier = 1.0
        
        # Total slippage
        base_slippage = spread_cost + market_impact + vol_slippage
        total_slippage = base_slippage * time_multiplier * type_multiplier
        
        # Executed price
        executed_price = price + total_slippage
        
        # Ensure price is positive
        executed_price = max(0.01, executed_price)
        
        slippage_pct = (total_slippage / price) * 100
        total_cost = abs(total_slippage) * quantity
        
        logger.debug(
            f"Slippage: {side} {quantity}x{price:.2f} -> {executed_price:.2f} "
            f"({slippage_pct:+.3f}%, condition={condition.value})"
        )
        
        return SlippageResult(
            expected_price=price,
            executed_price=executed_price,
            slippage_amount=total_slippage,
            slippage_pct=slippage_pct,
            total_cost=total_cost,
            components={
                "spread": spread_cost,
                "market_impact": market_impact,
                "volatility": vol_slippage,
                "time_multiplier": time_multiplier,
                "type_multiplier": type_multiplier,
                "condition": condition.value
            }
        )
    
    def simulate_limit_order_fill(
        self,
        limit_price: float,
        current_price: float,
        side: str,
        time_in_market_hours: float = 1.0
    ) -> tuple[bool, float]:
        """
        Симулировать исполнение лимитного ордера
        Returns: (filled, fill_price)
        """
        
        # Базовая вероятность исполнения
        base_fill_prob = self.config.limit_order_fill_rate
        
        # Adjust for distance from market
        distance_pct = abs(limit_price - current_price) / current_price * 100
        
        if side == "buy":
            # Покупка: лимит ниже рынка - выше шанс исполнения если цена упадёт
            if limit_price >= current_price:
                fill_prob = 0.99  # Сразу исполнится
            else:
                fill_prob = base_fill_prob * np.exp(-distance_pct * 0.5)
        else:
            # Продажа: лимит выше рынка
            if limit_price <= current_price:
                fill_prob = 0.99
            else:
                fill_prob = base_fill_prob * np.exp(-distance_pct * 0.5)
        
        # Adjust for time
        fill_prob = min(0.99, fill_prob * (1 + 0.1 * time_in_market_hours))
        
        filled = np.random.random() < fill_prob
        
        if filled:
            # Small improvement possible for limit orders
            improvement = np.random.uniform(0, 0.01) * limit_price
            fill_price = limit_price - improvement if side == "buy" else limit_price + improvement
        else:
            fill_price = 0
        
        return filled, fill_price

# Singleton
slippage_model = SlippageModel()
