"""Smart Executor - ML-оптимизированное исполнение ордеров"""
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

import numpy as np

logger = logging.getLogger("smart-executor")

class ExecutionStrategy(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"         # Time-weighted average price
    VWAP = "vwap"         # Volume-weighted average price
    ICEBERG = "iceberg"   # Скрытый объем

@dataclass
class Order:
    ticker: str
    side: str  # buy/sell
    quantity: int
    price: float
    account_id: str
    strategy: ExecutionStrategy = ExecutionStrategy.MARKET

@dataclass  
class ExecutionResult:
    success: bool
    filled_quantity: int
    avg_price: float
    slippage_pct: float
    strategy_used: str
    execution_time_ms: float
    child_orders: int
    message: str

class SlippagePredictor:
    """Предсказание slippage на основе рыночных условий"""
    
    def __init__(self):
        self.history = []
    
    def predict(self, price: float, quantity: int, side: str, 
                avg_daily_volume: float, volatility: float) -> float:
        """Предсказать ожидаемый slippage"""
        # Базовый slippage
        base_slippage = 0.05  # 0.05%
        
        # Размер относительно объема
        volume_ratio = quantity / max(avg_daily_volume / 390, 1)  # 390 минут в сессии
        volume_impact = min(volume_ratio * 0.1, 0.5)  # До 0.5%
        
        # Влияние волатильности
        volatility_impact = volatility * 0.5  # Половина дневной волатильности
        
        total = base_slippage + volume_impact + volatility_impact
        
        # Направление
        if side == "buy":
            return total
        else:
            return -total
    
    def record(self, expected: float, actual: float):
        self.history.append({"expected": expected, "actual": actual})
        if len(self.history) > 1000:
            self.history.pop(0)

class SmartExecutor:
    """Умное исполнение с выбором оптимальной стратегии"""
    
    LARGE_ORDER_THRESHOLD = 100000  # 100k руб
    HIGH_SLIPPAGE_THRESHOLD = 0.3   # 0.3%
    TWAP_INTERVALS = 5              # Разбить на 5 частей
    
    def __init__(self):
        self.slippage_predictor = SlippagePredictor()
        self.execution_stats = {
            "total_orders": 0,
            "total_volume": 0,
            "avg_slippage": 0,
            "strategies_used": {}
        }
    
    async def execute(self, order: Order, market_data: dict) -> ExecutionResult:
        """Умное исполнение с автовыбором стратегии"""
        start = datetime.now()
        
        order_value = order.quantity * order.price
        volatility = market_data.get("volatility", 0.02)
        avg_volume = market_data.get("avg_daily_volume", 1000000)
        
        # Предсказание slippage
        expected_slippage = self.slippage_predictor.predict(
            order.price, order.quantity, order.side, avg_volume, volatility
        )
        
        # Выбор стратегии
        if order.strategy != ExecutionStrategy.MARKET:
            strategy = order.strategy
        elif order_value > self.LARGE_ORDER_THRESHOLD:
            strategy = ExecutionStrategy.TWAP
            logger.info(f"Large order {order_value:.0f} -> TWAP")
        elif abs(expected_slippage) > self.HIGH_SLIPPAGE_THRESHOLD:
            strategy = ExecutionStrategy.LIMIT
            logger.info(f"High slippage {expected_slippage:.2f}% -> LIMIT")
        else:
            strategy = ExecutionStrategy.MARKET
        
        # Исполнение
        if strategy == ExecutionStrategy.TWAP:
            result = await self._execute_twap(order, market_data)
        elif strategy == ExecutionStrategy.LIMIT:
            result = await self._execute_limit(order, market_data)
        elif strategy == ExecutionStrategy.ICEBERG:
            result = await self._execute_iceberg(order, market_data)
        else:
            result = await self._execute_market(order, market_data)
        
        # Статистика
        execution_time = (datetime.now() - start).total_seconds() * 1000
        result.execution_time_ms = execution_time
        result.strategy_used = strategy.value
        
        self._update_stats(order, result)
        
        return result
    
    async def _execute_market(self, order: Order, market_data: dict) -> ExecutionResult:
        """Маркет ордер"""
        # Симуляция исполнения (в реальности через API брокера)
        slippage = np.random.uniform(0.01, 0.1)  # 0.01-0.1%
        fill_price = order.price * (1 + slippage/100 if order.side == "buy" else 1 - slippage/100)
        
        return ExecutionResult(
            success=True,
            filled_quantity=order.quantity,
            avg_price=fill_price,
            slippage_pct=slippage,
            strategy_used="market",
            execution_time_ms=0,
            child_orders=1,
            message="Market order filled"
        )
    
    async def _execute_limit(self, order: Order, market_data: dict) -> ExecutionResult:
        """Лимитный ордер с улучшением цены"""
        # Лимитная цена с небольшим улучшением
        improvement = 0.05  # 0.05%
        if order.side == "buy":
            limit_price = order.price * (1 - improvement/100)
        else:
            limit_price = order.price * (1 + improvement/100)
        
        # Симуляция: 80% вероятность исполнения
        filled = np.random.random() < 0.8
        
        if filled:
            # Исполнение по лимитной цене или лучше
            fill_price = limit_price
            slippage = (fill_price / order.price - 1) * 100
            
            return ExecutionResult(
                success=True,
                filled_quantity=order.quantity,
                avg_price=fill_price,
                slippage_pct=slippage,
                strategy_used="limit",
                execution_time_ms=0,
                child_orders=1,
                message="Limit order filled"
            )
        else:
            return ExecutionResult(
                success=False,
                filled_quantity=0,
                avg_price=0,
                slippage_pct=0,
                strategy_used="limit",
                execution_time_ms=0,
                child_orders=1,
                message="Limit order not filled"
            )
    
    async def _execute_twap(self, order: Order, market_data: dict) -> ExecutionResult:
        """TWAP - разбивка по времени"""
        intervals = self.TWAP_INTERVALS
        qty_per_interval = order.quantity // intervals
        remainder = order.quantity % intervals
        
        total_filled = 0
        total_cost = 0
        
        for i in range(intervals):
            qty = qty_per_interval + (1 if i < remainder else 0)
            if qty == 0:
                continue
            
            # Симуляция цены с небольшим дрейфом
            drift = np.random.uniform(-0.1, 0.1)
            interval_price = order.price * (1 + drift/100)
            
            total_filled += qty
            total_cost += qty * interval_price
            
            # Имитация задержки между интервалами
            await asyncio.sleep(0.01)
        
        avg_price = total_cost / total_filled if total_filled > 0 else order.price
        slippage = (avg_price / order.price - 1) * 100
        
        return ExecutionResult(
            success=True,
            filled_quantity=total_filled,
            avg_price=avg_price,
            slippage_pct=slippage,
            strategy_used="twap",
            execution_time_ms=0,
            child_orders=intervals,
            message=f"TWAP executed in {intervals} intervals"
        )
    
    async def _execute_iceberg(self, order: Order, market_data: dict) -> ExecutionResult:
        """Iceberg - скрытый объем"""
        visible_ratio = 0.1  # Показывать 10% объема
        visible_qty = max(1, int(order.quantity * visible_ratio))
        
        total_filled = 0
        total_cost = 0
        child_orders = 0
        
        remaining = order.quantity
        while remaining > 0:
            qty = min(visible_qty, remaining)
            
            # Исполнение видимой части
            price_impact = np.random.uniform(0, 0.05)  # Меньше impact
            fill_price = order.price * (1 + price_impact/100 if order.side == "buy" else 1 - price_impact/100)
            
            total_filled += qty
            total_cost += qty * fill_price
            remaining -= qty
            child_orders += 1
            
            await asyncio.sleep(0.005)
        
        avg_price = total_cost / total_filled if total_filled > 0 else order.price
        slippage = (avg_price / order.price - 1) * 100
        
        return ExecutionResult(
            success=True,
            filled_quantity=total_filled,
            avg_price=avg_price,
            slippage_pct=slippage,
            strategy_used="iceberg",
            execution_time_ms=0,
            child_orders=child_orders,
            message=f"Iceberg executed in {child_orders} slices"
        )
    
    def _update_stats(self, order: Order, result: ExecutionResult):
        self.execution_stats["total_orders"] += 1
        self.execution_stats["total_volume"] += order.quantity * order.price
        
        # Running average slippage
        n = self.execution_stats["total_orders"]
        old_avg = self.execution_stats["avg_slippage"]
        self.execution_stats["avg_slippage"] = old_avg + (result.slippage_pct - old_avg) / n
        
        # Strategy counts
        strategy = result.strategy_used
        if strategy not in self.execution_stats["strategies_used"]:
            self.execution_stats["strategies_used"][strategy] = 0
        self.execution_stats["strategies_used"][strategy] += 1
    
    def get_stats(self) -> dict:
        return self.execution_stats

# Global instance
smart_executor = SmartExecutor()
