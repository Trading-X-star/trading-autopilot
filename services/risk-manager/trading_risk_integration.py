"""
Integration layer for trading risk components
"""
import sys
sys.path.append('/app/shared')

from trading.regime_detector import regime_detector, MarketRegime
from trading.position_sizing import position_sizer, SizingMethod
from trading.correlation_analyzer import correlation_analyzer
from trading.slippage_model import slippage_model, OrderType

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeRiskAssessment:
    approved: bool
    position_size: int
    position_value: float
    stop_loss: float
    regime: str
    regime_multiplier: float
    correlation_ok: bool
    expected_slippage_pct: float
    warnings: list
    details: dict

class TradingRiskIntegration:
    def __init__(self):
        self.regime_detector = regime_detector
        self.position_sizer = position_sizer
        self.correlation_analyzer = correlation_analyzer
        self.slippage_model = slippage_model
    
    def assess_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        avg_daily_volume: float = 1_000_000
    ) -> TradeRiskAssessment:
        """Полная оценка рисков сделки"""
        
        warnings = []
        
        # 1. Определить режим рынка
        regime_analysis = self.regime_detector.detect_regime(high, low, close)
        regime_multiplier = regime_analysis.recommended_position_multiplier
        
        if regime_analysis.regime == MarketRegime.CRISIS:
            warnings.append("🚨 CRISIS режим - торговля приостановлена")
            return TradeRiskAssessment(
                approved=False,
                position_size=0,
                position_value=0,
                stop_loss=0,
                regime=regime_analysis.regime.value,
                regime_multiplier=0,
                correlation_ok=False,
                expected_slippage_pct=0,
                warnings=warnings,
                details={"regime_analysis": regime_analysis.__dict__}
            )
        
        if regime_analysis.regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("⚠️ Высокая волатильность - размер позиции уменьшен")
        
        # 2. Рассчитать размер позиции
        position = self.position_sizer.calculate_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            high=high,
            low=low,
            close=close,
            method=SizingMethod.ATR_BASED,
            side=side,
            regime_multiplier=regime_multiplier
        )
        
        if position.shares == 0:
            warnings.append("Размер позиции слишком мал")
            return TradeRiskAssessment(
                approved=False,
                position_size=0,
                position_value=0,
                stop_loss=0,
                regime=regime_analysis.regime.value,
                regime_multiplier=regime_multiplier,
                correlation_ok=True,
                expected_slippage_pct=0,
                warnings=warnings,
                details={}
            )
        
        # 3. Проверить корреляции
        self.correlation_analyzer.update_prices(symbol, close)
        correlation_ok, corr_message = self.correlation_analyzer.should_allow_trade(
            new_symbol=symbol,
            new_value=position.position_value,
            current_positions=current_positions,
            portfolio_value=portfolio_value
        )
        
        if not correlation_ok:
            warnings.append(corr_message)
        
        # 4. Оценить slippage
        slippage = self.slippage_model.calculate_slippage(
            price=entry_price,
            quantity=position.shares,
            side=side,
            avg_daily_volume=avg_daily_volume,
            volatility_ratio=regime_analysis.volatility_ratio,
            include_random=False
        )
        
        if slippage.slippage_pct > 0.5:
            warnings.append(f"⚠️ Высокий ожидаемый slippage: {slippage.slippage_pct:.2f}%")
        
        # Final decision
        approved = correlation_ok and position.shares > 0
        
        return TradeRiskAssessment(
            approved=approved,
            position_size=position.shares,
            position_value=position.position_value,
            stop_loss=position.stop_loss_price,
            regime=regime_analysis.regime.value,
            regime_multiplier=regime_multiplier,
            correlation_ok=correlation_ok,
            expected_slippage_pct=slippage.slippage_pct,
            warnings=warnings,
            details={
                "regime": regime_analysis.__dict__,
                "position": position.__dict__,
                "slippage": slippage.__dict__
            }
        )

# Singleton
trading_risk = TradingRiskIntegration()
