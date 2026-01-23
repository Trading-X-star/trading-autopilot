"""
Ensemble Orchestrator - объединяет 3 специализированные модели
- TrendModel: определяет направление (BULLISH/NEUTRAL/BEARISH)
- FlatModel: определяет флет (FLAT/TRENDING)  
- VolatilityModel: определяет волатильность (LOW/MEDIUM/HIGH)

Финальный сигнал формируется с учётом всех трёх режимов
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleOrchestrator")

class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class MarketRegime:
    """Текущий режим рынка"""
    trend: str           # BULLISH, NEUTRAL, BEARISH
    trend_confidence: float
    is_flat: bool
    flat_confidence: float
    volatility: str      # LOW, MEDIUM, HIGH
    volatility_confidence: float
    
    @property
    def regime_summary(self) -> str:
        if self.is_flat:
            return f"FLAT/{self.volatility}_VOL"
        return f"{self.trend}/{self.volatility}_VOL"

@dataclass 
class EnsembleSignal:
    """Финальный сигнал от ансамбля"""
    signal: Signal
    strength: float      # 0-1
    confidence: float    # 0-1
    regime: MarketRegime
    reasoning: str
    position_size_multiplier: float  # 0-1.5

class EnsembleOrchestrator:
    """
    Оркестратор ансамбля моделей
    
    Логика принятия решений:
    1. Если FLAT + LOW_VOL → HOLD (ждём пробоя)
    2. Если FLAT + HIGH_VOL → осторожный сигнал по тренду
    3. Если TRENDING + согласие моделей → усиленный сигнал
    4. Если HIGH_VOL → уменьшаем размер позиции
    """
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.trend_model = None
        self.flat_model = None
        self.volatility_model = None
        self.is_loaded = False
        
        # Веса моделей для финального решения
        self.weights = {
            'trend': 0.5,
            'flat': 0.3,
            'volatility': 0.2
        }
        
        # Матрица решений: (trend, is_flat, volatility) -> (signal, size_mult)
        self.decision_matrix = self._build_decision_matrix()
    
    def _build_decision_matrix(self) -> Dict:
        """Матрица решений на основе комбинации режимов"""
        return {
            # BULLISH scenarios
            ('BULLISH', False, 'LOW'):    (Signal.BUY, 1.0),      # Тренд вверх, спокойно
            ('BULLISH', False, 'MEDIUM'): (Signal.BUY, 1.2),      # Идеальные условия
            ('BULLISH', False, 'HIGH'):   (Signal.BUY, 0.7),      # Тренд, но риск
            ('BULLISH', True, 'LOW'):     (Signal.HOLD, 0.5),     # Флет, ждём пробоя
            ('BULLISH', True, 'MEDIUM'):  (Signal.BUY, 0.6),      # Начало движения?
            ('BULLISH', True, 'HIGH'):    (Signal.BUY, 0.5),      # Пробой с волой
            
            # NEUTRAL scenarios  
            ('NEUTRAL', False, 'LOW'):    (Signal.HOLD, 0.3),
            ('NEUTRAL', False, 'MEDIUM'): (Signal.HOLD, 0.3),
            ('NEUTRAL', False, 'HIGH'):   (Signal.HOLD, 0.2),     # Опасно
            ('NEUTRAL', True, 'LOW'):     (Signal.HOLD, 0.0),     # Полный флет
            ('NEUTRAL', True, 'MEDIUM'):  (Signal.HOLD, 0.2),
            ('NEUTRAL', True, 'HIGH'):    (Signal.HOLD, 0.1),
            
            # BEARISH scenarios
            ('BEARISH', False, 'LOW'):    (Signal.SELL, 1.0),
            ('BEARISH', False, 'MEDIUM'): (Signal.SELL, 1.2),
            ('BEARISH', False, 'HIGH'):   (Signal.SELL, 0.7),
            ('BEARISH', True, 'LOW'):     (Signal.HOLD, 0.5),
            ('BEARISH', True, 'MEDIUM'):  (Signal.SELL, 0.6),
            ('BEARISH', True, 'HIGH'):    (Signal.SELL, 0.5),
        }
    
    def load_models(self):
        """Загрузка всех моделей"""
        try:
            from trainers.trend_model import TrendModel
            from trainers.flat_model import FlatModel
            from trainers.volatility_model import VolatilityModel
            
            trend_path = self.models_dir / "trend_model.joblib"
            flat_path = self.models_dir / "flat_model.joblib"
            vol_path = self.models_dir / "volatility_model.joblib"
            
            if trend_path.exists():
                self.trend_model = TrendModel.load(str(trend_path))
                logger.info("✅ Trend model loaded")
            else:
                logger.warning(f"⚠️ Trend model not found: {trend_path}")
                
            if flat_path.exists():
                self.flat_model = FlatModel.load(str(flat_path))
                logger.info("✅ Flat model loaded")
            else:
                logger.warning(f"⚠️ Flat model not found: {flat_path}")
                
            if vol_path.exists():
                self.volatility_model = VolatilityModel.load(str(vol_path))
                logger.info("✅ Volatility model loaded")
            else:
                logger.warning(f"⚠️ Volatility model not found: {vol_path}")
            
            self.is_loaded = all([self.trend_model, self.flat_model, self.volatility_model])
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def analyze_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Анализ текущего режима рынка"""
        # Trend
        if self.trend_model:
            trend_pred = self.trend_model.predict(df)
            trend = trend_pred['trend_label']
            trend_conf = trend_pred['confidence']
        else:
            trend, trend_conf = 'NEUTRAL', 0.5
        
        # Flat
        if self.flat_model:
            flat_pred = self.flat_model.predict(df)
            is_flat = flat_pred['is_flat']
            flat_conf = flat_pred['confidence']
        else:
            is_flat, flat_conf = False, 0.5
        
        # Volatility
        if self.volatility_model:
            vol_pred = self.volatility_model.predict(df)
            volatility = vol_pred['regime_label']
            vol_conf = vol_pred['confidence']
        else:
            volatility, vol_conf = 'MEDIUM', 0.5
        
        return MarketRegime(
            trend=trend,
            trend_confidence=trend_conf,
            is_flat=is_flat,
            flat_confidence=flat_conf,
            volatility=volatility,
            volatility_confidence=vol_conf
        )
    
    def get_signal(self, df: pd.DataFrame) -> EnsembleSignal:
        """Получение финального сигнала"""
        regime = self.analyze_regime(df)
        
        # Lookup в матрице решений
        key = (regime.trend, regime.is_flat, regime.volatility)
        signal, size_mult = self.decision_matrix.get(key, (Signal.HOLD, 0.3))
        
        # Усиление сигнала при высокой уверенности
        avg_confidence = (
            regime.trend_confidence * self.weights['trend'] +
            regime.flat_confidence * self.weights['flat'] +
            regime.volatility_confidence * self.weights['volatility']
        )
        
        # Корректировка размера позиции по волатильности
        vol_adjustment = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.6}
        final_size_mult = size_mult * vol_adjustment.get(regime.volatility, 1.0)
        
        # Strong signals при высокой уверенности
        if avg_confidence > 0.8:
            if signal == Signal.BUY:
                signal = Signal.STRONG_BUY
            elif signal == Signal.SELL:
                signal = Signal.STRONG_SELL
        
        # Формируем reasoning
        reasoning = self._build_reasoning(regime, signal, avg_confidence)
        
        return EnsembleSignal(
            signal=signal,
            strength=abs(signal.value) / 2,  # 0-1
            confidence=avg_confidence,
            regime=regime,
            reasoning=reasoning,
            position_size_multiplier=min(final_size_mult, 1.5)
        )
    
    def _build_reasoning(self, regime: MarketRegime, signal: Signal, confidence: float) -> str:
        """Формирование объяснения решения"""
        parts = []
        
        # Trend
        parts.append(f"Trend: {regime.trend} ({regime.trend_confidence:.0%})")
        
        # Flat
        if regime.is_flat:
            parts.append(f"Market: FLAT ({regime.flat_confidence:.0%})")
        else:
            parts.append(f"Market: TRENDING ({regime.flat_confidence:.0%})")
        
        # Volatility
        parts.append(f"Volatility: {regime.volatility} ({regime.volatility_confidence:.0%})")
        
        # Decision
        parts.append(f"→ Signal: {signal.name}")
        
        return " | ".join(parts)
    
    def predict_batch(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Batch prediction для бэктеста"""
        results = []
        
        for i in range(window, len(df)):
            window_df = df.iloc[i-window:i+1]
            signal = self.get_signal(window_df)
            
            results.append({
                'date': df.index[i] if hasattr(df, 'index') else i,
                'signal': signal.signal.value,
                'signal_name': signal.signal.name,
                'confidence': signal.confidence,
                'regime': signal.regime.regime_summary,
                'size_mult': signal.position_size_multiplier
            })
        
        return pd.DataFrame(results)


# CLI для тестирования
if __name__ == "__main__":
    import sys
    
    orchestrator = EnsembleOrchestrator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Тест на синтетических данных
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 0,
            'low': 0,
            'close': 0,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        df['close'] = df['open'] + np.random.randn(100) * 0.3
        df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(100) * 0.2)
        df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(100) * 0.2)
        
        print("Testing EnsembleOrchestrator...")
        print(f"Models dir: {orchestrator.models_dir}")
        
        if orchestrator.load_models():
            signal = orchestrator.get_signal(df)
            print(f"\n📊 Market Regime: {signal.regime.regime_summary}")
            print(f"📈 Signal: {signal.signal.name}")
            print(f"💪 Confidence: {signal.confidence:.1%}")
            print(f"📐 Position Size: {signal.position_size_multiplier:.1%}")
            print(f"💡 Reasoning: {signal.reasoning}")
        else:
            print("⚠️ Models not trained yet. Run training first.")
    else:
        print("Usage: python ensemble_orchestrator.py test")
