"""
Enhanced Ensemble Orchestrator v2
- Adaptive weights based on regime
- Signal validation
- Drift monitoring
- Meta-model support
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import joblib
import logging
import json

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
    trend: str
    trend_confidence: float
    trend_probabilities: Dict[str, float]
    is_flat: bool
    flat_confidence: float
    flat_probability: float
    volatility: str
    volatility_confidence: float
    volatility_probabilities: Dict[str, float]
    
    @property
    def regime_summary(self) -> str:
        if self.is_flat:
            return f"FLAT/{self.volatility}_VOL"
        return f"{self.trend}/{self.volatility}_VOL"
    
    @property
    def risk_level(self) -> str:
        if self.volatility == 'HIGH':
            return 'HIGH'
        if self.is_flat:
            return 'LOW'
        return 'MEDIUM'

@dataclass 
class EnsembleSignal:
    signal: Signal
    strength: float
    confidence: float
    regime: MarketRegime
    reasoning: str
    position_size_multiplier: float
    validation_passed: bool
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class EnsembleOrchestrator:
    """Enhanced Ensemble Orchestrator with validation and monitoring"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.trend_model = None
        self.flat_model = None
        self.volatility_model = None
        self.signal_model = None
        self.is_loaded = False
        
        # Adaptive weights
        self.base_weights = {'trend': 0.5, 'flat': 0.3, 'volatility': 0.2}
        
        # Decision matrix
        self.decision_matrix = self._build_decision_matrix()
        
        # Validation thresholds
        self.min_confidence = 0.55
        self.high_vol_confidence = 0.75
        
        # Drift monitoring
        self.prediction_history = []
        self.drift_window = 100
    
    def _build_decision_matrix(self) -> Dict:
        return {
            # (trend, is_flat, volatility) -> (signal, size_mult)
            ('BULLISH', False, 'LOW'):    (Signal.BUY, 1.2),
            ('BULLISH', False, 'MEDIUM'): (Signal.BUY, 1.0),
            ('BULLISH', False, 'HIGH'):   (Signal.BUY, 0.5),
            ('BULLISH', True, 'LOW'):     (Signal.HOLD, 0.3),
            ('BULLISH', True, 'MEDIUM'):  (Signal.BUY, 0.5),
            ('BULLISH', True, 'HIGH'):    (Signal.BUY, 0.4),
            
            ('NEUTRAL', False, 'LOW'):    (Signal.HOLD, 0.2),
            ('NEUTRAL', False, 'MEDIUM'): (Signal.HOLD, 0.2),
            ('NEUTRAL', False, 'HIGH'):   (Signal.HOLD, 0.0),
            ('NEUTRAL', True, 'LOW'):     (Signal.HOLD, 0.0),
            ('NEUTRAL', True, 'MEDIUM'):  (Signal.HOLD, 0.1),
            ('NEUTRAL', True, 'HIGH'):    (Signal.HOLD, 0.0),
            
            ('BEARISH', False, 'LOW'):    (Signal.SELL, 1.2),
            ('BEARISH', False, 'MEDIUM'): (Signal.SELL, 1.0),
            ('BEARISH', False, 'HIGH'):   (Signal.SELL, 0.5),
            ('BEARISH', True, 'LOW'):     (Signal.HOLD, 0.3),
            ('BEARISH', True, 'MEDIUM'):  (Signal.SELL, 0.5),
            ('BEARISH', True, 'HIGH'):    (Signal.SELL, 0.4),
        }
    
    def _get_adaptive_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Adjust weights based on current regime"""
        weights = self.base_weights.copy()
        
        if regime.volatility == 'HIGH':
            weights['volatility'] = 0.4
            weights['trend'] = 0.4
            weights['flat'] = 0.2
        elif regime.is_flat:
            weights['flat'] = 0.5
            weights['trend'] = 0.3
            weights['volatility'] = 0.2
        
        return weights
    
    def load_models(self) -> bool:
        try:
            from trainers.trend_model import TrendModel
            from trainers.flat_model import FlatModel
            from trainers.volatility_model import VolatilityModel
            
            paths = {
                'trend': self.models_dir / "trend_model.joblib",
                'flat': self.models_dir / "flat_model.joblib",
                'volatility': self.models_dir / "volatility_model.joblib"
            }
            
            if paths['trend'].exists():
                self.trend_model = TrendModel.load(str(paths['trend']))
                logger.info(f"✅ Trend model loaded (v{self.trend_model.version})")
            
            if paths['flat'].exists():
                self.flat_model = FlatModel.load(str(paths['flat']))
                logger.info(f"✅ Flat model loaded (v{self.flat_model.version})")
            
            if paths['volatility'].exists():
                self.volatility_model = VolatilityModel.load(str(paths['volatility']))
                logger.info(f"✅ Volatility model loaded (v{self.volatility_model.version})")
            
            # Optional signal model
            signal_path = self.models_dir / "signal_model.joblib"
            if signal_path.exists():
                from trainers.signal_model import SignalModel
                self.signal_model = SignalModel.load(str(signal_path))
                logger.info("✅ Signal meta-model loaded")
            
            self.is_loaded = all([self.trend_model, self.flat_model, self.volatility_model])
            return self.is_loaded
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def analyze_regime(self, df: pd.DataFrame) -> MarketRegime:
        # Trend
        if self.trend_model:
            trend_pred = self.trend_model.predict(df)
        else:
            trend_pred = {'trend_label': 'NEUTRAL', 'confidence': 0.5, 
                         'probabilities': {'bearish': 0.33, 'neutral': 0.34, 'bullish': 0.33}}
        
        # Flat
        if self.flat_model:
            flat_pred = self.flat_model.predict(df)
        else:
            flat_pred = {'is_flat': False, 'confidence': 0.5, 'flat_probability': 0.3}
        
        # Volatility
        if self.volatility_model:
            vol_pred = self.volatility_model.predict(df)
        else:
            vol_pred = {'regime_label': 'MEDIUM', 'confidence': 0.5,
                       'probabilities': {'low': 0.33, 'medium': 0.34, 'high': 0.33}}
        
        return MarketRegime(
            trend=trend_pred['trend_label'],
            trend_confidence=trend_pred['confidence'],
            trend_probabilities=trend_pred['probabilities'],
            is_flat=flat_pred['is_flat'],
            flat_confidence=flat_pred['confidence'],
            flat_probability=flat_pred.get('flat_probability', 0.5),
            volatility=vol_pred['regime_label'],
            volatility_confidence=vol_pred['confidence'],
            volatility_probabilities=vol_pred['probabilities']
        )
    
    def validate_signal(self, signal: Signal, regime: MarketRegime, confidence: float) -> tuple:
        """Validate signal with warnings"""
        warnings = []
        passed = True
        
        # 1. Minimum confidence
        if confidence < self.min_confidence:
            warnings.append(f"Low confidence: {confidence:.1%} < {self.min_confidence:.1%}")
            passed = False
        
        # 2. Don't trade in flat with low confidence
        if regime.is_flat and signal != Signal.HOLD and regime.flat_confidence > 0.7:
            warnings.append("Trading in high-confidence FLAT market")
            passed = False
        
        # 3. High volatility needs higher confidence
        if regime.volatility == 'HIGH' and confidence < self.high_vol_confidence:
            warnings.append(f"High volatility requires confidence > {self.high_vol_confidence:.1%}")
            passed = False
        
        # 4. Conflicting signals
        if regime.trend == 'BULLISH' and signal in [Signal.SELL, Signal.STRONG_SELL]:
            warnings.append("Signal conflicts with bullish trend")
        elif regime.trend == 'BEARISH' and signal in [Signal.BUY, Signal.STRONG_BUY]:
            warnings.append("Signal conflicts with bearish trend")
        
        return passed, warnings
    
    def get_signal(self, df: pd.DataFrame) -> EnsembleSignal:
        regime = self.analyze_regime(df)
        
        # Lookup decision
        key = (regime.trend, regime.is_flat, regime.volatility)
        signal, size_mult = self.decision_matrix.get(key, (Signal.HOLD, 0.2))
        
        # Adaptive weights
        weights = self._get_adaptive_weights(regime)
        
        # Weighted confidence
        confidence = (
            regime.trend_confidence * weights['trend'] +
            regime.flat_confidence * weights['flat'] +
            regime.volatility_confidence * weights['volatility']
        )
        
        # Strong signals on high confidence
        if confidence > 0.8 and signal == Signal.BUY:
            signal = Signal.STRONG_BUY
        elif confidence > 0.8 and signal == Signal.SELL:
            signal = Signal.STRONG_SELL
        
        # Volatility adjustment
        vol_adj = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.5}
        final_size = min(size_mult * vol_adj[regime.volatility], 1.5)
        
        # Validate
        passed, warnings = self.validate_signal(signal, regime, confidence)
        
        # If validation fails, reduce to HOLD
        if not passed and signal != Signal.HOLD:
            original_signal = signal
            signal = Signal.HOLD
            final_size = 0.0
            warnings.append(f"Signal downgraded from {original_signal.name} to HOLD")
        
        # Build reasoning
        reasoning = self._build_reasoning(regime, signal, confidence, weights)
        
        # Track for drift
        self._track_prediction(regime, signal, confidence)
        
        return EnsembleSignal(
            signal=signal,
            strength=abs(signal.value) / 2,
            confidence=confidence,
            regime=regime,
            reasoning=reasoning,
            position_size_multiplier=final_size,
            validation_passed=passed,
            warnings=warnings
        )
    
    def _build_reasoning(self, regime: MarketRegime, signal: Signal, 
                        confidence: float, weights: Dict) -> str:
        parts = [
            f"Trend: {regime.trend} ({regime.trend_confidence:.0%}, w={weights['trend']:.1f})",
            f"Market: {'FLAT' if regime.is_flat else 'TRENDING'} ({regime.flat_confidence:.0%}, w={weights['flat']:.1f})",
            f"Vol: {regime.volatility} ({regime.volatility_confidence:.0%}, w={weights['volatility']:.1f})",
            f"→ {signal.name} ({confidence:.0%})"
        ]
        return " | ".join(parts)
    
    def _track_prediction(self, regime: MarketRegime, signal: Signal, confidence: float):
        """Track predictions for drift monitoring"""
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'trend': regime.trend,
            'is_flat': regime.is_flat,
            'volatility': regime.volatility,
            'signal': signal.name,
            'confidence': confidence
        })
        
        # Keep only recent
        if len(self.prediction_history) > self.drift_window:
            self.prediction_history = self.prediction_history[-self.drift_window:]
    
    def check_drift(self) -> Dict:
        """Check for distribution drift in predictions"""
        if len(self.prediction_history) < 20:
            return {'status': 'insufficient_data', 'n_samples': len(self.prediction_history)}
        
        recent = self.prediction_history[-20:]
        
        # Count distributions
        trend_dist = {}
        signal_dist = {}
        avg_confidence = []
        
        for p in recent:
            trend_dist[p['trend']] = trend_dist.get(p['trend'], 0) + 1
            signal_dist[p['signal']] = signal_dist.get(p['signal'], 0) + 1
            avg_confidence.append(p['confidence'])
        
        warnings = []
        
        # Check for stuck predictions
        for trend, count in trend_dist.items():
            if count / len(recent) > 0.9:
                warnings.append(f"Trend stuck at {trend}: {count/len(recent):.0%}")
        
        for sig, count in signal_dist.items():
            if count / len(recent) > 0.9:
                warnings.append(f"Signal stuck at {sig}: {count/len(recent):.0%}")
        
        # Low confidence
        if np.mean(avg_confidence) < 0.5:
            warnings.append(f"Low average confidence: {np.mean(avg_confidence):.0%}")
        
        return {
            'status': 'drift_detected' if warnings else 'ok',
            'warnings': warnings,
            'trend_distribution': {k: v/len(recent) for k, v in trend_dist.items()},
            'signal_distribution': {k: v/len(recent) for k, v in signal_dist.items()},
            'avg_confidence': np.mean(avg_confidence)
        }
    
    def get_model_info(self) -> Dict:
        """Get info about loaded models"""
        info = {'loaded': self.is_loaded, 'models': {}}
        
        if self.trend_model:
            info['models']['trend'] = {
                'version': self.trend_model.version,
                'metrics': getattr(self.trend_model, 'metrics', {})
            }
        if self.flat_model:
            info['models']['flat'] = {
                'version': self.flat_model.version,
                'metrics': getattr(self.flat_model, 'metrics', {})
            }
        if self.volatility_model:
            info['models']['volatility'] = {
                'version': self.volatility_model.version,
                'metrics': getattr(self.volatility_model, 'metrics', {})
            }
        
        return info
    
    def to_dict(self, signal: EnsembleSignal) -> Dict:
        """Convert signal to JSON-serializable dict"""
        return {
            'signal': signal.signal.name,
            'signal_value': signal.signal.value,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'position_size': signal.position_size_multiplier,
            'validation_passed': signal.validation_passed,
            'warnings': signal.warnings,
            'regime': {
                'summary': signal.regime.regime_summary,
                'trend': signal.regime.trend,
                'trend_confidence': signal.regime.trend_confidence,
                'is_flat': signal.regime.is_flat,
                'flat_confidence': signal.regime.flat_confidence,
                'volatility': signal.regime.volatility,
                'volatility_confidence': signal.regime.volatility_confidence,
                'risk_level': signal.regime.risk_level
            },
            'reasoning': signal.reasoning,
            'timestamp': signal.timestamp
        }
