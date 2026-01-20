"""ML Predictor v2 - Enhanced with feature engineering"""
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("ml-predictor-v2")

BASE_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
    'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
    'atr_14', 'volatility_20', 'volume_ratio',
    'pct_from_high', 'pct_from_low'
]

class MLPredictorV2:
    def __init__(self):
        self.model = None
        self.ready = False
        self.accuracy = 0
        self.version = 'v1'
        self.feature_names = BASE_FEATURES
        self._load_model()
    
    def _load_model(self):
        model_path = Path(__file__).parent / "model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.accuracy = data.get('accuracy', 0)
                    self.version = data.get('version', 'v1')
                    self.feature_names = data.get('features', BASE_FEATURES)
                    self.ready = True
                    logger.info(f"✅ ML model {self.version} loaded (accuracy: {self.accuracy:.1%}, features: {len(self.feature_names)})")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def _engineer_features(self, base_features: list) -> list:
        """Add derived features matching training"""
        features = base_features.copy()
        idx = {name: i for i, name in enumerate(BASE_FEATURES)}
        
        # 1. Momentum acceleration
        if 'return_1d' in idx and 'return_5d' in idx:
            features.append(base_features[idx['return_1d']] - base_features[idx['return_5d']] / 5)
        
        # 2. RSI zones
        if 'rsi_14' in idx:
            rsi = base_features[idx['rsi_14']]
            features.append(float(rsi < 30))
            features.append(float(rsi > 70))
            features.append((rsi - 50) / 50)
        
        # 3. MACD direction
        if 'macd_hist' in idx:
            features.append(np.sign(base_features[idx['macd_hist']]))
        
        # 4. Trend alignment
        ma_indices = [idx[c] for c in ['sma_5', 'sma_20', 'sma_50', 'sma_200'] if c in idx]
        if ma_indices:
            trend = sum(1 for mi in ma_indices if base_features[mi] > 0) / len(ma_indices)
            features.append(trend)
        
        # 5. Volatility regime
        if 'volatility_20' in idx:
            features.append(0)  # vol_zscore - need history, use 0
            features.append(0)  # high_vol_regime
        
        # 6. BB squeeze
        if 'bb_width' in idx:
            features.append(0)  # need history
        
        # 7. Mean reversion
        if 'bb_pct' in idx:
            bb_pct = base_features[idx['bb_pct']]
            features.append(max(0, 1 - abs(bb_pct - 0.5) * 2))
        
        # 8. RSI-MACD agreement
        if 'rsi_14' in idx and 'macd_hist' in idx:
            rsi_norm = (base_features[idx['rsi_14']] - 50) / 50
            macd_norm = np.tanh(base_features[idx['macd_hist']] * 100)
            features.append(rsi_norm * macd_norm)
        
        return features
    
    def predict(self, indicators: dict, close: float) -> tuple:
        """Predict signal with confidence"""
        if not self.ready:
            return 0, 0.0
        
        try:
            # Extract base features
            base_X = []
            for feat in BASE_FEATURES:
                val = float(indicators.get(feat, 0) or 0)
                if feat in ['sma_5','sma_10','sma_20','sma_50','sma_200','ema_12','ema_26',
                           'bb_upper','bb_middle','bb_lower','atr_14']:
                    val = (val / close - 1) if close > 0 else 0
                base_X.append(val)
            
            # Add engineered features for v2
            if self.version == 'v2':
                X = self._engineer_features(base_X)
            else:
                X = base_X
            
            # Pad or truncate to match model
            while len(X) < len(self.feature_names):
                X.append(0)
            X = X[:len(self.feature_names)]
            
            # Predict
            X = np.array([X])
            proba = self.model.predict(X)[0]
            
            signal = int(np.argmax(proba) - 1)  # -1, 0, 1
            confidence = float(max(proba))
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0
    
    def get_info(self):
        return {
            "ready": self.ready,
            "accuracy": self.accuracy,
            "features": len(self.feature_names),
            "version": self.version
        }

# Global instance
predictor = MLPredictorV2()
