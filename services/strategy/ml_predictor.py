"""ML Predictor - Load and use trained model"""
import pickle
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("ml-predictor")

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
    'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
    'atr_14', 'volatility_20', 'volume_ratio',
    'pct_from_high', 'pct_from_low'
]

class MLPredictor:
    def __init__(self):
        self.model = None
        self.ready = False
        self.accuracy = 0
        self._load_model()
    
    def _load_model(self):
        model_path = Path(__file__).parent / "model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.accuracy = data.get('accuracy', 0)
                    self.ready = True
                    logger.info(f"✅ ML model loaded (accuracy: {self.accuracy:.1%})")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def predict(self, indicators: dict, close: float) -> tuple:
        if not self.ready: return 0, 0.0
        try:
            X = []
            for feat in FEATURE_COLS:
                val = float(indicators.get(feat, 0) or 0)
                if feat in ['sma_5','sma_10','sma_20','sma_50','sma_200','ema_12','ema_26','bb_upper','bb_middle','bb_lower','atr_14']:
                    val = (val / close - 1) if close > 0 else 0
                X.append(val)
            proba = self.model.predict(np.array([X]))[0]
            return int(np.argmax(proba) - 1), float(max(proba))
        except: return 0, 0.0
    
    def get_info(self): return {"ready": self.ready, "accuracy": self.accuracy, "features": len(FEATURE_COLS)}

predictor = MLPredictor()
