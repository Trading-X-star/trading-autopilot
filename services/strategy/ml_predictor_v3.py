"""ML Predictor v3 - с валидацией и мониторингом (joblib)"""
import numpy as np
import joblib
import time
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

logger = logging.getLogger("ml_predictor_v3")

@dataclass
class Prediction:
    signal: int          # -1, 0, 1
    confidence: float    # 0.0 - 1.0
    probabilities: dict  # {-1: 0.2, 0: 0.3, 1: 0.5}
    is_valid: bool
    warnings: list
    latency_ms: float
    model_version: str

class MLPredictorV3:
    FEATURES = [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_pct', 'bb_width', 'atr_14',
        'return_1d', 'return_5d', 'return_10d',
        'volatility_20', 'volume_ratio',
        'sma_20', 'sma_50', 'ema_12', 'ema_26'
    ]
    
    BOUNDS = {
        'rsi_14': (0, 100),
        'bb_pct': (-1, 2),
        'volume_ratio': (0, 50),
        'return_1d': (-0.3, 0.3),
        'volatility_20': (0, 0.5),
    }
    
    def __init__(self, model_path: str = "model_v3_macro.joblib"):
        self.model = None
        self.scaler = None
        self.version = "unknown"
        self.ready = False
        self.predictions = 0
        self.accuracy = None
        self.feature_names = self.FEATURES
        self._load(model_path)
    
    def _load(self, path: str):
        try:
            # Try multiple paths
            paths_to_try = [
                Path(path),
                Path("/app/models") / path,
                Path("/app") / path,
                Path("models") / path,
            ]
            
            found_path = None
            for p in paths_to_try:
                if p.exists():
                    found_path = p
                    break
            
            if not found_path:
                logger.warning(f"Model not found in any path: {paths_to_try}")
                return
            
            logger.info(f"Loading model from: {found_path}")
            data = joblib.load(found_path)
            
            if isinstance(data, dict):
                self.model = data.get('model')
                self.scaler = data.get('scaler')
                self.version = data.get('version', 'v3')
                self.accuracy = data.get('accuracy') or (data.get('results') or {}).get('accuracy')
                self.feature_names = data.get('features', self.FEATURES)
            else:
                self.model = data
                self.version = "legacy"
            
            # Load accuracy from results.json if not in pkl
            if self.accuracy is None:
                try:
                    results_paths = [
                        found_path.parent / "model_v7_ultimate_results.json",
                        Path("/app/models/model_v7_ultimate_results.json"),
                        Path("model_v7_ultimate_results.json"),
                    ]
                    for rp in results_paths:
                        if rp.exists():
                            import json
                            with open(rp) as f:
                                results = json.load(f)
                                self.accuracy = results.get("accuracy")
                                logger.info(f"📊 Loaded accuracy from {rp}: {self.accuracy}")
                            break
                except Exception as e:
                    logger.warning(f"Could not load accuracy from results: {e}")
            
            if self.model is not None:
                self.ready = True
                logger.info(f"✅ ML model loaded: {self.version}, accuracy={self.accuracy}")
            else:
                logger.error("Model object is None")
                
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            import traceback
            traceback.print_exc()
    
    def validate(self, features: dict) -> tuple:
        warnings = []
        
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            return False, [f"Missing: {missing}"]
        
        for feat, (lo, hi) in self.BOUNDS.items():
            val = features.get(feat)
            if val is not None and (val < lo or val > hi):
                warnings.append(f"{feat}={val:.2f} out [{lo},{hi}]")
        
        for f in self.feature_names:
            v = features.get(f)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return False, [f"Invalid {f}: {v}"]
        
        return True, warnings
    
    def predict(self, features: dict) -> Prediction:
        start = time.time()
        
        if not self.ready or self.model is None:
            return Prediction(0, 0, {}, False, ["Model not loaded"], 0, "none")
        
        valid, warnings = self.validate(features)
        if not valid:
            return Prediction(0, 0, {}, False, warnings, 0, self.version)
        
        try:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            # Apply scaler if available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            
            prob_dict = {int(c): float(p) for c, p in zip(classes, probs)}
            signal = int(classes[np.argmax(probs)])
            conf = float(np.max(probs))
            
            if warnings:
                conf *= 0.9
            
            self.predictions += 1
            latency = (time.time() - start) * 1000
            
            return Prediction(signal, conf, prob_dict, True, warnings, latency, self.version)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return Prediction(0, 0, {}, False, [str(e)], 0, self.version)
    
    def info(self) -> dict:
        return {
            "ready": self.ready,
            "version": self.version,
            "predictions": self.predictions,
            "accuracy": self.accuracy,
            "features": len(self.feature_names)
        }
