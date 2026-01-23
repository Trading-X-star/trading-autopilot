"""ML Predictor v4 - Batch predictions, caching, drift detection"""
import numpy as np
import joblib
import time
import logging
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import random

logger = logging.getLogger("ml_predictor_v4")

@dataclass
class Prediction:
    signal: int
    confidence: float
    probabilities: dict
    is_valid: bool
    warnings: list
    latency_ms: float
    model_version: str
    model_name: str = "default"

class TTLCache:
    """Simple TTL cache"""
    def __init__(self, maxsize: int = 1000, ttl: int = 60):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def _hash(self, features: dict) -> str:
        return hashlib.md5(json.dumps(features, sort_keys=True, default=str).encode()).hexdigest()
    
    def get(self, features: dict) -> Optional[Prediction]:
        key = self._hash(features)
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self.ttl:
                    return self._cache[key]
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, features: dict, prediction: Prediction):
        key = self._hash(features)
        with self._lock:
            if len(self._cache) >= self.maxsize:
                oldest = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest]
                del self._timestamps[oldest]
            self._cache[key] = prediction
            self._timestamps[key] = time.time()

class FeatureDriftDetector:
    """Детекция дрифта фичей"""
    
    def __init__(self, reference_stats: dict = None, window_size: int = 1000):
        self.reference = reference_stats or {}
        self.current_window = []
        self.window_size = window_size
        self.drift_threshold = 2.0
        self._lock = threading.Lock()
    
    def update(self, features: dict):
        with self._lock:
            self.current_window.append(features)
            if len(self.current_window) > self.window_size:
                self.current_window.pop(0)
    
    def set_reference(self, features_list: list):
        """Установить референсные статистики из списка фичей"""
        if not features_list:
            return
        
        self.reference = {}
        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())
        
        for key in all_keys:
            values = [f.get(key, 0) for f in features_list if f.get(key) is not None]
            if values:
                self.reference[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) or 1.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
    
    def check_drift(self) -> dict:
        with self._lock:
            if len(self.current_window) < 100:
                return {"drift_detected": False, "reason": "insufficient_data", "alerts": []}
            
            alerts = []
            for feat, ref_stats in self.reference.items():
                values = [f.get(feat, 0) for f in self.current_window if f.get(feat) is not None]
                if not values:
                    continue
                    
                current_mean = np.mean(values)
                ref_mean = ref_stats.get('mean', current_mean)
                ref_std = ref_stats.get('std', 1.0)
                
                if ref_std > 0:
                    z_score = abs(current_mean - ref_mean) / ref_std
                    if z_score > self.drift_threshold:
                        alerts.append({
                            "feature": feat,
                            "z_score": round(z_score, 2),
                            "current_mean": round(current_mean, 4),
                            "reference_mean": round(ref_mean, 4)
                        })
            
            return {
                "drift_detected": len(alerts) > 0,
                "alerts": alerts,
                "window_size": len(self.current_window),
                "checked_at": datetime.now().isoformat()
            }

class ModelABTester:
    """A/B тестирование моделей"""
    
    def __init__(self):
        self.models: Dict[str, 'MLPredictorV4'] = {}
        self.traffic_split: Dict[str, float] = {}
        self.results = defaultdict(lambda: {
            "correct": 0, "total": 0, "pnl": 0.0,
            "signals": defaultdict(int), "latencies": []
        })
        self._lock = threading.Lock()
    
    def add_model(self, name: str, model: 'MLPredictorV4', traffic: float = 0.5):
        with self._lock:
            self.models[name] = model
            self.traffic_split[name] = traffic
            self._normalize_traffic()
    
    def _normalize_traffic(self):
        total = sum(self.traffic_split.values())
        if total > 0:
            for k in self.traffic_split:
                self.traffic_split[k] /= total
    
    def predict(self, features: dict) -> tuple:
        model_name = self._select_model()
        if model_name not in self.models:
            return Prediction(0, 0, {}, False, ["No model"], 0, "none"), "none"
        
        prediction = self.models[model_name].predict(features)
        
        with self._lock:
            self.results[model_name]["total"] += 1
            self.results[model_name]["signals"][prediction.signal] += 1
            self.results[model_name]["latencies"].append(prediction.latency_ms)
        
        return prediction, model_name
    
    def record_outcome(self, model_name: str, predicted: int, actual: int, pnl: float = 0):
        with self._lock:
            if predicted == actual:
                self.results[model_name]["correct"] += 1
            self.results[model_name]["pnl"] += pnl
    
    def get_stats(self) -> dict:
        with self._lock:
            stats = {}
            for name, r in self.results.items():
                total = r["total"] or 1
                latencies = r["latencies"][-1000:] if r["latencies"] else [0]
                stats[name] = {
                    "accuracy": r["correct"] / total,
                    "total_predictions": r["total"],
                    "total_pnl": round(r["pnl"], 2),
                    "signal_distribution": dict(r["signals"]),
                    "avg_latency_ms": round(np.mean(latencies), 2),
                    "p99_latency_ms": round(np.percentile(latencies, 99), 2) if len(latencies) > 10 else 0,
                    "traffic_weight": self.traffic_split.get(name, 0)
                }
            return stats
    
    def _select_model(self) -> str:
        r = random.random()
        cumsum = 0
        for name, weight in self.traffic_split.items():
            cumsum += weight
            if r <= cumsum:
                return name
        return list(self.models.keys())[0] if self.models else "none"

class MLPredictorV4:
    """ML Predictor V4 с батч-предсказаниями, кэшированием и drift detection"""
    
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
        'return_5d': (-0.5, 0.5),
        'volatility_20': (0, 1.0),
    }
    
    def __init__(self, model_path: str = "model_v3_macro.joblib", cache_ttl: int = 60):
        self.model = None
        self.scaler = None
        self.version = "unknown"
        self.ready = False
        self.predictions = 0
        self.accuracy = None
        self.feature_names = self.FEATURES
        self.model_name = Path(model_path).stem
        
        # V4 additions
        self._cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.drift_detector = FeatureDriftDetector()
        self._metrics = {
            "total_predictions": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0
        }
        self._lock = threading.Lock()
        
        self._load(model_path)
    
    def _load(self, path: str):
        try:
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
                logger.warning(f"Model not found in paths: {[str(p) for p in paths_to_try]}")
                return
            
            logger.info(f"Loading model from: {found_path}")
            data = joblib.load(found_path)
            
            if isinstance(data, dict):
                self.model = data.get('model')
                self.scaler = data.get('scaler')
                self.version = data.get('version', 'v4')
                self.accuracy = data.get('accuracy') or (data.get('results') or {}).get('accuracy')
                self.feature_names = data.get('features', self.FEATURES)
                
                # Load reference stats for drift detection
                if 'feature_stats' in data:
                    self.drift_detector.reference = data['feature_stats']
            else:
                self.model = data
                self.version = "legacy"
            
            if self.model is not None:
                self.ready = True
                logger.info(f"✅ ML model loaded: {self.version}, accuracy={self.accuracy}")
                
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            import traceback
            traceback.print_exc()
    
    def validate(self, features: dict) -> tuple:
        warnings = []
        
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            return False, [f"Missing features: {missing}"]
        
        for feat, (lo, hi) in self.BOUNDS.items():
            val = features.get(feat)
            if val is not None and (val < lo or val > hi):
                warnings.append(f"{feat}={val:.3f} outside [{lo},{hi}]")
        
        for f in self.feature_names:
            v = features.get(f)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return False, [f"Invalid value for {f}: {v}"]
        
        return True, warnings
    
    def predict(self, features: dict, use_cache: bool = True) -> Prediction:
        """Single prediction with caching"""
        start = time.time()
        
        if not self.ready or self.model is None:
            return Prediction(0, 0, {}, False, ["Model not loaded"], 0, "none", self.model_name)
        
        # Check cache
        if use_cache:
            cached = self._cache.get(features)
            if cached:
                with self._lock:
                    self._metrics["cache_hits"] += 1
                return cached
        
        # Update drift detector
        self.drift_detector.update(features)
        
        valid, warnings = self.validate(features)
        if not valid:
            return Prediction(0, 0, {}, False, warnings, 0, self.version, self.model_name)
        
        try:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            probs = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            
            prob_dict = {int(c): float(p) for c, p in zip(classes, probs)}
            signal = int(classes[np.argmax(probs)])
            conf = float(np.max(probs))
            
            if warnings:
                conf *= 0.9
            
            latency = (time.time() - start) * 1000
            
            with self._lock:
                self._metrics["total_predictions"] += 1
                self._metrics["total_latency_ms"] += latency
                self.predictions += 1
            
            prediction = Prediction(signal, conf, prob_dict, True, warnings, latency, self.version, self.model_name)
            
            if use_cache:
                self._cache.set(features, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            with self._lock:
                self._metrics["errors"] += 1
            return Prediction(0, 0, {}, False, [str(e)], 0, self.version, self.model_name)
    
    def predict_batch(self, features_list: List[dict]) -> List[Prediction]:
        """Batch predictions - 10x faster"""
        if not features_list:
            return []
        
        if not self.ready or self.model is None:
            return [Prediction(0, 0, {}, False, ["Model not loaded"], 0, "none", self.model_name)] * len(features_list)
        
        start = time.time()
        
        # Validate all
        valid_indices = []
        warnings_list = []
        for i, f in enumerate(features_list):
            valid, warns = self.validate(f)
            warnings_list.append(warns)
            if valid:
                valid_indices.append(i)
            self.drift_detector.update(f)
        
        # Prepare batch
        if valid_indices:
            X = np.array([
                [features_list[i].get(f, 0) for f in self.feature_names]
                for i in valid_indices
            ])
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            probs_batch = self.model.predict_proba(X)
            classes = self.model.classes_
        else:
            probs_batch = []
        
        # Collect results
        results = []
        prob_idx = 0
        total_latency = (time.time() - start) * 1000
        latency_per = total_latency / max(len(features_list), 1)
        
        for i in range(len(features_list)):
            if i in valid_indices:
                probs = probs_batch[prob_idx]
                prob_idx += 1
                signal = int(classes[np.argmax(probs)])
                conf = float(np.max(probs))
                if warnings_list[i]:
                    conf *= 0.9
                prob_dict = {int(c): float(p) for c, p in zip(classes, probs)}
                results.append(Prediction(signal, conf, prob_dict, True, warnings_list[i], latency_per, self.version, self.model_name))
            else:
                results.append(Prediction(0, 0, {}, False, warnings_list[i], 0, self.version, self.model_name))
        
        with self._lock:
            self._metrics["total_predictions"] += len(features_list)
            self._metrics["total_latency_ms"] += total_latency
            self.predictions += len(features_list)
        
        return results
    
    def get_drift_status(self) -> dict:
        return self.drift_detector.check_drift()
    
    def info(self) -> dict:
        with self._lock:
            total = self._metrics["total_predictions"] or 1
            return {
                "ready": self.ready,
                "version": self.version,
                "model_name": self.model_name,
                "predictions": self.predictions,
                "accuracy": self.accuracy,
                "features_count": len(self.feature_names),
                "cache_hit_rate": round(self._metrics["cache_hits"] / total, 3),
                "avg_latency_ms": round(self._metrics["total_latency_ms"] / total, 2),
                "error_rate": round(self._metrics["errors"] / total, 4),
                "drift_status": self.drift_detector.check_drift()
            }

# Global instances
predictor_v4: Optional[MLPredictorV4] = None
ab_tester: Optional[ModelABTester] = None

def get_predictor() -> MLPredictorV4:
    global predictor_v4
    if predictor_v4 is None:
        predictor_v4 = MLPredictorV4()
    return predictor_v4

def get_ab_tester() -> ModelABTester:
    global ab_tester
    if ab_tester is None:
        ab_tester = ModelABTester()
    return ab_tester
