"""Meta Signal Model - combines all model outputs"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import joblib
from pathlib import Path

class SignalModel:
    """
    Meta-model that takes outputs from Trend, Flat, Volatility models
    and produces final trading signal: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    """
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.version = "signal_v1_gpu"
        self.use_gpu = use_gpu
        self.metrics = {}
    
    def prepare_meta_features(self, trend_pred: dict, flat_pred: dict, vol_pred: dict) -> np.ndarray:
        """Combine model outputs into meta-features"""
        features = [
            # Trend
            trend_pred['probabilities']['bearish'],
            trend_pred['probabilities']['neutral'],
            trend_pred['probabilities']['bullish'],
            trend_pred['confidence'],
            
            # Flat
            flat_pred['flat_probability'],
            flat_pred['confidence'],
            1 if flat_pred['is_flat'] else 0,
            
            # Volatility
            vol_pred['probabilities']['low'],
            vol_pred['probabilities']['medium'],
            vol_pred['probabilities']['high'],
            vol_pred['confidence'],
            
            # Interactions
            trend_pred['probabilities']['bullish'] * (1 - flat_pred['flat_probability']),
            trend_pred['probabilities']['bearish'] * (1 - flat_pred['flat_probability']),
            trend_pred['confidence'] * (1 - vol_pred['probabilities']['high']),
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train on historical meta-features and returns"""
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        scores = []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 5,  # STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'verbosity': 0,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cuda:0' if self.use_gpu else 'cpu'
        }
        
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        self.metrics = {
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'n_samples': len(X)
        }
        return self.metrics
    
    def predict(self, meta_features: np.ndarray) -> dict:
        proba = self.model.predict_proba(meta_features)[0]
        pred = int(self.model.predict(meta_features)[0])
        labels = {0: 'STRONG_SELL', 1: 'SELL', 2: 'HOLD', 3: 'BUY', 4: 'STRONG_BUY'}
        return {
            'signal': pred - 2,  # -2 to 2
            'signal_label': labels[pred],
            'confidence': float(max(proba)),
            'probabilities': {labels[i]: float(proba[i]) for i in range(5)}
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'version': self.version, 'metrics': self.metrics}, path)
    
    @classmethod
    def load(cls, path: str) -> 'SignalModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
