"""Meta Model - Stacking ensemble for final signal"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import joblib
from pathlib import Path

class MetaModel:
    """
    Stacking meta-learner that combines outputs from base models
    Input: probabilities from Trend, Flat, Volatility models + market features
    Output: Final trading signal
    """
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_names = None
        self.version = "meta_v1"
        self.use_gpu = use_gpu
        self.metrics = {}
    
    def prepare_meta_features(self, trend_pred: dict, flat_pred: dict, vol_pred: dict,
                              market_features: dict = None) -> np.ndarray:
        """Combine all model outputs into meta-features"""
        features = []
        
        # Trend model outputs
        features.extend([
            trend_pred['probabilities']['bearish'],
            trend_pred['probabilities']['neutral'],
            trend_pred['probabilities']['bullish'],
            trend_pred['confidence'],
            trend_pred['trend'],  # -1, 0, 1
        ])
        
        # Flat model outputs
        features.extend([
            flat_pred['flat_probability'],
            flat_pred['confidence'],
            1.0 if flat_pred['is_flat'] else 0.0,
        ])
        
        # Volatility model outputs
        features.extend([
            vol_pred['probabilities']['low'],
            vol_pred['probabilities']['medium'],
            vol_pred['probabilities']['high'],
            vol_pred['confidence'],
            vol_pred.get('predicted_volatility', 0.2),
            vol_pred['regime'],  # 0, 1, 2
        ])
        
        # Interactions
        features.extend([
            # Trend strength in trending market
            trend_pred['confidence'] * (1 - flat_pred['flat_probability']),
            # Bullish with low volatility
            trend_pred['probabilities']['bullish'] * vol_pred['probabilities']['low'],
            # Bearish with high volatility  
            trend_pred['probabilities']['bearish'] * vol_pred['probabilities']['high'],
            # Agreement score
            trend_pred['confidence'] * flat_pred['confidence'] * vol_pred['confidence'],
        ])
        
        # Market features (if provided)
        if market_features:
            features.extend([
                market_features.get('rsi_14', 50) / 100,
                market_features.get('adx_14', 25) / 100,
                market_features.get('return_5d', 0),
                market_features.get('volume_ratio', 1),
            ])
        
        self.feature_names = [
            'trend_bearish', 'trend_neutral', 'trend_bullish', 'trend_conf', 'trend_dir',
            'flat_prob', 'flat_conf', 'is_flat',
            'vol_low', 'vol_med', 'vol_high', 'vol_conf', 'vol_value', 'vol_regime',
            'trend_strength', 'bull_low_vol', 'bear_high_vol', 'agreement',
            'rsi', 'adx', 'return_5d', 'volume_ratio'
        ][:len(features)]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train meta-model
        y: 0=STRONG_SELL, 1=SELL, 2=HOLD, 3=BUY, 4=STRONG_BUY
        """
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        scores, f1_scores = [], []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 5,
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'verbosity': 0,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cuda:0' if self.use_gpu else 'cpu'
        }
        
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(accuracy_score(y[val_idx], pred))
            f1_scores.append(f1_score(y[val_idx], pred, average='weighted'))
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        # Feature importance
        importance = dict(zip(self.feature_names or [f'f{i}' for i in range(X.shape[1])], 
                             self.model.feature_importances_))
        
        self.metrics = {
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'f1_score': np.mean(f1_scores),
            'n_samples': len(X),
            'feature_importance': dict(sorted(importance.items(), key=lambda x: -x[1])[:10])
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
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'version': self.version,
            'metrics': self.metrics
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'MetaModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_names = data.get('feature_names')
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
