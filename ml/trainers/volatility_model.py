"""Volatility Regime Detection Model with GPU"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.features import FeatureBuilder

class VolatilityModel:
    """Detects: LOW (0) / MEDIUM (1) / HIGH (2)"""
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_cols = None
        self.version = "volatility_v1_gpu"
        self.use_gpu = use_gpu
    
    def prepare_target(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = df['close'].pct_change()
        vol = returns.rolling(window).std() * np.sqrt(252)
        vol_25, vol_75 = vol.quantile(0.25), vol.quantile(0.75)
        target = pd.Series(1, index=df.index)
        target[vol < vol_25] = 0
        target[vol > vol_75] = 2
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.volatility_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & target.notna()
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 150,
            'max_depth': 5,
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
            scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        return {
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'regime_distribution': {'low': (y==0).mean(), 'medium': (y==1).mean(), 'high': (y==2).mean()},
            'n_samples': len(X),
            'gpu_used': self.use_gpu
        }
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.volatility_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        return {
            'regime': pred,
            'regime_label': {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}[pred],
            'confidence': float(max(proba)),
            'probabilities': {'low': float(proba[0]), 'medium': float(proba[1]), 'high': float(proba[2])}
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols, 'version': self.version}, path)
    
    @classmethod
    def load(cls, path: str) -> 'VolatilityModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        return instance
