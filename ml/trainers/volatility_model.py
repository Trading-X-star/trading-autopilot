"""Enhanced Volatility Regime Model - Fixed inf values"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
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
        self.version = "volatility_v2_gpu"
        self.use_gpu = use_gpu
        self.metrics = {}
    
    def prepare_target(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        close = df['close'].astype(float)
        returns = close.pct_change()
        future_vol = returns.rolling(window).std().shift(-window) * np.sqrt(252)
        
        vol_33 = future_vol.quantile(0.33)
        vol_66 = future_vol.quantile(0.66)
        
        target = pd.Series(1, index=df.index)
        target[future_vol < vol_33] = 0
        target[future_vol > vol_66] = 2
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.volatility_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & target.notna()
        valid_idx.iloc[-10:] = False
        
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        # FIX: Replace inf
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
        scores = []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
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
            'regime_distribution': {
                'low': float((y == 0).mean()),
                'medium': float((y == 1).mean()),
                'high': float((y == 2).mean())
            },
            'n_samples': len(X),
            'gpu_used': self.use_gpu
        }
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.volatility_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        return {
            'regime': pred,
            'regime_label': {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}[pred],
            'confidence': float(max(proba)),
            'probabilities': {'low': float(proba[0]), 'medium': float(proba[1]), 'high': float(proba[2])}
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols,
                    'version': self.version, 'metrics': self.metrics}, path)
    
    @classmethod
    def load(cls, path: str) -> 'VolatilityModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
