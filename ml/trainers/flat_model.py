"""Enhanced Flat/Sideways Detection Model - Fixed inf values"""
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

class FlatModel:
    """Detects: FLAT (1) / TRENDING (0)"""
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_cols = None
        self.version = "flat_v2_gpu"
        self.use_gpu = use_gpu
        self.metrics = {}
    
    def prepare_target(self, df: pd.DataFrame, window: int = 10, future_window: int = 5) -> pd.Series:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        future_high = high.rolling(future_window).max().shift(-future_window)
        future_low = low.rolling(future_window).min().shift(-future_window)
        future_range = (future_high - future_low) / close
        
        range_threshold = future_range.quantile(0.3)
        target = pd.Series(0, index=df.index)
        target[future_range < range_threshold] = 1
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.flat_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & target.notna()
        valid_idx.iloc[-10:] = False
        
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        # FIX: Replace inf with nan, then fill
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
        scores = []
        
        # Balance classes
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        
        params = {
            'objective': 'binary:logistic',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': pos_weight,
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
            'flat_ratio': y.mean(),
            'n_samples': len(X),
            'gpu_used': self.use_gpu
        }
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.flat_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        return {
            'is_flat': bool(pred),
            'regime': 'FLAT' if pred else 'TRENDING',
            'flat_probability': float(proba[1]) if len(proba) > 1 else 0.5,
            'confidence': float(max(proba))
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols, 
                    'version': self.version, 'metrics': self.metrics}, path)
    
    @classmethod
    def load(cls, path: str) -> 'FlatModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
