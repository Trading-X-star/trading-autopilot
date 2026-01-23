"""Flat/Sideways Market Detection Model with GPU"""
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

class FlatModel:
    """Detects: FLAT (1) / TRENDING (0)"""
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_cols = None
        self.version = "flat_v1_gpu"
        self.use_gpu = use_gpu
    
    def prepare_target(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        high_max = df['high'].rolling(window).max()
        low_min = df['low'].rolling(window).min()
        range_pct = (high_max - low_min) / df['close']
        adx = df.get('adx_14', pd.Series(20, index=df.index))
        target = pd.Series(0, index=df.index)
        target[(range_pct < 0.03) | (adx < 25)] = 1
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.flat_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & target.notna()
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        params = {
            'objective': 'binary:logistic',
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
            'flat_ratio': y.mean(),
            'n_samples': len(X),
            'gpu_used': self.use_gpu
        }
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.flat_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0])
        return {
            'is_flat': bool(pred),
            'regime': 'FLAT' if pred else 'TRENDING',
            'flat_probability': float(proba[1]),
            'confidence': float(max(proba))
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols, 'version': self.version}, path)
    
    @classmethod
    def load(cls, path: str) -> 'FlatModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        return instance
