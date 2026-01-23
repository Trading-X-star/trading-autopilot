"""Trend Detection Model with GPU support"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.features import FeatureBuilder

# GPU: use XGBoost instead of LightGBM
import xgboost as xgb

class TrendModel:
    """Detects: BULLISH (1) / NEUTRAL (0) / BEARISH (-1)"""
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_cols = None
        self.version = "trend_v1_gpu"
        self.use_gpu = use_gpu
        
    def prepare_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        future_return = df['close'].pct_change(horizon).shift(-horizon)
        target = pd.Series(0, index=df.index)
        target[future_return > 0.02] = 1    # BULLISH
        target[future_return < -0.02] = -1  # BEARISH
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.trend_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & target.notna()
        X = features[valid_idx].values
        y = target[valid_idx].values + 1  # -1,0,1 -> 0,1,2
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda:0' if self.use_gpu else 'cpu'
        }
        
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            scores.append(accuracy_score(y[val_idx], y_pred))
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        return {
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'gpu_used': self.use_gpu
        }
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.trend_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        proba = self.model.predict_proba(X)[0]
        pred = int(self.model.predict(X)[0]) - 1
        return {
            'trend': pred,
            'trend_label': {-1: 'BEARISH', 0: 'NEUTRAL', 1: 'BULLISH'}[pred],
            'confidence': float(max(proba)),
            'probabilities': {'bearish': float(proba[0]), 'neutral': float(proba[1]), 'bullish': float(proba[2])}
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols, 'version': self.version}, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrendModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        return instance
