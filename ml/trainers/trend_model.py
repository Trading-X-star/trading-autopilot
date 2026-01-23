"""Enhanced Trend Detection Model with GPU"""
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

class TrendModel:
    """Detects: BULLISH (1) / NEUTRAL (0) / BEARISH (-1)"""
    
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_cols = None
        self.version = "trend_v2_gpu"
        self.use_gpu = use_gpu
        self.metrics = {}
        
    def prepare_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Create trend labels - NO LEAKAGE: use shifted returns"""
        close = df['close'].astype(float)
        future_return = close.pct_change(horizon).shift(-horizon)
        
        # Adaptive thresholds based on volatility
        vol = close.pct_change().rolling(20).std()
        bull_thresh = vol * 2
        bear_thresh = -vol * 2
        
        target = pd.Series(0, index=df.index)  # NEUTRAL
        target[future_return > bull_thresh] = 1   # BULLISH
        target[future_return < bear_thresh] = -1  # BEARISH
        
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.trend_features(df)
        self.feature_cols = features.columns.tolist()
        target = self.prepare_target(df)
        
        # Remove future data (last horizon rows)
        valid_idx = features.notna().all(axis=1) & target.notna()
        valid_idx.iloc[-5:] = False  # Remove last 5 rows
        
        X = features[valid_idx].values
        y = target[valid_idx].values + 1  # -1,0,1 -> 0,1,2
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # Gap to prevent leakage
        scores, f1_scores = [], []
        
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda:0' if self.use_gpu else 'cpu'
        }
        
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx], 
                     eval_set=[(X[val_idx], y[val_idx])],
                     verbose=False)
            y_pred = model.predict(X[val_idx])
            scores.append(accuracy_score(y[val_idx], y_pred))
            f1_scores.append(f1_score(y[val_idx], y_pred, average='weighted'))
        
        # Final model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        # Feature importance
        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        self.metrics = {
            'accuracy': np.mean(scores),
            'accuracy_std': np.std(scores),
            'f1_score': np.mean(f1_scores),
            'n_samples': len(X),
            'n_features': len(self.feature_cols),
            'top_features': top_features,
            'class_distribution': {
                'bearish': (y == 0).mean(),
                'neutral': (y == 1).mean(),
                'bullish': (y == 2).mean()
            },
            'gpu_used': self.use_gpu
        }
        return self.metrics
    
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
        joblib.dump({
            'model': self.model, 
            'feature_cols': self.feature_cols, 
            'version': self.version,
            'metrics': self.metrics
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'TrendModel':
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_cols = data['feature_cols']
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
