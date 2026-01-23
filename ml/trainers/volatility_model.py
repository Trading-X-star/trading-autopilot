"""Volatility Model v3 - Regression + Classification hybrid"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.features import FeatureBuilder

class VolatilityModel:
    """Hybrid model: predicts volatility value, then classifies regime"""
    
    def __init__(self, use_gpu: bool = True):
        self.regressor = None
        self.classifier = None
        self.feature_cols = None
        self.vol_thresholds = None
        self.version = "volatility_v3_hybrid"
        self.use_gpu = use_gpu
        self.metrics = {}
    
    def prepare_target(self, df: pd.DataFrame, horizon: int = 5) -> tuple:
        """Prepare both regression and classification targets"""
        close = df['close'].astype(float)
        returns = close.pct_change()
        
        # Future realized volatility (regression target)
        future_vol = returns.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
        
        # Dynamic thresholds based on historical distribution
        vol_33 = future_vol.quantile(0.33)
        vol_66 = future_vol.quantile(0.66)
        self.vol_thresholds = (vol_33, vol_66)
        
        # Classification target
        regime = pd.Series(1, index=df.index)  # MEDIUM
        regime[future_vol < vol_33] = 0  # LOW
        regime[future_vol > vol_66] = 2  # HIGH
        
        return future_vol, regime
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        features = FeatureBuilder.volatility_features(df)
        self.feature_cols = features.columns.tolist()
        vol_target, regime_target = self.prepare_target(df)
        
        valid_idx = features.notna().all(axis=1) & vol_target.notna() & regime_target.notna()
        valid_idx.iloc[-10:] = False
        
        X = features[valid_idx].values
        y_vol = vol_target[valid_idx].values
        y_regime = regime_target[valid_idx].values
        
        device = 'cuda:0' if self.use_gpu else 'cpu'
        
        # === Stage 1: Regression ===
        reg_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0,
            'random_state': 42,
            'tree_method': 'hist',
            'device': device
        }
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
        reg_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            model = xgb.XGBRegressor(**reg_params)
            model.fit(X[train_idx], y_vol[train_idx])
            pred = model.predict(X[val_idx])
            reg_scores.append(mean_absolute_error(y_vol[val_idx], pred))
        
        self.regressor = xgb.XGBRegressor(**reg_params)
        self.regressor.fit(X, y_vol)
        
        # === Stage 2: Classification with regression output ===
        vol_pred = self.regressor.predict(X).reshape(-1, 1)
        X_combined = np.hstack([X, vol_pred])
        
        clf_params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'verbosity': 0,
            'random_state': 42,
            'tree_method': 'hist',
            'device': device
        }
        
        clf_scores = []
        for train_idx, val_idx in tscv.split(X_combined):
            model = xgb.XGBClassifier(**clf_params)
            model.fit(X_combined[train_idx], y_regime[train_idx])
            clf_scores.append(accuracy_score(y_regime[val_idx], model.predict(X_combined[val_idx])))
        
        self.classifier = xgb.XGBClassifier(**clf_params)
        self.classifier.fit(X_combined, y_regime)
        
        self.metrics = {
            'regression_mae': np.mean(reg_scores),
            'classification_accuracy': np.mean(clf_scores),
            'accuracy_std': np.std(clf_scores),
            'vol_thresholds': {'low': float(self.vol_thresholds[0]), 'high': float(self.vol_thresholds[1])},
            'regime_distribution': {
                'low': float((y_regime == 0).mean()),
                'medium': float((y_regime == 1).mean()),
                'high': float((y_regime == 2).mean())
            },
            'n_samples': len(X),
            'gpu_used': self.use_gpu
        }
        return self.metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        features = FeatureBuilder.volatility_features(df)
        X = features[self.feature_cols].iloc[-1:].values
        
        # Stage 1: Predict volatility value
        vol_pred = self.regressor.predict(X)[0]
        
        # Stage 2: Classify regime
        X_combined = np.hstack([X, [[vol_pred]]])
        proba = self.classifier.predict_proba(X_combined)[0]
        regime = int(self.classifier.predict(X_combined)[0])
        
        return {
            'regime': regime,
            'regime_label': {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}[regime],
            'predicted_volatility': float(vol_pred),
            'confidence': float(max(proba)),
            'probabilities': {
                'low': float(proba[0]),
                'medium': float(proba[1]),
                'high': float(proba[2])
            }
        }
    
    def save(self, path: str):
        joblib.dump({
            'regressor': self.regressor,
            'classifier': self.classifier,
            'feature_cols': self.feature_cols,
            'vol_thresholds': self.vol_thresholds,
            'version': self.version,
            'metrics': self.metrics
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'VolatilityModel':
        data = joblib.load(path)
        instance = cls()
        instance.regressor = data['regressor']
        instance.classifier = data['classifier']
        instance.feature_cols = data['feature_cols']
        instance.vol_thresholds = data['vol_thresholds']
        instance.version = data['version']
        instance.metrics = data.get('metrics', {})
        return instance
