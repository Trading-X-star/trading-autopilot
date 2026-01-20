#!/usr/bin/env python3
"""Fast GPU Trainer - оптимизированная версия"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sqlalchemy import create_engine
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fast_trainer")

# Исправленные названия колонок
FEATURES = [
    'rsi_14', 'macd', 'macd_hist', 'bb_pct', 'bb_width', 'atr_14',
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'volatility_20', 'volume_ratio', 'pct_from_high', 'pct_from_low'
]

def train():
    db_url = os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")
    engine = create_engine(db_url)
    
    logger.info("Loading recent data...")
    cols = ', '.join(FEATURES)
    df = pd.read_sql(f"""
        SELECT date, ticker, {cols}, signal_class
        FROM features 
        WHERE signal_class IS NOT NULL AND date >= '2020-01-01'
        ORDER BY date
    """, engine)
    
    df = df.dropna()
    logger.info(f"Loaded {len(df):,} rows")
    
    # Prepare
    X = df[FEATURES].values.astype(np.float32)
    y = (df['signal_class'].values + 1).astype(np.int32)  # -1,0,1 -> 0,1,2
    
    # Time split 80/20
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'tree_method': 'hist',
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_child_weight': 50,
        'subsample': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    logger.info("Training XGBoost (CPU)...")
    model = xgb.train(params, dtrain, num_boost_round=150,
                      evals=[(dtest, 'test')], 
                      early_stopping_rounds=20, verbose_eval=20)
    
    # Evaluate
    y_pred = model.predict(dtest).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info("=" * 50)
    logger.info(f"Accuracy: {acc:.1%} | F1: {f1:.3f}")
    logger.info(f"Edge over random (33%): {(acc-0.333)*100:+.1f}%")
    logger.info("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test-1, y_pred-1, target_names=['Sell (-1)','Hold (0)','Buy (1)']))
    
    # Save
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': FEATURES,
        'accuracy': acc,
        'f1_score': f1,
        'version': f"xgb_{datetime.now():%Y%m%d_%H%M}",
        'trained_at': datetime.now().isoformat(),
        'label_offset': 1
    }
    
    joblib.dump(model_data, '/app/models/model_v3_macro.joblib')
    logger.info("✅ Model saved to /app/models/model_v3_macro.joblib")

if __name__ == "__main__":
    train()
