#!/usr/bin/env python3
"""Simple ML Trainer - для быстрого обучения"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sqlalchemy import create_engine
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trainer")

# Features для модели
FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_pct', 'bb_width', 'atr_14',
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'volatility_20', 'volume_ratio',
    'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'pct_from_high_20', 'pct_from_low_20'
]

def train():
    logger.info("=" * 60)
    logger.info("Starting ML Training")
    logger.info("=" * 60)
    
    # Connect to DB
    db_url = os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading")
    engine = create_engine(db_url)
    
    # Load data
    logger.info("Loading data from database...")
    
    # Выберем только нужные колонки которые точно есть
    available_features = []
    for f in FEATURES:
        try:
            test = pd.read_sql(f"SELECT {f} FROM features LIMIT 1", engine)
            available_features.append(f)
        except:
            logger.warning(f"Feature {f} not found, skipping")
    
    logger.info(f"Available features: {len(available_features)}")
    
    cols = ', '.join(available_features)
    query = f"""
        SELECT date, ticker, {cols}, signal_class
        FROM features 
        WHERE signal_class IS NOT NULL
        AND rsi_14 IS NOT NULL
        AND close IS NOT NULL
        AND date >= '2020-01-01'
        ORDER BY date
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Clean data
    df = df.dropna(subset=available_features + ['signal_class'])
    logger.info(f"After dropna: {len(df):,} rows")
    
    # Class distribution
    class_dist = df['signal_class'].value_counts().sort_index()
    logger.info(f"Class distribution:\n{class_dist}")
    
    # Prepare X, y
    X = df[available_features].values
    y = df['signal_class'].values
    
    # Train/test split (time-based)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training GradientBoosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=100,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info("=" * 60)
    logger.info(f"RESULTS:")
    logger.info(f"  Accuracy: {accuracy:.1%}")
    logger.info(f"  F1 Score: {f1:.3f}")
    logger.info(f"  Edge over random: {(accuracy - 0.333) * 100:+.1f}%")
    logger.info("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': available_features,
        'version': f"v4_{datetime.now().strftime('%Y%m%d_%H%M')}",
        'accuracy': accuracy,
        'f1_score': f1,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    model_path = '/app/models/model_v3_macro.joblib'
    joblib.dump(model_data, model_path)
    logger.info(f"✅ Model saved to {model_path}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'version': model_data['version']
    }

if __name__ == "__main__":
    train()
