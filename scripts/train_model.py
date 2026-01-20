#!/usr/bin/env python3
"""Train ML model for signal prediction"""
import asyncio
import asyncpg
import pickle
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml-train")

DB_DSN = "postgresql://trading:trading123@postgres:5432/trading"

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
    'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
    'atr_14', 'volatility_20', 'volume_ratio',
    'pct_from_high', 'pct_from_low'
]


async def load_data(pool):
    """Load features from database"""
    cols = ', '.join(FEATURE_COLS)
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT {cols}, signal_class, date, ticker
            FROM features 
            WHERE signal_class IS NOT NULL
            ORDER BY date
        """)
    logger.info(f"📊 Loaded {len(rows):,} samples")
    return rows


def prepare_data(rows):
    """Prepare X, y arrays"""
    X, y, dates, tickers = [], [], [], []
    
    for row in rows:
        features = [float(row[col] or 0) for col in FEATURE_COLS]
        # Normalize price-based features relative to close
        close = float(row['sma_5'] or 1)  # approximate close
        if close > 0:
            for i, col in enumerate(FEATURE_COLS):
                if col in ['sma_5','sma_10','sma_20','sma_50','sma_200','ema_12','ema_26','bb_upper','bb_middle','bb_lower','atr_14']:
                    features[i] = features[i] / close - 1
        
        X.append(features)
        y.append(row['signal_class'])
        dates.append(row['date'])
        tickers.append(row['ticker'])
    
    return np.array(X), np.array(y), dates, tickers


def train_model(X_train, y_train):
    """Train LightGBM model"""
    import lightgbm as lgb
    
    # Map classes: -1,0,1 -> 0,1,2
    y_mapped = y_train + 1
    
    train_data = lgb.Dataset(X_train, label=y_mapped)
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1) - 1  # Map back: 0,1,2 -> -1,0,1
    
    accuracy = np.mean(y_pred == y_test)
    
    # Per-class metrics
    for cls, name in [(-1, 'Sell'), (0, 'Hold'), (1, 'Buy')]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = np.mean(y_pred[mask] == cls)
            pred_mask = y_pred == cls
            precision = np.mean(y_test[pred_mask] == cls) if pred_mask.sum() > 0 else 0
            logger.info(f"  {name}: Recall={cls_acc:.1%}, Precision={precision:.1%}, Support={mask.sum():,}")
    
    return accuracy, y_pred, y_pred_proba


def backtest_signals(y_test, y_pred, target_returns, dates):
    """Simple backtest of predicted signals"""
    # y_test and target are actual values
    # y_pred are predictions
    
    total_return = 0
    trades = 0
    wins = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == 1:  # Buy signal
            ret = target_returns[i] if target_returns[i] else 0
            total_return += ret
            trades += 1
            if ret > 0: wins += 1
        elif y_pred[i] == -1:  # Sell/short signal
            ret = target_returns[i] if target_returns[i] else 0
            total_return -= ret  # Inverse for short
            trades += 1
            if ret < 0: wins += 1
    
    win_rate = wins / trades if trades > 0 else 0
    avg_return = total_return / trades if trades > 0 else 0
    
    return {
        'total_return': total_return,
        'trades': trades,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_return
    }


async def main():
    logger.info("🚀 ML Training started")
    
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
    
    # Load data
    rows = await load_data(pool)
    X, y, dates, tickers = prepare_data(rows)
    
    # Get target returns for backtest
    async with pool.acquire() as conn:
        target_rows = await conn.fetch("""
            SELECT target_5d FROM features WHERE signal_class IS NOT NULL ORDER BY date
        """)
    target_returns = [float(r['target_5d'] or 0) for r in target_rows]
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    target_test = target_returns[split_idx:]
    
    logger.info(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"📅 Test period: {dates_test[0]} to {dates_test[-1]}")
    
    # Train
    logger.info("🔧 Training LightGBM...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    logger.info("📈 Evaluation:")
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    logger.info(f"  Overall Accuracy: {accuracy:.1%}")
    
    # Backtest
    logger.info("💰 Backtest results:")
    bt = backtest_signals(y_test, y_pred, target_test, dates_test)
    logger.info(f"  Trades: {bt['trades']:,}")
    logger.info(f"  Win Rate: {bt['win_rate']:.1%}")
    logger.info(f"  Avg Return/Trade: {bt['avg_return_per_trade']*100:.2f}%")
    logger.info(f"  Total Return: {bt['total_return']*100:.1f}%")
    
    # Feature importance
    logger.info("🔍 Top 10 Features:")
    importance = dict(zip(FEATURE_COLS, model.feature_importance()))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {feat}: {imp}")
    
    # Save model
    with open('/scripts/model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'features': FEATURE_COLS,
            'trained_at': datetime.now().isoformat(),
            'accuracy': accuracy,
            'backtest': bt
        }, f)
    logger.info("💾 Model saved to /scripts/model.pkl")
    
    await pool.close()
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
