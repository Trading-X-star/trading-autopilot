#!/usr/bin/env python3
"""Enhanced ML Training with Feature Engineering & Hyperparameter Tuning"""
import asyncio
import asyncpg
import pickle
import logging
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml-train-v2")

DB_DSN = "postgresql://${DB_USER:-trading}:${DB_PASSWORD:-trading123}@${DB_HOST:-postgres}:5432/trading"

# === EXPANDED FEATURES ===
BASE_FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_hist',
    'rsi_14', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
    'atr_14', 'volatility_20', 'volume_ratio',
    'pct_from_high', 'pct_from_low'
]


def engineer_features(X_base, feature_names):
    """Create additional derived features"""
    X = X_base.copy()
    new_features = list(feature_names)
    
    # Feature indices
    idx = {name: i for i, name in enumerate(feature_names)}
    
    additions = []
    
    # 1. Momentum combinations
    if 'return_1d' in idx and 'return_5d' in idx:
        momentum_accel = X[:, idx['return_1d']] - X[:, idx['return_5d']] / 5
        additions.append(momentum_accel)
        new_features.append('momentum_accel')
    
    # 2. RSI zones (oversold/overbought encoding)
    if 'rsi_14' in idx:
        rsi = X[:, idx['rsi_14']]
        additions.append((rsi < 30).astype(float))  # oversold
        additions.append((rsi > 70).astype(float))  # overbought
        additions.append((rsi - 50) / 50)  # normalized RSI
        new_features.extend(['rsi_oversold', 'rsi_overbought', 'rsi_norm'])
    
    # 3. MACD momentum
    if 'macd_hist' in idx:
        macd_h = X[:, idx['macd_hist']]
        additions.append(np.sign(macd_h))  # direction
        new_features.append('macd_direction')
    
    # 4. Trend strength (price vs multiple MAs)
    ma_cols = ['sma_5', 'sma_20', 'sma_50', 'sma_200']
    ma_indices = [idx[c] for c in ma_cols if c in idx]
    if len(ma_indices) >= 2:
        # Count how many MAs price is above (trend alignment)
        trend_count = np.zeros(len(X))
        for mi in ma_indices:
            trend_count += (X[:, mi] > 0).astype(float)  # already normalized
        additions.append(trend_count / len(ma_indices))
        new_features.append('trend_alignment')
    
    # 5. Volatility regime
    if 'volatility_20' in idx and 'atr_14' in idx:
        vol = X[:, idx['volatility_20']]
        vol_zscore = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)
        additions.append(vol_zscore)
        additions.append((vol_zscore > 1).astype(float))  # high vol regime
        new_features.extend(['vol_zscore', 'high_vol_regime'])
    
    # 6. BB squeeze detection
    if 'bb_width' in idx:
        bbw = X[:, idx['bb_width']]
        bbw_ma = np.convolve(bbw, np.ones(20)/20, mode='same')
        squeeze = (bbw < bbw_ma * 0.8).astype(float)
        additions.append(squeeze)
        new_features.append('bb_squeeze')
    
    # 7. Mean reversion signal
    if 'bb_pct' in idx:
        bb_pct = X[:, idx['bb_pct']]
        additions.append(np.clip(1 - abs(bb_pct - 0.5) * 2, 0, 1))  # distance from middle
        new_features.append('mean_reversion_score')
    
    # 8. Cross-feature interactions
    if 'rsi_14' in idx and 'macd_hist' in idx:
        rsi_norm = (X[:, idx['rsi_14']] - 50) / 50
        macd_norm = np.tanh(X[:, idx['macd_hist']] * 100)
        additions.append(rsi_norm * macd_norm)  # agreement score
        new_features.append('rsi_macd_agreement')
    
    # Combine all features
    if additions:
        X_extended = np.column_stack([X] + additions)
    else:
        X_extended = X
    
    logger.info(f"📊 Features: {len(feature_names)} → {len(new_features)}")
    return X_extended, new_features


async def load_data(pool):
    """Load features from database"""
    cols = ', '.join(BASE_FEATURES)
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT {cols}, signal_class, target_5d, close, date, ticker
            FROM features 
            WHERE signal_class IS NOT NULL
            ORDER BY ticker, date
        """)
    logger.info(f"📊 Loaded {len(rows):,} samples")
    return rows


def prepare_data(rows):
    """Prepare X, y arrays with improved target"""
    X, y, y_orig, dates, tickers, targets = [], [], [], [], [], []
    
    for row in rows:
        features = []
        close = float(row['close'] or row['sma_5'] or 1)
        
        for col in BASE_FEATURES:
            val = float(row[col] or 0)
            # Normalize price-based features
            if col in ['sma_5','sma_10','sma_20','sma_50','sma_200','ema_12','ema_26',
                      'bb_upper','bb_middle','bb_lower','atr_14'] and close > 0:
                val = val / close - 1
            features.append(val)
        
        X.append(features)
        y_orig.append(row['signal_class'])
        targets.append(float(row['target_5d'] or 0))
        dates.append(row['date'])
        tickers.append(row['ticker'])
    
    X = np.array(X)
    y_orig = np.array(y_orig)
    targets = np.array(targets)
    
    # === IMPROVED TARGET LABELING ===
    # Filter out noise: only keep strong signals
    threshold = 0.015  # 1.5% minimum move
    
    y_improved = np.zeros(len(targets))
    y_improved[targets > threshold] = 1   # Strong buy
    y_improved[targets < -threshold] = -1  # Strong sell
    # Zeros remain as hold
    
    # Use improved labels but keep mask for evaluation
    strong_signal_mask = np.abs(targets) > threshold * 0.5
    
    logger.info(f"📊 Target distribution:")
    logger.info(f"   Original: Sell={np.mean(y_orig==-1):.1%}, Hold={np.mean(y_orig==0):.1%}, Buy={np.mean(y_orig==1):.1%}")
    logger.info(f"   Improved: Sell={np.mean(y_improved==-1):.1%}, Hold={np.mean(y_improved==0):.1%}, Buy={np.mean(y_improved==1):.1%}")
    
    return X, y_improved, y_orig, targets, dates, tickers, strong_signal_mask


def tune_hyperparams(X_train, y_train, n_trials=30):
    """Simple grid search for key hyperparameters"""
    logger.info("🔧 Tuning hyperparameters...")
    
    y_mapped = y_train + 1
    
    best_score = 0
    best_params = {}
    
    # Key parameters to tune
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.02, 0.05, 0.1],
        'feature_fraction': [0.6, 0.8, 1.0],
        'min_data_in_leaf': [10, 20, 50],
    }
    
    base_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
    }
    
    # TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    trial = 0
    for num_leaves in param_grid['num_leaves']:
        for lr in param_grid['learning_rate']:
            for ff in param_grid['feature_fraction']:
                for mdl in param_grid['min_data_in_leaf']:
                    trial += 1
                    if trial > n_trials:
                        break
                    
                    params = {**base_params, 
                             'num_leaves': num_leaves,
                             'learning_rate': lr,
                             'feature_fraction': ff,
                             'min_data_in_leaf': mdl}
                    
                    scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_t, X_v = X_train[train_idx], X_train[val_idx]
                        y_t, y_v = y_mapped[train_idx], y_mapped[val_idx]
                        
                        train_data = lgb.Dataset(X_t, label=y_t)
                        val_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
                        
                        model = lgb.train(
                            params, train_data, num_boost_round=300,
                            valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(30, verbose=False)]
                        )
                        
                        y_pred = np.argmax(model.predict(X_v), axis=1)
                        acc = np.mean(y_pred == y_v)
                        scores.append(acc)
                    
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params.copy()
                        logger.info(f"   Trial {trial}: {avg_score:.4f} ✓")
    
    logger.info(f"✅ Best CV score: {best_score:.4f}")
    return best_params


def train_model(X_train, y_train, params):
    """Train final model with best params"""
    y_mapped = y_train + 1
    train_data = lgb.Dataset(X_train, label=y_mapped)
    
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model


def evaluate_model(model, X_test, y_test, targets_test):
    """Enhanced evaluation with trading metrics"""
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1) - 1
    
    accuracy = np.mean(y_pred == y_test)
    
    # Confidence-weighted accuracy (high confidence predictions)
    confidence = np.max(y_pred_proba, axis=1)
    high_conf_mask = confidence > 0.5
    if high_conf_mask.sum() > 0:
        high_conf_acc = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask])
        logger.info(f"  High-confidence accuracy (>{0.5}): {high_conf_acc:.1%} ({high_conf_mask.sum()} samples)")
    
    # Per-class metrics
    for cls, name in [(-1, 'Sell'), (0, 'Hold'), (1, 'Buy')]:
        mask = y_test == cls
        if mask.sum() > 0:
            recall = np.mean(y_pred[mask] == cls)
            pred_mask = y_pred == cls
            precision = np.mean(y_test[pred_mask] == cls) if pred_mask.sum() > 0 else 0
            
            # Trading performance for this class
            if cls != 0:  # For buy/sell signals
                trades_mask = y_pred == cls
                if trades_mask.sum() > 0:
                    returns = targets_test[trades_mask] * (1 if cls == 1 else -1)
                    win_rate = np.mean(returns > 0)
                    avg_ret = np.mean(returns) * 100
                    logger.info(f"  {name}: Prec={precision:.1%}, Recall={recall:.1%} | WinRate={win_rate:.1%}, AvgRet={avg_ret:+.2f}%")
                else:
                    logger.info(f"  {name}: Prec={precision:.1%}, Recall={recall:.1%}")
            else:
                logger.info(f"  {name}: Prec={precision:.1%}, Recall={recall:.1%}")
    
    return accuracy, y_pred, y_pred_proba


def backtest_signals(y_pred, y_pred_proba, targets, min_confidence=0.4):
    """Enhanced backtest with confidence filter"""
    confidence = np.max(y_pred_proba, axis=1)
    
    total_return = 0
    trades = 0
    wins = 0
    returns_list = []
    
    for i in range(len(y_pred)):
        if confidence[i] < min_confidence:
            continue
            
        if y_pred[i] == 1:  # Buy
            ret = targets[i]
            total_return += ret
            trades += 1
            if ret > 0: wins += 1
            returns_list.append(ret)
        elif y_pred[i] == -1:  # Sell/short
            ret = -targets[i]
            total_return += ret
            trades += 1
            if ret > 0: wins += 1
            returns_list.append(ret)
    
    win_rate = wins / trades if trades > 0 else 0
    avg_return = total_return / trades if trades > 0 else 0
    sharpe = np.mean(returns_list) / (np.std(returns_list) + 1e-8) * np.sqrt(252/5) if returns_list else 0
    
    return {
        'total_return': total_return,
        'trades': trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe': sharpe
    }


async def main():
    logger.info("🚀 Enhanced ML Training v2")
    
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
    
    # Load data
    rows = await load_data(pool)
    X_base, y, y_orig, targets, dates, tickers, strong_mask = prepare_data(rows)
    
    # Feature engineering
    X, feature_names = engineer_features(X_base, BASE_FEATURES)
    
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    targets_test = targets[split_idx:]
    dates_test = dates[split_idx:]
    
    logger.info(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"📅 Test period: {dates_test[0]} to {dates_test[-1]}")
    
    # Hyperparameter tuning
    best_params = tune_hyperparams(X_train, y_train, n_trials=27)
    
    # Train final model
    logger.info("🎯 Training final model...")
    model = train_model(X_train, y_train, best_params)
    
    # Evaluate
    logger.info("📈 Evaluation:")
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, targets_test)
    logger.info(f"  Overall Accuracy: {accuracy:.1%}")
    
    # Backtest with different confidence thresholds
    logger.info("💰 Backtest (confidence filtered):")
    for conf in [0.3, 0.4, 0.5]:
        bt = backtest_signals(y_pred, y_pred_proba, targets_test, min_confidence=conf)
        logger.info(f"  Conf>{conf}: Trades={bt['trades']}, WinRate={bt['win_rate']:.1%}, "
                   f"Return={bt['total_return']*100:.1f}%, Sharpe={bt['sharpe']:.2f}")
    
    # Feature importance
    logger.info("🔝 Top 15 Features:")
    importance = dict(zip(feature_names, model.feature_importance()))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"   {feat}: {imp}")
    
    # Save model
    model_data = {
        'model': model,
        'features': feature_names,
        'base_features': BASE_FEATURES,
        'params': best_params,
        'trained_at': datetime.now().isoformat(),
        'accuracy': accuracy,
        'version': 'v2'
    }
    
    with open('/app/services/strategy/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    logger.info("💾 Model saved!")
    
    await pool.close()
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
