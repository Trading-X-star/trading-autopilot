import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sqlalchemy import create_engine
import os

class WalkForwardTrainer:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.model = None
        
    def load_features(self) -> pd.DataFrame:
        query = """
            SELECT * FROM features
            WHERE signal_class IS NOT NULL
            ORDER BY date, ticker
        """
        df = pd.read_sql(query, self.engine)
        print(f"Signal class distribution:\n{df['signal_class'].value_counts().sort_index()}")
        return df
    
    def add_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляем дополнительные признаки"""
        print("Adding extra features...")
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df.loc[mask].copy()
            
            # Volatility на разных периодах
            returns = ticker_data['close'].pct_change()
            df.loc[mask, 'volatility_5'] = returns.rolling(5).std()
            df.loc[mask, 'volatility_10'] = returns.rolling(10).std()
            
            # Vol ratio (краткосрочная / долгосрочная волатильность)
            df.loc[mask, 'vol_ratio'] = df.loc[mask, 'volatility_5'] / df.loc[mask, 'volatility_20']
            
            # Momentum
            df.loc[mask, 'momentum_10'] = ticker_data['close'] / ticker_data['close'].shift(10) - 1
            df.loc[mask, 'momentum_20'] = ticker_data['close'] / ticker_data['close'].shift(20) - 1
            
            # RSI momentum
            df.loc[mask, 'rsi_change'] = ticker_data['rsi_14'].diff(5)
            
            # Price position relative to moving averages
            df.loc[mask, 'price_vs_sma20'] = ticker_data['close'] / ticker_data['sma_20'] - 1
            df.loc[mask, 'price_vs_sma50'] = ticker_data['close'] / ticker_data['sma_50'] - 1
            
            # MACD momentum
            df.loc[mask, 'macd_momentum'] = ticker_data['macd_hist'].diff(3)
            
        return df.dropna()
    
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 20):
        df = df.sort_values('date')
        dates = sorted(df['date'].unique())
        
        for i in range(n_splits):
            train_end_idx = len(dates) - (n_splits - i) * 21
            test_end_idx = train_end_idx + 21
            
            if train_end_idx < 252:  # минимум год для обучения
                continue
                
            train_end = dates[min(train_end_idx, len(dates)-1)]
            test_end = dates[min(test_end_idx, len(dates)-1)]
            
            train_mask = df['date'] <= train_end
            test_mask = (df['date'] > train_end) & (df['date'] <= test_end)
            
            if test_mask.sum() < 100:  # минимум 100 примеров в тесте
                continue
                
            yield df[train_mask], df[test_mask], i
    
    def train_walk_forward(self):
        print("Loading features...")
        df = self.load_features()
        print(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
        
        df = self.add_extra_features(df)
        print(f"After feature engineering: {len(df):,} rows")
        
        # Фичи для модели
        feature_cols = [
            # Базовые
            'rsi_14', 'macd', 'macd_hist', 'macd_signal',
            'bb_pct', 'bb_width', 'atr_14',
            # Returns
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            # Volatility
            'volatility_20', 'volatility_5', 'volatility_10', 'vol_ratio',
            # Volume
            'volume_ratio',
            # Position
            'pct_from_high', 'pct_from_low',
            'price_vs_sma20', 'price_vs_sma50',
            # Momentum
            'momentum_10', 'momentum_20', 'rsi_change', 'macd_momentum',
            # Targets (как фичи - только прошлые!)
            'target_1d',
        ]
        
        # Оставляем только существующие колонки
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"\nUsing {len(feature_cols)} features")
        
        results = []
        
        print("\n" + "=" * 60)
        print("Walk-Forward Validation")
        print("=" * 60)
        
        for train_df, test_df, fold in self.walk_forward_split(df):
            X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_train = train_df['signal_class']
            X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_test = test_df['signal_class']
            
            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results.append({
                'fold': fold, 
                'accuracy': accuracy,
                'train_size': len(train_df), 
                'test_size': len(test_df),
                'test_period': f"{test_df['date'].min()} - {test_df['date'].max()}"
            })
            
            print(f"Fold {fold:2d}: Acc={accuracy:.1%} | Train={len(train_df):,} | Test={len(test_df):,}")
        
        if not results:
            print("ERROR: No valid folds. Check data size.")
            return None
            
        results_df = pd.DataFrame(results)
        avg_acc = results_df['accuracy'].mean()
        std_acc = results_df['accuracy'].std()
        
        print("=" * 60)
        print(f"\n📊 RESULTS:")
        print(f"   Walk-Forward Accuracy: {avg_acc:.1%} ± {std_acc:.1%}")
        print(f"   Previous accuracy:     43.5%")
        print(f"   Improvement:          {(avg_acc - 0.435) * 100:+.1f}%")
        print(f"   Min/Max:              {results_df['accuracy'].min():.1%} / {results_df['accuracy'].max():.1%}")
        
        # Финальная модель на всех данных
        print("\n🔧 Training final model on all data...")
        X_all = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_all = df['signal_class']
        
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42
        )
        self.model.fit(X_all, y_all)
        
        # Сохранение
        os.makedirs('/app/models', exist_ok=True)
        model_path = '/app/models/model_wf.joblib'
        joblib.dump({
            'model': self.model, 
            'features': feature_cols,
            'wf_accuracy': avg_acc,
            'wf_std': std_acc,
            'n_folds': len(results),
            'trained_samples': len(df)
        }, model_path)
        
        print(f"\n✅ Model saved: {model_path}")
        
        # Feature importance
        print("\n📈 Top 15 Feature Importance:")
        importance = sorted(zip(feature_cols, self.model.feature_importances_),
                          key=lambda x: x[1], reverse=True)[:15]
        for feat, imp in importance:
            bar = '█' * int(imp * 40)
            print(f"   {feat:18s} {imp:.4f} {bar}")
        
        # Classification report на последнем фолде
        print(f"\n📋 Classification Report (last fold):")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
        
        return results_df

if __name__ == "__main__":
    db_url = os.environ.get('DATABASE_URL') or os.environ.get('DB_DSN',
        'postgresql://trading:trading123@postgres:5432/trading')
    
    trainer = WalkForwardTrainer(db_url)
    trainer.train_walk_forward()
