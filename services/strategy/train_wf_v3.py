#!/usr/bin/env python3
"""
Walk-Forward Trainer v4 - Self-Learning Edition
Features DB + CBR Macro + MOEX News + Auto-Retrain
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine, text
import httpx
import asyncio
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from threading import Lock
import os
import json
import warnings
import schedule
import time
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trainer_v4")


# ============================================================
# CBR CLIENT (Центральный Банк России)
# ============================================================
class CBRClient:
    """Загрузка макро-данных ЦБ РФ"""
    
    BASE_URL = "https://www.cbr.ru"
    CURRENCY_CODES = {'USD': 'R01235', 'EUR': 'R01239', 'CNY': 'R01375'}
    
    def __init__(self):
        self.client = httpx.Client(timeout=30, follow_redirects=True)
    
    def get_currency_history(self, currency: str = 'USD', days: int = 2500) -> pd.DataFrame:
        """История курса валюты"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        code = self.CURRENCY_CODES.get(currency, 'R01235')
        url = f"{self.BASE_URL}/scripts/XML_dynamic.asp"
        params = {
            'date_req1': start_date.strftime('%d/%m/%Y'),
            'date_req2': end_date.strftime('%d/%m/%Y'),
            'VAL_NM_RQ': code
        }
        
        try:
            resp = self.client.get(url, params=params)
            root = ET.fromstring(resp.text)
            
            records = []
            for record in root.findall('.//Record'):
                records.append({
                    'date': datetime.strptime(record.get('Date'), '%d.%m.%Y'),
                    f'{currency.lower()}_rate': float(record.find('Value').text.replace(',', '.'))
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"CBR {currency} error: {e}")
            return pd.DataFrame()
    
    def get_key_rate_history(self) -> pd.DataFrame:
        """История ключевой ставки"""
        url = f"{self.BASE_URL}/scripts/XML_key_rate.asp"
        
        try:
            resp = self.client.get(url)
            root = ET.fromstring(resp.text)
            
            records = []
            for record in root.findall('.//Record'):
                records.append({
                    'date': datetime.strptime(record.get('Date'), '%d.%m.%Y'),
                    'key_rate': float(record.find('Rate').text.replace(',', '.'))
                })
            return pd.DataFrame(records).sort_values('date')
        except Exception as e:
            logger.warning(f"CBR key rate error: {e}")
            return pd.DataFrame()
    
    def load_all_macro(self) -> pd.DataFrame:
        """Загрузка всех макро-данных"""
        logger.info("Loading CBR macro data...")
        
        usd = self.get_currency_history('USD')
        eur = self.get_currency_history('EUR')
        key_rate = self.get_key_rate_history()
        
        if usd.empty:
            return pd.DataFrame()
        
        macro = usd.copy()
        if not eur.empty:
            macro = macro.merge(eur, on='date', how='outer')
        if not key_rate.empty:
            macro = macro.merge(key_rate, on='date', how='outer')
        
        macro = macro.sort_values('date').ffill()
        return macro
    
    def close(self):
        self.client.close()


# ============================================================
# MOEX NEWS CLIENT
# ============================================================
class MOEXNewsClient:
    """Загрузка и анализ новостей MOEX"""
    
    ISS_URL = "https://iss.moex.com/iss/sitenews.json"
    
    TICKER_PATTERN = re.compile(r'\b([A-Z]{4})\b')
    
    # Включаем делистинговые тикеры для survivorship bias
    KNOWN_TICKERS = {
        'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'VTBR', 'MTSS',
        'MGNT', 'TATN', 'SNGS', 'NLMK', 'ALRS', 'CHMF', 'MAGN', 'PLZL',
        'YNDX', 'POLY', 'MOEX', 'FIVE', 'TCSG', 'AFKS', 'IRAO', 'HYDR',
        # Делистинговые/исторические
        'MAIL', 'PIKK', 'MVID', 'DSKY', 'RUAL', 'AFLT', 'OZON'
    }
    
    POSITIVE_WORDS = {
        'рост', 'прибыль', 'дивиденд', 'увеличение', 'повышение', 'рекорд',
        'успех', 'выручка', 'покупка', 'инвестиции', 'развитие', 'расширение'
    }
    NEGATIVE_WORDS = {
        'падение', 'убыток', 'снижение', 'сокращение', 'риск', 'потери',
        'штраф', 'санкции', 'дефолт', 'банкротство', 'продажа', 'закрытие'
    }
    
    def __init__(self):
        self.client = httpx.Client(timeout=30)
    
    def fetch_iss_news(self, pages: int = 100) -> pd.DataFrame:
        """Загрузка новостей через ISS API"""
        all_news = []
        
        for start in range(0, pages * 50, 50):
            try:
                resp = self.client.get(self.ISS_URL, params={'start': start})
                data = resp.json()
                
                sitenews = data.get('sitenews', {})
                columns = sitenews.get('columns', [])
                rows = sitenews.get('data', [])
                
                if not rows:
                    break
                
                for row in rows:
                    item = dict(zip(columns, row))
                    title = item.get('title', '')
                    
                    try:
                        pub_date = datetime.strptime(item.get('published_at', ''), '%Y-%m-%d %H:%M:%S')
                    except:
                        continue
                    
                    tickers = self._extract_tickers(title)
                    sentiment = self._analyze_sentiment(title)
                    
                    all_news.append({
                        'date': pub_date.date(),
                        'title': title,
                        'tickers': tickers,
                        'sentiment': sentiment
                    })
                    
            except Exception as e:
                logger.warning(f"News page {start} error: {e}")
                break
        
        return pd.DataFrame(all_news)
    
    def aggregate_daily_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация sentiment по дням и тикерам с ЛАГОМ +1 день"""
        if news_df.empty:
            return pd.DataFrame()
        
        expanded = []
        for _, row in news_df.iterrows():
            for ticker in row['tickers']:
                expanded.append({
                    # ✅ FIX: Сдвигаем на +1 день чтобы избежать look-ahead bias
                    'date': row['date'] + timedelta(days=1),
                    'ticker': ticker,
                    'sentiment': row['sentiment']
                })
        
        if not expanded:
            return pd.DataFrame()
        
        exp_df = pd.DataFrame(expanded)
        
        daily = exp_df.groupby(['date', 'ticker']).agg({
            'sentiment': ['mean', 'sum', 'count']
        }).reset_index()
        daily.columns = ['date', 'ticker', 'news_sentiment', 'news_sentiment_sum', 'news_count']
        
        return daily
    
    def _extract_tickers(self, text: str) -> List[str]:
        found = self.TICKER_PATTERN.findall(text.upper())
        return [t for t in found if t in self.KNOWN_TICKERS]
    
    def _analyze_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0
    
    def close(self):
        self.client.close()


# ============================================================
# SECTOR MAPPING
# ============================================================
SECTOR_MAP = {
    # Нефтегаз (экспортёры)
    'GAZP': 'oil_gas', 'LKOH': 'oil_gas', 'ROSN': 'oil_gas',
    'NVTK': 'oil_gas', 'TATN': 'oil_gas', 'SNGS': 'oil_gas',
    # Банки
    'SBER': 'banks', 'VTBR': 'banks', 'TCSG': 'banks', 'CBOM': 'banks',
    # Металлы (экспортёры)
    'GMKN': 'metals', 'NLMK': 'metals', 'MAGN': 'metals',
    'CHMF': 'metals', 'ALRS': 'metals', 'PLZL': 'metals', 'POLY': 'metals',
    # Ритейл
    'MGNT': 'retail', 'FIVE': 'retail',
    # Телеком
    'MTSS': 'telecom', 'RTKM': 'telecom',
    # IT
    'YNDX': 'tech', 'OZON': 'tech',
    # Электроэнергетика
    'IRAO': 'energy', 'HYDR': 'energy',
}


# ============================================================
# PERFORMANCE TRACKER (для самообучения)
# ============================================================
class PerformanceTracker:
    """Отслеживание метрик модели для автоматического переобучения"""
    
    def __init__(self, db_engine, threshold_accuracy: float = 0.38,
                 threshold_sharpe: float = 0.3, lookback_days: int = 30):
        self.engine = db_engine
        self.threshold_accuracy = threshold_accuracy
        self.threshold_sharpe = threshold_sharpe
        self.lookback_days = lookback_days
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Создание таблиц для мониторинга"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    ticker VARCHAR(20),
                    predicted_class INTEGER,
                    predicted_proba FLOAT,
                    actual_class INTEGER,
                    actual_return FLOAT,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    accuracy FLOAT,
                    f1_score FLOAT,
                    sharpe_ratio FLOAT,
                    max_drawdown FLOAT,
                    predictions_count INTEGER,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS retrain_history (
                    id SERIAL PRIMARY KEY,
                    triggered_at TIMESTAMPTZ DEFAULT NOW(),
                    reason VARCHAR(100),
                    old_accuracy FLOAT,
                    new_accuracy FLOAT,
                    old_sharpe FLOAT,
                    new_sharpe FLOAT,
                    model_version VARCHAR(50)
                )
            """))
            conn.commit()
    
    def log_prediction(self, date, ticker: str, predicted_class: int,
                       predicted_proba: float, model_version: str):
        """Логирование предсказания"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_predictions (date, ticker, predicted_class, predicted_proba, model_version)
                VALUES (:date, :ticker, :pred_class, :pred_proba, :version)
            """), {
                'date': date, 'ticker': ticker, 'pred_class': predicted_class,
                'pred_proba': predicted_proba, 'version': model_version
            })
            conn.commit()
    
    def update_actuals(self):
        """Обновление фактических результатов из features DB"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE model_predictions mp
                SET actual_class = f.signal_class,
                    actual_return = f.return_5d
                FROM features f
                WHERE mp.date = f.date 
                  AND mp.ticker = f.ticker
                  AND mp.actual_class IS NULL
            """))
            conn.commit()
    
    def calculate_metrics(self) -> Dict:
        """Расчёт метрик за последние N дней"""
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT predicted_class, actual_class, predicted_proba, actual_return
                FROM model_predictions
                WHERE date >= :cutoff AND actual_class IS NOT NULL
            """), {'cutoff': cutoff_date})
            rows = result.fetchall()
        
        if len(rows) < 50:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        df = pd.DataFrame(rows, columns=['predicted', 'actual', 'proba', 'return'])
        
        # Accuracy & F1
        accuracy = accuracy_score(df['actual'], df['predicted'])
        f1 = f1_score(df['actual'], df['predicted'], average='weighted')
        
        # Sharpe Ratio
        positions = df['predicted'].map({-1: -1, 0: 0, 1: 1})
        strategy_returns = positions * df['return']
        sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-8)) * np.sqrt(252)
        
        # Max Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        metrics = {
            'valid': True,
            'accuracy': accuracy,
            'f1_score': f1,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'predictions_count': len(df)
        }
        
        # Сохраняем метрики
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_metrics (date, accuracy, f1_score, sharpe_ratio, max_drawdown, predictions_count, model_version)
                VALUES (:date, :acc, :f1, :sharpe, :mdd, :count, :version)
            """), {
                'date': datetime.now().date(), 'acc': accuracy, 'f1': f1,
                'sharpe': sharpe, 'mdd': max_dd, 'count': len(df), 'version': 'v4'
            })
            conn.commit()
        
        return metrics
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Проверка необходимости переобучения"""
        metrics = self.calculate_metrics()
        
        if not metrics.get('valid'):
            return False, metrics.get('reason', 'unknown')
        
        # Условия для переобучения
        if metrics['accuracy'] < self.threshold_accuracy:
            return True, f"accuracy_degraded_{metrics['accuracy']:.2%}"
        
        if metrics['sharpe_ratio'] < self.threshold_sharpe:
            return True, f"sharpe_degraded_{metrics['sharpe_ratio']:.2f}"
        
        if metrics['max_drawdown'] < -0.15:
            return True, f"high_drawdown_{metrics['max_drawdown']:.2%}"
        
        return False, "metrics_ok"
    
    def log_retrain(self, reason: str, old_metrics: Dict, new_metrics: Dict, version: str):
        """Логирование переобучения"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO retrain_history (reason, old_accuracy, new_accuracy, old_sharpe, new_sharpe, model_version)
                VALUES (:reason, :old_acc, :new_acc, :old_sharpe, :new_sharpe, :version)
            """), {
                'reason': reason,
                'old_acc': old_metrics.get('accuracy'),
                'new_acc': new_metrics.get('accuracy'),
                'old_sharpe': old_metrics.get('sharpe_ratio'),
                'new_sharpe': new_metrics.get('sharpe_ratio'),
                'version': version
            })
            conn.commit()


# ============================================================
# WALK-FORWARD TRAINER v4
# ============================================================
class WalkForwardTrainerV4:
    """Walk-Forward Trainer с самообучением"""
    
    VERSION = "v4_self_learning"
    
    def __init__(self, db_url: str, models_dir: str = "/app/models"):
        self.engine = create_engine(db_url)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
        self.cbr = CBRClient()
        self.news_client = MOEXNewsClient()
        self.tracker = PerformanceTracker(self.engine)
        
        self._lock = Lock()
    
    def load_features(self) -> pd.DataFrame:
        """Загрузка фичей из БД с проверкой returns"""
        query = """
            SELECT * FROM features
            WHERE signal_class IS NOT NULL
            ORDER BY date, ticker
        """
        df = pd.read_sql(query, self.engine)
        
        # ✅ Проверяем что returns - это ПРОШЛЫЕ returns
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Signal distribution:\n{df['signal_class'].value_counts().sort_index()}")
        
        return df
    
    def load_macro_data(self) -> pd.DataFrame:
        """Загрузка макро-данных ЦБ РФ"""
        logger.info("Loading CBR macro data...")
        macro = self.cbr.load_all_macro()
        
        if macro.empty:
            return pd.DataFrame()
        
        # Производные фичи
        if 'usd_rate' in macro.columns:
            macro['usd_change_1d'] = macro['usd_rate'].pct_change(1)
            macro['usd_change_5d'] = macro['usd_rate'].pct_change(5)
            macro['usd_change_20d'] = macro['usd_rate'].pct_change(20)
            macro['usd_volatility'] = macro['usd_rate'].rolling(20).std() / macro['usd_rate'].rolling(20).mean()
            macro['usd_ma_20'] = macro['usd_rate'].rolling(20).mean()
            macro['usd_vs_ma'] = macro['usd_rate'] / macro['usd_ma_20'] - 1
        
        if 'eur_rate' in macro.columns:
            macro['eur_change_5d'] = macro['eur_rate'].pct_change(5)
        
        if 'key_rate' in macro.columns:
            macro['rate_change'] = macro['key_rate'].diff()
            macro['rate_high'] = (macro['key_rate'] > 12).astype(int)
            macro['rate_rising'] = (macro['rate_change'] > 0).astype(int)
        
        logger.info(f"Loaded {len(macro)} days of macro data")
        return macro
    
    def load_news_sentiment(self) -> pd.DataFrame:
        """Загрузка новостного sentiment"""
        logger.info("Loading MOEX news sentiment...")
        
        news_df = self.news_client.fetch_iss_news(pages=50)
        
        if news_df.empty:
            return pd.DataFrame()
        
        # ✅ FIX: Sentiment сдвинут на +1 день внутри aggregate_daily_sentiment
        daily_sentiment = self.news_client.aggregate_daily_sentiment(news_df)
        
        logger.info(f"Loaded {len(news_df)} news, aggregated to {len(daily_sentiment)} ticker-day records")
        return daily_sentiment
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Технические фичи"""
        logger.info("Adding technical features...")
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df.loc[mask].copy()
            
            returns = ticker_data['close'].pct_change()
            
            # Volatility
            df.loc[mask, 'volatility_5'] = returns.rolling(5).std()
            df.loc[mask, 'volatility_10'] = returns.rolling(10).std()
            vol_20 = df.loc[mask, 'volatility_20'].replace(0, np.nan)
            df.loc[mask, 'vol_ratio'] = df.loc[mask, 'volatility_5'] / vol_20
            
            # Momentum (используем ПРОШЛЫЕ данные!)
            df.loc[mask, 'momentum_10'] = ticker_data['close'].shift(1) / ticker_data['close'].shift(11) - 1
            df.loc[mask, 'momentum_20'] = ticker_data['close'].shift(1) / ticker_data['close'].shift(21) - 1
            
            # RSI dynamics
            df.loc[mask, 'rsi_change_5'] = ticker_data['rsi_14'].diff(5)
            df.loc[mask, 'rsi_ma_5'] = ticker_data['rsi_14'].rolling(5).mean()
            df.loc[mask, 'rsi_oversold'] = (ticker_data['rsi_14'] < 30).astype(int)
            df.loc[mask, 'rsi_overbought'] = (ticker_data['rsi_14'] > 70).astype(int)
            
            # Price vs MAs
            df.loc[mask, 'price_vs_sma20'] = ticker_data['close'] / ticker_data['sma_20'] - 1
            df.loc[mask, 'price_vs_sma50'] = ticker_data['close'] / ticker_data['sma_50'] - 1
            df.loc[mask, 'sma20_vs_sma50'] = ticker_data['sma_20'] / ticker_data['sma_50'] - 1
            df.loc[mask, 'above_sma200'] = (ticker_data['close'] > ticker_data['sma_200']).astype(int)
            
            # MACD dynamics
            df.loc[mask, 'macd_change'] = ticker_data['macd_hist'].diff(3)
            df.loc[mask, 'macd_positive'] = (ticker_data['macd_hist'] > 0).astype(int)
            
            # Volume dynamics
            df.loc[mask, 'volume_change'] = ticker_data['volume_ratio'].diff(3)
            df.loc[mask, 'volume_spike'] = (ticker_data['volume_ratio'] > 2).astype(int)
            
            # BB dynamics
            df.loc[mask, 'bb_pct_change'] = ticker_data['bb_pct'].diff(3)
            
            # ATR normalized
            df.loc[mask, 'atr_pct'] = ticker_data['atr_14'] / ticker_data['close']
        
        return df
    
    def add_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Секторные фичи"""
        df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('other')
        
        # Sector-macro interactions
        if 'usd_change_5d' in df.columns:
            df['sector_usd_impact'] = 0.0
            exporters = df['sector'].isin(['oil_gas', 'metals'])
            df.loc[exporters, 'sector_usd_impact'] = df.loc[exporters, 'usd_change_5d'].fillna(0)
            importers = df['sector'].isin(['retail', 'tech'])
            df.loc[importers, 'sector_usd_impact'] = -df.loc[importers, 'usd_change_5d'].fillna(0)
        
        if 'key_rate' in df.columns:
            df['sector_rate_impact'] = 0.0
            banks = df['sector'] == 'banks'
            df.loc[banks, 'sector_rate_impact'] = df.loc[banks, 'key_rate'].fillna(0) / 100
            retail = df['sector'] == 'retail'
            df.loc[retail, 'sector_rate_impact'] = -df.loc[retail, 'key_rate'].fillna(0) / 100
        
        # One-hot encoding
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        return df
    
    def check_feature_stability(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """✅ Проверка стабильности фичей во времени"""
        logger.info("Checking feature stability...")
        
        warnings = []
        median_date = df['date'].median()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            early = df[df['date'] < median_date][col].mean()
            late = df[df['date'] >= median_date][col].mean()
            
            if abs(early) < 1e-8 and abs(late) < 1e-8:
                continue
            
            drift = abs(early - late) / (abs(early) + abs(late) + 1e-8)
            
            if drift > 0.5:
                warnings.append(f"⚠️ Feature drift: {col} (early={early:.3f}, late={late:.3f}, drift={drift:.1%})")
        
        for w in warnings[:10]:
            logger.warning(w)
        
        return warnings
    
    def merge_all_data(self, features_df: pd.DataFrame, macro_df: pd.DataFrame,
                       news_df: pd.DataFrame) -> pd.DataFrame:
        """Объединение всех источников"""
        df = features_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if not macro_df.empty:
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            df = df.merge(macro_df, on='date', how='left')
        
        if not news_df.empty:
            news_df['date'] = pd.to_datetime(news_df['date'])
            df = df.merge(news_df, on=['date', 'ticker'], how='left')
            df['news_sentiment'] = df['news_sentiment'].fillna(0)
            df['news_count'] = df['news_count'].fillna(0)
        
        # Forward fill macro columns
        macro_cols = ['usd_rate', 'eur_rate', 'key_rate', 'usd_change_1d', 'usd_change_5d',
                      'usd_change_20d', 'usd_volatility', 'usd_vs_ma', 'eur_change_5d',
                      'rate_change', 'rate_high', 'rate_rising']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        return df
    
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 20, embargo_days: int = 5):
        """✅ Walk-Forward с embargo периодом"""
        df = df.sort_values('date')
        dates = sorted(df['date'].unique())
        
        for i in range(n_splits):
            train_end_idx = len(dates) - (n_splits - i) * 21
            # ✅ Embargo: пропускаем дни между train и test
            test_start_idx = train_end_idx + embargo_days
            test_end_idx = train_end_idx + 21 + embargo_days
            
            if train_end_idx < 252 or test_start_idx >= len(dates):
                continue
            
            train_end = dates[min(train_end_idx, len(dates) - 1)]
            test_start = dates[min(test_start_idx, len(dates) - 1)]
            test_end = dates[min(test_end_idx, len(dates) - 1)]
            
            train_mask = df['date'] <= train_end
            test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
            
            if test_mask.sum() < 50:
                continue
            
            yield df[train_mask], df[test_mask], i
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   returns: np.ndarray) -> Dict:
        """✅ Trading-based метрики"""
        positions = pd.Series(y_pred).map({-1: -1, 0: 0, 1: 1}).values
        
        # Shift positions to avoid look-ahead
        strategy_returns = np.roll(positions, 1)[1:] * returns[1:]
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) < 1e-8:
            return {'sharpe': 0, 'max_drawdown': 0, 'total_return': 0}
        
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        cumulative = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = np.min(drawdown)
        
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return
        }
    
    def create_ensemble_model(self):
        """✅ Ensemble с class_weight и калибровкой"""
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42
        )
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=50,
            class_weight='balanced',  # ✅ Fix class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        ensemble = VotingClassifier(
            estimators=[('gb', gb), ('rf', rf)],
            voting='soft'
        )
        
        return ensemble
    
    def train(self, save_model: bool = True) -> Dict:
        """Основной метод обучения"""
        with self._lock:
            return self._train_internal(save_model)
    
    def _train_internal(self, save_model: bool = True) -> Dict:
        logger.info("=" * 70)
        logger.info("WALK-FORWARD TRAINING v4 - Self Learning")
        logger.info("=" * 70)
        
        # 1. Load data
        features_df = self.load_features()
        macro_df = self.load_macro_data()
        news_df = self.load_news_sentiment()
        
        # 2. Merge
        df = self.merge_all_data(features_df, macro_df, news_df)
        
        # 3. Feature engineering
        df = self.add_technical_features(df)
        df = self.add_sector_features(df)
        
        # 4. Clean
        df = df.dropna(subset=['signal_class'])
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"Final dataset: {len(df):,} rows")
        
        # 5. Define features
        self.feature_cols = [
            # Technical
            'rsi_14', 'rsi_change_5', 'rsi_ma_5', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_hist', 'macd_signal', 'macd_change', 'macd_positive',
            'bb_pct', 'bb_width', 'bb_pct_change',
            'atr_14', 'atr_pct',
            # Returns (past!)
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            # Volatility
            'volatility_20', 'volatility_5', 'volatility_10', 'vol_ratio',
            # Volume
            'volume_ratio', 'volume_change', 'volume_spike',
            # Price position
            'pct_from_high', 'pct_from_low',
            'price_vs_sma20', 'price_vs_sma50', 'sma20_vs_sma50', 'above_sma200',
            # Momentum
            'momentum_10', 'momentum_20',
            # Macro
            'usd_rate', 'usd_change_1d', 'usd_change_5d', 'usd_change_20d',
            'usd_volatility', 'usd_vs_ma', 'eur_change_5d',
            'key_rate', 'rate_change', 'rate_high', 'rate_rising',
            # News
            'news_sentiment', 'news_count',
            # Sector
            'sector_usd_impact', 'sector_rate_impact',
        ]
        
        # Add sector dummies
        sector_cols = [c for c in df.columns if c.startswith('sector_')
                       and c not in ['sector_usd_impact', 'sector_rate_impact']]
        self.feature_cols.extend(sector_cols)
        
        # Filter existing
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        logger.info(f"Using {len(self.feature_cols)} features")
        
        # Check feature stability
        self.check_feature_stability(df, self.feature_cols)
        
        # 6. Walk-Forward Validation
        results = []
        all_predictions = []
        
        logger.info("-" * 70)
        logger.info("Walk-Forward Validation (with embargo):")
        
        for train_df, test_df, fold in self.walk_forward_split(df, embargo_days=5):
            X_train = train_df[self.feature_cols].fillna(0)
            y_train = train_df['signal_class']
            X_test = test_df[self.feature_cols].fillna(0)
            y_test = test_df['signal_class']
            returns_test = test_df['return_5d'].values
            
            model = self.create_ensemble_model()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            trading = self.calculate_trading_metrics(y_test.values, y_pred, returns_test)
            
            results.append({
                'fold': fold,
                'accuracy': accuracy,
                'f1': f1,
                'sharpe': trading['sharpe'],
                'max_dd': trading['max_drawdown'],
                'train_size': len(train_df),
                'test_size': len(test_df),
            })
            
            # Сохраняем предсказания
            for i, (_, row) in enumerate(test_df.iterrows()):
                all_predictions.append({
                    'date': row['date'],
                    'ticker': row['ticker'],
                    'predicted': y_pred[i],
                    'actual': y_test.iloc[i],
                    'proba': np.max(y_proba[i]),
                    'return': returns_test[i]
                })
            
            logger.info(f"Fold {fold:2d}: Acc={accuracy:.1%} F1={f1:.2f} Sharpe={trading['sharpe']:.2f} | Train={len(train_df):,} Test={len(test_df):,}")
        
        results_df = pd.DataFrame(results)
        
        # 7. Results
        metrics = {
            'accuracy': results_df['accuracy'].mean(),
            'accuracy_std': results_df['accuracy'].std(),
            'f1': results_df['f1'].mean(),
            'sharpe': results_df['sharpe'].mean(),
            'max_drawdown': results_df['max_dd'].mean(),
            'n_folds': len(results)
        }
        
        logger.info("-" * 70)
        logger.info(f"RESULTS:")
        logger.info(f"  Walk-Forward Accuracy: {metrics['accuracy']:.1%} ± {metrics['accuracy_std']:.1%}")
        logger.info(f"  Walk-Forward F1:       {metrics['f1']:.3f}")
        logger.info(f"  Walk-Forward Sharpe:   {metrics['sharpe']:.2f}")
        logger.info(f"  Walk-Forward Max DD:   {metrics['max_drawdown']:.1%}")
        logger.info(f"  Baseline (random):     33.3%")
        logger.info(f"  Edge over random:      {(metrics['accuracy'] - 0.333) * 100:+.1f}%")
        
        # 8. Train final model with calibration
        logger.info("Training final calibrated model...")
        X_all = df[self.feature_cols].fillna(0)
        y_all = df['signal_class']
        
        # Fit scaler
        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)
        
        base_model = self.create_ensemble_model()
        
        # ✅ Calibrated probabilities
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        self.model.fit(X_scaled, y_all)
        
        # 9. Save model
        if save_model:
            model_path = self.models_dir / f"model_{self.VERSION}.joblib"
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_cols,
                'metrics': metrics,
                'version': self.VERSION,
                'trained_at': datetime.now().isoformat(),
                'data_sources': ['features_db', 'cbr_macro', 'moex_news']
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"✅ Model saved: {model_path}")
            
            # Symlink to latest
            latest_path = self.models_dir / "model_latest.joblib"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(model_path.name)
        
        # 10. Feature importance
        logger.info("\nTop 15 Feature Importance:")
        try:
            # Get base estimator from calibrated model
            base = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base, 'named_estimators_'):
                gb_model = base.named_estimators_['gb']
                importance = sorted(zip(self.feature_cols, gb_model.feature_importances_),
                                   key=lambda x: x[1], reverse=True)[:15]
                for feat, imp in importance:
                    bar = '█' * int(imp * 50)
                    logger.info(f"  {feat:22s} {imp:.4f} {bar}")
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        # Cleanup
        self.cbr.close()
        self.news_client.close()
        
        return metrics
    
    def predict(self, features: Dict) -> Dict:
        """Предсказание для одного образца"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        X = np.array([[features.get(f, 0) for f in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_
        
        pred_class = int(classes[np.argmax(proba)])
        confidence = float(np.max(proba))
        
        return {
            'signal': pred_class,
            'confidence': confidence,
            'probabilities': {int(c): float(p) for c, p in zip(classes, proba)}
        }
    
    def load_model(self, path: str = None):
        """Загрузка модели"""
        if path is None:
            path = self.models_dir / "model_latest.joblib"
        
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data.get('scaler', StandardScaler())
            self.feature_cols = data['features']
            logger.info(f"✅ Model loaded: {data.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# ============================================================
# SELF-LEARNING SCHEDULER
# ============================================================
class SelfLearningScheduler:
    """Планировщик самообучения"""
    
    def __init__(self, trainer: WalkForwardTrainerV4, 
                 check_interval_hours: int = 6,
                 retrain_day: str = "sunday",
                 retrain_time: str = "03:00"):
        self.trainer = trainer
        self.check_interval = check_interval_hours
        self.running = False
    
    def check_and_retrain(self):
        """Проверка метрик и переобучение при необходимости"""
        logger.info("🔍 Checking model performance...")
        
        # Обновляем фактические результаты
        self.trainer.tracker.update_actuals()
        
        # Проверяем нужно ли переобучать
        should_retrain, reason = self.trainer.tracker.should_retrain()
        
        if should_retrain:
            logger.info(f"⚠️ Retrain triggered: {reason}")
            
            old_metrics = self.trainer.tracker.calculate_metrics()
            
            # Переобучаем
            new_metrics = self.trainer.train(save_model=True)
            
            # Логируем
            self.trainer.tracker.log_retrain(
                reason=reason,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                version=self.trainer.VERSION
            )
            
            logger.info(f"✅ Retrain complete. New accuracy: {new_metrics['accuracy']:.1%}")
        else:
            logger.info(f"✅ Model performance OK: {reason}")
    
    def scheduled_retrain(self):
        """Плановое еженедельное переобучение"""
        logger.info("📅 Scheduled weekly retrain...")
        
        old_metrics = self.trainer.tracker.calculate_metrics()
        new_metrics = self.trainer.train(save_model=True)
        
        self.trainer.tracker.log_retrain(
            reason="scheduled_weekly",
            old_metrics=old_metrics if old_metrics.get('valid') else {},
            new_metrics=new_metrics,
            version=self.trainer.VERSION
        )
        
        logger.info(f"✅ Weekly retrain complete")
    
    def start(self):
        """Запуск планировщика"""
        self.running = True
        
        # Проверка каждые N часов
        schedule.every(self.check_interval).hours.do(self.check_and_retrain)
        
        # Еженедельное переобучение
        schedule.every().sunday.at("03:00").do(self.scheduled_retrain)
        
        logger.info(f"🚀 Self-learning scheduler started")
        logger.info(f"   - Performance check: every {self.check_interval} hours")
        logger.info(f"   - Scheduled retrain: Sunday 03:00")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop(self):
        """Остановка планировщика"""
        self.running = False
        logger.info("Scheduler stopped")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'serve', 'self-learn'], default='train')
    parser.add_argument('--db-url', default=os.environ.get('DATABASE_URL', 
        'postgresql://${DB_USER:-trading}:${DB_PASSWORD:-trading123}@${DB_HOST:-postgres}:5432/trading'))
    args = parser.parse_args()
    
    trainer = WalkForwardTrainerV4(args.db_url)
    
    if args.mode == 'train':
        # Однократное обучение
        trainer.train(save_model=True)
    
    elif args.mode == 'self-learn':
        # Запуск с самообучением
        trainer.train(save_model=True)  # Initial train
        
        scheduler = SelfLearningScheduler(trainer)
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.stop()
    
    elif args.mode == 'serve':
        # Только загрузка модели для serving
        trainer.load_model()
        print(f"Model ready. Features: {len(trainer.feature_cols)}")
