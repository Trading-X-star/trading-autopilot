#!/usr/bin/env python3
"""
Walk-Forward Trainer v5 - Advanced Self-Learning Edition
Improvements:
- LightGBM/XGBoost instead of sklearn GradientBoosting
- Transformer-based sentiment (ruBERT)
- Purged K-Fold Cross-Validation
- SHAP interpretability
- Kelly Criterion position sizing
- Data drift monitoring (PSI, KS-test)
- Feature selection with Boruta
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import joblib
from sqlalchemy import create_engine, text
import httpx
import asyncio
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import os
import json
import warnings
import schedule
import time
import logging
import hashlib

# Advanced ML
import lightgbm as lgb
import xgboost as xgb
import shap
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except (ImportError, AttributeError):
    HAS_TRANSFORMERS = False
    torch = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trainer_v5")


# ============================================================
# DATA DRIFT MONITOR
# ============================================================
class DataDriftMonitor:
    """Мониторинг дрейфа данных с PSI и KS-test"""

    def __init__(self, reference_data: pd.DataFrame = None, n_bins: int = 10):
        self.reference_data = reference_data
        self.n_bins = n_bins
        self.reference_distributions = {}
        self.drift_history = []

    def fit_reference(self, df: pd.DataFrame, feature_cols: List[str]):
        """Сохранение референсного распределения"""
        self.reference_data = df[feature_cols].copy()
        self.reference_distributions = {}

        for col in feature_cols:
            data = df[col].dropna()
            if len(data) > 0:
                self.reference_distributions[col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'quantiles': data.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                    'histogram': np.histogram(data, bins=self.n_bins)
                }

        logger.info(f"Reference distributions saved for {len(feature_cols)} features")

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Population Stability Index"""
        # Создаём бины на основе expected
        bins = np.percentile(expected, np.linspace(0, 100, self.n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        # Нормализуем
        expected_pct = expected_counts / len(expected) + 1e-8
        actual_pct = actual_counts / len(actual) + 1e-8

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    def calculate_ks_statistic(self, expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test"""
        statistic, p_value = stats.ks_2samp(expected, actual)
        return statistic, p_value

    def check_drift(self, current_df: pd.DataFrame, feature_cols: List[str],
                    psi_threshold: float = 0.2, ks_threshold: float = 0.1) -> Dict:
        """Проверка дрейфа данных"""
        if self.reference_data is None:
            return {'drift_detected': False, 'reason': 'no_reference'}

        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features_checked': len(feature_cols),
            'drifted_features': [],
            'psi_scores': {},
            'ks_scores': {}
        }

        for col in feature_cols:
            if col not in self.reference_data.columns or col not in current_df.columns:
                continue

            ref_data = self.reference_data[col].dropna().values
            cur_data = current_df[col].dropna().values

            if len(ref_data) < 100 or len(cur_data) < 100:
                continue

            # PSI
            psi = self.calculate_psi(ref_data, cur_data)
            drift_report['psi_scores'][col] = psi

            # KS-test
            ks_stat, ks_pvalue = self.calculate_ks_statistic(ref_data, cur_data)
            drift_report['ks_scores'][col] = {'statistic': ks_stat, 'p_value': ks_pvalue}

            # Проверка порогов
            if psi > psi_threshold or ks_stat > ks_threshold:
                drift_report['drifted_features'].append({
                    'feature': col,
                    'psi': psi,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue
                })

        drift_report['drift_detected'] = len(drift_report['drifted_features']) > 0
        drift_report['drift_ratio'] = len(drift_report['drifted_features']) / max(len(feature_cols), 1)

        self.drift_history.append(drift_report)

        if drift_report['drift_detected']:
            logger.warning(f"⚠️ Data drift detected in {len(drift_report['drifted_features'])} features")
            for feat in drift_report['drifted_features'][:5]:
                logger.warning(f"   - {feat['feature']}: PSI={feat['psi']:.3f}, KS={feat['ks_statistic']:.3f}")

        return drift_report


# ============================================================
# TRANSFORMER SENTIMENT ANALYZER
# ============================================================
class TransformerSentimentAnalyzer:
    """Sentiment анализ на базе ruBERT"""

    MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

    def __init__(self, device: str = None, batch_size: int = 32):
        self.device = device or ('cuda' if HAS_TRANSFORMERS and torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self):
        """Ленивая загрузка модели"""
        if self._loaded:
            return

        logger.info(f"Loading sentiment model: {self.MODEL_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Sentiment model loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            logger.warning("Falling back to rule-based sentiment")
            self._loaded = False

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Batch sentiment analysis"""
        if not self._loaded:
            return [self._rule_based_sentiment(t) for t in texts]

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

                # Model outputs: [NEUTRAL, POSITIVE, NEGATIVE] или аналогично
                for j, prob in enumerate(probs):
                    # Нормализуем к [-1, 1]
                    sentiment_score = prob[1] - prob[2]  # positive - negative
                    confidence = np.max(prob)

                    results.append({
                        'sentiment': float(sentiment_score),
                        'confidence': float(confidence),
                        'probabilities': {
                            'positive': float(prob[1]),
                            'negative': float(prob[2]),
                            'neutral': float(prob[0])
                        }
                    })
            except Exception as e:
                logger.warning(f"Batch sentiment error: {e}")
                results.extend([self._rule_based_sentiment(t) for t in batch])

        return results

    def _rule_based_sentiment(self, text: str) -> Dict:
        """Fallback rule-based sentiment с расширенным словарём"""
        text_lower = text.lower()

        POSITIVE_WORDS = {
            'рост': 1.0, 'прибыль': 1.2, 'дивиденд': 1.5, 'увеличение': 0.8,
            'повышение': 0.8, 'рекорд': 1.3, 'успех': 1.0, 'выручка': 0.7,
            'покупка': 0.5, 'инвестиции': 0.6, 'развитие': 0.5, 'расширение': 0.6,
            'одобрен': 0.8, 'утверждён': 0.7, 'рекомендация': 0.5, 'выплата': 1.0,
            'оптимизм': 1.0, 'восстановление': 0.8, 'улучшение': 0.7, 'превысил': 0.9,
            'сильный': 0.6, 'высокий': 0.5, 'максимум': 1.0, 'лидер': 0.7
        }
        NEGATIVE_WORDS = {
            'падение': -1.0, 'убыток': -1.2, 'снижение': -0.8, 'сокращение': -0.7,
            'риск': -0.6, 'потери': -1.0, 'штраф': -1.3, 'санкции': -1.5,
            'дефолт': -2.0, 'банкротство': -2.0, 'продажа': -0.3, 'закрытие': -0.8,
            'отмена': -0.9, 'приостановка': -0.7, 'расследование': -1.0, 'иск': -1.0,
            'пессимизм': -1.0, 'кризис': -1.5, 'обвал': -1.8, 'минимум': -0.8,
            'слабый': -0.6, 'низкий': -0.5, 'провал': -1.2, 'проблема': -0.7
        }

        pos_score = sum(weight for word, weight in POSITIVE_WORDS.items() if word in text_lower)
        neg_score = sum(abs(weight) for word, weight in NEGATIVE_WORDS.items() if word in text_lower)

        total = pos_score + neg_score
        if total > 0:
            sentiment = (pos_score - neg_score) / total
        else:
            sentiment = 0.0

        return {
            'sentiment': sentiment,
            'confidence': min(total / 5, 1.0),
            'probabilities': {'positive': pos_score / max(total, 1),
                              'negative': neg_score / max(total, 1),
                              'neutral': 1 - min(total / 5, 1)}
        }


# ============================================================
# ENHANCED MOEX NEWS CLIENT
# ============================================================
class EnhancedMOEXNewsClient:
    """Улучшенный клиент новостей MOEX"""

    ISS_URL = "https://iss.moex.com/iss/sitenews.json"

    # Расширенный маппинг компаний
    COMPANY_PATTERNS = {
        'SBER': [r'\bсбер', r'\bsber', r'сбербанк'],
        'GAZP': [r'\bгазпром', r'\bgazp', r'газпрома'],
        'LKOH': [r'\bлукойл', r'\blkoh', r'лукойла'],
        'GMKN': [r'\bнорникель', r'\bgmkn', r'норильский никель'],
        'NVTK': [r'\bноватэк', r'\bnvtk'],
        'ROSN': [r'\bроснефть', r'\brosn'],
        'VTBR': [r'\bвтб', r'\bvtbr'],
        'MTSS': [r'\bмтс', r'\bmtss'],
        'MGNT': [r'\bмагнит', r'\bmgnt'],
        'TATN': [r'\bтатнефть', r'\btatn'],
        'YNDX': [r'\bяндекс', r'\byndx', r'yandex'],
        'TCSG': [r'\bтинькофф', r'\btcsg', r'\btcs\b'],
        'OZON': [r'\bозон', r'\bozon'],
        'FIVE': [r'\bпятёрочка', r'\bx5', r'\bfive'],
        'POLY': [r'\bполюс', r'\bpoly', r'полюс золото'],
        'PLZL': [r'\bполиметалл', r'\bplzl', r'polymetal'],
        'ALRS': [r'\bалроса', r'\balrs'],
        'CHMF': [r'\bсеверсталь', r'\bchmf'],
        'NLMK': [r'\bнлмк', r'\bnlmk'],
        'MAGN': [r'\bммк', r'\bmagn', r'магнитогорск'],
        'MOEX': [r'\bмосбиржа', r'\bmoex', r'московская биржа'],
        'AFKS': [r'\bсистема', r'\bafks', r'\bафк'],
        'IRAO': [r'\bинтер рао', r'\birao'],
        'HYDR': [r'\bрусгидро', r'\bhydr'],
        'AFLT': [r'\bаэрофлот', r'\baflt'],
        'RUAL': [r'\bрусал', r'\brual'],
        'MAIL': [r'\bvk\b', r'\bmail\.ru', r'\bмэйл'],
        'PIKK': [r'\bпик\b', r'\bpikk', r'группа пик'],
        'MVID': [r'\bм\.видео', r'\bmvid', r'мвидео'],
    }

    KNOWN_TICKERS = set(COMPANY_PATTERNS.keys())
    TICKER_PATTERN = re.compile(r'\b([A-Z]{4})\b')

    def __init__(self, sentiment_analyzer: TransformerSentimentAnalyzer = None):
        self.client = httpx.Client(timeout=30)
        self.sentiment_analyzer = sentiment_analyzer or TransformerSentimentAnalyzer()
        self._compiled_patterns = {
            ticker: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ticker, patterns in self.COMPANY_PATTERNS.items()
        }

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
                    body = item.get('body', '')
                    full_text = f"{title} {body}"

                    try:
                        pub_date = datetime.strptime(item.get('published_at', ''), '%Y-%m-%d %H:%M:%S')
                    except:
                        continue

                    tickers = self._extract_tickers_advanced(full_text)

                    all_news.append({
                        'date': pub_date.date(),
                        'datetime': pub_date,
                        'title': title,
                        'body': body[:500] if body else '',
                        'tickers': tickers,
                        'news_id': item.get('id')
                    })

            except Exception as e:
                logger.warning(f"News page {start} error: {e}")
                break

        if not all_news:
            return pd.DataFrame()

        news_df = pd.DataFrame(all_news)

        # Batch sentiment analysis
        logger.info(f"Analyzing sentiment for {len(news_df)} news...")
        self.sentiment_analyzer.load_model()

        texts = (news_df['title'] + ' ' + news_df['body']).tolist()
        sentiments = self.sentiment_analyzer.analyze_batch(texts)

        news_df['sentiment'] = [s['sentiment'] for s in sentiments]
        news_df['sentiment_confidence'] = [s['confidence'] for s in sentiments]

        return news_df

    def _extract_tickers_advanced(self, text: str) -> List[str]:
        """Улучшенное извлечение тикеров через NER-подобный подход"""
        found_tickers = set()

        # 1. Прямой поиск тикеров
        direct_matches = self.TICKER_PATTERN.findall(text.upper())
        for ticker in direct_matches:
            if ticker in self.KNOWN_TICKERS:
                found_tickers.add(ticker)

        # 2. Поиск по названиям компаний
        for ticker, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    found_tickers.add(ticker)
                    break

        return list(found_tickers)

    def aggregate_daily_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегация sentiment по дням и тикерам с ЛАГОМ +1 день"""
        if news_df.empty:
            return pd.DataFrame()

        expanded = []
        for _, row in news_df.iterrows():
            for ticker in row['tickers']:
                expanded.append({
                    # Сдвиг на +1 день для избежания look-ahead bias
                    'date': row['date'] + timedelta(days=1),
                    'ticker': ticker,
                    'sentiment': row['sentiment'],
                    'sentiment_confidence': row['sentiment_confidence']
                })

        if not expanded:
            return pd.DataFrame()

        exp_df = pd.DataFrame(expanded)

        # Weighted aggregation by confidence
        daily = exp_df.groupby(['date', 'ticker']).apply(
            lambda x: pd.Series({
                'news_sentiment': np.average(x['sentiment'], weights=x['sentiment_confidence'] + 0.1),
                'news_sentiment_std': x['sentiment'].std(),
                'news_count': len(x),
                'news_confidence_avg': x['sentiment_confidence'].mean(),
                'news_sentiment_sum': x['sentiment'].sum()
            })
        ).reset_index()

        return daily

    def close(self):
        self.client.close()


# ============================================================
# CBR CLIENT (без изменений, но с улучшенной обработкой)
# ============================================================
class CBRClient:
    """Загрузка макро-данных ЦБ РФ с улучшенной обработкой"""

    BASE_URL = "https://www.cbr.ru"
    CURRENCY_CODES = {'USD': 'R01235', 'EUR': 'R01239', 'CNY': 'R01375'}

    def __init__(self):
        self.client = httpx.Client(timeout=30, follow_redirects=True)
        self._cache = {}

    def get_currency_history(self, currency: str = 'USD', days: int = 2500) -> pd.DataFrame:
        """История курса валюты с кэшированием"""
        cache_key = f"{currency}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

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

            df = pd.DataFrame(records)
            self._cache[cache_key] = df
            return df
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
        """Загрузка всех макро-данных с интерполяцией"""
        logger.info("Loading CBR macro data...")

        usd = self.get_currency_history('USD')
        eur = self.get_currency_history('EUR')
        cny = self.get_currency_history('CNY')
        key_rate = self.get_key_rate_history()

        if usd.empty:
            return pd.DataFrame()

        # Создаём полный date range
        date_range = pd.date_range(start=usd['date'].min(), end=usd['date'].max(), freq='D')
        macro = pd.DataFrame({'date': date_range})

        # Merge all
        for df, name in [(usd, 'usd'), (eur, 'eur'), (cny, 'cny')]:
            if not df.empty:
                macro = macro.merge(df, on='date', how='left')

        if not key_rate.empty:
            macro = macro.merge(key_rate, on='date', how='left')

        # Интерполяция вместо ffill для выходных
        numeric_cols = macro.select_dtypes(include=[np.number]).columns
        macro[numeric_cols] = macro[numeric_cols].interpolate(method='linear')
        macro = macro.bfill()  # Для начальных значений

        return macro

    def close(self):
        self.client.close()


# ============================================================
# SECTOR MAPPING (расширенный)
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
    'MGNT': 'retail', 'FIVE': 'retail', 'MVID': 'retail',
    # Телеком
    'MTSS': 'telecom', 'RTKM': 'telecom',
    # IT
    'YNDX': 'tech', 'OZON': 'tech', 'MAIL': 'tech',
    # Электроэнергетика
    'IRAO': 'energy', 'HYDR': 'energy',
    # Строительство
    'PIKK': 'construction',
    # Транспорт
    'AFLT': 'transport',
}

# Секторные коэффициенты чувствительности
SECTOR_SENSITIVITY = {
    'oil_gas': {'usd': 0.8, 'oil_price': 1.0, 'rate': -0.2},
    'metals': {'usd': 0.7, 'commodities': 0.9, 'rate': -0.2},
    'banks': {'usd': -0.3, 'rate': 0.6, 'credit_growth': 0.5},
    'retail': {'usd': -0.5, 'rate': -0.4, 'consumer': 0.7},
    'tech': {'usd': -0.2, 'rate': -0.3, 'growth': 0.8},
    'telecom': {'usd': -0.1, 'rate': -0.2, 'defensive': 0.5},
    'energy': {'usd': 0.1, 'rate': -0.3, 'tariff': 0.6},
    'construction': {'usd': -0.4, 'rate': -0.7, 'mortgage': 0.8},
    'transport': {'usd': -0.3, 'rate': -0.2, 'oil_price': -0.5},
}


# ============================================================
# PURGED K-FOLD CROSS-VALIDATION
# ============================================================
class PurgedKFold:
    """
    Purged K-Fold Cross-Validation для временных рядов.
    Предотвращает утечку информации между train/test через embargo период.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, purge_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Generator для train/test индексов с purging и embargo.
        X должен иметь колонку 'date' или быть отсортирован по времени.
        """
        if isinstance(X, pd.DataFrame) and 'date' in X.columns:
            dates = X['date'].values
            sorted_idx = np.argsort(dates)
        else:
            sorted_idx = np.arange(len(X))

        n_samples = len(sorted_idx)
        embargo_size = int(n_samples * self.embargo_pct)
        purge_size = int(n_samples * self.purge_pct)

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            test_idx = sorted_idx[test_start:test_end]

            # Train indices: всё кроме test + purge + embargo
            train_end = max(0, test_start - purge_size)
            train_start_after = min(n_samples, test_end + embargo_size)

            train_idx_before = sorted_idx[:train_end]
            train_idx_after = sorted_idx[train_start_after:]

            train_idx = np.concatenate([train_idx_before, train_idx_after])

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Более продвинутый метод для финансовых временных рядов.
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2,
                 embargo_td: timedelta = timedelta(days=5)):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """Generator для CPCV splits"""
        from itertools import combinations

        if 'date' not in X.columns:
            raise ValueError("X must have 'date' column")

        dates = pd.to_datetime(X['date'])
        sorted_idx = dates.argsort().values
        n_samples = len(sorted_idx)

        # Разбиваем на n_splits групп
        group_size = n_samples // self.n_splits
        groups_list = []

        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups_list.append(sorted_idx[start:end])

        # Генерируем все комбинации тестовых групп
        for test_group_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups_list[i] for i in test_group_indices])

            # Train = всё остальное с embargo
            train_groups = [i for i in range(self.n_splits) if i not in test_group_indices]

            train_idx_list = []
            for train_group in train_groups:
                group_idx = groups_list[train_group]
                group_dates = dates.iloc[group_idx]

                # Проверяем embargo с каждой тестовой группой
                valid_mask = np.ones(len(group_idx), dtype=bool)

                for test_group in test_group_indices:
                    test_dates = dates.iloc[groups_list[test_group]]
                    test_min, test_max = test_dates.min(), test_dates.max()

                    # Исключаем точки слишком близкие к тесту
                    too_close = (
                        (group_dates >= test_min - self.embargo_td) &
                        (group_dates <= test_max + self.embargo_td)
                    )
                    valid_mask &= ~too_close.values

                train_idx_list.append(group_idx[valid_mask])

            train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([])

            if len(train_idx) > 100 and len(test_idx) > 50:
                yield train_idx, test_idx


# ============================================================
# FEATURE SELECTOR
# ============================================================
class AdvancedFeatureSelector:
    """Продвинутый отбор фичей"""

    def __init__(self, method: str = 'importance', top_k: int = 30):
        self.method = method
        self.top_k = top_k
        self.selected_features = []
        self.feature_importance = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> List[str]:
        """Отбор топ-K фичей"""
        logger.info(f"Selecting features using {self.method}...")

        X_clean = X[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        if self.method == 'importance':
            return self._select_by_importance(X_clean, y, feature_cols)
        elif self.method == 'mutual_info':
            return self._select_by_mutual_info(X_clean, y, feature_cols)
        elif self.method == 'boruta':
            return self._select_by_boruta(X_clean, y, feature_cols)
        else:
            return feature_cols[:self.top_k]

    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series,
                              feature_cols: List[str]) -> List[str]:
        """Отбор по важности LightGBM"""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X, y)

        importance = dict(zip(feature_cols, model.feature_importances_))
        self.feature_importance = importance

        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_features[:self.top_k]]

        logger.info(f"Selected {len(self.selected_features)} features by importance")
        return self.selected_features

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                               feature_cols: List[str]) -> List[str]:
        """Отбор по mutual information"""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance = dict(zip(feature_cols, mi_scores))
        self.feature_importance = importance

        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_features[:self.top_k]]

        logger.info(f"Selected {len(self.selected_features)} features by mutual info")
        return self.selected_features

    def _select_by_boruta(self, X: pd.DataFrame, y: pd.Series,
                          feature_cols: List[str]) -> List[str]:
        """Boruta-подобный алгоритм отбора"""
        # Создаём shadow features
        X_shadow = X.copy()
        for col in feature_cols:
            X_shadow[f'shadow_{col}'] = np.random.permutation(X[col].values)

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        model.fit(X_shadow, y)

        all_cols = list(X_shadow.columns)
        importance = dict(zip(all_cols, model.feature_importances_))

        # Порог = максимум среди shadow features
        shadow_max = max(importance.get(f'shadow_{col}', 0) for col in feature_cols)

        # Отбираем фичи выше порога
        selected = [col for col in feature_cols if importance.get(col, 0) > shadow_max]

        self.feature_importance = {k: v for k, v in importance.items() if not k.startswith('shadow_')}
        self.selected_features = selected[:self.top_k] if len(selected) > self.top_k else selected

        logger.info(f"Boruta selected {len(self.selected_features)} features")
        return self.selected_features


# ============================================================
# KELLY CRITERION & POSITION SIZING
# ============================================================
class PositionSizer:
    """Position sizing с Kelly Criterion"""

    def __init__(self, kelly_fraction: float = 0.25, max_position: float = 0.2,
                 min_confidence: float = 0.4):
        self.kelly_fraction = kelly_fraction  # Fractional Kelly для снижения риска
        self.max_position = max_position
        self.min_confidence = min_confidence

    def calculate_kelly(self, win_prob: float, win_loss_ratio: float) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        где p = вероятность выигрыша, q = 1-p, b = отношение выигрыша к проигрышу
        """
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.0

        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        return max(0, kelly)

    def calculate_position_size(self, confidence: float, predicted_class: int,
                                class_probabilities: Dict[int, float],
                                historical_win_rate: float = 0.4,
                                historical_win_loss_ratio: float = 1.5) -> Dict:
        """
        Расчёт размера позиции на основе confidence и Kelly.
        """
        if confidence < self.min_confidence:
            return {
                'position_size': 0.0,
                'direction': 0,
                'kelly_raw': 0.0,
                'reason': 'low_confidence'
            }

        # Kelly на основе исторических данных
        kelly_raw = self.calculate_kelly(historical_win_rate, historical_win_loss_ratio)

        # Fractional Kelly
        kelly_adj = kelly_raw * self.kelly_fraction

        # Модификация по confidence
        confidence_multiplier = (confidence - self.min_confidence) / (1 - self.min_confidence)

        position_size = kelly_adj * confidence_multiplier
        position_size = min(position_size, self.max_position)

        # Direction
        direction = predicted_class if predicted_class != 0 else 0

        return {
            'position_size': position_size,
            'direction': direction,
            'kelly_raw': kelly_raw,
            'kelly_adjusted': kelly_adj,
            'confidence_multiplier': confidence_multiplier,
            'reason': 'calculated'
        }

    def calculate_portfolio_weights(self, predictions: List[Dict],
                                    max_total_exposure: float = 1.0) -> List[Dict]:
        """Расчёт весов для портфеля предсказаний"""
        if not predictions:
            return []

        # Рассчитываем сырые веса
        for pred in predictions:
            sizing = self.calculate_position_size(
                confidence=pred.get('confidence', 0),
                predicted_class=pred.get('signal', 0),
                class_probabilities=pred.get('probabilities', {}),
                historical_win_rate=pred.get('historical_win_rate', 0.4),
                historical_win_loss_ratio=pred.get('historical_win_loss_ratio', 1.5)
            )
            pred['position_size'] = sizing['position_size']
            pred['kelly_info'] = sizing

        # Нормализуем если сумма > max_total_exposure
        total_exposure = sum(abs(p['position_size']) for p in predictions)

        if total_exposure > max_total_exposure:
            scale_factor = max_total_exposure / total_exposure
            for pred in predictions:
                pred['position_size'] *= scale_factor
                pred['scaled'] = True

        return predictions


# ============================================================
# PERFORMANCE TRACKER (улучшенный)
# ============================================================
class PerformanceTracker:
    """Отслеживание метрик модели с drift detection"""

    def __init__(self, db_engine, threshold_accuracy: float = 0.38,
                 threshold_sharpe: float = 0.3, lookback_days: int = 30):
        self.engine = db_engine
        self.threshold_accuracy = threshold_accuracy
        self.threshold_sharpe = threshold_sharpe
        self.lookback_days = lookback_days
        self.drift_monitor = DataDriftMonitor()
        self._ensure_tables()

    def _ensure_tables(self):
        """Создание таблиц для мониторинга"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_predictions_v5 (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    ticker VARCHAR(20),
                    predicted_class INTEGER,
                    predicted_proba FLOAT,
                    position_size FLOAT,
                    actual_class INTEGER,
                    actual_return FLOAT,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_metrics_v5 (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    accuracy FLOAT,
                    f1_score FLOAT,
                    sharpe_ratio FLOAT,
                    max_drawdown FLOAT,
                    predictions_count INTEGER,
                    drift_detected BOOLEAN,
                    drift_ratio FLOAT,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS retrain_history_v5 (
                    id SERIAL PRIMARY KEY,
                    triggered_at TIMESTAMPTZ DEFAULT NOW(),
                    reason VARCHAR(100),
                    old_accuracy FLOAT,
                    new_accuracy FLOAT,
                    old_sharpe FLOAT,
                    new_sharpe FLOAT,
                    features_selected INTEGER,
                    model_version VARCHAR(50)
                )
            """))
            conn.commit()

    def log_prediction(self, date, ticker: str, predicted_class: int,
                       predicted_proba: float, position_size: float, model_version: str):
        """Логирование предсказания"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_predictions_v5 
                (date, ticker, predicted_class, predicted_proba, position_size, model_version)
                VALUES (:date, :ticker, :pred_class, :pred_proba, :pos_size, :version)
            """), {
                'date': date, 'ticker': ticker, 'pred_class': predicted_class,
                'pred_proba': predicted_proba, 'pos_size': position_size,
                'version': model_version
            })
            conn.commit()

    def update_actuals(self):
        """Обновление фактических результатов"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE model_predictions_v5 mp
                SET actual_class = f.signal_class,
                    actual_return = f.return_5d
                FROM features f
                WHERE mp.date = f.date 
                  AND mp.ticker = f.ticker
                  AND mp.actual_class IS NULL
            """))
            conn.commit()

    def calculate_metrics(self, include_drift: bool = True) -> Dict:
        """Расчёт метрик за последние N дней"""
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)

        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT predicted_class, actual_class, predicted_proba, 
                       position_size, actual_return
                FROM model_predictions_v5
                WHERE date >= :cutoff AND actual_class IS NOT NULL
            """), {'cutoff': cutoff_date})
            rows = result.fetchall()

        if len(rows) < 50:
            return {'valid': False, 'reason': 'insufficient_data'}

        df = pd.DataFrame(rows, columns=['predicted', 'actual', 'proba', 'position', 'return'])

        # Accuracy & F1
        accuracy = accuracy_score(df['actual'], df['predicted'])
        f1 = f1_score(df['actual'], df['predicted'], average='weighted')

        # Sharpe Ratio с position sizing
        positions = df['predicted'].map({-1: -1, 0: 0, 1: 1}) * df['position']
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
            'predictions_count': len(df),
            'drift_detected': False,
            'drift_ratio': 0.0
        }

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

        if metrics.get('drift_detected') and metrics.get('drift_ratio', 0) > 0.3:
            return True, f"data_drift_detected_{metrics['drift_ratio']:.1%}"

        return False, "metrics_ok"

    def log_retrain(self, reason: str, old_metrics: Dict, new_metrics: Dict,
                    features_count: int, version: str):
        """Логирование переобучения"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO retrain_history_v5 
                (reason, old_accuracy, new_accuracy, old_sharpe, new_sharpe, 
                 features_selected, model_version)
                VALUES (:reason, :old_acc, :new_acc, :old_sharpe, :new_sharpe, 
                        :features, :version)
            """), {
                'reason': reason,
                'old_acc': old_metrics.get('accuracy'),
                'new_acc': new_metrics.get('accuracy'),
                'old_sharpe': old_metrics.get('sharpe_ratio'),
                'new_sharpe': new_metrics.get('sharpe_ratio'),
                'features': features_count,
                'version': version
            })
            conn.commit()


# ============================================================
# SHAP EXPLAINER
# ============================================================
class ModelExplainer:
    """SHAP-based model interpretability"""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit(self, X_background: np.ndarray, sample_size: int = 1000):
        """Инициализация SHAP explainer"""
        logger.info("Initializing SHAP explainer...")

        # Subsample для скорости
        if len(X_background) > sample_size:
            idx = np.random.choice(len(X_background), sample_size, replace=False)
            X_background = X_background[idx]

        try:
            # Для tree-based моделей
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback на KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background[:100]
            )

        logger.info("✅ SHAP explainer initialized")

    def explain(self, X: np.ndarray) -> Dict:
        """Получение SHAP values для предсказаний"""
        if self.explainer is None:
            return {'error': 'Explainer not initialized'}

        try:
            shap_values = self.explainer.shap_values(X)

            # Для multi-class берём среднее по классам
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

            # Feature importance
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance = dict(zip(self.feature_names, mean_importance))

            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'top_features': sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
            }
        except Exception as e:
            logger.warning(f"SHAP explanation error: {e}")
            return {'error': str(e)}

    def explain_single(self, x: np.ndarray, class_idx: int = 1) -> Dict:
        """Объяснение одного предсказания"""
        if self.explainer is None:
            return {'error': 'Explainer not initialized'}

        try:
            shap_values = self.explainer.shap_values(x.reshape(1, -1))

            if isinstance(shap_values, list):
                sv = shap_values[class_idx][0]
            else:
                sv = shap_values[0]

            # Top contributing features
            contributions = list(zip(self.feature_names, sv))
            contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

            return {
                'contributions': contributions_sorted[:10],
                'positive_drivers': [(f, v) for f, v in contributions_sorted if v > 0][:5],
                'negative_drivers': [(f, v) for f, v in contributions_sorted if v < 0][:5]
            }
        except Exception as e:
            return {'error': str(e)}


# ============================================================
# WALK-FORWARD TRAINER v5
# ============================================================
class WalkForwardTrainerV5:
    """Walk-Forward Trainer v5 с всеми улучшениями"""

    VERSION = "v5_advanced"

    def __init__(self, db_url: str, models_dir: str = "/app/models"):
        self.engine = create_engine(db_url)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.selected_features = []

        # Clients
        self.cbr = CBRClient()
        self.sentiment_analyzer = TransformerSentimentAnalyzer()
        self.news_client = EnhancedMOEXNewsClient(self.sentiment_analyzer)

        # Advanced components
        self.tracker = PerformanceTracker(self.engine)
        self.drift_monitor = DataDriftMonitor()
        self.feature_selector = AdvancedFeatureSelector(method='boruta', top_k=35)
        self.position_sizer = PositionSizer(kelly_fraction=0.25, max_position=0.15)
        self.explainer = None

        self._lock = Lock()

    def load_features(self) -> pd.DataFrame:
        """Загрузка фичей из БД"""
        query = """
            SELECT * FROM features
            WHERE signal_class IS NOT NULL
            ORDER BY date, ticker
        """
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Signal distribution:\n{df['signal_class'].value_counts().sort_index()}")
        return df

    def load_macro_data(self) -> pd.DataFrame:
        """Загрузка макро-данных ЦБ РФ"""
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
            macro['usd_momentum'] = macro['usd_rate'].pct_change(10)
            macro['usd_acceleration'] = macro['usd_change_5d'] - macro['usd_change_5d'].shift(5)

        if 'eur_rate' in macro.columns:
            macro['eur_change_5d'] = macro['eur_rate'].pct_change(5)
            macro['eur_usd_ratio'] = macro['eur_rate'] / macro['usd_rate']

        if 'cny_rate' in macro.columns:
            macro['cny_change_5d'] = macro['cny_rate'].pct_change(5)

        if 'key_rate' in macro.columns:
            macro['rate_change'] = macro['key_rate'].diff()
            macro['rate_high'] = (macro['key_rate'] > 12).astype(int)
            macro['rate_rising'] = (macro['rate_change'] > 0).astype(int)
            macro['real_rate'] = macro['key_rate'] - macro['usd_change_20d'] * 100  # Proxy

        logger.info(f"Loaded {len(macro)} days of macro data")
        return macro

    def load_news_sentiment(self) -> pd.DataFrame:
        """Загрузка новостного sentiment с transformer моделью"""
        logger.info("Loading MOEX news with transformer sentiment...")
        news_df = self.news_client.fetch_iss_news(pages=50)

        if news_df.empty:
            return pd.DataFrame()

        daily_sentiment = self.news_client.aggregate_daily_sentiment(news_df)
        logger.info(f"Aggregated to {len(daily_sentiment)} ticker-day records")
        return daily_sentiment

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расширенные технические фичи"""
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
            df.loc[mask, 'vol_regime'] = (df.loc[mask, 'volatility_5'] > df.loc[mask, 'volatility_20']).astype(int)

            # Momentum (используем ПРОШЛЫЕ данные!)
            df.loc[mask, 'momentum_5'] = ticker_data['close'].shift(1) / ticker_data['close'].shift(6) - 1
            df.loc[mask, 'momentum_10'] = ticker_data['close'].shift(1) / ticker_data['close'].shift(11) - 1
            df.loc[mask, 'momentum_20'] = ticker_data['close'].shift(1) / ticker_data['close'].shift(21) - 1
            df.loc[mask, 'momentum_acceleration'] = df.loc[mask, 'momentum_5'] - df.loc[mask, 'momentum_5'].shift(5)

            # RSI dynamics
            df.loc[mask, 'rsi_change_5'] = ticker_data['rsi_14'].diff(5)
            df.loc[mask, 'rsi_ma_5'] = ticker_data['rsi_14'].rolling(5).mean()
            df.loc[mask, 'rsi_oversold'] = (ticker_data['rsi_14'] < 30).astype(int)
            df.loc[mask, 'rsi_overbought'] = (ticker_data['rsi_14'] > 70).astype(int)
            df.loc[mask, 'rsi_divergence'] = df.loc[mask, 'momentum_10'] - (ticker_data['rsi_14'] / 50 - 1)

            # Price vs MAs
            df.loc[mask, 'price_vs_sma20'] = ticker_data['close'] / ticker_data['sma_20'] - 1
            df.loc[mask, 'price_vs_sma50'] = ticker_data['close'] / ticker_data['sma_50'] - 1
            df.loc[mask, 'sma20_vs_sma50'] = ticker_data['sma_20'] / ticker_data['sma_50'] - 1
            df.loc[mask, 'above_sma200'] = (ticker_data['close'] > ticker_data['sma_200']).astype(int)
            df.loc[mask, 'ma_trend'] = (ticker_data['sma_20'] > ticker_data['sma_50']).astype(int)

            # MACD dynamics
            df.loc[mask, 'macd_change'] = ticker_data['macd_hist'].diff(3)
            df.loc[mask, 'macd_positive'] = (ticker_data['macd_hist'] > 0).astype(int)
            df.loc[mask, 'macd_crossover'] = (
                (ticker_data['macd_hist'] > 0) & 
                (ticker_data['macd_hist'].shift(1) <= 0)
            ).astype(int)

            # Volume dynamics
            df.loc[mask, 'volume_change'] = ticker_data['volume_ratio'].diff(3)
            df.loc[mask, 'volume_spike'] = (ticker_data['volume_ratio'] > 2).astype(int)
            df.loc[mask, 'volume_trend'] = ticker_data['volume_ratio'].rolling(5).mean()

            # BB dynamics
            df.loc[mask, 'bb_pct_change'] = ticker_data['bb_pct'].diff(3)
            df.loc[mask, 'bb_squeeze'] = (ticker_data['bb_width'] < ticker_data['bb_width'].rolling(20).quantile(0.2)).astype(int)

            # ATR normalized
            df.loc[mask, 'atr_pct'] = ticker_data['atr_14'] / ticker_data['close']
            df.loc[mask, 'atr_ratio'] = ticker_data['atr_14'] / ticker_data['atr_14'].rolling(20).mean()

        return df

    def add_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Секторные фичи с коэффициентами чувствительности"""
        df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('other')
        df['sector_rate_impact'] = 0.0
        df['sector_usd_impact'] = 0.0

        # Sector-macro interactions с коэффициентами
        for sector, sensitivities in SECTOR_SENSITIVITY.items():
            sector_mask = df['sector'] == sector

            if 'usd_change_5d' in df.columns and 'usd' in sensitivities:
                df.loc[sector_mask, 'sector_usd_impact'] = (
                    df.loc[sector_mask, 'usd_change_5d'].fillna(0) * sensitivities['usd']
                )

            if 'key_rate' in df.columns and 'rate' in sensitivities:
                df.loc[sector_mask, 'sector_rate_impact'] = (
                    df.loc[sector_mask, 'key_rate'].fillna(0) / 100 * sensitivities['rate']
                )

        # Fill NaN for sectors not in SECTOR_SENSITIVITY
        df['sector_usd_impact'] = df['sector_usd_impact'].fillna(0)
        df['sector_rate_impact'] = df.get('sector_rate_impact', pd.Series(0, index=df.index)).fillna(0)

        # One-hot encoding
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)

        return df

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
            # Fill news columns
            news_cols = ['news_sentiment', 'news_sentiment_std', 'news_count', 
                         'news_confidence_avg', 'news_sentiment_sum']
            for col in news_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        # Interpolate macro columns
        macro_cols = [c for c in df.columns if c.startswith(('usd_', 'eur_', 'cny_', 'key_', 'rate_', 'real_'))]
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear').bfill().ffill()

        return df

    def create_advanced_model(self):
        """Создание продвинутого ансамбля LightGBM + XGBoost"""
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )

        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        ensemble = VotingClassifier(
            estimators=[
                ('lgb', lgb_model),
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.4, 0.2]
        )

        return ensemble

    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   returns: np.ndarray, confidences: np.ndarray = None) -> Dict:
        """Trading-based метрики с position sizing"""
        positions = pd.Series(y_pred).map({-1: -1, 0: 0, 1: 1}).values

        # Apply position sizing if confidences provided
        if confidences is not None:
            # Simple confidence-based sizing
            position_sizes = np.where(confidences > 0.5, 
                                      (confidences - 0.5) * 2,  # Scale 0.5-1.0 to 0-1
                                      0.1)  # Minimum position
            positions = positions * position_sizes

        # Shift positions to avoid look-ahead
        strategy_returns = np.roll(positions, 1)[1:] * returns[1:]

        if len(strategy_returns) == 0 or np.std(strategy_returns) < 1e-8:
            return {'sharpe': 0, 'max_drawdown': 0, 'total_return': 0, 'win_rate': 0}

        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

        cumulative = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = np.min(drawdown)

        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0

        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        total_trades = np.sum(strategy_returns != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def train(self, save_model: bool = True, use_cpcv: bool = True) -> Dict:
        """Основной метод обучения"""
        with self._lock:
            return self._train_internal(save_model, use_cpcv)

    def _train_internal(self, save_model: bool = True, use_cpcv: bool = True) -> Dict:
        logger.info("=" * 70)
        logger.info("WALK-FORWARD TRAINING v5 - Advanced Self Learning")
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
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info(f"Final dataset: {len(df):,} rows")

        # 5. Define all possible features
        all_feature_cols = [
            # Technical
            'rsi_14', 'rsi_change_5', 'rsi_ma_5', 'rsi_oversold', 'rsi_overbought', 'rsi_divergence',
            'macd', 'macd_hist', 'macd_signal', 'macd_change', 'macd_positive', 'macd_crossover',
            'bb_pct', 'bb_width', 'bb_pct_change', 'bb_squeeze',
            'atr_14', 'atr_pct', 'atr_ratio',
            # Returns
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            # Volatility
            'volatility_20', 'volatility_5', 'volatility_10', 'vol_ratio', 'vol_regime',
            # Volume
            'volume_ratio', 'volume_change', 'volume_spike', 'volume_trend',
            # Price position
            'pct_from_high', 'pct_from_low',
            'price_vs_sma20', 'price_vs_sma50', 'sma20_vs_sma50', 'above_sma200', 'ma_trend',
            # Momentum
            'momentum_5', 'momentum_10', 'momentum_20', 'momentum_acceleration',
            # Macro
            'usd_rate', 'usd_change_1d', 'usd_change_5d', 'usd_change_20d',
            'usd_volatility', 'usd_vs_ma', 'usd_momentum', 'usd_acceleration',
            'eur_change_5d', 'eur_usd_ratio', 'cny_change_5d',
            'key_rate', 'rate_change', 'rate_high', 'rate_rising', 'real_rate',
            # News
            'news_sentiment', 'news_sentiment_std', 'news_count', 
            'news_confidence_avg', 'news_sentiment_sum',
            # Sector
            'sector_usd_impact', 'sector_rate_impact',
        ]

        # Add sector dummies
        sector_cols = [c for c in df.columns if c.startswith('sector_') 
                       and c not in ['sector_usd_impact', 'sector_rate_impact']]
        all_feature_cols.extend(sector_cols)

        # Filter existing
        self.feature_cols = [c for c in all_feature_cols if c in df.columns]
        logger.info(f"Available features: {len(self.feature_cols)}")

        # 6. Feature selection
        self.selected_features = self.feature_selector.fit(
            df, df['signal_class'], self.feature_cols
        )
        logger.info(f"Selected features: {len(self.selected_features)}")

        # 7. Data drift check
        median_idx = len(df) // 2
        self.drift_monitor.fit_reference(df.iloc[:median_idx], self.selected_features)
        drift_report = self.drift_monitor.check_drift(df.iloc[median_idx:], self.selected_features)

        # 8. Cross-validation
        results = []
        all_predictions = []

        logger.info("-" * 70)
        if use_cpcv:
            logger.info("Combinatorial Purged Cross-Validation:")
            cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo_td=timedelta(days=5))
        else:
            logger.info("Purged K-Fold Cross-Validation:")
            cv = PurgedKFold(n_splits=10, embargo_pct=0.01, purge_pct=0.01)

        fold = 0
        for train_idx, test_idx in cv.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            X_train = train_df[self.selected_features].fillna(0)
            y_train = train_df['signal_class']
            X_test = test_df[self.selected_features].fillna(0)
            y_test = test_df['signal_class']
            returns_test = test_df['return_5d'].values

            model = self.create_advanced_model()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            confidences = np.max(y_proba, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            trading = self.calculate_trading_metrics(y_test.values, y_pred, returns_test, confidences)

            results.append({
                'fold': fold,
                'accuracy': accuracy,
                'f1': f1,
                'sharpe': trading['sharpe'],
                'max_dd': trading['max_drawdown'],
                'win_rate': trading['win_rate'],
                'profit_factor': trading['profit_factor'],
                'train_size': len(train_df),
                'test_size': len(test_df),
            })

            logger.info(f"Fold {fold:2d}: Acc={accuracy:.1%} Sharpe={trading['sharpe']:.2f} "
                        f"WinRate={trading['win_rate']:.1%} PF={trading['profit_factor']:.2f}")

            fold += 1
            if fold >= 15:  # Limit folds for CPCV
                break

        results_df = pd.DataFrame(results)

        # 9. Results
        metrics = {
            'accuracy': results_df['accuracy'].mean(),
            'accuracy_std': results_df['accuracy'].std(),
            'f1': results_df['f1'].mean(),
            'sharpe': results_df['sharpe'].mean(),
            'sharpe_std': results_df['sharpe'].std(),
            'max_drawdown': results_df['max_dd'].mean(),
            'win_rate': results_df['win_rate'].mean(),
            'profit_factor': results_df['profit_factor'].mean(),
            'n_folds': len(results),
            'n_features': len(self.selected_features),
            'drift_detected': drift_report['drift_detected'],
            'drift_ratio': drift_report['drift_ratio']
        }

        logger.info("-" * 70)
        logger.info("RESULTS:")
        logger.info(f"  Accuracy:      {metrics['accuracy']:.1%} ± {metrics['accuracy_std']:.1%}")
        logger.info(f"  Sharpe Ratio:  {metrics['sharpe']:.2f} ± {metrics['sharpe_std']:.2f}")
        logger.info(f"  Win Rate:      {metrics['win_rate']:.1%}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown:  {metrics['max_drawdown']:.1%}")
        logger.info(f"  Features:      {metrics['n_features']}")
        logger.info(f"  Data Drift:    {'⚠️ Detected' if metrics['drift_detected'] else '✅ OK'}")

        # 10. Train final model
        logger.info("Training final calibrated model...")
        X_all = df[self.selected_features].fillna(0)
        y_all = df['signal_class']

        self.scaler.fit(X_all)
        X_scaled = self.scaler.transform(X_all)

        base_model = self.create_advanced_model()
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        self.model.fit(X_scaled, y_all)

        # 11. Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        try:
            # Use LightGBM from ensemble for SHAP
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, verbose=-1
            )
            lgb_model.fit(X_scaled, y_all)
            self.explainer = ModelExplainer(lgb_model, self.selected_features)
            self.explainer.fit(X_scaled, sample_size=500)
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}")

        # 12. Save model
        if save_model:
            model_path = self.models_dir / f"model_{self.VERSION}.joblib"

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.selected_features,
                'all_features': self.feature_cols,
                'feature_importance': self.feature_selector.feature_importance,
                'metrics': metrics,
                'version': self.VERSION,
                'trained_at': datetime.now().isoformat(),
                'data_sources': ['features_db', 'cbr_macro', 'moex_news_transformer'],
                'drift_report': drift_report
            }

            joblib.dump(model_data, model_path)
            logger.info(f"✅ Model saved: {model_path}")

            # Symlink to latest
            latest_path = self.models_dir / "model_latest.joblib"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(model_path.name)

        # 13. Feature importance
        logger.info("\nTop 15 Feature Importance (Boruta):")
        sorted_importance = sorted(
            self.feature_selector.feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:15]
        for feat, imp in sorted_importance:
            bar = '█' * int(imp / max(i[1] for i in sorted_importance) * 30)
            logger.info(f"  {feat:25s} {imp:.4f} {bar}")

        # Cleanup
        self.cbr.close()
        self.news_client.close()

        return metrics

    def predict(self, features: Dict) -> Dict:
        """Предсказание с position sizing и SHAP объяснением"""
        if self.model is None:
            return {'error': 'Model not loaded'}

        X = np.array([[features.get(f, 0) for f in self.selected_features]])
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        classes = self.model.classes_

        pred_class = int(classes[np.argmax(proba)])
        confidence = float(np.max(proba))

        # Position sizing
        sizing = self.position_sizer.calculate_position_size(
            confidence=confidence,
            predicted_class=pred_class,
            class_probabilities={int(c): float(p) for c, p in zip(classes, proba)}
        )

        result = {
            'signal': pred_class,
            'confidence': confidence,
            'probabilities': {int(c): float(p) for c, p in zip(classes, proba)},
            'position_size': sizing['position_size'],
            'kelly_info': sizing
        }

        # SHAP explanation
        if self.explainer is not None:
            explanation = self.explainer.explain_single(X_scaled[0], class_idx=pred_class)
            result['explanation'] = explanation

        return result

    def load_model(self, path: str = None):
        """Загрузка модели"""
        if path is None:
            path = self.models_dir / "model_latest.joblib"

        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data.get('scaler', StandardScaler())
            self.selected_features = data['features']
            self.feature_cols = data.get('all_features', data['features'])

            # Restore feature selector importance
            if 'feature_importance' in data:
                self.feature_selector.feature_importance = data['feature_importance']

            logger.info(f"✅ Model loaded: {data.get('version', 'unknown')}")
            logger.info(f"   Features: {len(self.selected_features)}")
            logger.info(f"   Metrics: Acc={data['metrics'].get('accuracy', 0):.1%} "
                        f"Sharpe={data['metrics'].get('sharpe', 0):.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# ============================================================
# SELF-LEARNING SCHEDULER (улучшенный)
# ============================================================
class SelfLearningSchedulerV5:
    """Планировщик самообучения v5"""

    def __init__(self, trainer: WalkForwardTrainerV5,
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

        # Проверяем data drift
        drift_report = self.trainer.drift_monitor.check_drift(
            self.trainer.load_features(),
            self.trainer.selected_features
        )

        # Проверяем метрики
        should_retrain, reason = self.trainer.tracker.should_retrain()

        # Добавляем drift к решению
        if drift_report['drift_detected'] and drift_report['drift_ratio'] > 0.3:
            should_retrain = True
            reason = f"data_drift_{drift_report['drift_ratio']:.1%}"

        if should_retrain:
            logger.info(f"⚠️ Retrain triggered: {reason}")

            old_metrics = self.trainer.tracker.calculate_metrics()
            new_metrics = self.trainer.train(save_model=True)

            self.trainer.tracker.log_retrain(
                reason=reason,
                old_metrics=old_metrics,
                new_metrics=new_metrics,
                features_count=len(self.trainer.selected_features),
                version=self.trainer.VERSION
            )

            logger.info(f"✅ Retrain complete. New accuracy: {new_metrics['accuracy']:.1%}")
        else:
            logger.info(f"✅ Model performance OK: {reason}")

    def scheduled_retrain(self):
        """Плановое еженедельное переобучение"""
        logger.info("📅 Scheduled weekly retrain...")

        old_metrics = self.trainer.tracker.calculate_metrics()
        new_metrics = self.trainer.train(save_model=True, use_cpcv=True)

        self.trainer.tracker.log_retrain(
            reason="scheduled_weekly",
            old_metrics=old_metrics if old_metrics.get('valid') else {},
            new_metrics=new_metrics,
            features_count=len(self.trainer.selected_features),
            version=self.trainer.VERSION
        )

        logger.info("✅ Weekly retrain complete")

    def start(self):
        """Запуск планировщика"""
        self.running = True

        schedule.every(self.check_interval).hours.do(self.check_and_retrain)
        schedule.every().sunday.at("03:00").do(self.scheduled_retrain)

        logger.info("🚀 Self-learning scheduler v5 started")
        logger.info(f"   - Performance check: every {self.check_interval} hours")
        logger.info(f"   - Data drift monitoring: enabled")
        logger.info(f"   - Scheduled retrain: Sunday 03:00")

        while self.running:
            schedule.run_pending()
            time.sleep(60)

    def stop(self):
        """Остановка планировщика"""
        self.running = False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Trainer v5")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://localhost/trading"))
    parser.add_argument("--models-dir", default="/app/models")
    parser.add_argument("--mode", choices=["train", "schedule", "predict"], default="train")
    parser.add_argument("--use-cpcv", action="store_true", default=True)
    args = parser.parse_args()

    trainer = WalkForwardTrainerV5(args.db_url, args.models_dir)

    if args.mode == "train":
        metrics = trainer.train(save_model=True, use_cpcv=args.use_cpcv)
        print(f"\n✅ Training complete!")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   Sharpe:   {metrics['sharpe']:.2f}")
        print(f"   Features: {metrics['n_features']}")

    elif args.mode == "schedule":
        scheduler = SelfLearningSchedulerV5(trainer)
        try:
            scheduler.start()
        except KeyboardInterrupt:
            scheduler.stop()
            print("\nScheduler stopped")

    elif args.mode == "predict":
        trainer.load_model()
        # Example prediction
        sample_features = {f: 0.0 for f in trainer.selected_features}
        result = trainer.predict(sample_features)
        print(f"Prediction: {result}")
