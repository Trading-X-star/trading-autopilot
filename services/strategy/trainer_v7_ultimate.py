#!/usr/bin/env python3
"""
Trainer v7 Ultimate - Advanced ML Pipeline
==========================================
Улучшения:
1. Knowledge Distillation (Teacher-Student)
2. Расширенная выборка данных (3 года, multi-timeframe)
3. Auto-Augmentation для временных рядов
4. Экзогенные факторы (макро, глобальные рынки, commodities)
5. Все улучшения из v6 (VIF, SMOTE, Optuna, etc.)
"""

import os
import json
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score, 
                            roc_auc_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================
KNOWN_TICKERS = [
    'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'VTBR', 'MTSS',
    'MGNT', 'TATN', 'YNDX', 'TCSG', 'NLMK', 'CHMF', 'PLZL', 'ALRS',
    'POLY', 'FIVE', 'IRAO', 'HYDR', 'PHOR', 'RUAL', 'MAGN', 'AFLT'
]

SECTORS = {
    'banks': ['SBER', 'VTBR', 'TCSG'],
    'oil_gas': ['GAZP', 'LKOH', 'ROSN', 'NVTK', 'TATN'],
    'metals': ['GMKN', 'NLMK', 'CHMF', 'MAGN', 'PLZL', 'ALRS', 'POLY', 'RUAL'],
    'retail': ['MGNT', 'FIVE'],
    'tech': ['YNDX'],
    'telecom': ['MTSS'],
    'energy': ['IRAO', 'HYDR'],
    'fertilizers': ['PHOR'],
    'transport': ['AFLT'],
}

CBR_MEETINGS_2026 = [
    '2026-02-14', '2026-03-21', '2026-04-25', '2026-06-06',
    '2026-07-25', '2026-09-12', '2026-10-24', '2026-12-19'
]


# ============================================================
# 1. TIME SERIES AUTO-AUGMENTATION
# ============================================================
class TimeSeriesAutoAugment:
    """Автоматическая аугментация для финансовых временных рядов"""

    def __init__(self, augment_ratio: float = 0.3, n_augmentations: int = 3):
        self.augment_ratio = augment_ratio
        self.n_augmentations = n_augmentations

    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Применить аугментацию к данным"""
        n_samples = len(X)
        n_augment = int(n_samples * self.augment_ratio)

        X_aug_list = [X.copy()]
        y_aug_list = [y.copy()]

        augment_funcs = [
            self._jitter,
            self._scaling,
            self._magnitude_warp,
            self._window_slice,
            self._permutation,
        ]

        for i in range(n_augment):
            idx = np.random.randint(n_samples)
            x_orig = X[idx].copy()

            # Применить случайные аугментации
            n_apply = np.random.randint(1, self.n_augmentations + 1)
            funcs = np.random.choice(augment_funcs, n_apply, replace=False)

            x_aug = x_orig
            for func in funcs:
                x_aug = func(x_aug)

            X_aug_list.append(x_aug.reshape(1, -1))
            y_aug_list.append(y[idx])

        X_augmented = np.vstack(X_aug_list)
        y_augmented = np.hstack(y_aug_list)

        logger.info(f"Augmentation: {n_samples} → {len(X_augmented)} samples")
        return X_augmented, y_augmented

    def _jitter(self, x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        """Добавить гауссовский шум"""
        noise = np.random.normal(0, sigma * np.std(x), x.shape)
        return x + noise

    def _scaling(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Случайное масштабирование"""
        factor = np.random.normal(1, sigma)
        return x * factor

    def _magnitude_warp(self, x: np.ndarray, sigma: float = 0.2, knots: int = 4) -> np.ndarray:
        """Искажение амплитуды с помощью сплайна"""
        orig_steps = np.arange(len(x))
        random_warps = np.random.normal(1, sigma, knots + 2)
        warp_steps = np.linspace(0, len(x) - 1, knots + 2)

        if len(warp_steps) > 1:
            warper = np.interp(orig_steps, warp_steps, random_warps)
            return x * warper
        return x

    def _window_slice(self, x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Случайный срез окна с интерполяцией"""
        target_len = max(int(len(x) * reduce_ratio), 2)
        if target_len >= len(x):
            return x

        start = np.random.randint(0, len(x) - target_len + 1)
        sliced = x[start:start + target_len]

        # Интерполировать обратно до исходной длины
        old_indices = np.linspace(0, 1, len(sliced))
        new_indices = np.linspace(0, 1, len(x))
        return np.interp(new_indices, old_indices, sliced)

    def _permutation(self, x: np.ndarray, n_segments: int = 4) -> np.ndarray:
        """Перестановка сегментов (для небольшого шума)"""
        if len(x) < n_segments:
            return x

        segment_len = len(x) // n_segments
        segments = [x[i*segment_len:(i+1)*segment_len] for i in range(n_segments)]

        # Небольшая перестановка (swap 2 соседних)
        if len(segments) >= 2:
            i = np.random.randint(len(segments) - 1)
            segments[i], segments[i+1] = segments[i+1], segments[i]

        result = np.concatenate(segments)
        # Дополнить если нужно
        if len(result) < len(x):
            result = np.concatenate([result, x[len(result):]])
        return result[:len(x)]


# ============================================================
# 2. EXOGENOUS FACTORS
# ============================================================
class ExogenousFactors:
    """Сбор экзогенных факторов"""

    def __init__(self):
        self.cbr_key_rate = 21.0  # Текущая ставка ЦБ

    async def collect_all(self, date: datetime = None) -> Dict[str, float]:
        """Собрать все экзогенные факторы"""
        if date is None:
            date = datetime.now()

        factors = {}

        # Макро факторы
        factors.update(self._macro_factors())

        # Сезонность
        factors.update(self._seasonality(date))

        # Календарные события
        factors.update(self._calendar_events(date))

        # Глобальные рынки (заглушки если нет API)
        factors.update(await self._global_markets())

        # Commodities
        factors.update(await self._commodities())

        logger.info(f"Collected {len(factors)} exogenous factors")
        return factors

    def _macro_factors(self) -> Dict[str, float]:
        """Макроэкономические факторы РФ"""
        return {
            'cbr_key_rate': self.cbr_key_rate,
            'cbr_rate_high': 1.0 if self.cbr_key_rate >= 15 else 0.0,
            'inflation_yoy': 8.5,
            'rub_volatility': 0.5,  # Нормализованная волатильность рубля
        }

    def _seasonality(self, date: datetime) -> Dict[str, float]:
        """Сезонные факторы"""
        return {
            'month': date.month,
            'day_of_week': date.weekday(),
            'day_of_month': date.day,
            'week_of_year': date.isocalendar()[1],
            'quarter': (date.month - 1) // 3 + 1,
            'is_monday': 1.0 if date.weekday() == 0 else 0.0,
            'is_friday': 1.0 if date.weekday() == 4 else 0.0,
            'is_month_start': 1.0 if date.day <= 5 else 0.0,
            'is_month_end': 1.0 if date.day >= 25 else 0.0,
            'is_quarter_end': 1.0 if date.month in [3, 6, 9, 12] and date.day >= 20 else 0.0,
            'january_effect': 1.0 if date.month == 1 else 0.0,
            'sell_in_may': 1.0 if date.month in [5, 6, 7, 8, 9] else 0.0,
            'santa_rally': 1.0 if date.month == 12 and date.day >= 15 else 0.0,
        }

    def _calendar_events(self, date: datetime) -> Dict[str, float]:
        """Календарные события"""
        # Дни до заседания ЦБ
        days_to_cbr = 30
        for meeting_str in CBR_MEETINGS_2026:
            meeting_date = datetime.strptime(meeting_str, '%Y-%m-%d')
            diff = (meeting_date - date).days
            if diff > 0:
                days_to_cbr = min(days_to_cbr, diff)
                break

        return {
            'days_to_cbr_meeting': days_to_cbr,
            'cbr_meeting_soon': 1.0 if days_to_cbr <= 7 else 0.0,
            'is_dividend_season': 1.0 if date.month in [4, 5, 6, 7] else 0.0,
            'is_reporting_season': 1.0 if date.month in [2, 3, 4, 8, 9, 10, 11] else 0.0,
            'days_to_month_end': max(0, 28 - date.day),
        }

    async def _global_markets(self) -> Dict[str, float]:
        """Глобальные рынки"""
        # В реальности здесь API запросы
        return {
            'sp500_trend': 0.0,  # -1 to +1
            'vix_level': 15.0,
            'vix_high': 0.0,  # 1 if VIX > 25
            'dxy_trend': 0.0,
            'em_sentiment': 0.0,  # Emerging markets sentiment
        }

    async def _commodities(self) -> Dict[str, float]:
        """Сырьевые товары"""
        return {
            'brent_price_norm': 0.5,  # Нормализованная цена
            'brent_trend': 0.0,  # -1 to +1
            'gold_trend': 0.0,
            'gas_europe_trend': 0.0,
            'metals_index_trend': 0.0,
        }

    def get_sync(self, date: datetime = None) -> Dict[str, float]:
        """Синхронная версия для использования без async"""
        if date is None:
            date = datetime.now()

        factors = {}
        factors.update(self._macro_factors())
        factors.update(self._seasonality(date))
        factors.update(self._calendar_events(date))

        # Заглушки для async данных
        factors.update({
            'sp500_trend': 0.0,
            'vix_level': 15.0,
            'vix_high': 0.0,
            'dxy_trend': 0.0,
            'em_sentiment': 0.0,
            'brent_price_norm': 0.5,
            'brent_trend': 0.0,
            'gold_trend': 0.0,
            'gas_europe_trend': 0.0,
            'metals_index_trend': 0.0,
        })

        return factors


# ============================================================
# 3. DATA EXPANSION
# ============================================================
class DataExpansion:
    """Расширение датасета"""

    def __init__(self, lookback_years: int = 3):
        self.lookback_years = lookback_years
        self.lookback_days = lookback_years * 365

    def expand_with_synthetic(self, df: pd.DataFrame, 
                              expansion_factor: float = 2.0) -> pd.DataFrame:
        """Расширить датасет синтетическими данными"""

        original_len = len(df)

        # 1. Bootstrap sampling с шумом
        n_bootstrap = int(original_len * (expansion_factor - 1) * 0.5)
        bootstrap_indices = np.random.choice(original_len, n_bootstrap, replace=True)
        df_bootstrap = df.iloc[bootstrap_indices].copy()

        # Добавить небольшой шум к числовым колонкам
        numeric_cols = df_bootstrap.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            noise = np.random.normal(0, df_bootstrap[col].std() * 0.01, len(df_bootstrap))
            df_bootstrap[col] = df_bootstrap[col] + noise

        # 2. Интерполяция между точками
        n_interp = int(original_len * (expansion_factor - 1) * 0.5)
        interp_data = []

        for _ in range(n_interp):
            idx1, idx2 = np.random.choice(original_len, 2, replace=False)
            alpha = np.random.uniform(0.3, 0.7)

            row = {}
            for col in numeric_cols:
                row[col] = alpha * df.iloc[idx1][col] + (1 - alpha) * df.iloc[idx2][col]

            # Сохранить target от первой точки
            if 'target' in df.columns:
                row['target'] = df.iloc[idx1]['target']

            interp_data.append(row)

        df_interp = pd.DataFrame(interp_data)

        # Объединить
        df_expanded = pd.concat([df, df_bootstrap, df_interp], ignore_index=True)

        logger.info(f"Data expansion: {original_len} → {len(df_expanded)} samples")
        return df_expanded

    def add_lagged_features(self, df: pd.DataFrame, 
                            lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Добавить лаговые признаки"""

        df = df.copy()

        # Основные колонки для лагов
        lag_cols = ['close', 'volume', 'return_1d', 'volatility_20', 'rsi_14']
        lag_cols = [c for c in lag_cols if c in df.columns]

        for col in lag_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Удалить NaN
        df = df.dropna()

        logger.info(f"Added lagged features: {len(lags)} lags × {len(lag_cols)} columns")
        return df


# ============================================================
# 4. KNOWLEDGE DISTILLATION
# ============================================================
class KnowledgeDistillation:
    """Teacher-Student обучение для трейдинга"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha  # Вес soft labels
        self.teacher = None
        self.student = None
        self.teacher_accuracy = 0.0
        self.student_accuracy = 0.0

    def build_teacher(self, optimal_params: Dict = None) -> VotingClassifier:
        """Создать тяжёлый ensemble-учитель"""

        if optimal_params is None:
            optimal_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 64,
            }

        estimators = [
            ('lgb1', lgb.LGBMClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                max_depth=optimal_params.get('max_depth', 8),
                learning_rate=optimal_params.get('learning_rate', 0.05),
                num_leaves=optimal_params.get('num_leaves', 64),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
            ('lgb2', lgb.LGBMClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                max_depth=optimal_params.get('max_depth', 8) + 2,
                learning_rate=optimal_params.get('learning_rate', 0.05) * 0.8,
                num_leaves=optimal_params.get('num_leaves', 64) * 2,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=43,
                verbose=-1,
                n_jobs=-1
            )),
            ('xgb', xgb.XGBClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                max_depth=optimal_params.get('max_depth', 8),
                learning_rate=optimal_params.get('learning_rate', 0.05),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                max_depth=optimal_params.get('max_depth', 8) + 4,
                random_state=42,
                n_jobs=-1
            )),
        ]

        if CATBOOST_AVAILABLE:
            estimators.append(('cb', CatBoostClassifier(
                iterations=optimal_params.get('n_estimators', 200),
                depth=min(optimal_params.get('max_depth', 8), 10),
                learning_rate=optimal_params.get('learning_rate', 0.05),
                random_state=42,
                verbose=0
            )))

        self.teacher = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        return self.teacher

    def build_student(self) -> lgb.LGBMClassifier:
        """Создать лёгкую модель-ученика"""
        self.student = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        return self.student

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            optimal_params: Dict = None) -> 'KnowledgeDistillation':
        """Обучить teacher и distill в student"""

        # Convert labels to 0/1 if needed
        # Convert to binary
        y_train = np.where(np.array(y_train) > 0, 1, 0).astype(np.int32)
        # Convert to binary
        y_val = np.where(np.array(y_val) > 0, 1, 0).astype(np.int32)

        # 1. Построить и обучить учителя
        logger.info("Training TEACHER ensemble (5 models)...")
        self.build_teacher(optimal_params)
        self.teacher.fit(X_train, y_train)

        # Оценить учителя
        teacher_pred = self.teacher.predict(X_val)
        self.teacher_accuracy = accuracy_score(y_val, teacher_pred)
        logger.info(f"Teacher accuracy: {self.teacher_accuracy:.4f}")

        # 2. Получить soft labels от учителя
        teacher_proba = self.teacher.predict_proba(X_train)
        soft_labels = self._apply_temperature(teacher_proba)

        # 3. Построить ученика
        logger.info("Training STUDENT model with knowledge distillation...")
        self.build_student()

        # 4. Обучить ученика (используем hard labels, т.к. LightGBM не поддерживает soft напрямую)
        # Альтернатива: sample weights на основе confidence учителя
        teacher_confidence = np.max(teacher_proba, axis=1)
        sample_weights = 0.5 + 0.5 * teacher_confidence  # Больше вес для уверенных примеров

        self.student.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # 5. Оценить ученика
        student_pred = self.student.predict(X_val)
        self.student_accuracy = accuracy_score(y_val, student_pred)
        logger.info(f"Student accuracy: {self.student_accuracy:.4f}")
        logger.info(f"Distillation gap: {self.teacher_accuracy - self.student_accuracy:.4f}")

        return self

    def _apply_temperature(self, proba: np.ndarray) -> np.ndarray:
        """Применить temperature scaling к вероятностям"""
        # Конвертировать в logits
        logits = np.log(proba + 1e-10)

        # Применить temperature
        scaled_logits = logits / self.temperature

        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание (использует student)"""
        return self.student.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности (использует student)"""
        return self.student.predict_proba(X)


# ============================================================
# 5. ADVANCED FEATURE ENGINEERING
# ============================================================
class AdvancedFeatureEngineer:
    """Продвинутый feature engineering"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать продвинутые признаки"""
        df = df.copy()

        # Базовые признаки
        if 'close' in df.columns:
            # Returns
            for period in [1, 2, 3, 5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(period)

            # Volatility
            df['volatility_5'] = df['return_1d'].rolling(5).std()
            df['volatility_10'] = df['return_1d'].rolling(10).std()
            df['volatility_20'] = df['return_1d'].rolling(20).std()

            # Volatility ratio
            df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)

            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'close_to_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10) - 1

            # EMA
            for period in [12, 26]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

            # Momentum
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

            # Rate of Change
            df['roc_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)
            df['roc_10'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)

        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
            df['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-10)

        # High/Low features
        if 'high' in df.columns and 'low' in df.columns:
            df['daily_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
            df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)

            # ATR
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                        n_features: int = 20) -> List[str]:
        """Отбор лучших признаков"""

        # Convert labels
        y_binary = np.where(np.array(y) > 0, 1, 0).astype(np.int32)

        # Удалить константные и NaN колонки
        valid_cols = []
        for col in X.columns:
            if X[col].std() > 1e-10 and X[col].notna().sum() > len(X) * 0.5:
                valid_cols.append(col)

        X_valid = X[valid_cols].fillna(0)

        # Mutual Information
        mi_scores = mutual_info_classif(X_valid, y_binary, random_state=42)
        mi_df = pd.DataFrame({'feature': valid_cols, 'mi_score': mi_scores})

        # LightGBM importance
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(X_valid, y_binary)
        lgb_importance = lgb_model.feature_importances_
        mi_df['lgb_score'] = lgb_importance

        # Combined score
        mi_df['combined'] = (
            mi_df['mi_score'] / (mi_df['mi_score'].max() + 1e-10) +
            mi_df['lgb_score'] / (mi_df['lgb_score'].max() + 1e-10)
        )

        # Top features
        top_features = mi_df.nlargest(n_features, 'combined')['feature'].tolist()

        logger.info(f"Selected {len(top_features)} features")
        for i, feat in enumerate(top_features[:10]):
            score = mi_df[mi_df['feature'] == feat]['combined'].values[0]
            logger.info(f"  {i+1}. {feat}: {score:.4f}")

        self.feature_names = top_features
        return top_features


# ============================================================
# 6. HYPERPARAMETER OPTIMIZER
# ============================================================
class HyperparameterOptimizer:
    """Optuna-based оптимизация"""

    def __init__(self, n_trials: int = 30):
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = 0.0

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров"""

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default params")
            return self._get_default_params()

        # Convert labels
        # Convert to binary
        y_train = np.where(np.array(y_train) > 0, 1, 0).astype(np.int32)
        # Convert to binary
        y_val = np.where(np.array(y_val) > 0, 1, 0).astype(np.int32)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }

            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )

            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)

        logger.info(f"Starting Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    @staticmethod
    def _get_default_params() -> Dict:
        return {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 48,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
        }


# ============================================================
# 7. MAIN TRAINER V7
# ============================================================
class TrainerV7Ultimate:
    """Trainer v7 Ultimate с продвинутыми техниками"""

    def __init__(self, n_features: int = 20, n_trials: int = 30):
        self.n_features = n_features
        self.n_trials = n_trials

        # Components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.augmenter = TimeSeriesAutoAugment(augment_ratio=0.3)
        self.exogenous = ExogenousFactors()
        self.data_expander = DataExpansion()
        self.optimizer = HyperparameterOptimizer(n_trials=n_trials)
        self.distiller = KnowledgeDistillation(temperature=3.0, alpha=0.5)

        # Results
        self.selected_features = []
        self.best_params = {}
        self.results = {}

    def train(self, df: pd.DataFrame, use_augmentation: bool = True,
              use_distillation: bool = True, use_optuna: bool = True) -> Dict:
        """Полный pipeline обучения"""

        logger.info("=" * 80)
        logger.info("TRAINER V7 ULTIMATE - ADVANCED ML PIPELINE")
        logger.info("=" * 80)

        # ========================================
        # PHASE 1: Feature Engineering
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: FEATURE ENGINEERING")
        logger.info("=" * 60)

        df = self.feature_engineer.create_features(df)
        logger.info(f"Created features: {len(df.columns)} columns")

        # ========================================
        # PHASE 2: Add Exogenous Factors
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: EXOGENOUS FACTORS")
        logger.info("=" * 60)

        exo_factors = self.exogenous.get_sync()
        for k, v in exo_factors.items():
            df[k] = v
        logger.info(f"Added {len(exo_factors)} exogenous factors")

        # ========================================
        # PHASE 3: Create Target
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: TARGET ENGINEERING")
        logger.info("=" * 60)

        if 'target' not in df.columns:
            if 'return_5d' in df.columns:
                df['target'] = np.where(df['return_5d'] > 0.01, 1,
                               np.where(df['return_5d'] < -0.01, -1, 0))
            else:
                df['target'] = np.where(df['close'].shift(-5) > df['close'] * 1.01, 1,
                               np.where(df['close'].shift(-5) < df['close'] * 0.99, -1, 0))

        # Remove neutral
        df = df[df['target'] != 0].copy()
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

        # ========================================
        # PHASE 4: Data Expansion
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: DATA EXPANSION")
        logger.info("=" * 60)

        original_len = len(df)
        df = self.data_expander.expand_with_synthetic(df, expansion_factor=1.5)
        logger.info(f"Expanded: {original_len} → {len(df)}")

        # ========================================
        # PHASE 5: Prepare Data
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: DATA PREPARATION")
        logger.info("=" * 60)

        # Remove non-feature columns
        exclude_cols = ['target', 'date', 'ticker', 'timestamp']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df['target'].copy()

        # Fill NaN and Inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        logger.info(f"Features: {X.shape[1]}, Samples: {len(X)}")

        # ========================================
        # PHASE 6: Feature Selection
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6: FEATURE SELECTION")
        logger.info("=" * 60)

        self.selected_features = self.feature_engineer.select_features(
            X, y, n_features=self.n_features
        )
        X = X[self.selected_features]

        # ========================================
        # PHASE 7: Train/Val Split
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 7: TRAIN/VALIDATION SPLIT")
        logger.info("=" * 60)

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, test_size=0.2, shuffle=False
        )
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # ========================================
        # PHASE 8: Augmentation
        # ========================================
        if use_augmentation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 8: DATA AUGMENTATION")
            logger.info("=" * 60)

            X_train, y_train = self.augmenter.augment(X_train, y_train)

        # ========================================
        # PHASE 9: Class Balancing (SMOTE)
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 9: CLASS BALANCING (SMOTE)")
        logger.info("=" * 60)

        # Convert to 0/1 for SMOTE
        y_train_binary = np.where(np.array(y_train) > 0, 1, 0).astype(np.int32)


        if IMBLEARN_AVAILABLE:
            smote = SMOTE(random_state=42)
            X_train, y_train_binary = smote.fit_resample(X_train, y_train_binary)
            logger.info(f"After SMOTE: {len(X_train)} samples")

        # Convert validation labels
        y_val_binary = np.where(np.array(y_val) > 0, 1, 0).astype(np.int32)

        # ========================================
        # PHASE 10: Hyperparameter Optimization
        # ========================================
        if use_optuna:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 10: HYPERPARAMETER OPTIMIZATION")
            logger.info("=" * 60)

            self.best_params = self.optimizer.optimize(
                X_train, y_train_binary, X_val, y_val_binary
            )


            self.best_params = self.optimizer.optimize(
                X_train, y_train_binary, X_val, y_val_binary
            )
        else:
            self.best_params = HyperparameterOptimizer._get_default_params()

        # ========================================
        # PHASE 11: Knowledge Distillation
        # ========================================
        if use_distillation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 11: KNOWLEDGE DISTILLATION")
            logger.info("=" * 60)


            self.distiller.fit(X_train, y_train_binary, X_val, y_val_binary, self.best_params)

            # Final model is the student
            self.model = self.distiller.student
        else:
            # Train single model
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 11: TRAINING FINAL MODEL")
            logger.info("=" * 60)


            self.model = lgb.LGBMClassifier(**self.best_params, random_state=42, verbose=-1)
            self.model.fit(
                X_train, y_train_binary,
                eval_set=[(X_val, y_val_binary)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

        # ========================================
        # PHASE 12: Evaluation
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 12: FINAL EVALUATION")
        logger.info("=" * 60)


        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val_binary, y_pred)
        f1 = f1_score(y_val_binary, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_val_binary, y_pred)

        try:
            roc_auc = roc_auc_score(y_val_binary, y_proba)
        except:
            roc_auc = 0.5

        # Results
        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'roc_auc': roc_auc,
            'n_features': len(self.selected_features),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'best_params': self.best_params,
            'selected_features': self.selected_features,
            'use_augmentation': use_augmentation,
            'use_distillation': use_distillation,
        }

        if use_distillation:
            self.results['teacher_accuracy'] = self.distiller.teacher_accuracy
            self.results['student_accuracy'] = self.distiller.student_accuracy
            self.results['distillation_gap'] = self.distiller.teacher_accuracy - self.distiller.student_accuracy

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Accuracy: {accuracy:.4f}")
        logger.info(f"✅ F1-Score: {f1:.4f}")
        logger.info(f"✅ Cohen Kappa: {kappa:.4f}")
        logger.info(f"✅ ROC-AUC: {roc_auc:.4f}")

        if use_distillation:
            logger.info(f"📚 Teacher Accuracy: {self.distiller.teacher_accuracy:.4f}")
            logger.info(f"🎓 Student Accuracy: {self.distiller.student_accuracy:.4f}")

        return self.results

    def save_model(self, path: str = "model_v7_ultimate.joblib"):
        """Сохранить модель"""
        model_data = {
            'model': self.model,
            'features': self.selected_features,
            'params': self.best_params,
            'results': self.results,
            'version': f"v7_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'timestamp': datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        logger.info(f"💾 Model saved to {path}")

        # Save results JSON
        results_path = path.replace('.joblib', '_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                k: v if not isinstance(v, np.ndarray) else v.tolist() 
                for k, v in self.results.items()
            }, f, indent=2, default=str)
        logger.info(f"📄 Results saved to {results_path}")


# ============================================================
# MAIN
# ============================================================
def create_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Создать синтетические данные для демонстрации"""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='h')

    # Генерация цен с трендом и шумом
    trend = np.cumsum(np.random.randn(n_samples) * 0.001)
    seasonality = 0.02 * np.sin(np.arange(n_samples) * 2 * np.pi / 24)  # Daily pattern
    noise = np.random.randn(n_samples) * 0.01

    close = 100 * np.exp(trend + seasonality + noise)

    df = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.randn(n_samples) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'close': close,
        'volume': np.random.exponential(1000000, n_samples),
    })

    return df


def main():
    parser = argparse.ArgumentParser(description='Trainer V7 Ultimate')
    parser.add_argument('--data', type=str, default='/app/data/training_data.csv', help='Path to data CSV')
    parser.add_argument('--features', type=int, default=20, help='Number of features')
    parser.add_argument('--trials', type=int, default=30, help='Optuna trials')
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--no-distillation', action='store_true')
    parser.add_argument('--no-optuna', action='store_true')
    parser.add_argument('--output', type=str, default='model_v7_ultimate.joblib')
    args = parser.parse_args()

    # Load or create data
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data, parse_dates=['date'] if 'date' in pd.read_csv(args.data, nrows=1).columns else None)
    else:
        logger.info("Creating synthetic data for demonstration...")
        df = create_synthetic_data(10000)

    logger.info(f"Data shape: {df.shape}")

    # Train
    trainer = TrainerV7Ultimate(n_features=args.features, n_trials=args.trials)
    results = trainer.train(
        df,
        use_augmentation=not args.no_augmentation,
        use_distillation=not args.no_distillation,
        use_optuna=not args.no_optuna
    )

    # Save
    trainer.save_model(args.output)

    logger.info("\n" + "=" * 60)
    logger.info("✨ TRAINING COMPLETED ✨")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
