#!/usr/bin/env python3
"""
Trading Brain v5.0 - Enhanced Market Intelligence
Улучшения:
- 10+ паттернов (double bottom, head & shoulders, breakout, etc.)
- Sector rotation tracking
- Macro signals (ЦБ, отчёты)
- Расширенный Fear & Greed (8 компонентов)
- Smart position sizing
- Anomaly detection
"""
import asyncio
import os
import logging
import json
import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque, defaultdict
import asyncpg
import redis.asyncio as aioredis
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading_brain")


def to_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, Decimal): return float(val)
    try: return float(val)
    except: return default


# ============================================================
# PROMETHEUS METRICS
# ============================================================
JOBS_TOTAL = Counter('scheduler_jobs_total', 'Jobs executed', ['job', 'status'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
MODEL_LAST_TRAIN = Gauge('model_last_train_timestamp', 'Last training timestamp')
FEATURES_COUNT = Gauge('features_count', 'Total features in database')
FEAR_GREED_INDEX = Gauge('fear_greed_index', 'Current Fear & Greed Index')
MARKET_REGIME = Gauge('market_regime', 'Market regime')
PATTERNS_DETECTED = Counter('patterns_detected', 'Patterns detected', ['pattern'])
SECTOR_ROTATION = Gauge('sector_rotation_score', 'Sector rotation', ['sector'])
SIGNAL_LATENCY = Histogram('signal_enhancement_latency', 'Signal enhancement time')


# ============================================================
# CONSTANTS
# ============================================================
CURRENCY_CODES = {'USD': 'R01235', 'EUR': 'R01239', 'CNY': 'R01375'}
KNOWN_TICKERS = {
    'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'VTBR', 'MTSS',
    'MGNT', 'TATN', 'YNDX', 'TCSG', 'NLMK', 'CHMF', 'PLZL', 'ALRS',
    'POLY', 'FIVE', 'IRAO', 'HYDR', 'PHOR', 'RUAL', 'MAGN', 'AFLT',
    'FEES', 'CBOM', 'RTKM', 'MOEX', 'AFKS', 'SNGS', 'OZON'
}

POSITIVE_WORDS = {'рост', 'прибыль', 'дивиденд', 'увеличение', 'повышение', 'рекорд', 
                  'успех', 'выручка', 'buyback', 'выкуп', 'рекомендация', 'апгрейд'}
NEGATIVE_WORDS = {'падение', 'убыток', 'снижение', 'сокращение', 'санкции', 'риск', 
                  'дефолт', 'штраф', 'делистинг', 'даунгрейд', 'дивгэп'}

SECTORS = {
    'banks': ['SBER', 'VTBR', 'TCSG', 'CBOM'],
    'oil_gas': ['GAZP', 'LKOH', 'ROSN', 'NVTK', 'TATN', 'SNGS'],
    'metals': ['GMKN', 'NLMK', 'CHMF', 'MAGN', 'PLZL', 'ALRS', 'POLY', 'RUAL'],
    'retail': ['MGNT', 'FIVE', 'OZON'],
    'tech': ['YNDX', 'OZON'],
    'telecom': ['MTSS', 'RTKM'],
    'energy': ['IRAO', 'HYDR', 'FEES'],
    'fertilizers': ['PHOR'],
    'transport': ['AFLT', 'NMTP'],
}

SECTOR_CYCLE = ['banks', 'tech', 'retail', 'oil_gas', 'metals', 'energy']

CBR_KEY_RATE = 21.0
MIN_ACCURACY_THRESHOLD = 0.38
MIN_SHARPE_THRESHOLD = 0.2
RETRAIN_COOLDOWN_HOURS = 24

# CBR Meeting dates 2026 (примерные)
CBR_MEETINGS_2026 = [
    '2026-02-14', '2026-03-21', '2026-04-25', '2026-06-06',
    '2026-07-25', '2026-09-12', '2026-10-24', '2026-12-19'
]


# ============================================================
# ENUMS & DATACLASSES
# ============================================================

class MarketRegime(Enum):
    CRISIS = "crisis"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    TRENDING_UP = "trending_up"
    HIGH_VOLATILITY = "high_vol"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class MarketEmotion(Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class PatternType(Enum):
    # Reversal patterns
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inv_head_shoulders"
    
    # Continuation patterns
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    TRIANGLE = "triangle"
    
    # Breakout patterns
    RESISTANCE_BREAKOUT = "resistance_breakout"
    SUPPORT_BREAKDOWN = "support_breakdown"
    VOLUME_BREAKOUT = "volume_breakout"
    
    # Behavioral patterns
    PANIC_SELLING = "panic_selling"
    FOMO_BUYING = "fomo_buying"
    CAPITULATION = "capitulation"
    SMART_MONEY_ACCUMULATION = "smart_accumulation"
    SMART_MONEY_DISTRIBUTION = "smart_distribution"
    DIVERGENCE_BULLISH = "divergence_bullish"
    DIVERGENCE_BEARISH = "divergence_bearish"


@dataclass
class TradingConfig:
    min_confidence: float = 0.45
    max_position_pct: float = 0.10
    max_daily_trades: int = 20
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    trailing_stop_pct: float = 0.015
    # New fields
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    risk_budget: float = 0.02  # Max daily loss
    correlation_limit: float = 0.7  # Max portfolio correlation


@dataclass
class GlobalConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    regime: MarketRegime = MarketRegime.SIDEWAYS
    fear_greed: float = 50.0
    volatility_percentile: float = 50.0
    updated_at: str = ""
    corrections_count: int = 0
    active_sectors: List[str] = field(default_factory=list)
    macro_bias: float = 0.0  # -1 to +1


@dataclass
class FearGreedIndex:
    value: float
    emotion: MarketEmotion
    components: Dict[str, float]
    timestamp: datetime


@dataclass
class BehaviorPattern:
    name: str
    pattern_type: PatternType
    strength: float
    description: str
    signal: str
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class LiquidityMetrics:
    ticker: str
    liquidity_score: float
    bid_ask_spread_pct: float
    avg_daily_volume: float
    slippage_estimate_pct: float
    can_trade_size: int
    depth_score: float = 50.0


@dataclass 
class SectorAnalysis:
    sector: str
    performance_1d: float
    performance_5d: float
    performance_20d: float
    relative_strength: float
    flow_direction: str  # "inflow", "outflow", "neutral"
    top_ticker: str
    momentum_score: float


@dataclass
class MacroSignal:
    event: str
    date: str
    impact: str  # "high", "medium", "low"
    direction: str  # "bullish", "bearish", "neutral"
    affected_sectors: List[str]


# ============================================================
# TRADING BRAIN - MAIN CLASS
# ============================================================

class TradingBrain:
    """Главный класс - мозг торговой системы v5.0"""
    
    def __init__(self):
        self.pg = None
        self.redis = None
        self.config = GlobalConfig()
        self.last_retrain_time = None
        self.fear_greed_history = deque(maxlen=200)
        self._correlation_cache = {}
        self._liquidity_cache = {}
        self._sector_performance = {}
        self._pattern_history = defaultdict(list)
        self._anomaly_scores = {}
    
    async def start(self):
        self.pg = await asyncpg.create_pool(
            os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"),
            min_size=2, max_size=10
        )
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True
        )
        
        await self._ensure_tables()
        await self._load_config()
        
        logger.info("✅ Trading Brain v5.0 initialized")
    
    async def _ensure_tables(self):
        async with self.pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_daily (
                    date DATE PRIMARY KEY,
                    usd_rate FLOAT, eur_rate FLOAT, cny_rate FLOAT,
                    key_rate FLOAT, usd_change_1d FLOAT, usd_change_5d FLOAT,
                    brent_price FLOAT, gold_price FLOAT,
                    updated_at TIMESTAMPTZ
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id VARCHAR(50) PRIMARY KEY,
                    title TEXT, published TIMESTAMPTZ, sentiment FLOAT,
                    importance VARCHAR(20) DEFAULT 'normal'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_tickers (
                    news_id VARCHAR(50), ticker VARCHAR(20),
                    PRIMARY KEY (news_id, ticker)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment_daily (
                    date DATE, ticker VARCHAR(20),
                    sentiment_avg FLOAT, news_count INTEGER, updated_at TIMESTAMPTZ,
                    PRIMARY KEY (date, ticker)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE, ticker VARCHAR(20),
                    predicted_class INTEGER, predicted_proba FLOAT,
                    actual_class INTEGER, actual_return FLOAT,
                    model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    date DATE, accuracy FLOAT, f1_score FLOAT,
                    sharpe_ratio FLOAT, max_drawdown FLOAT,
                    predictions_count INTEGER, model_version VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS retrain_history (
                    id SERIAL PRIMARY KEY,
                    triggered_at TIMESTAMPTZ DEFAULT NOW(),
                    reason VARCHAR(100),
                    old_accuracy FLOAT, new_accuracy FLOAT,
                    model_version VARCHAR(50)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS config_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    config JSONB, reason VARCHAR(200)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS fear_greed_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    value FLOAT, emotion VARCHAR(20),
                    components JSONB
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_regime_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    regime VARCHAR(50), indicators JSONB
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_rotation (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    sector VARCHAR(50),
                    performance_1d FLOAT, performance_5d FLOAT, performance_20d FLOAT,
                    relative_strength FLOAT, flow_direction VARCHAR(20),
                    UNIQUE (date, sector)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns_detected (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ticker VARCHAR(20),
                    pattern_type VARCHAR(50),
                    strength FLOAT, signal VARCHAR(10),
                    price_target FLOAT, stop_loss FLOAT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_events (
                    id SERIAL PRIMARY KEY,
                    event_date DATE,
                    event_type VARCHAR(50),
                    description TEXT,
                    impact VARCHAR(20),
                    direction VARCHAR(20)
                )
            """)
        logger.info("✅ Database tables ensured")
    
    async def _load_config(self):
        try:
            data = await self.redis.get("global_config")
            if data:
                d = json.loads(data)
                self.config.trading = TradingConfig(**d.get('trading', {}))
                self.config.regime = MarketRegime(d.get('regime', 'sideways'))
                self.config.fear_greed = d.get('fear_greed', 50)
                self.config.active_sectors = d.get('active_sectors', [])
                self.config.macro_bias = d.get('macro_bias', 0)
                logger.info(f"📥 Config loaded: regime={self.config.regime.value}")
        except Exception as e:
            logger.warning(f"Load config failed: {e}, using defaults")
    
    async def save_config(self, reason: str = "auto"):
        self.config.updated_at = datetime.now().isoformat()
        
        config_dict = {
            'trading': asdict(self.config.trading),
            'regime': self.config.regime.value,
            'fear_greed': self.config.fear_greed,
            'volatility_percentile': self.config.volatility_percentile,
            'updated_at': self.config.updated_at,
            'corrections_count': self.config.corrections_count,
            'active_sectors': self.config.active_sectors,
            'macro_bias': self.config.macro_bias
        }
        
        await self.redis.set("global_config", json.dumps(config_dict))
        
        async with self.pg.acquire() as conn:
            await conn.execute(
                "INSERT INTO config_history (config, reason) VALUES ($1, $2)",
                json.dumps(config_dict), reason
            )
        
        logger.info(f"💾 Config saved: {reason}")

    # ============================================================
    # DATA COLLECTION
    # ============================================================
    
    async def load_cbr_macro(self):
        """Загрузка макро-данных ЦБ РФ + commodities"""
        logger.info("🏦 Loading CBR macro...")
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                end, start = datetime.now(), datetime.now() - timedelta(days=30)
                currencies = {}
                
                for cur, code in CURRENCY_CODES.items():
                    try:
                        resp = await client.get(
                            "https://www.cbr.ru/scripts/XML_dynamic.asp",
                            params={
                                'date_req1': start.strftime('%d/%m/%Y'),
                                'date_req2': end.strftime('%d/%m/%Y'),
                                'VAL_NM_RQ': code
                            }
                        )
                        for rec in ET.fromstring(resp.text).findall('.//Record'):
                            d = datetime.strptime(rec.get('Date'), '%d.%m.%Y').date()
                            if d not in currencies:
                                currencies[d] = {}
                            currencies[d][cur.lower()] = float(rec.find('Value').text.replace(',', '.'))
                    except Exception as e:
                        logger.warning(f"Currency {cur}: {e}")
                
                # Fallback to daily API
                if not currencies:
                    resp = await client.get("https://www.cbr-xml-daily.ru/daily_json.js")
                    data = resp.json()
                    currencies[datetime.now().date()] = {
                        'usd': data['Valute']['USD']['Value'],
                        'eur': data['Valute']['EUR']['Value'],
                        'cny': data['Valute']['CNY']['Value']
                    }
                
                # Load Brent price (proxy via MOEX futures)
                brent_price = None
                try:
                    resp = await client.get("https://iss.moex.com/iss/engines/futures/markets/forts/securities/BRF6.json")
                    data = resp.json()
                    if data.get('marketdata', {}).get('data'):
                        for row in data['marketdata']['data']:
                            if row[0]:  # LAST price
                                brent_price = float(row[8]) if row[8] else None
                                break
                except:
                    pass
                
                async with self.pg.acquire() as conn:
                    for d, rates in currencies.items():
                        await conn.execute("""
                            INSERT INTO macro_daily (date, usd_rate, eur_rate, cny_rate, key_rate, brent_price, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, NOW())
                            ON CONFLICT (date) DO UPDATE SET
                                usd_rate=EXCLUDED.usd_rate, eur_rate=EXCLUDED.eur_rate,
                                cny_rate=EXCLUDED.cny_rate, key_rate=EXCLUDED.key_rate,
                                brent_price=EXCLUDED.brent_price, updated_at=NOW()
                        """, d, rates.get('usd'), rates.get('eur'), rates.get('cny'), CBR_KEY_RATE, brent_price)
                    
                    await conn.execute("""
                        UPDATE macro_daily m SET
                            usd_change_1d = m.usd_rate - COALESCE(
                                (SELECT usd_rate FROM macro_daily WHERE date < m.date ORDER BY date DESC LIMIT 1), m.usd_rate),
                            usd_change_5d = m.usd_rate - COALESCE(
                                (SELECT usd_rate FROM macro_daily WHERE date <= m.date - 5 ORDER BY date DESC LIMIT 1), m.usd_rate)
                        WHERE m.date >= CURRENT_DATE - 30
                    """)
            
            JOBS_TOTAL.labels(job='cbr_macro', status='success').inc()
            logger.info(f"✅ CBR macro: {len(currencies)} days, key_rate={CBR_KEY_RATE}%")
        except Exception as e:
            JOBS_TOTAL.labels(job='cbr_macro', status='failed').inc()
            logger.error(f"❌ CBR macro: {e}")
    
    async def load_moex_news(self):
        """Загрузка новостей MOEX с важностью"""
        logger.info("📰 Loading MOEX news...")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                news = []
                for start in range(0, 200, 50):
                    resp = await client.get("https://iss.moex.com/iss/sitenews.json", params={'start': start})
                    data = resp.json()
                    rows = data.get('sitenews', {}).get('data', [])
                    if not rows:
                        break
                    
                    for row in rows:
                        item = dict(zip(data['sitenews']['columns'], row))
                        title = item.get('title', '')
                        tickers = [t for t in re.findall(r'\b([A-Z]{4})\b', title.upper()) if t in KNOWN_TICKERS]
                        
                        # Enhanced sentiment
                        pos = sum(1 for w in POSITIVE_WORDS if w in title.lower())
                        neg = sum(1 for w in NEGATIVE_WORDS if w in title.lower())
                        
                        # Importance detection
                        importance = 'normal'
                        high_importance_words = ['дивиденд', 'buyback', 'выкуп', 'слияние', 'поглощение', 
                                                  'санкции', 'делистинг', 'ipo', 'spo']
                        if any(w in title.lower() for w in high_importance_words):
                            importance = 'high'
                        
                        try:
                            published = dateparser.parse(item.get('published_at', '')) or datetime.now()
                        except:
                            published = datetime.now()
                        
                        news.append({
                            'id': str(item.get('id')),
                            'title': title[:500],
                            'published': published,
                            'sentiment': (pos - neg) / (pos + neg) if pos + neg else 0,
                            'importance': importance,
                            'tickers': tickers
                        })
                
                async with self.pg.acquire() as conn:
                    inserted = 0
                    for n in news:
                        r = await conn.execute(
                            "INSERT INTO news (id, title, published, sentiment, importance) VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING",
                            n['id'], n['title'], n['published'], n['sentiment'], n['importance']
                        )
                        if 'INSERT' in r:
                            inserted += 1
                        for t in n['tickers']:
                            await conn.execute(
                                "INSERT INTO news_tickers (news_id, ticker) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                                n['id'], t
                            )
                    
                    await conn.execute("""
                        INSERT INTO news_sentiment_daily (date, ticker, sentiment_avg, news_count, updated_at)
                        SELECT DATE(n.published), nt.ticker, AVG(n.sentiment), COUNT(*), NOW()
                        FROM news n JOIN news_tickers nt ON n.id = nt.news_id
                        WHERE n.published > NOW() - INTERVAL '7 days'
                        GROUP BY DATE(n.published), nt.ticker
                        ON CONFLICT (date, ticker) DO UPDATE SET
                            sentiment_avg=EXCLUDED.sentiment_avg, news_count=EXCLUDED.news_count, updated_at=NOW()
                    """)
            
            JOBS_TOTAL.labels(job='moex_news', status='success').inc()
            logger.info(f"✅ MOEX news: {len(news)} fetched, {inserted} new")
        except Exception as e:
            JOBS_TOTAL.labels(job='moex_news', status='failed').inc()
            logger.error(f"❌ MOEX news: {e}")
    
    async def collect_features(self):
        """Сбор фичей для обучения"""
        logger.info("📊 Collecting features...")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                for ticker in KNOWN_TICKERS:
                    try:
                        resp = await client.get(f"http://datafeed:8006/history/{ticker}?days=5")
                        if resp.status_code != 200:
                            continue
                        data = resp.json()
                        if not data:
                            continue
                        
                        async with self.pg.acquire() as conn:
                            for candle in data:
                                await conn.execute("""
                                    INSERT INTO features (date, ticker, open, high, low, close, volume, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                                    ON CONFLICT (date, ticker) DO UPDATE SET
                                        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                                        close=EXCLUDED.close, volume=EXCLUDED.volume, updated_at=NOW()
                                """, candle.get('date'), ticker,
                                    candle.get('open'), candle.get('high'),
                                    candle.get('low'), candle.get('close'),
                                    candle.get('volume'))
                    except:
                        pass
                
                async with self.pg.acquire() as conn:
                    count = await conn.fetchval("SELECT COUNT(*) FROM features")
                    FEATURES_COUNT.set(count or 0)
            
            JOBS_TOTAL.labels(job='collect_features', status='success').inc()
            logger.info("✅ Features collected")
        except Exception as e:
            JOBS_TOTAL.labels(job='collect_features', status='failed').inc()
            logger.error(f"❌ Collect features: {e}")

    # ============================================================
    # FEAR & GREED INDEX v2.0 (8 components)
    # ============================================================
    
    async def calculate_fear_greed(self) -> FearGreedIndex:
        """Расчёт индекса страха и жадности (8 компонентов)"""
        components = {
            'momentum': 50.0,
            'volatility': 50.0,
            'breadth': 50.0,
            'volume': 50.0,
            'safe_haven': 50.0,
            'news': 50.0,
            'put_call_proxy': 50.0,  # New: approximation via volatility skew
            'junk_bond_proxy': 50.0  # New: risk appetite via sector spread
        }
        
        try:
            async with self.pg.acquire() as conn:
                # 1. Momentum (20%)
                row = await conn.fetchrow("""
                    WITH recent AS (
                        SELECT ticker, close,
                               AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS 125 PRECEDING) as ma125
                        FROM features WHERE date >= CURRENT_DATE - 130
                    )
                    SELECT AVG(CASE WHEN close > ma125 THEN 1.0 ELSE 0.0 END) as above_ma
                    FROM recent WHERE ma125 IS NOT NULL
                """)
                if row and row['above_ma'] is not None:
                    components['momentum'] = to_float(row['above_ma'], 0.5) * 100
                
                # 2. Volatility (15%)
                row = await conn.fetchrow("""
                    SELECT AVG(volatility_20) as current_vol,
                           (SELECT AVG(volatility_20) FROM features 
                            WHERE date BETWEEN CURRENT_DATE - 90 AND CURRENT_DATE - 30) as avg_vol
                    FROM features WHERE date >= CURRENT_DATE - 5
                """)
                if row and row['current_vol'] and row['avg_vol']:
                    ratio = to_float(row['current_vol']) / to_float(row['avg_vol'], 1)
                    components['volatility'] = 100 - min(100, max(0, (ratio - 0.5) * 80))
                
                # 3. Market Breadth (15%)
                row = await conn.fetchrow("""
                    SELECT SUM(CASE WHEN close > sma_20 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as breadth
                    FROM features WHERE date = (SELECT MAX(date) FROM features) AND sma_20 IS NOT NULL
                """)
                if row and row['breadth'] is not None:
                    components['breadth'] = to_float(row['breadth'], 0.5) * 100
                
                # 4. Volume Sentiment (10%)
                row = await conn.fetchrow("""
                    SELECT SUM(CASE WHEN return_1d > 0 THEN volume ELSE 0 END) as up_vol,
                           SUM(CASE WHEN return_1d < 0 THEN volume ELSE 0 END) as down_vol
                    FROM features WHERE date >= CURRENT_DATE - 5
                """)
                if row and row['up_vol'] and row['down_vol']:
                    up = to_float(row['up_vol'])
                    down = to_float(row['down_vol'])
                    total = up + down
                    if total > 0:
                        components['volume'] = (up / total) * 100
                
                # 5. Safe Haven Demand (10%)
                row = await conn.fetchrow("SELECT usd_change_5d FROM macro_daily ORDER BY date DESC LIMIT 1")
                if row and row['usd_change_5d'] is not None:
                    change = to_float(row['usd_change_5d'])
                    components['safe_haven'] = max(0.0, min(100.0, 50 - change * 10))
                
                # 6. News Sentiment (10%)
                row = await conn.fetchrow("""
                    SELECT AVG(sentiment_avg) as avg_sent FROM news_sentiment_daily WHERE date >= CURRENT_DATE - 7
                """)
                if row and row['avg_sent'] is not None:
                    components['news'] = (to_float(row['avg_sent']) + 1) * 50
                
                # 7. Put/Call Proxy - volatility skew (10%)
                row = await conn.fetchrow("""
                    WITH vol_data AS (
                        SELECT ticker, volatility_20,
                               LAG(volatility_20, 5) OVER (PARTITION BY ticker ORDER BY date) as vol_5d_ago
                        FROM features WHERE date >= CURRENT_DATE - 10
                    )
                    SELECT AVG(CASE WHEN volatility_20 > vol_5d_ago THEN 0 ELSE 1 END) as vol_declining
                    FROM vol_data WHERE vol_5d_ago IS NOT NULL
                """)
                if row and row['vol_declining'] is not None:
                    components['put_call_proxy'] = to_float(row['vol_declining'], 0.5) * 100
                
                # 8. Risk Appetite Proxy - sector spread (10%)
                row = await conn.fetchrow("""
                    WITH sector_perf AS (
                        SELECT 
                            CASE 
                                WHEN ticker IN ('SBER', 'VTBR', 'TCSG') THEN 'risky'
                                WHEN ticker IN ('GAZP', 'LKOH', 'ROSN') THEN 'defensive'
                            END as sector_type,
                            AVG(return_1d) as avg_return
                        FROM features 
                        WHERE date >= CURRENT_DATE - 5 
                        AND ticker IN ('SBER', 'VTBR', 'TCSG', 'GAZP', 'LKOH', 'ROSN')
                        GROUP BY sector_type
                    )
                    SELECT 
                        (SELECT avg_return FROM sector_perf WHERE sector_type = 'risky') -
                        (SELECT avg_return FROM sector_perf WHERE sector_type = 'defensive') as spread
                """)
                if row and row['spread'] is not None:
                    spread = to_float(row['spread'])
                    components['junk_bond_proxy'] = min(100, max(0, 50 + spread * 500))
        
        except Exception as e:
            logger.warning(f"Fear&Greed calc error: {e}, using defaults")
        
        # Weighted average (8 components)
        weights = {
            'momentum': 0.20, 
            'volatility': 0.15, 
            'breadth': 0.15, 
            'volume': 0.10, 
            'safe_haven': 0.10, 
            'news': 0.10,
            'put_call_proxy': 0.10,
            'junk_bond_proxy': 0.10
        }
        value = sum(float(components.get(k, 50)) * w for k, w in weights.items())
        
        # Determine emotion
        if value <= 20:
            emotion = MarketEmotion.EXTREME_FEAR
        elif value <= 40:
            emotion = MarketEmotion.FEAR
        elif value <= 60:
            emotion = MarketEmotion.NEUTRAL
        elif value <= 80:
            emotion = MarketEmotion.GREED
        else:
            emotion = MarketEmotion.EXTREME_GREED
        
        fg = FearGreedIndex(value=round(value, 1), emotion=emotion, components=components, timestamp=datetime.now())
        
        FEAR_GREED_INDEX.set(value)
        self.config.fear_greed = value
        self.fear_greed_history.append(fg)
        
        try:
            async with self.pg.acquire() as conn:
                await conn.execute(
                    "INSERT INTO fear_greed_history (value, emotion, components) VALUES ($1, $2, $3)",
                    value, emotion.value, json.dumps(components)
                )
        except:
            pass
        
        return fg

    # ============================================================
    # PATTERN DETECTION v2.0 (10+ patterns)
    # ============================================================
    
    async def detect_patterns(self, ticker: str) -> List[BehaviorPattern]:
        """Обнаружение паттернов поведения (расширенный набор)"""
        patterns = []
        
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT date, open, high, low, close, volume, return_1d, volatility_20, volume_ratio, rsi_14
                    FROM features WHERE ticker = $1 AND date >= CURRENT_DATE - 60 ORDER BY date
                """, ticker)
                
                if len(rows) < 20:
                    return patterns
                
                df = pd.DataFrame([dict(r) for r in rows])
                for col in ['open', 'high', 'low', 'close', 'volume', 'return_1d', 'volatility_20', 'volume_ratio', 'rsi_14']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: to_float(x) if x is not None else None)
                
                df = df.dropna(subset=['close'])
                if len(df) < 20:
                    return patterns
                
                last = df.iloc[-1]
                closes = df['close'].values
                highs = df['high'].values if 'high' in df else closes
                lows = df['low'].values if 'low' in df else closes
                volumes = df['volume'].values if 'volume' in df else np.ones(len(closes))
                
                # === BEHAVIORAL PATTERNS ===
                
                # 1. Panic Selling
                if last.get('return_1d') and last['return_1d'] < -0.05 and last.get('volume_ratio') and last['volume_ratio'] > 3:
                    strength = min(1.0, abs(last['return_1d']) * 10)
                    patterns.append(BehaviorPattern(
                        name="panic_selling", pattern_type=PatternType.PANIC_SELLING, strength=strength,
                        description=f"Паника: -{abs(last['return_1d'])*100:.1f}% на объёме {last['volume_ratio']:.1f}x",
                        signal="buy", confidence=0.6 * strength,
                        stop_loss=last['close'] * 0.95
                    ))
                
                # 2. FOMO Buying
                if len(df) >= 5:
                    five_day_return = (closes[-1] / closes[-5]) - 1 if closes[-5] != 0 else 0
                    today_return = last.get('return_1d', 0) or 0
                    vol_ratio = last.get('volume_ratio', 1) or 1
                    
                    if five_day_return > 0.10 and today_return > 0.03 and vol_ratio > 2.5:
                        strength = min(1.0, five_day_return * 5)
                        patterns.append(BehaviorPattern(
                            name="fomo_buying", pattern_type=PatternType.FOMO_BUYING, strength=strength,
                            description=f"FOMO: +{five_day_return*100:.1f}% за 5д, объём {vol_ratio:.1f}x",
                            signal="sell", confidence=0.5 * strength
                        ))
                
                # 3. Capitulation
                if len(df) >= 10:
                    ten_day_return = (closes[-1] / closes[-10]) - 1 if closes[-10] != 0 else 0
                    rsi = last.get('rsi_14', 50) or 50
                    
                    if ten_day_return < -0.15 and rsi < 25:
                        strength = min(1.0, abs(ten_day_return) * 4)
                        patterns.append(BehaviorPattern(
                            name="capitulation", pattern_type=PatternType.CAPITULATION, strength=strength,
                            description=f"Капитуляция: {ten_day_return*100:.1f}% за 10д, RSI={rsi:.0f}",
                            signal="buy", confidence=0.7 * strength,
                            price_target=closes[-1] * 1.15
                        ))
                
                # === TECHNICAL PATTERNS ===
                
                # 4. Double Bottom
                if len(df) >= 30:
                    window = lows[-30:]
                    min_idx = np.argmin(window)
                    if 5 < min_idx < 25:
                        left_min = np.min(window[:min_idx])
                        right_min = np.min(window[min_idx+3:]) if min_idx + 3 < len(window) else window[-1]
                        current = window[-1]
                        
                        if abs(left_min - right_min) / left_min < 0.03 and current > left_min * 1.02:
                            neckline = np.max(window[min_idx-3:min_idx+3])
                            strength = min(1.0, (neckline - left_min) / left_min * 10)
                            patterns.append(BehaviorPattern(
                                name="double_bottom", pattern_type=PatternType.DOUBLE_BOTTOM, strength=strength,
                                description=f"Двойное дно на {left_min:.2f}",
                                signal="buy", confidence=0.65 * strength,
                                price_target=current + (neckline - left_min),
                                stop_loss=left_min * 0.98
                            ))
                
                # 5. Double Top
                if len(df) >= 30:
                    window = highs[-30:]
                    max_idx = np.argmax(window)
                    if 5 < max_idx < 25:
                        left_max = np.max(window[:max_idx])
                        right_max = np.max(window[max_idx+3:]) if max_idx + 3 < len(window) else window[-1]
                        current = window[-1]
                        
                        if abs(left_max - right_max) / left_max < 0.03 and current < left_max * 0.98:
                            neckline = np.min(window[max_idx-3:max_idx+3])
                            strength = min(1.0, (left_max - neckline) / neckline * 10)
                            patterns.append(BehaviorPattern(
                                name="double_top", pattern_type=PatternType.DOUBLE_TOP, strength=strength,
                                description=f"Двойная вершина на {left_max:.2f}",
                                signal="sell", confidence=0.65 * strength,
                                price_target=current - (left_max - neckline)
                            ))
                
                # 6. Head & Shoulders (simplified)
                if len(df) >= 40:
                    window = highs[-40:]
                    peaks = []
                    for i in range(2, len(window) - 2):
                        if window[i] > window[i-1] and window[i] > window[i-2] and \
                           window[i] > window[i+1] and window[i] > window[i+2]:
                            peaks.append((i, window[i]))
                    
                    if len(peaks) >= 3:
                        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:3]
                        peaks = sorted(peaks, key=lambda x: x[0])
                        
                        if len(peaks) == 3:
                            left, head, right = peaks
                            if head[1] > left[1] and head[1] > right[1] and abs(left[1] - right[1]) / left[1] < 0.05:
                                neckline = min(lows[left[0]:right[0]+1])
                                if closes[-1] < neckline:
                                    strength = min(1.0, (head[1] - neckline) / neckline * 5)
                                    patterns.append(BehaviorPattern(
                                        name="head_shoulders", pattern_type=PatternType.HEAD_SHOULDERS, strength=strength,
                                        description=f"Голова и плечи, neckline={neckline:.2f}",
                                        signal="sell", confidence=0.7 * strength,
                                        price_target=neckline - (head[1] - neckline)
                                    ))
                
                # 7. Inverse Head & Shoulders
                if len(df) >= 40:
                    window = lows[-40:]
                    troughs = []
                    for i in range(2, len(window) - 2):
                        if window[i] < window[i-1] and window[i] < window[i-2] and \
                           window[i] < window[i+1] and window[i] < window[i+2]:
                            troughs.append((i, window[i]))
                    
                    if len(troughs) >= 3:
                        troughs = sorted(troughs, key=lambda x: x[1])[:3]
                        troughs = sorted(troughs, key=lambda x: x[0])
                        
                        if len(troughs) == 3:
                            left, head, right = troughs
                            if head[1] < left[1] and head[1] < right[1] and abs(left[1] - right[1]) / left[1] < 0.05:
                                neckline = max(highs[left[0]:right[0]+1])
                                if closes[-1] > neckline:
                                    strength = min(1.0, (neckline - head[1]) / head[1] * 5)
                                    patterns.append(BehaviorPattern(
                                        name="inv_head_shoulders", pattern_type=PatternType.INV_HEAD_SHOULDERS, strength=strength,
                                        description=f"Перевёрнутая голова и плечи, neckline={neckline:.2f}",
                                        signal="buy", confidence=0.7 * strength,
                                        price_target=neckline + (neckline - head[1])
                                    ))
                
                # 8. Resistance Breakout
                if len(df) >= 20:
                    resistance = np.percentile(highs[-20:], 95)
                    if closes[-1] > resistance and closes[-2] <= resistance:
                        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
                        if vol_ratio > 1.5:
                            strength = min(1.0, vol_ratio / 3)
                            patterns.append(BehaviorPattern(
                                name="resistance_breakout", pattern_type=PatternType.RESISTANCE_BREAKOUT, strength=strength,
                                description=f"Пробой сопротивления {resistance:.2f} на объёме {vol_ratio:.1f}x",
                                signal="buy", confidence=0.6 * strength,
                                stop_loss=resistance * 0.98
                            ))
                
                # 9. Support Breakdown
                if len(df) >= 20:
                    support = np.percentile(lows[-20:], 5)
                    if closes[-1] < support and closes[-2] >= support:
                        vol_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
                        if vol_ratio > 1.5:
                            strength = min(1.0, vol_ratio / 3)
                            patterns.append(BehaviorPattern(
                                name="support_breakdown", pattern_type=PatternType.SUPPORT_BREAKDOWN, strength=strength,
                                description=f"Пробой поддержки {support:.2f} на объёме {vol_ratio:.1f}x",
                                signal="sell", confidence=0.6 * strength
                            ))
                
                # 10. Volume Breakout (accumulation)
                if len(df) >= 10:
                    avg_vol = np.mean(volumes[-10:-1])
                    if avg_vol > 0 and volumes[-1] > avg_vol * 3:
                        price_change = (closes[-1] / closes[-2]) - 1 if closes[-2] != 0 else 0
                        if abs(price_change) < 0.02:  # Low price change but huge volume
                            patterns.append(BehaviorPattern(
                                name="volume_breakout", pattern_type=PatternType.VOLUME_BREAKOUT, strength=0.7,
                                description=f"Аномальный объём {volumes[-1]/avg_vol:.1f}x без движения цены",
                                signal="buy" if last.get('return_1d', 0) >= 0 else "sell",
                                confidence=0.5
                            ))
                
                # 11. Bullish Divergence (price down, RSI up)
                if len(df) >= 14:
                    price_trend = (closes[-1] - closes[-14]) / closes[-14] if closes[-14] != 0 else 0
                    rsi_values = df['rsi_14'].dropna().values
                    if len(rsi_values) >= 14:
                        rsi_trend = rsi_values[-1] - rsi_values[-14] if len(rsi_values) >= 14 else 0
                        if price_trend < -0.05 and rsi_trend > 10:
                            strength = min(1.0, abs(price_trend) * 10)
                            patterns.append(BehaviorPattern(
                                name="bullish_divergence", pattern_type=PatternType.DIVERGENCE_BULLISH, strength=strength,
                                description=f"Бычья дивергенция: цена {price_trend*100:.1f}%, RSI +{rsi_trend:.0f}",
                                signal="buy", confidence=0.55 * strength
                            ))
                
                # 12. Bearish Divergence (price up, RSI down)
                if len(df) >= 14:
                    price_trend = (closes[-1] - closes[-14]) / closes[-14] if closes[-14] != 0 else 0
                    rsi_values = df['rsi_14'].dropna().values
                    if len(rsi_values) >= 14:
                        rsi_trend = rsi_values[-1] - rsi_values[-14] if len(rsi_values) >= 14 else 0
                        if price_trend > 0.05 and rsi_trend < -10:
                            strength = min(1.0, price_trend * 10)
                            patterns.append(BehaviorPattern(
                                name="bearish_divergence", pattern_type=PatternType.DIVERGENCE_BEARISH, strength=strength,
                                description=f"Медвежья дивергенция: цена +{price_trend*100:.1f}%, RSI {rsi_trend:.0f}",
                                signal="sell", confidence=0.55 * strength
                            ))
                
                # 13. Smart Money Accumulation
                if len(df) >= 20:
                    price_range = (np.max(closes[-20:]) - np.min(closes[-20:])) / np.mean(closes[-20:])
                    vol_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:-5]) if np.mean(volumes[-20:-5]) > 0 else 1
                    
                    if price_range < 0.05 and vol_trend > 1.5:
                        patterns.append(BehaviorPattern(
                            name="smart_accumulation", pattern_type=PatternType.SMART_MONEY_ACCUMULATION, strength=0.6,
                            description=f"Накопление: узкий диапазон {price_range*100:.1f}%, растущий объём",
                            signal="buy", confidence=0.5
                        ))
                
                # Save patterns to DB
                for p in patterns:
                    try:
                        await conn.execute("""
                            INSERT INTO patterns_detected (ticker, pattern_type, strength, signal, price_target, stop_loss)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, ticker, p.pattern_type.value, p.strength, p.signal, p.price_target, p.stop_loss)
                        PATTERNS_DETECTED.labels(pattern=p.pattern_type.value).inc()
                    except:
                        pass
                
        except Exception as e:
            logger.warning(f"Pattern detection error for {ticker}: {e}")
        
        return patterns

    # ============================================================
    # SECTOR ROTATION
    # ============================================================
    
    async def analyze_sector_rotation(self) -> Dict[str, SectorAnalysis]:
        """Анализ ротации секторов"""
        logger.info("🔄 Analyzing sector rotation...")
        results = {}
        
        try:
            async with self.pg.acquire() as conn:
                for sector, tickers in SECTORS.items():
                    tickers_str = "','".join(tickers)
                    
                    row = await conn.fetchrow(f"""
                        WITH sector_data AS (
                            SELECT date, ticker, close, volume,
                                   LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_1d,
                                   LAG(close, 5) OVER (PARTITION BY ticker ORDER BY date) as prev_5d,
                                   LAG(close, 20) OVER (PARTITION BY ticker ORDER BY date) as prev_20d
                            FROM features
                            WHERE ticker IN ('{tickers_str}')
                            AND date >= CURRENT_DATE - 25
                        )
                        SELECT 
                            AVG((close - prev_1d) / NULLIF(prev_1d, 0)) as perf_1d,
                            AVG((close - prev_5d) / NULLIF(prev_5d, 0)) as perf_5d,
                            AVG((close - prev_20d) / NULLIF(prev_20d, 0)) as perf_20d,
                            AVG(volume) as avg_volume
                        FROM sector_data
                        WHERE date = (SELECT MAX(date) FROM sector_data)
                    """)
                    
                    if row:
                        perf_1d = to_float(row['perf_1d'], 0) * 100
                        perf_5d = to_float(row['perf_5d'], 0) * 100
                        perf_20d = to_float(row['perf_20d'], 0) * 100
                        
                        # Momentum score
                        momentum = perf_1d * 0.5 + perf_5d * 0.3 + perf_20d * 0.2
                        
                        # Flow direction
                        if perf_5d > 2 and perf_1d > 0:
                            flow = "inflow"
                        elif perf_5d < -2 and perf_1d < 0:
                            flow = "outflow"
                        else:
                            flow = "neutral"
                        
                        # Find top ticker in sector
                        top_row = await conn.fetchrow(f"""
                            SELECT ticker, (close - LAG(close, 5) OVER (ORDER BY date)) / NULLIF(LAG(close, 5) OVER (ORDER BY date), 0) as ret
                            FROM features
                            WHERE ticker IN ('{tickers_str}')
                            AND date = (SELECT MAX(date) FROM features WHERE ticker IN ('{tickers_str}'))
                            ORDER BY ret DESC
                            LIMIT 1
                        """)
                        
                        results[sector] = SectorAnalysis(
                            sector=sector,
                            performance_1d=round(perf_1d, 2),
                            performance_5d=round(perf_5d, 2),
                            performance_20d=round(perf_20d, 2),
                            relative_strength=round(momentum, 2),
                            flow_direction=flow,
                            top_ticker=top_row['ticker'] if top_row else tickers[0],
                            momentum_score=round(momentum, 2)
                        )
                        
                        SECTOR_ROTATION.labels(sector=sector).set(momentum)
                        
                        # Save to DB
                        await conn.execute("""
                            INSERT INTO sector_rotation (date, sector, performance_1d, performance_5d, performance_20d, relative_strength, flow_direction)
                            VALUES (CURRENT_DATE, $1, $2, $3, $4, $5, $6)
                            ON CONFLICT (date, sector) DO UPDATE SET
                                performance_1d=EXCLUDED.performance_1d, performance_5d=EXCLUDED.performance_5d,
                                performance_20d=EXCLUDED.performance_20d, relative_strength=EXCLUDED.relative_strength,
                                flow_direction=EXCLUDED.flow_direction
                        """, sector, perf_1d, perf_5d, perf_20d, momentum, flow)
                
                # Determine active sectors (top 3 by momentum)
                sorted_sectors = sorted(results.items(), key=lambda x: x[1].momentum_score, reverse=True)
                self.config.active_sectors = [s[0] for s in sorted_sectors[:3]]
                
                self._sector_performance = results
                
        except Exception as e:
            logger.error(f"Sector rotation error: {e}")
        
        JOBS_TOTAL.labels(job='sector_rotation', status='success').inc()
        logger.info(f"✅ Sector rotation: active={self.config.active_sectors}")
        
        return results

    # ============================================================
    # MACRO SIGNALS
    # ============================================================
    
    async def check_macro_events(self) -> List[MacroSignal]:
        """Проверка макро-событий"""
        signals = []
        today = datetime.now().date()
        
        # Check CBR meetings
        for meeting_date in CBR_MEETINGS_2026:
            meeting = datetime.strptime(meeting_date, '%Y-%m-%d').date()
            days_until = (meeting - today).days
            
            if 0 <= days_until <= 3:
                signals.append(MacroSignal(
                    event="CBR_MEETING",
                    date=meeting_date,
                    impact="high",
                    direction="neutral",  # Unknown until decision
                    affected_sectors=['banks', 'retail']
                ))
                
                # Adjust config before meeting
                if days_until <= 1:
                    self.config.trading.max_position_pct = min(self.config.trading.max_position_pct, 0.07)
                    self.config.trading.min_confidence = max(self.config.trading.min_confidence, 0.50)
        
        # Check for dividend season (May-July)
        if today.month in [5, 6, 7]:
            signals.append(MacroSignal(
                event="DIVIDEND_SEASON",
                date=today.isoformat(),
                impact="medium",
                direction="bullish",
                affected_sectors=['oil_gas', 'metals', 'banks']
            ))
        
        # Check macro from DB
        try:
            async with self.pg.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT usd_change_5d, brent_price FROM macro_daily ORDER BY date DESC LIMIT 1
                """)
                
                if row:
                    usd_change = to_float(row['usd_change_5d'], 0)
                    
                    # Strong ruble move
                    if abs(usd_change) > 3:
                        signals.append(MacroSignal(
                            event="USD_VOLATILITY",
                            date=today.isoformat(),
                            impact="high",
                            direction="bearish" if usd_change > 0 else "bullish",
                            affected_sectors=['retail', 'tech'] if usd_change > 0 else ['oil_gas', 'metals']
                        ))
        except:
            pass
        
        # Update macro bias
        bullish_count = sum(1 for s in signals if s.direction == "bullish")
        bearish_count = sum(1 for s in signals if s.direction == "bearish")
        self.config.macro_bias = (bullish_count - bearish_count) / max(1, len(signals)) if signals else 0
        
        return signals

    # ============================================================
    # LIQUIDITY ANALYSIS
    # ============================================================
    
    async def analyze_liquidity(self, ticker: str) -> LiquidityMetrics:
        """Анализ ликвидности"""
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT date, open, high, low, close, volume
                    FROM features WHERE ticker = $1 AND date >= CURRENT_DATE - 30 ORDER BY date
                """, ticker)
                
                if not rows:
                    return LiquidityMetrics(ticker, 0, 0.01, 0, 0.01, 0)
                
                df = pd.DataFrame([dict(r) for r in rows])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: to_float(x) if x is not None else 0)
                
                df['turnover'] = df['close'] * df['volume']
                avg_daily_volume = df['turnover'].mean()
                
                df['hl_spread'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
                bid_ask_spread = df['hl_spread'].mean() * 0.3
                
                volume_score = min(100, avg_daily_volume / 1_000_000_000 * 100)
                spread_score = max(0, 100 - bid_ask_spread * 1000)
                
                # Depth score based on volume consistency
                vol_std = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1
                depth_score = max(0, 100 - vol_std * 50)
                
                liquidity_score = volume_score * 0.5 + spread_score * 0.3 + depth_score * 0.2
                
                can_trade_size = int(avg_daily_volume * 0.01)
                slippage = bid_ask_spread / 2 * max(0.5, (100 - liquidity_score) / 100 + 0.5)
                
                result = LiquidityMetrics(
                    ticker=ticker,
                    liquidity_score=round(liquidity_score, 1),
                    bid_ask_spread_pct=round(bid_ask_spread, 4),
                    avg_daily_volume=round(avg_daily_volume, 0),
                    slippage_estimate_pct=round(slippage, 4),
                    can_trade_size=can_trade_size,
                    depth_score=round(depth_score, 1)
                )
                
                self._liquidity_cache[ticker] = result
                return result
                
        except Exception as e:
            logger.warning(f"Liquidity analysis error: {e}")
            return LiquidityMetrics(ticker, 0, 0.01, 0, 0.01, 0)

    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    
    async def analyze_correlation(self, ticker: str) -> Dict:
        """Корреляционный анализ"""
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT ticker, date, return_1d FROM features
                    WHERE date >= CURRENT_DATE - 60 AND return_1d IS NOT NULL
                    ORDER BY ticker, date
                """)
                
                if not rows:
                    return {'ticker': ticker, 'correlations': {}, 'beta': 1.0}
                
                returns = {}
                for row in rows:
                    t = row['ticker']
                    if t not in returns:
                        returns[t] = []
                    returns[t].append(to_float(row['return_1d']))
                
                if ticker not in returns:
                    return {'ticker': ticker, 'correlations': {}, 'beta': 1.0}
                
                ticker_returns = np.array(returns[ticker])
                correlations = {}
                
                for other, other_returns in returns.items():
                    if other != ticker and len(other_returns) == len(ticker_returns):
                        corr = np.corrcoef(ticker_returns, np.array(other_returns))[0, 1]
                        if not np.isnan(corr):
                            correlations[other] = round(corr, 3)
                
                beta = 1.0
                if 'SBER' in returns and len(returns['SBER']) == len(ticker_returns):
                    cov = np.cov(ticker_returns, np.array(returns['SBER']))[0, 1]
                    var = np.var(returns['SBER'])
                    if var > 0:
                        beta = round(cov / var, 3)
                
                top = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
                
                result = {'ticker': ticker, 'correlations': top, 'beta': beta}
                self._correlation_cache[ticker] = result
                
                return result
                
        except Exception as e:
            logger.warning(f"Correlation analysis error: {e}")
            return {'ticker': ticker, 'correlations': {}, 'beta': 1.0}

    # ============================================================
    # SIGNAL ENHANCEMENT v2.0
    # ============================================================
    
    async def enhance_signal(self, ticker: str, base_signal: int, base_confidence: float) -> Dict:
        """Улучшение сигнала с учётом всех факторов"""
        with SIGNAL_LATENCY.time():
            adjustments = []
            confidence = base_confidence
            
            # 1. Fear & Greed adjustment
            fg = self.config.fear_greed
            if fg <= 20 and base_signal == 1:  # Extreme fear + buy
                confidence += 0.20
                adjustments.append(f"extreme_fear_buy:+20%")
            elif fg >= 80 and base_signal == -1:  # Extreme greed + sell
                confidence += 0.15
                adjustments.append(f"extreme_greed_sell:+15%")
            elif fg <= 30 and base_signal == -1:  # Fear + sell = reduce
                confidence -= 0.10
                adjustments.append(f"fear_sell:-10%")
            elif fg >= 70 and base_signal == 1:  # Greed + buy = reduce
                confidence -= 0.10
                adjustments.append(f"greed_buy:-10%")
            
            # 2. Pattern alignment
            patterns = await self.detect_patterns(ticker)
            for p in patterns:
                if (p.signal == "buy" and base_signal == 1) or (p.signal == "sell" and base_signal == -1):
                    boost = 0.30 * p.strength * p.confidence
                    confidence += boost
                    adjustments.append(f"{p.name}:+{boost*100:.0f}%")
            
            # 3. Sector momentum
            sector = None
            for s, tickers in SECTORS.items():
                if ticker in tickers:
                    sector = s
                    break
            
            if sector and sector in self._sector_performance:
                sp = self._sector_performance[sector]
                if sp.flow_direction == "inflow" and base_signal == 1:
                    confidence += 0.10
                    adjustments.append(f"sector_inflow:+10%")
                elif sp.flow_direction == "outflow" and base_signal == -1:
                    confidence += 0.10
                    adjustments.append(f"sector_outflow:+10%")
            
            # 4. Macro bias
            if self.config.macro_bias > 0.3 and base_signal == 1:
                confidence += 0.05
                adjustments.append(f"macro_bullish:+5%")
            elif self.config.macro_bias < -0.3 and base_signal == -1:
                confidence += 0.05
                adjustments.append(f"macro_bearish:+5%")
            
            # 5. Liquidity penalty
            liquidity = self._liquidity_cache.get(ticker) or await self.analyze_liquidity(ticker)
            if liquidity.liquidity_score < 30:
                confidence -= 0.20
                adjustments.append(f"low_liquidity:-20%")
            elif liquidity.liquidity_score < 50:
                confidence -= 0.10
                adjustments.append(f"med_liquidity:-10%")
            
            # 6. Regime adjustment
            regime = self.config.regime
            if regime == MarketRegime.CRISIS:
                if base_signal == 1:
                    confidence -= 0.15
                    adjustments.append(f"crisis_buy:-15%")
            elif regime == MarketRegime.TRENDING_UP and base_signal == 1:
                confidence += 0.05
                adjustments.append(f"uptrend_buy:+5%")
            elif regime == MarketRegime.TRENDING_DOWN and base_signal == -1:
                confidence += 0.05
                adjustments.append(f"downtrend_sell:+5%")
            
            # 7. Correlation check (avoid concentrated risk)
            corr_data = self._correlation_cache.get(ticker) or await self.analyze_correlation(ticker)
            high_corr_count = sum(1 for c in corr_data['correlations'].values() if abs(c) > 0.7)
            if high_corr_count > 3:
                confidence -= 0.05
                adjustments.append(f"high_correlation:-5%")
            
            # Final bounds
            final_confidence = min(0.95, max(0.05, confidence))
            
            return {
                'ticker': ticker,
                'signal': base_signal,
                'base_confidence': base_confidence,
                'enhanced_confidence': round(final_confidence, 3),
                'adjustments': adjustments,
                'patterns': [asdict(p) for p in patterns],
                'sector': sector,
                'liquidity_score': liquidity.liquidity_score,
                'fear_greed': fg,
                'regime': regime.value
            }

    # ============================================================
    # AUTO-CORRECTION
    # ============================================================
    
    async def detect_market_regime(self) -> MarketRegime:
        """Определение режима рынка"""
        try:
            async with self.pg.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT date, close, volume FROM features
                    WHERE ticker IN ('SBER', 'GAZP', 'LKOH') ORDER BY date DESC LIMIT 30
                """)
                
                if len(rows) < 15:
                    return MarketRegime.SIDEWAYS
                
                closes = [to_float(r['close']) for r in rows if r['close']]
                if not closes or len(closes) < 10:
                    return MarketRegime.SIDEWAYS
                
                returns = np.diff(closes) / np.array(closes[:-1])
                volatility = np.std(returns) * np.sqrt(252)
                trend = (closes[0] - closes[-1]) / closes[-1] if closes[-1] != 0 else 0
                
                # Volume trend
                volumes = [to_float(r['volume']) for r in rows if r['volume']]
                vol_trend = np.mean(volumes[:5]) / np.mean(volumes[10:15]) if len(volumes) >= 15 and np.mean(volumes[10:15]) > 0 else 1
                
                # Accumulation/Distribution
                if volatility < 0.2 and abs(trend) < 0.03 and vol_trend > 1.3:
                    return MarketRegime.ACCUMULATION
                if volatility < 0.2 and abs(trend) < 0.03 and vol_trend < 0.7:
                    return MarketRegime.DISTRIBUTION
                
                if volatility > 0.4:
                    return MarketRegime.CRISIS if trend < -0.1 else MarketRegime.HIGH_VOLATILITY
                if trend > 0.05:
                    return MarketRegime.TRENDING_UP
                if trend < -0.05:
                    return MarketRegime.TRENDING_DOWN
                return MarketRegime.SIDEWAYS
                
        except:
            return MarketRegime.SIDEWAYS
    
    async def auto_correct_config(self) -> Dict:
        """Автокоррекция конфигурации"""
        logger.info("🔧 Running auto-correction...")
        corrections = []
        
        new_regime = await self.detect_market_regime()
        if new_regime != self.config.regime:
            old = self.config.regime
            self.config.regime = new_regime
            corrections.append(f"regime: {old.value} → {new_regime.value}")
            
            # Adjust config based on regime
            if new_regime == MarketRegime.CRISIS:
                self.config.trading.min_confidence = 0.65
                self.config.trading.max_position_pct = 0.05
                self.config.trading.max_daily_trades = 5
                self.config.trading.stop_loss_pct = 0.03
            elif new_regime == MarketRegime.HIGH_VOLATILITY:
                self.config.trading.min_confidence = 0.55
                self.config.trading.max_position_pct = 0.07
                self.config.trading.max_daily_trades = 10
            elif new_regime == MarketRegime.TRENDING_UP:
                self.config.trading.min_confidence = 0.42
                self.config.trading.max_position_pct = 0.12
                self.config.trading.max_daily_trades = 25
                self.config.trading.trailing_stop_pct = 0.02
            elif new_regime == MarketRegime.TRENDING_DOWN:
                self.config.trading.min_confidence = 0.50
                self.config.trading.max_position_pct = 0.08
                self.config.trading.max_daily_trades = 15
            elif new_regime == MarketRegime.ACCUMULATION:
                self.config.trading.min_confidence = 0.40
                self.config.trading.max_position_pct = 0.10
            elif new_regime == MarketRegime.DISTRIBUTION:
                self.config.trading.min_confidence = 0.55
                self.config.trading.max_position_pct = 0.07
            else:  # SIDEWAYS
                self.config.trading.min_confidence = 0.45
                self.config.trading.max_position_pct = 0.10
                self.config.trading.max_daily_trades = 20
        
        # Check macro events
        macro_signals = await self.check_macro_events()
        for sig in macro_signals:
            if sig.impact == "high":
                corrections.append(f"macro_event: {sig.event}")
        
        # Update sector allocation
        await self.analyze_sector_rotation()
        if self.config.active_sectors:
            corrections.append(f"active_sectors: {self.config.active_sectors}")
        
        if corrections:
            self.config.corrections_count += 1
            await self.save_config(f"auto_correct: {', '.join(corrections[:3])}")
            
            # Save regime to DB
            try:
                async with self.pg.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO market_regime_history (regime, indicators) VALUES ($1, $2)",
                        new_regime.value, json.dumps({'corrections': corrections})
                    )
            except:
                pass
        
        MARKET_REGIME.set(list(MarketRegime).index(new_regime))
        JOBS_TOTAL.labels(job='auto_correct', status='success').inc()
        
        logger.info(f"✅ Auto-correction: {len(corrections)} changes")
        return {'corrections': corrections, 'regime': new_regime.value, 'config': asdict(self.config.trading)}

    # ============================================================
    # SELF-LEARNING
    # ============================================================
    
    async def check_model_performance(self) -> Dict:
        """Проверка качества модели"""
        try:
            async with self.pg.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT accuracy, f1_score, sharpe_ratio, max_drawdown, predictions_count
                    FROM model_metrics WHERE date >= CURRENT_DATE - 7 ORDER BY date DESC LIMIT 1
                """)
                
                if not row:
                    return {'valid': False, 'reason': 'no_metrics'}
                
                return {
                    'valid': True,
                    'accuracy': to_float(row['accuracy']),
                    'f1_score': to_float(row['f1_score']),
                    'sharpe_ratio': to_float(row['sharpe_ratio']),
                    'max_drawdown': to_float(row['max_drawdown']),
                    'predictions_count': row['predictions_count']
                }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    async def trigger_retrain(self, reason: str) -> bool:
        """Запуск переобучения"""
        if self.last_retrain_time:
            hours_since = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since < RETRAIN_COOLDOWN_HOURS:
                logger.info(f"⏳ Retrain cooldown: {RETRAIN_COOLDOWN_HOURS - hours_since:.1f}h remaining")
                return False
        
        logger.info(f"🔄 Triggering retrain: {reason}")
        
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post("http://strategy:8005/retrain", json={'reason': reason})
                
                if resp.status_code == 200:
                    result = resp.json()
                    self.last_retrain_time = datetime.now()
                    
                    MODEL_ACCURACY.set(result.get('accuracy', 0))
                    MODEL_LAST_TRAIN.set(datetime.now().timestamp())
                    
                    async with self.pg.acquire() as conn:
                        await conn.execute(
                            "INSERT INTO retrain_history (reason, new_accuracy, model_version) VALUES ($1, $2, $3)",
                            reason, result.get('accuracy'), result.get('version', 'v5')
                        )
                    
                    JOBS_TOTAL.labels(job='retrain', status='success').inc()
                    logger.info(f"✅ Retrain complete: accuracy={result.get('accuracy', 0):.1%}")
                    return True
                else:
                    JOBS_TOTAL.labels(job='retrain', status='failed').inc()
                    logger.error(f"❌ Retrain failed: {resp.status_code}")
                    return False
        except Exception as e:
            JOBS_TOTAL.labels(job='retrain', status='failed').inc()
            logger.error(f"❌ Retrain error: {e}")
            return False
    
    async def auto_retrain_check(self):
        """Автоматическая проверка и переобучение"""
        logger.info("🔍 Checking if retrain needed...")
        
        metrics = await self.check_model_performance()
        
        if not metrics.get('valid'):
            logger.info(f"   ⚠️ No valid metrics")
            return
        
        accuracy = metrics.get('accuracy', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        MODEL_ACCURACY.set(accuracy)
        
        if accuracy < MIN_ACCURACY_THRESHOLD:
            await self.trigger_retrain(f"accuracy_low_{accuracy:.2%}")
        elif sharpe < MIN_SHARPE_THRESHOLD:
            await self.trigger_retrain(f"sharpe_low_{sharpe:.2f}")
        else:
            logger.info(f"   ✅ Model OK: accuracy={accuracy:.1%}, sharpe={sharpe:.2f}")
    
    async def update_actuals(self):
        """Обновление фактических результатов"""
        logger.info("📈 Updating prediction actuals...")
        try:
            async with self.pg.acquire() as conn:
                await conn.execute("""
                    UPDATE model_predictions mp SET actual_class = f.signal_class, actual_return = f.return_5d
                    FROM features f WHERE mp.date = f.date AND mp.ticker = f.ticker
                    AND mp.actual_class IS NULL AND f.signal_class IS NOT NULL
                """)
            JOBS_TOTAL.labels(job='update_actuals', status='success').inc()
        except Exception as e:
            JOBS_TOTAL.labels(job='update_actuals', status='failed').inc()
            logger.error(f"❌ Update actuals: {e}")


# ============================================================
# FASTAPI APP
# ============================================================

brain = TradingBrain()
scheduler = AsyncIOScheduler()

app = FastAPI(title="Trading Brain v5.0", version="5.0")

@app.on_event("startup")
async def startup():
    await brain.start()
    
    # Data collection
    scheduler.add_job(brain.load_cbr_macro, CronTrigger(hour=8, minute=0), id='cbr_macro')
    scheduler.add_job(brain.load_moex_news, CronTrigger(minute='*/15'), id='moex_news')
    scheduler.add_job(brain.collect_features, CronTrigger(hour='*/2'), id='features')
    
    # Intelligence
    scheduler.add_job(brain.calculate_fear_greed, CronTrigger(minute='*/30'), id='fear_greed')
    scheduler.add_job(brain.analyze_sector_rotation, CronTrigger(hour='*/4'), id='sector_rotation')
    
    # Auto-correction
    scheduler.add_job(brain.auto_correct_config, CronTrigger(hour='8,14,20', minute=0), id='auto_correct')
    
    # Self-learning
    scheduler.add_job(brain.auto_retrain_check, CronTrigger(hour='7,13,19,1'), id='retrain_check')
    scheduler.add_job(brain.update_actuals, CronTrigger(hour=23, minute=30), id='update_actuals')
    
    # Weekly full retrain
    scheduler.add_job(lambda: asyncio.create_task(brain.trigger_retrain("weekly_scheduled")), 
                      CronTrigger(day_of_week='sun', hour=3), id='weekly_retrain')
    
    scheduler.start()
    start_http_server(9102)
    
    logger.info("✅ Trading Brain v5.0 fully started")

@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()
    if brain.pg:
        await brain.pg.close()
    if brain.redis:
        await brain.redis.close()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "5.0",
        "regime": brain.config.regime.value,
        "fear_greed": round(brain.config.fear_greed, 1)
    }

@app.get("/config")
async def get_config():
    return {
        **asdict(brain.config.trading),
        'regime': brain.config.regime.value,
        'fear_greed': brain.config.fear_greed,
        'active_sectors': brain.config.active_sectors,
        'macro_bias': brain.config.macro_bias,
        'updated_at': brain.config.updated_at
    }

@app.get("/sentiment")
async def sentiment():
    fg = brain.fear_greed_history[-1] if brain.fear_greed_history else await brain.calculate_fear_greed()
    return {
        'value': fg.value,
        'emotion': fg.emotion.value,
        'components': fg.components,
        'timestamp': fg.timestamp.isoformat()
    }

@app.get("/patterns/{ticker}")
async def patterns(ticker: str):
    return await brain.detect_patterns(ticker.upper())

@app.get("/liquidity/{ticker}")
async def liquidity(ticker: str):
    return asdict(await brain.analyze_liquidity(ticker.upper()))

@app.get("/correlation/{ticker}")
async def correlation(ticker: str):
    return await brain.analyze_correlation(ticker.upper())

@app.get("/sectors")
async def sectors():
    if not brain._sector_performance:
        await brain.analyze_sector_rotation()
    return {k: asdict(v) for k, v in brain._sector_performance.items()}

@app.get("/analysis/{ticker}")
async def full_analysis(ticker: str):
    ticker = ticker.upper()
    return {
        'patterns': await brain.detect_patterns(ticker),
        'liquidity': asdict(await brain.analyze_liquidity(ticker)),
        'correlation': await brain.analyze_correlation(ticker)
    }

@app.post("/enhance")
async def enhance(ticker: str, signal: int, confidence: float):
    return await brain.enhance_signal(ticker.upper(), signal, confidence)

@app.post("/correct")
async def correct():
    return await brain.auto_correct_config()

@app.post("/retrain")
async def retrain():
    triggered = await brain.trigger_retrain("manual_request")
    return {"triggered": triggered}

@app.get("/macro")
async def macro():
    signals = await brain.check_macro_events()
    return {
        'signals': [asdict(s) for s in signals] if signals else [],
        'macro_bias': brain.config.macro_bias,
        'cbr_meetings': CBR_MEETINGS_2026
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8009")))
