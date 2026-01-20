#!/usr/bin/env python3
"""
Datafeed Service v2 - Real-time MOEX data + CBR Macro + News
Интеграция: MOEX ISS API, ЦБ РФ API, MOEX News RSS
"""
import asyncio
import os
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import httpx
import redis.asyncio as aioredis
import asyncpg
from fastapi import Response, FastAPI, Query
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("datafeed")

# ============================================================
# PROMETHEUS METRICS
# ============================================================
PRICES_UPDATED = Counter("prices_updated_total", "Price updates", ["ticker"])
PRICE_VALUE = Gauge("current_price", "Current price", ["ticker"])
MACRO_VALUE = Gauge("macro_value", "Macro indicator", ["indicator"])
NEWS_COUNT = Counter("news_fetched_total", "News items fetched", ["source"])
API_LATENCY = Histogram("api_latency_seconds", "API call latency", ["endpoint"])

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = [
    # Blue chips
    "SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "PLZL", "TCSG", "MGNT", "VTBR",
    # Second tier
    "MTSS", "ALRS", "CHMF", "NLMK", "TATN", "SNGS", "MOEX", "AFKS", "POLY", "YNDX",
    # Third tier
    "FIVE", "IRAO", "HYDR", "RTKM", "PHOR", "PIKK", "RUAL", "CBOM", "MAGN", "AFLT",
]

MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
MOEX_CANDLES_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
MOEX_NEWS_RSS = "https://moex.com/export/news.aspx?cat=100"
MOEX_NEWS_ISS = "https://iss.moex.com/iss/sitenews.json"

CBR_CURRENCY_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
CBR_KEY_RATE_URL = "https://www.cbr.ru/scripts/XML_key_rate.asp"
CBR_DAILY_JSON = "https://www.cbr-xml-daily.ru/daily_json.js"

CURRENCY_CODES = {'USD': 'R01235', 'EUR': 'R01239', 'CNY': 'R01375'}


# ============================================================
# DATA MODELS
# ============================================================
@dataclass
class PriceData:
    ticker: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    change: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MacroData:
    usd_rate: float
    eur_rate: float
    cny_rate: float
    key_rate: float
    usd_change: float
    eur_change: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class NewsItem:
    id: str
    title: str
    published: str
    category: str
    tickers: List[str]
    sentiment: float
    url: str


# ============================================================
# CBR CLIENT - Центральный Банк России
# ============================================================
class CBRClient:
    """Клиент для API Центрального Банка России"""
    
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
    
    async def get_current_rates(self) -> Dict:
        """Текущие курсы валют (JSON API)"""
        try:
            with API_LATENCY.labels(endpoint="cbr_daily").time():
                resp = await self.http.get(CBR_DAILY_JSON)
                data = resp.json()
            
            return {
                'usd_rate': data['Valute']['USD']['Value'],
                'eur_rate': data['Valute']['EUR']['Value'],
                'cny_rate': data['Valute']['CNY']['Value'],
                'usd_change': data['Valute']['USD']['Value'] - data['Valute']['USD']['Previous'],
                'eur_change': data['Valute']['EUR']['Value'] - data['Valute']['EUR']['Previous'],
                'date': data['Date'],
            }
        except Exception as e:
            logger.error(f"CBR rates error: {e}")
            return {}
    
    async def get_key_rate(self) -> float:
        """Текущая ключевая ставка"""
        try:
            with API_LATENCY.labels(endpoint="cbr_keyrate").time():
                resp = await self.http.get(CBR_KEY_RATE_URL)
                root = ET.fromstring(resp.text)
            
            rates = []
            for record in root.findall('.//Record'):
                date_str = record.get('Date')
                rate = float(record.find('Rate').text.replace(',', '.'))
                date = datetime.strptime(date_str, '%d.%m.%Y')
                rates.append((date, rate))
            
            if rates:
                latest = max(rates, key=lambda x: x[0])
                return latest[1]
            return 0.0
        except Exception as e:
            logger.error(f"CBR key rate error: {e}")
            return 0.0
    
    async def get_currency_history(self, currency: str = 'USD', days: int = 30) -> List[Dict]:
        """История курса валюты"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            code = CURRENCY_CODES.get(currency, 'R01235')
            
            params = {
                'date_req1': start_date.strftime('%d/%m/%Y'),
                'date_req2': end_date.strftime('%d/%m/%Y'),
                'VAL_NM_RQ': code
            }
            
            with API_LATENCY.labels(endpoint="cbr_history").time():
                resp = await self.http.get(CBR_CURRENCY_URL, params=params)
                root = ET.fromstring(resp.text)
            
            records = []
            for record in root.findall('.//Record'):
                records.append({
                    'date': record.get('Date'),
                    'rate': float(record.find('Value').text.replace(',', '.'))
                })
            return records
        except Exception as e:
            logger.error(f"CBR history error: {e}")
            return []


# ============================================================
# MOEX NEWS CLIENT
# ============================================================
class MOEXNewsClient:
    """Клиент для новостей MOEX"""
    
    KNOWN_TICKERS = set(TICKERS)
    TICKER_PATTERN = re.compile(r'\b([A-Z]{4})\b')
    
    POSITIVE_WORDS = {'рост', 'прибыль', 'дивиденд', 'увеличение', 'повышение', 'рекорд', 'успех'}
    NEGATIVE_WORDS = {'падение', 'убыток', 'снижение', 'сокращение', 'риск', 'потери', 'штраф', 'санкции'}
    
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
    
    async def fetch_rss(self, limit: int = 50) -> List[NewsItem]:
        """Получить новости из RSS"""
        try:
            with API_LATENCY.labels(endpoint="moex_news_rss").time():
                resp = await self.http.get(MOEX_NEWS_RSS)
            
            news = []
            root = ET.fromstring(resp.text)
            
            for item in root.findall('.//item')[:limit]:
                title = item.find('title').text or ''
                link = item.find('link').text or ''
                pub_date = item.find('pubDate').text or ''
                description = item.find('description').text or ''
                
                try:
                    published = datetime.strptime(
                        pub_date.replace(' +0300', '').strip(),
                        '%a, %d %b %Y %H:%M:%S'
                    )
                except:
                    published = datetime.now()
                
                text = f"{title} {description}"
                tickers = self._extract_tickers(text)
                sentiment = self._analyze_sentiment(text)
                
                news.append(NewsItem(
                    id=link.split('/')[-1] if link else str(hash(title)),
                    title=title,
                    published=published.isoformat(),
                    category='rss',
                    tickers=tickers,
                    sentiment=sentiment,
                    url=link
                ))
            
            NEWS_COUNT.labels(source="rss").inc(len(news))
            return news
            
        except Exception as e:
            logger.error(f"MOEX RSS error: {e}")
            return []
    
    async def fetch_iss_news(self, start: int = 0, limit: int = 50) -> List[NewsItem]:
        """Получить новости через ISS API"""
        try:
            with API_LATENCY.labels(endpoint="moex_news_iss").time():
                resp = await self.http.get(MOEX_NEWS_ISS, params={'start': start})
                data = resp.json()
            
            news = []
            sitenews = data.get('sitenews', {})
            columns = sitenews.get('columns', [])
            rows = sitenews.get('data', [])[:limit]
            
            for row in rows:
                item = dict(zip(columns, row))
                title = item.get('title', '')
                
                try:
                    published = datetime.strptime(item.get('published_at', ''), '%Y-%m-%d %H:%M:%S')
                except:
                    published = datetime.now()
                
                tickers = self._extract_tickers(title)
                sentiment = self._analyze_sentiment(title)
                
                news.append(NewsItem(
                    id=str(item.get('id', '')),
                    title=title,
                    published=published.isoformat(),
                    category=item.get('tag', 'general'),
                    tickers=tickers,
                    sentiment=sentiment,
                    url=f"https://www.moex.com/n{item.get('id', '')}"
                ))
            
            NEWS_COUNT.labels(source="iss").inc(len(news))
            return news
            
        except Exception as e:
            logger.error(f"MOEX ISS news error: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        found = self.TICKER_PATTERN.findall(text.upper())
        return [t for t in found if t in self.KNOWN_TICKERS]
    
    def _analyze_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        pos = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0
    
    async def get_sentiment_by_ticker(self) -> Dict[str, float]:
        """Агрегированный sentiment по тикерам"""
        news = await self.fetch_rss(100)
        
        ticker_sentiment = {}
        ticker_count = {}
        
        for item in news:
            for ticker in item.tickers:
                if ticker not in ticker_sentiment:
                    ticker_sentiment[ticker] = 0.0
                    ticker_count[ticker] = 0
                ticker_sentiment[ticker] += item.sentiment
                ticker_count[ticker] += 1
        
        return {
            t: ticker_sentiment[t] / ticker_count[t]
            for t in ticker_sentiment if ticker_count[t] > 0
        }


# ============================================================
# MAIN DATAFEED SERVICE
# ============================================================
class DatafeedService:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.pg: Optional[asyncpg.Pool] = None
        self.http: Optional[httpx.AsyncClient] = None
        self.cbr: Optional[CBRClient] = None
        self.news: Optional[MOEXNewsClient] = None
        
        self.running = False
        self.last_update = None
        self.prices: Dict[str, PriceData] = {}
        self.macro: Optional[MacroData] = None
    
    async def start(self):
        # Redis
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True
        )
        logger.info("✅ Redis connected")
        
        # PostgreSQL
        try:
            self.pg = await asyncpg.create_pool(
                os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"),
                min_size=2, max_size=10
            )
            await self._init_db()
            logger.info("✅ PostgreSQL connected")
        except Exception as e:
            logger.warning(f"PostgreSQL not available: {e}")
            self.pg = None
        
        # HTTP client
        self.http = httpx.AsyncClient(timeout=30.0)
        
        # Sub-clients
        self.cbr = CBRClient(self.http)
        self.news = MOEXNewsClient(self.http)
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._price_loop())
        asyncio.create_task(self._macro_loop())
        asyncio.create_task(self._news_loop())
        
        logger.info(f"✅ Datafeed v2 started, tracking {len(TICKERS)} tickers")
    
    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        if self.pg:
            await self.pg.close()
        logger.info("Datafeed stopped")
    
    async def _init_db(self):
        async with self.pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    price DECIMAL(20,4),
                    open DECIMAL(20,4),
                    high DECIMAL(20,4),
                    low DECIMAL(20,4),
                    volume BIGINT,
                    change_pct DECIMAL(10,4),
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_ph_ticker_ts ON price_history(ticker, timestamp DESC);
                
                CREATE TABLE IF NOT EXISTS macro_history (
                    id SERIAL PRIMARY KEY,
                    usd_rate DECIMAL(10,4),
                    eur_rate DECIMAL(10,4),
                    cny_rate DECIMAL(10,4),
                    key_rate DECIMAL(6,2),
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS news_cache (
                    id VARCHAR(50) PRIMARY KEY,
                    title TEXT,
                    published TIMESTAMPTZ,
                    tickers TEXT[],
                    sentiment DECIMAL(5,4),
                    url TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_news_published ON news_cache(published DESC);
            """)
    
    def _is_trading_hours(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return time(7, 0) <= t <= time(21, 0)  # MSK adjusted
    
    # ============================================================
    # PRICE FETCHING
    # ============================================================
    async def fetch_prices(self) -> Dict[str, PriceData]:
        """Получить текущие цены с MOEX"""
        try:
            params = {
                "iss.meta": "off",
                "iss.only": "marketdata",
                "marketdata.columns": "SECID,LAST,OPEN,HIGH,LOW,VOLTODAY,LASTTOPREVPRICE"
            }
            
            with API_LATENCY.labels(endpoint="moex_prices").time():
                resp = await self.http.get(MOEX_URL, params=params)
                data = resp.json()
            
            prices = {}
            md_cols = data["marketdata"]["columns"]
            
            for row in data["marketdata"]["data"]:
                d = dict(zip(md_cols, row))
                ticker = d.get("SECID")
                
                if ticker in TICKERS and d.get("LAST"):
                    prices[ticker] = PriceData(
                        ticker=ticker,
                        price=float(d["LAST"] or 0),
                        open=float(d["OPEN"] or 0),
                        high=float(d["HIGH"] or 0),
                        low=float(d["LOW"] or 0),
                        volume=int(d["VOLTODAY"] or 0),
                        change=float(d["LASTTOPREVPRICE"] or 0),
                        timestamp=datetime.now().isoformat()
                    )
            
            return prices
        except Exception as e:
            logger.error(f"Fetch prices error: {e}")
            return {}
    
    async def fetch_candles(self, ticker: str, interval: int = 24, 
                           start: str = None, end: str = None) -> List[Dict]:
        """Получить свечи с MOEX ISS"""
        try:
            url = MOEX_CANDLES_URL.format(ticker=ticker)
            params = {
                "interval": interval,
                "iss.meta": "off",
                "candles.columns": "begin,open,high,low,close,volume"
            }
            if start:
                params["from"] = start
            if end:
                params["till"] = end
            
            with API_LATENCY.labels(endpoint="moex_candles").time():
                resp = await self.http.get(url, params=params)
                data = resp.json()
            
            candles = []
            cols = data.get("candles", {}).get("columns", [])
            rows = data.get("candles", {}).get("data", [])
            
            for row in rows:
                candles.append(dict(zip(cols, row)))
            
            return candles
        except Exception as e:
            logger.error(f"Fetch candles error: {e}")
            return []
    
    async def update_prices_cache(self, prices: Dict[str, PriceData]):
        """Обновить кэш цен в Redis"""
        pipe = self.redis.pipeline()
        
        for ticker, data in prices.items():
            # Текущая цена
            pipe.set(f"price:{ticker}", json.dumps(data.to_dict()))
            
            # История (последние 300 записей)
            pipe.zadd(f"history:{ticker}", {json.dumps(data.to_dict()): datetime.now().timestamp()})
            pipe.zremrangebyrank(f"history:{ticker}", 0, -301)
        
        # Общий snapshot
        pipe.set("prices:all", json.dumps({t: d.to_dict() for t, d in prices.items()}))
        pipe.set("prices:updated", datetime.now().isoformat())
        
        await pipe.execute()
        
        self.prices = prices
        self.last_update = datetime.now()
    
    async def _price_loop(self):
        """Фоновый цикл обновления цен"""
        while self.running:
            try:
                prices = await self.fetch_prices()
                
                if prices:
                    await self.update_prices_cache(prices)
                    
                    # Update metrics
                    for ticker, data in prices.items():
                        PRICES_UPDATED.labels(ticker=ticker).inc()
                        PRICE_VALUE.labels(ticker=ticker).set(data.price)
                    
                    logger.info(f"📊 Updated {len(prices)} prices")
                
                # Interval: 5s during trading, 60s otherwise
                interval = 5 if self._is_trading_hours() else 60
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Price loop error: {e}")
                await asyncio.sleep(10)
    
    # ============================================================
    # MACRO DATA
    # ============================================================
    async def fetch_macro(self) -> Optional[MacroData]:
        """Получить макро-данные от ЦБ РФ"""
        try:
            rates = await self.cbr.get_current_rates()
            key_rate = await self.cbr.get_key_rate()
            
            if rates:
                return MacroData(
                    usd_rate=rates.get('usd_rate', 0),
                    eur_rate=rates.get('eur_rate', 0),
                    cny_rate=rates.get('cny_rate', 0),
                    key_rate=key_rate,
                    usd_change=rates.get('usd_change', 0),
                    eur_change=rates.get('eur_change', 0),
                    timestamp=datetime.now().isoformat()
                )
            return None
        except Exception as e:
            logger.error(f"Fetch macro error: {e}")
            return None
    
    async def update_macro_cache(self, macro: MacroData):
        """Обновить кэш макро-данных"""
        await self.redis.set("macro:current", json.dumps(macro.to_dict()))
        await self.redis.zadd("macro:history", {json.dumps(macro.to_dict()): datetime.now().timestamp()})
        
        # Metrics
        MACRO_VALUE.labels(indicator="usd").set(macro.usd_rate)
        MACRO_VALUE.labels(indicator="eur").set(macro.eur_rate)
        MACRO_VALUE.labels(indicator="key_rate").set(macro.key_rate)
        
        self.macro = macro
    
    async def _macro_loop(self):
        """Фоновый цикл обновления макро-данных (каждые 5 минут)"""
        while self.running:
            try:
                macro = await self.fetch_macro()
                
                if macro:
                    await self.update_macro_cache(macro)
                    logger.info(f"🏦 Macro updated: USD={macro.usd_rate:.2f}, Key={macro.key_rate}%")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Macro loop error: {e}")
                await asyncio.sleep(60)
    
    # ============================================================
    # NEWS
    # ============================================================
    async def _news_loop(self):
        """Фоновый цикл обновления новостей (каждые 15 минут)"""
        while self.running:
            try:
                # RSS news
                rss_news = await self.news.fetch_rss(50)
                
                if rss_news:
                    pipe = self.redis.pipeline()
                    for item in rss_news:
                        pipe.hset(f"news:{item.id}", mapping={
                            'title': item.title,
                            'published': item.published,
                            'tickers': ','.join(item.tickers),
                            'sentiment': str(item.sentiment),
                            'url': item.url
                        })
                        pipe.expire(f"news:{item.id}", 86400 * 7)  # 7 days TTL
                    await pipe.execute()
                    
                    logger.info(f"📰 News updated: {len(rss_news)} items")
                
                # Sentiment by ticker
                sentiment = await self.news.get_sentiment_by_ticker()
                if sentiment:
                    await self.redis.set("news:sentiment", json.dumps(sentiment))
                
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"News loop error: {e}")
                await asyncio.sleep(300)
    
    # ============================================================
    # PUBLIC API
    # ============================================================
    async def get_price(self, ticker: str) -> dict:
        data = await self.redis.get(f"price:{ticker}")
        return json.loads(data) if data else {}
    
    async def get_all_prices(self) -> dict:
        data = await self.redis.get("prices:all")
        return json.loads(data) if data else {}
    
    async def get_price_history(self, ticker: str, limit: int = 100) -> list:
        data = await self.redis.zrevrange(f"history:{ticker}", 0, limit - 1)
        return [json.loads(d) for d in data]
    
    async def get_macro(self) -> dict:
        data = await self.redis.get("macro:current")
        return json.loads(data) if data else {}
    
    async def get_news(self, limit: int = 50) -> List[dict]:
        return [asdict(n) for n in await self.news.fetch_rss(limit)]
    
    async def get_news_sentiment(self) -> dict:
        data = await self.redis.get("news:sentiment")
        return json.loads(data) if data else {}


# ============================================================
# FASTAPI APPLICATION
# ============================================================
svc = DatafeedService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(title="Datafeed v2", description="MOEX + CBR + News", lifespan=lifespan)
# ============================================================
# METRICS ENDPOINT (fixed - no 307 redirects)
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# OLD: metrics mount removed


# Health
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0",
        "tickers": len(TICKERS),
        "cached": len(svc.prices),
        "last_update": svc.last_update.isoformat() if svc.last_update else None,
        "trading_hours": svc._is_trading_hours(),
        "macro": svc.macro.to_dict() if svc.macro else None
    }


# Prices
@app.get("/price/{ticker}")
async def get_price(ticker: str):
    return await svc.get_price(ticker.upper())


@app.get("/prices")
async def get_prices():
    return await svc.get_all_prices()


@app.get("/history/{ticker}")
async def get_history(ticker: str, limit: int = Query(100, le=500)):
    return await svc.get_price_history(ticker.upper(), limit)


@app.get("/candles/{ticker}")
async def get_candles(
    ticker: str,
    interval: int = Query(24, description="1=1m, 10=10m, 60=1h, 24=1d"),
    start: str = None,
    end: str = None
):
    return await svc.fetch_candles(ticker.upper(), interval, start, end)


@app.post("/refresh")
async def refresh():
    prices = await svc.fetch_prices()
    if prices:
        await svc.update_prices_cache(prices)
    return {"updated": len(prices)}


# Macro (CBR)
@app.get("/macro")
async def get_macro():
    return await svc.get_macro()


@app.get("/macro/history/{currency}")
async def get_currency_history(currency: str = "USD", days: int = Query(30, le=365)):
    return await svc.cbr.get_currency_history(currency.upper(), days)


@app.get("/macro/keyrate")
async def get_key_rate():
    rate = await svc.cbr.get_key_rate()
    return {"key_rate": rate, "timestamp": datetime.now().isoformat()}


# News
@app.get("/news")
async def get_news(limit: int = Query(50, le=200)):
    return await svc.get_news(limit)


@app.get("/news/sentiment")
async def get_news_sentiment():
    return await svc.get_news_sentiment()


@app.get("/news/ticker/{ticker}")
async def get_news_for_ticker(ticker: str, limit: int = 20):
    news = await svc.news.fetch_rss(100)
    ticker_news = [asdict(n) for n in news if ticker.upper() in n.tickers]
    return ticker_news[:limit]


# Tickers list
@app.get("/tickers")
async def get_tickers():
    return {"tickers": TICKERS, "count": len(TICKERS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
