#!/usr/bin/env python3
"""Data Ingestion - MOEX market data fetcher"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data-ingestion")

# Tickers to track
TICKERS = ["SBER", "GAZP", "LKOH", "YNDX", "TCSG", "ROSN", "NVTK", "GMKN", "MGNT", "PLZL"]

# Metrics
QUOTES_FETCHED = Counter("quotes_fetched_total", "Total quotes fetched", ["ticker"])
QUOTES_ERRORS = Counter("quotes_errors_total", "Quote fetch errors", ["ticker"])
LAST_PRICE = Gauge("last_price", "Last price", ["ticker"])


class DataIngestion:
    def __init__(self):
        self.pool = None
        self.redis = None
        self.http = None
        self.running = False

    async def start(self):
        # Database connection
        self.pool = await asyncpg.create_pool(
            os.getenv("DB_DSN", "postgresql://trading:trading123@localhost:5433/trading"),
            min_size=2,
            max_size=10
        )

        # Redis connection
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )

        # HTTP client
        self.http = httpx.AsyncClient(timeout=30.0)

        # Create table
        await self._init_schema()

        # Start background fetcher
        self.running = True
        asyncio.create_task(self._fetch_loop())

        logger.info(f"✅ Data Ingestion started (tracking {len(TICKERS)} tickers)")

    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.close()
        logger.info("🛑 Data Ingestion stopped")

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    open NUMERIC(12,4),
                    high NUMERIC(12,4),
                    low NUMERIC(12,4),
                    close NUMERIC(12,4),
                    volume BIGINT,
                    ts TIMESTAMP DEFAULT NOW(),
                    UNIQUE(ticker, ts)
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_data(ticker)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_market_ts ON market_data(ts DESC)")

    async def _fetch_loop(self):
        """Background loop to fetch quotes"""
        while self.running:
            try:
                for ticker in TICKERS:
                    data = await self._fetch_quote(ticker)
                    if data:
                        await self._store_quote(ticker, data)
                        QUOTES_FETCHED.labels(ticker=ticker).inc()
                        LAST_PRICE.labels(ticker=ticker).set(data["close"] or 0)
                    await asyncio.sleep(0.5)  # Rate limiting

                await asyncio.sleep(60)  # Fetch every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fetch loop error: {e}")
                await asyncio.sleep(30)

    async def _fetch_quote(self, ticker: str) -> dict | None:
        """Fetch quote from MOEX ISS API"""
        try:
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            response = await self.http.get(url)

            if response.status_code != 200:
                QUOTES_ERRORS.labels(ticker=ticker).inc()
                return None

            data = response.json()
            marketdata = data.get("marketdata", {}).get("data", [])

            if not marketdata:
                return None

            columns = data.get("marketdata", {}).get("columns", [])
            row = dict(zip(columns, marketdata[0]))

            return {
                "open": row.get("OPEN") or row.get("LAST"),
                "high": row.get("HIGH") or row.get("LAST"),
                "low": row.get("LOW") or row.get("LAST"),
                "close": row.get("LAST"),
                "volume": row.get("VOLTODAY", 0)
            }

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            QUOTES_ERRORS.labels(ticker=ticker).inc()
            return None

    async def _store_quote(self, ticker: str, data: dict):
        """Store quote in DB and Redis"""
        ts = datetime.now()

        # Store in PostgreSQL
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data (ticker, open, high, low, close, volume, ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker, ts) DO NOTHING
            """, ticker, data["open"], data["high"], data["low"], data["close"], data["volume"], ts)

        # Cache in Redis (5 min TTL)
        cache_data = {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "ts": ts.isoformat()
        }
        await self.redis.set(f"price:{ticker}", json.dumps(cache_data), ex=300)

        logger.debug(f"📊 {ticker}: {data['close']}")

    async def get_price(self, ticker: str) -> dict:
        """Get latest price from cache"""
        data = await self.redis.get(f"price:{ticker.upper()}")
        if data:
            return json.loads(data)
        return {"error": "not found", "ticker": ticker}

    async def get_history(self, ticker: str, limit: int = 100) -> list:
        """Get price history from DB"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ticker, open, high, low, close, volume, ts
                FROM market_data
                WHERE ticker = $1
                ORDER BY ts DESC
                LIMIT $2
            """, ticker.upper(), limit)
        return [dict(r) for r in rows]


# Initialize service
svc = DataIngestion()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Data Ingestion",
    description="MOEX market data service",
    version="1.0.0",
    lifespan=lifespan
)
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


@app.get("/health")
async def health():
    return {"status": "healthy", "tickers": len(TICKERS)}


@app.get("/tickers")
async def list_tickers():
    """List tracked tickers"""
    return {"tickers": TICKERS}


@app.get("/price/{ticker}")
async def get_price(ticker: str):
    """Get current price"""
    return await svc.get_price(ticker)


@app.get("/history/{ticker}")
async def get_history(ticker: str, limit: int = 100):
    """Get price history"""
    return await svc.get_history(ticker, limit)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
