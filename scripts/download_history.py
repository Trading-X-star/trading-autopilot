#!/usr/bin/env python3
"""Download historical OHLCV data from MOEX ISS"""
import asyncio
import httpx
import asyncpg
from datetime import datetime, date
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("history-loader")

MOEX_BASE = "https://iss.moex.com/iss"
DB_DSN = "postgresql://${DB_USER:-trading}:${DB_PASSWORD:-trading123}@${DB_HOST:-postgres}:5432/trading"

START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


async def init_db(pool):
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker VARCHAR(20) PRIMARY KEY,
                name VARCHAR(255),
                isin VARCHAR(20),
                lot_size INT DEFAULT 1,
                list_level INT,
                added_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open NUMERIC(12,4),
                high NUMERIC(12,4),
                low NUMERIC(12,4),
                close NUMERIC(12,4),
                volume BIGINT,
                value NUMERIC(18,2),
                PRIMARY KEY (ticker, date)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_daily(date)")
    logger.info("✅ Database tables created")


async def get_all_shares(http):
    url = f"{MOEX_BASE}/engines/stock/markets/shares/boards/TQBR/securities.json"
    resp = await http.get(url, params={"iss.meta": "off"})
    data = resp.json()
    securities = data.get("securities", {})
    columns = securities.get("columns", [])
    rows = securities.get("data", [])
    
    shares = []
    for row in rows:
        item = dict(zip(columns, row))
        shares.append({
            "ticker": item.get("SECID"),
            "name": item.get("SHORTNAME"),
            "isin": item.get("ISIN"),
            "lot_size": item.get("LOTSIZE", 1),
            "list_level": item.get("LISTLEVEL", 3)
        })
    logger.info(f"📋 Found {len(shares)} shares on MOEX")
    return shares


async def save_tickers(pool, shares):
    async with pool.acquire() as conn:
        for share in shares:
            await conn.execute("""
                INSERT INTO tickers (ticker, name, isin, lot_size, list_level)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name
            """, share["ticker"], share["name"], share["isin"], share["lot_size"], share["list_level"])
    logger.info(f"💾 Saved {len(shares)} tickers")


def parse_date(date_str):
    """Convert string to date object"""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


async def download_daily(http, ticker, start, end):
    url = f"{MOEX_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    all_candles = []
    cursor = start
    
    while True:
        try:
            resp = await http.get(url, params={"from": cursor, "till": end, "iss.meta": "off"})
            if resp.status_code != 200:
                break
            data = resp.json()
            rows = data.get("history", {}).get("data", [])
            columns = data.get("history", {}).get("columns", [])
            if not rows:
                break
            
            for row in rows:
                item = dict(zip(columns, row))
                if item.get("CLOSE"):
                    all_candles.append((
                        ticker,
                        parse_date(item["TRADEDATE"]),
                        float(item["OPEN"]) if item.get("OPEN") else None,
                        float(item["HIGH"]) if item.get("HIGH") else None,
                        float(item["LOW"]) if item.get("LOW") else None,
                        float(item["CLOSE"]),
                        int(item.get("VOLUME", 0) or 0),
                        float(item.get("VALUE", 0) or 0)
                    ))
            
            last = rows[-1][columns.index("TRADEDATE")]
            if last == cursor or len(rows) < 100:
                break
            cursor = last
        except Exception as e:
            logger.error(f"{ticker}: {e}")
            break
    return all_candles


async def save_candles(pool, candles):
    if not candles:
        return
    async with pool.acquire() as conn:
        await conn.executemany("""
            INSERT INTO ohlcv_daily (ticker, date, open, high, low, close, volume, value)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ticker, date) DO UPDATE SET close = EXCLUDED.close, volume = EXCLUDED.volume
        """, candles)


async def main():
    logger.info(f"🚀 MOEX History Download: {START_DATE} to {END_DATE}")
    
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
    async with httpx.AsyncClient(timeout=30.0) as http:
        await init_db(pool)
        shares = await get_all_shares(http)
        await save_tickers(pool, shares)
        
        total = len(shares)
        total_candles = 0
        
        for i, share in enumerate(shares):
            ticker = share["ticker"]
            candles = await download_daily(http, ticker, START_DATE, END_DATE)
            if candles:
                await save_candles(pool, candles)
                total_candles += len(candles)
            logger.info(f"[{i+1}/{total}] {ticker}: {len(candles)} candles | Total: {total_candles:,}")
            await asyncio.sleep(0.2)
        
        logger.info(f"✅ Done! {total} tickers, {total_candles:,} candles")
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
