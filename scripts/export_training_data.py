#!/usr/bin/env python3
"""Export OHLCV data to CSV for model training"""
import asyncio
import asyncpg
import pandas as pd

async def main():
    pool = await asyncpg.create_pool(
        "postgresql://trading:trading123@postgres:5432/trading",
        min_size=2, max_size=5
    )
    
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT ticker, date, open, high, low, close, volume
            FROM ohlcv_daily
            WHERE close IS NOT NULL
            ORDER BY ticker, date
        """)
    
    df = pd.DataFrame([dict(r) for r in rows])
    df.to_csv('/data/training_data.csv', index=False)
    print(f"✅ Exported {len(df)} rows to /data/training_data.csv")
    print(f"   Tickers: {df['ticker'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
