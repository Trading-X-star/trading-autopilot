#!/usr/bin/env python3
"""Feature Engineering - Calculate technical indicators"""
import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("feature-eng")

DB_DSN = "postgresql://${DB_USER:-trading}:${DB_PASSWORD:-trading123}@${DB_HOST:-postgres}:5432/trading"


async def init_features_table(pool):
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS features")
        await conn.execute("""
            CREATE TABLE features (
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open NUMERIC(12,4), high NUMERIC(12,4), low NUMERIC(12,4), close NUMERIC(12,4), volume BIGINT,
                return_1d NUMERIC(10,6), return_5d NUMERIC(10,6), return_10d NUMERIC(10,6), return_20d NUMERIC(10,6),
                sma_5 NUMERIC(12,4), sma_10 NUMERIC(12,4), sma_20 NUMERIC(12,4), sma_50 NUMERIC(12,4), sma_200 NUMERIC(12,4),
                ema_12 NUMERIC(12,4), ema_26 NUMERIC(12,4),
                macd NUMERIC(12,6), macd_signal NUMERIC(12,6), macd_hist NUMERIC(12,6),
                rsi_14 NUMERIC(8,4),
                bb_upper NUMERIC(12,4), bb_middle NUMERIC(12,4), bb_lower NUMERIC(12,4), bb_width NUMERIC(10,6), bb_pct NUMERIC(10,6),
                atr_14 NUMERIC(12,4), volatility_20 NUMERIC(10,6),
                volume_sma_20 NUMERIC(18,2), volume_ratio NUMERIC(10,4),
                high_52w NUMERIC(12,4), low_52w NUMERIC(12,4), pct_from_high NUMERIC(10,6), pct_from_low NUMERIC(10,6),
                target_1d NUMERIC(10,6), target_5d NUMERIC(10,6),
                signal_class INT,
                PRIMARY KEY (ticker, date)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON features(date)")
    logger.info("✅ Features table created")


async def get_tickers(pool):
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT ticker FROM ohlcv_daily GROUP BY ticker HAVING COUNT(*) >= 250")
    return [r["ticker"] for r in rows]


def calc_ema(prices, period):
    if len(prices) < period: return prices[-1] if prices else 0
    m = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]: ema = p * m + ema * (1 - m)
    return ema

def calc_rsi(prices):
    if len(prices) < 2: return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains)/len(gains) if gains else 0
    al = sum(losses)/len(losses) if losses else 0
    if al == 0: return 100.0 if ag > 0 else 50.0
    if ag == 0: return 0.0
    return 100 - 100/(1 + ag/al)

def calc_std(v):
    if len(v) < 2: return 0
    m = sum(v)/len(v)
    return (sum((x-m)**2 for x in v)/len(v))**0.5

def calc_atr(h, l, c):
    if len(h) < 2: return 0
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(h))]
    return sum(trs)/len(trs) if trs else 0


async def calculate_features(pool, ticker):
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT date,open,high,low,close,volume FROM ohlcv_daily WHERE ticker=$1 ORDER BY date", ticker)
    
    if len(rows) < 250: return 0
    
    dates = [r["date"] for r in rows]
    O = [float(r["open"] or r["close"] or 0) for r in rows]
    H = [float(r["high"] or r["close"] or 0) for r in rows]
    L = [float(r["low"] or r["close"] or 0) for r in rows]
    C = [float(r["close"] or 0) for r in rows]
    V = [int(r["volume"] or 0) for r in rows]
    n = len(C)
    
    data = []
    for i in range(250, n):
        try:
            c = C[i]
            if c == 0: continue
            
            ret_1d = (c/C[i-1]-1) if C[i-1]>0 else 0
            ret_5d = (c/C[i-5]-1) if C[i-5]>0 else 0
            ret_10d = (c/C[i-10]-1) if C[i-10]>0 else 0
            ret_20d = (c/C[i-20]-1) if C[i-20]>0 else 0
            
            sma_5 = sum(C[i-4:i+1])/5
            sma_10 = sum(C[i-9:i+1])/10
            sma_20 = sum(C[i-19:i+1])/20
            sma_50 = sum(C[i-49:i+1])/50
            sma_200 = sum(C[i-199:i+1])/200
            
            ema_12 = calc_ema(C[:i+1], 12)
            ema_26 = calc_ema(C[:i+1], 26)
            macd = ema_12 - ema_26
            macd_hist_series = [calc_ema(C[:j+1],12)-calc_ema(C[:j+1],26) for j in range(max(26,i-20),i+1)]
            macd_signal = calc_ema(macd_hist_series, 9) if len(macd_hist_series) >= 9 else macd
            macd_hist = macd - macd_signal
            
            rsi = calc_rsi(C[i-14:i+1])
            
            bb_std = calc_std(C[i-19:i+1])
            bb_upper = sma_20 + 2*bb_std
            bb_middle = sma_20
            bb_lower = sma_20 - 2*bb_std
            bb_width = (bb_upper-bb_lower)/bb_middle if bb_middle>0 else 0
            bb_pct = (c-bb_lower)/(bb_upper-bb_lower) if (bb_upper-bb_lower)>0 else 0.5
            
            atr = calc_atr(H[i-13:i+1], L[i-13:i+1], C[i-14:i+1])
            rets = [C[j]/C[j-1]-1 for j in range(i-19,i+1) if C[j-1]>0]
            vol20 = calc_std(rets) if rets else 0
            
            vol_sma = sum(V[i-19:i+1])/20
            vol_ratio = V[i]/vol_sma if vol_sma>0 else 1
            
            p = min(252, i)
            high_52w = max(H[i-p:i+1]) or c
            low_52w = min([x for x in L[i-p:i+1] if x > 0]) if any(x>0 for x in L[i-p:i+1]) else c
            pct_high = (c/high_52w-1) if high_52w>0 else 0
            pct_low = (c/low_52w-1) if low_52w>0 else 0
            
            t1d = (C[i+1]/c-1) if i+1<n and c>0 and C[i+1]>0 else None
            t5d = (C[i+5]/c-1) if i+5<n and c>0 and C[i+5]>0 else None
            
            if t5d is not None:
                sig = 1 if t5d > 0.02 else (-1 if t5d < -0.02 else 0)
            else:
                sig = None
            
            data.append((ticker,dates[i],O[i],H[i],L[i],C[i],V[i],ret_1d,ret_5d,ret_10d,ret_20d,
                sma_5,sma_10,sma_20,sma_50,sma_200,ema_12,ema_26,macd,macd_signal,macd_hist,rsi,
                bb_upper,bb_middle,bb_lower,bb_width,bb_pct,atr,vol20,vol_sma,vol_ratio,
                high_52w,low_52w,pct_high,pct_low,t1d,t5d,sig))
        except Exception as e:
            continue
    
    if data:
        async with pool.acquire() as conn:
            await conn.executemany("""INSERT INTO features VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38) ON CONFLICT DO NOTHING""", data)
    return len(data)


async def main():
    logger.info("🚀 Feature Engineering started")
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=10)
    await init_features_table(pool)
    
    tickers = await get_tickers(pool)
    logger.info(f"📋 Processing {len(tickers)} tickers")
    
    total = 0
    for i, t in enumerate(tickers):
        cnt = await calculate_features(pool, t)
        total += cnt
        if (i+1) % 25 == 0 or i == len(tickers)-1:
            logger.info(f"[{i+1}/{len(tickers)}] Total: {total:,}")
    
    async with pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT COUNT(*) as t, 
                   COUNT(*) FILTER(WHERE signal_class=1) as buy, 
                   COUNT(*) FILTER(WHERE signal_class=0) as hold, 
                   COUNT(*) FILTER(WHERE signal_class=-1) as sell 
            FROM features WHERE signal_class IS NOT NULL
        """)
    
    logger.info("=" * 50)
    logger.info(f"✅ Done! {stats['t']:,} rows")
    logger.info(f"📈 Buy: {stats['buy']:,} | ⏸️ Hold: {stats['hold']:,} | 📉 Sell: {stats['sell']:,}")
    logger.info("=" * 50)
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
