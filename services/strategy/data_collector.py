#!/usr/bin/env python3
"""
Data Collector - автоматический сбор данных для ML trainer
- В торговые часы: загружает с Datafeed и дописывает в CSV
- Запускает retrain по расписанию
"""
import os
import asyncio
import logging
import httpx
import pandas as pd
from datetime import datetime, time, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_collector")

DATA_DIR = Path("/app/data")
TRAINING_CSV = DATA_DIR / "training_data.csv"
DATAFEED_URL = os.getenv("DATAFEED_URL", "http://datafeed:8006")
STRATEGY_URL = os.getenv("STRATEGY_URL", "http://localhost:8005")

TICKERS = [
    'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'VTBR', 'MTSS',
    'MGNT', 'TATN', 'NLMK', 'CHMF', 'PLZL', 'ALRS', 'IRAO', 'HYDR',
    'PHOR', 'MOEX', 'AFKS', 'SNGS', 'PIKK', 'MAGN', 'FIVE', 'RUAL'
]


def is_trading_hours() -> bool:
    """Проверка торговых часов MOEX (10:00-18:50 MSK, пн-пт)"""
    now = datetime.now()
    if now.weekday() >= 5:  # Выходные
        return False
    t = now.time()
    return time(10, 0) <= t <= time(18, 50)


async def fetch_candles_from_datafeed(http: httpx.AsyncClient, ticker: str, days: int = 30) -> list:
    """Загрузить свечи с Datafeed"""
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        resp = await http.get(
            f"{DATAFEED_URL}/candles/{ticker}",
            params={"interval": 24, "start": start_date, "end": end_date},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
    return []


async def fetch_macro(http: httpx.AsyncClient) -> dict:
    """Загрузить макро-данные"""
    try:
        resp = await http.get(f"{DATAFEED_URL}/macro", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {}


async def collect_training_data(days: int = 30) -> pd.DataFrame:
    """Собрать данные для обучения с Datafeed"""
    logger.info(f"📥 Collecting {days} days of data from Datafeed...")
    
    all_data = []
    
    async with httpx.AsyncClient() as http:
        macro = await fetch_macro(http)
        usd_rate = macro.get('usd_rate', 90.0)
        key_rate = macro.get('key_rate', 21.0)
        
        for ticker in TICKERS:
            candles = await fetch_candles_from_datafeed(http, ticker, days)
            
            for c in candles:
                all_data.append({
                    'date': c.get('begin', '')[:10],
                    'ticker': ticker,
                    'open': c.get('open', 0),
                    'high': c.get('high', 0),
                    'low': c.get('low', 0),
                    'close': c.get('close', 0),
                    'volume': c.get('volume', 0),
                    'usd_rate': usd_rate,
                    'key_rate': key_rate,
                })
            
            logger.info(f"  {ticker}: {len(candles)} candles")
            await asyncio.sleep(0.1)  # Rate limiting
    
    df = pd.DataFrame(all_data)
    logger.info(f"✅ Collected {len(df)} rows")
    return df


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Объединить новые данные с существующим CSV"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if TRAINING_CSV.exists():
        existing = pd.read_csv(TRAINING_CSV)
        logger.info(f"📂 Existing data: {len(existing)} rows")
        
        # Объединяем и убираем дубликаты
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
        combined = combined.sort_values(['ticker', 'date'])
        
        logger.info(f"📊 Combined: {len(combined)} rows")
        return combined
    else:
        logger.info("📄 Creating new training file")
        return new_df


async def update_training_data(days: int = 7):
    """Обновить training_data.csv свежими данными"""
    new_data = await collect_training_data(days)
    
    if len(new_data) > 0:
        combined = merge_with_existing(new_data)
        combined.to_csv(TRAINING_CSV, index=False)
        logger.info(f"💾 Saved {len(combined)} rows to {TRAINING_CSV}")
        return len(combined)
    return 0


async def trigger_retrain():
    """Запустить переобучение модели"""
    try:
        async with httpx.AsyncClient() as http:
            logger.info("🔄 Triggering model retrain...")
            resp = await http.post(
                f"{STRATEGY_URL}/retrain",
                params={"reason": "auto_scheduled"},
                timeout=7200  # 2 hours max
            )
            result = resp.json()
            logger.info(f"✅ Retrain result: {result}")
            return result
    except Exception as e:
        logger.error(f"❌ Retrain failed: {e}")
        return {"error": str(e)}


async def daily_routine():
    """Ежедневная рутина: сбор данных + переобучение"""
    logger.info("=" * 50)
    logger.info("🚀 Starting daily data collection routine")
    logger.info("=" * 50)
    
    # 1. Собрать свежие данные (последние 7 дней для обновления)
    rows = await update_training_data(days=7)
    
    if rows > 0:
        # 2. Запустить переобучение
        await trigger_retrain()
    else:
        logger.warning("⚠️ No new data collected, skipping retrain")


async def main_loop():
    """Основной цикл с расписанием"""
    logger.info("📅 Data Collector started")
    
    last_collection = None
    last_retrain_date = None
    
    while True:
        now = datetime.now()
        
        # Сбор данных каждый час в торговые часы
        if is_trading_hours():
            if last_collection is None or (now - last_collection).seconds >= 3600:
                await update_training_data(days=1)
                last_collection = now
        
        # Переобучение раз в день в 19:00 (после закрытия рынка)
        if now.hour == 19 and now.minute < 5:
            today = now.date()
            if last_retrain_date != today:
                await daily_routine()
                last_retrain_date = today
        
        await asyncio.sleep(60)  # Проверка каждую минуту


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "collect":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            asyncio.run(update_training_data(days))
        
        elif cmd == "retrain":
            asyncio.run(trigger_retrain())
        
        elif cmd == "daily":
            asyncio.run(daily_routine())
        
        elif cmd == "loop":
            asyncio.run(main_loop())
        
        else:
            print("Usage: python data_collector.py [collect|retrain|daily|loop] [days]")
    else:
        # По умолчанию запускаем loop
        asyncio.run(main_loop())
