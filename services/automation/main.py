#!/usr/bin/env python3
"""Trading Autopilot - Automation Engine"""
import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List
from dataclasses import dataclass
import os
import httpx
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Automation Engine")
scheduler = AsyncIOScheduler()

# Config
CONFIG = {
    "auto_trade": os.getenv("AUTO_TRADE_ENABLED", "true").lower() == "true",
    "max_daily_trades": 50,
    "ml_threshold": 0.65,
    "max_position_pct": 0.10,
    "max_daily_loss_pct": 0.03,
}

SERVICES = {
    "orchestrator": "http://orchestrator:8000",
    "strategy": "http://trading-autopilot-strategy-gpu-1:8005",
    "executor": "http://executor:8007",
    "risk_manager": "http://risk-manager:8001",
    "kill_switch": "http://kill-switch:8020",
    "searxng": "http://searxng:8080",
}

class TradingEngine:
    def __init__(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.client = httpx.AsyncClient(timeout=30)
    
    async def call_service(self, service: str, endpoint: str, method="GET", data=None):
        url = f"{SERVICES[service]}{endpoint}"
        try:
            if method == "GET":
                r = await self.client.get(url)
            else:
                r = await self.client.post(url, json=data)
            return r.json() if r.status_code == 200 else {"error": r.text}
        except Exception as e:
            logger.error(f"Service {service} error: {e}")
            return {"error": str(e)}
    
    def is_market_open(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        return time(10, 0) <= t <= time(18, 45)
    
    async def run_trading_cycle(self):
        if not CONFIG["auto_trade"] or not self.is_market_open():
            return
        
        # Check kill switch
        ks = await self.call_service("kill_switch", "/status")
        if ks.get("active"):
            logger.warning("Kill switch active - skipping")
            return
        
        # Get signals
        signals = await self.call_service("strategy", "/signals")
        if "error" in signals:
            return
        
        good_signals = [s for s in signals.get("signals", []) 
                       if s.get("confidence", 0) >= CONFIG["ml_threshold"]]
        
        logger.info(f"🔄 Trading cycle: {len(good_signals)} signals above threshold")
        
        executed = 0
        for signal in sorted(good_signals, key=lambda x: x["confidence"], reverse=True)[:5]:
            if self.daily_trades >= CONFIG["max_daily_trades"]:
                break
            
            if signal["direction"] in ["BUY", "SELL"]:
                result = await self.execute_trade(signal)
                if result:
                    executed += 1
        
        logger.info(f"✅ Executed {executed} trades")
    
    async def execute_trade(self, signal: dict) -> bool:
        # Risk check
        risk = await self.call_service("risk_manager", "/check", "POST", {
            "ticker": signal["ticker"],
            "direction": signal["direction"]
        })
        if not risk.get("approved", False):
            return False
        
        # Execute
        order = {
            "ticker": signal["ticker"],
            "direction": signal["direction"],
            "quantity": signal.get("suggested_quantity", 10),
            "order_type": "MARKET"
        }
        result = await self.call_service("executor", "/order", "POST", order)
        
        if result.get("status") == "executed":
            self.daily_trades += 1
            logger.info(f"✓ {signal['direction']} {signal['ticker']}")
            return True
        return False
    
    async def check_risk_levels(self):
        status = await self.call_service("risk_manager", "/status")
        drawdown = status.get("current_drawdown", 0)
        
        if drawdown > 0.15:
            logger.critical(f"🚨 Critical drawdown {drawdown:.1%} - activating kill switch")
            await self.call_service("kill_switch", "/activate", "POST")
        elif drawdown > 0.10:
            logger.warning(f"⚠️ High drawdown: {drawdown:.1%}")
    
    async def analyze_news(self):
        positions = await self.call_service("orchestrator", "/positions")
        for pos in positions.get("positions", [])[:10]:
            ticker = pos["ticker"]
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.get(f"http://searxng:8080/search",
                                   params={"q": f"{ticker} новости", "format": "json"})
                    news = r.json().get("results", [])[:5]
                    for article in news:
                        if any(w in article.get("title", "").lower() 
                              for w in ["падение", "убыток", "риск", "санкции"]):
                            logger.warning(f"⚠️ Negative news for {ticker}: {article['title'][:50]}")
            except:
                pass
    
    async def daily_reset(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0
        logger.info("🔄 Daily counters reset")

engine = TradingEngine()

@app.on_event("startup")
async def startup():
    # Trading cycle - every 5 min during market hours
    scheduler.add_job(engine.run_trading_cycle, 
                     CronTrigger(day_of_week="mon-fri", hour="10-18", minute="*/5"),
                     id="trading")
    # Risk check - every minute
    scheduler.add_job(engine.check_risk_levels,
                     CronTrigger(minute="*"), id="risk")
    # News analysis - every 15 min
    scheduler.add_job(engine.analyze_news,
                     CronTrigger(day_of_week="mon-fri", hour="9-19", minute="*/15"),
                     id="news")
    # Daily reset
    scheduler.add_job(engine.daily_reset, CronTrigger(hour=0, minute=0), id="reset")
    
    scheduler.start()
    logger.info("🚀 Automation Engine started")
    logger.info(f"📋 Jobs: {[j.id for j in scheduler.get_jobs()]}")

@app.get("/health")
async def health():
    return {"status": "healthy", "trades_today": engine.daily_trades}

@app.get("/status")
async def status():
    return {
        "running": True,
        "auto_trade": CONFIG["auto_trade"],
        "daily_trades": engine.daily_trades,
        "jobs": [{"id": j.id, "next": str(j.next_run_time)} for j in scheduler.get_jobs()]
    }

@app.post("/toggle")
async def toggle(enabled: bool = True):
    CONFIG["auto_trade"] = enabled
    return {"auto_trade": CONFIG["auto_trade"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8025)
