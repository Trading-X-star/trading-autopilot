#!/usr/bin/env python3
"""Trailing Stop Service - Dynamic stop-loss management with breakeven"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trailing-stop")

# Metrics
STOPS_TRIGGERED = Counter("trailing_stops_triggered", "Stops triggered", ["account_id", "ticker"])
BREAKEVEN_SET = Counter("breakeven_set_total", "Breakeven activations", ["account_id"])
ACTIVE_STOPS = Gauge("active_trailing_stops", "Active trailing stops")

# Config
TRAILING_PCT = float(os.getenv("TRAILING_PCT", "3.0"))  # 3% trailing
BREAKEVEN_TRIGGER_PCT = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "2.0"))  # Move to breakeven at +2%
BREAKEVEN_OFFSET_PCT = float(os.getenv("BREAKEVEN_OFFSET_PCT", "0.5"))  # Breakeven + 0.5%


class TrailingStopConfig(BaseModel):
    ticker: str
    account_id: str
    entry_price: float
    trailing_pct: float = TRAILING_PCT
    breakeven_trigger_pct: float = BREAKEVEN_TRIGGER_PCT
    breakeven_offset_pct: float = BREAKEVEN_OFFSET_PCT


class TrailingStopService:
    def __init__(self):
        self.redis = None
        self.http = None
        self.running = False
        self.stops: dict[str, dict] = {}  # key: "account_id:ticker"

    async def start(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)

        # Load existing stops from Redis
        await self._load_stops()

        self.running = True
        asyncio.create_task(self._monitor_loop())

        logger.info(f"✅ Trailing Stop started (trail: {TRAILING_PCT}%, breakeven at: +{BREAKEVEN_TRIGGER_PCT}%)")

    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("🛑 Trailing Stop stopped")

    async def _load_stops(self):
        """Load stops from Redis"""
        keys = await self.redis.keys("trailing:*")
        for key in keys:
            data = await self.redis.get(key)
            if data:
                stop = json.loads(data)
                stop_key = f"{stop['account_id']}:{stop['ticker']}"
                self.stops[stop_key] = stop

        ACTIVE_STOPS.set(len(self.stops))
        logger.info(f"📂 Loaded {len(self.stops)} trailing stops")

    async def _save_stop(self, stop: dict):
        """Save stop to Redis"""
        key = f"trailing:{stop['account_id']}:{stop['ticker']}"
        await self.redis.set(key, json.dumps(stop))

    async def _delete_stop(self, account_id: str, ticker: str):
        """Delete stop from Redis"""
        key = f"trailing:{account_id}:{ticker}"
        await self.redis.delete(key)
        stop_key = f"{account_id}:{ticker}"
        if stop_key in self.stops:
            del self.stops[stop_key]
        ACTIVE_STOPS.set(len(self.stops))

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                for stop_key, stop in list(self.stops.items()):
                    await self._check_stop(stop)
                    await asyncio.sleep(0.1)

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)

    async def _check_stop(self, stop: dict):
        """Check and update trailing stop"""
        ticker = stop["ticker"]
        account_id = stop["account_id"]

        # Get current price
        price_data = await self.redis.get(f"price:{ticker}")
        if not price_data:
            return

        current_price = json.loads(price_data).get("close", 0)
        if current_price <= 0:
            return

        entry_price = stop["entry_price"]
        current_stop = stop.get("stop_price", 0)
        highest_price = stop.get("highest_price", entry_price)
        breakeven_active = stop.get("breakeven_active", False)

        # Calculate profit percentage
        profit_pct = ((current_price / entry_price) - 1) * 100

        # Update highest price
        if current_price > highest_price:
            highest_price = current_price
            stop["highest_price"] = highest_price

            # Update trailing stop
            trailing_pct = stop.get("trailing_pct", TRAILING_PCT)
            new_stop = highest_price * (1 - trailing_pct / 100)

            if new_stop > current_stop:
                stop["stop_price"] = new_stop
                await self._save_stop(stop)
                logger.debug(f"📈 {ticker} trail updated: {new_stop:.2f} (high: {highest_price:.2f})")

        # Check breakeven activation
        if not breakeven_active and profit_pct >= stop.get("breakeven_trigger_pct", BREAKEVEN_TRIGGER_PCT):
            breakeven_price = entry_price * (1 + stop.get("breakeven_offset_pct", BREAKEVEN_OFFSET_PCT) / 100)

            if breakeven_price > current_stop:
                stop["stop_price"] = breakeven_price
                stop["breakeven_active"] = True
                await self._save_stop(stop)

                BREAKEVEN_SET.labels(account_id=account_id).inc()
                logger.info(f"🔒 {ticker} breakeven set at {breakeven_price:.2f} (+{profit_pct:.1f}%)")

                # Notify
                await self.redis.xadd("stream:alerts", {
                    "name": "breakeven_activated",
                    "severity": "info",
                    "message": f"{ticker}: Breakeven at {breakeven_price:.2f}",
                    "account_id": account_id
                }, maxlen=1000)

        # Check stop hit
        current_stop = stop.get("stop_price", 0)
        if current_stop > 0 and current_price <= current_stop:
            await self._trigger_stop(stop, current_price)

    async def _trigger_stop(self, stop: dict, current_price: float):
        """Trigger stop-loss sell"""
        ticker = stop["ticker"]
        account_id = stop["account_id"]

        logger.warning(f"🛑 STOP TRIGGERED: {ticker} @ {current_price:.2f} (stop: {stop['stop_price']:.2f})")

        STOPS_TRIGGERED.labels(account_id=account_id, ticker=ticker).inc()

        # Get position quantity
        positions = await self.redis.hget(f"positions:{account_id}", ticker)
        if not positions:
            await self._delete_stop(account_id, ticker)
            return

        pos = json.loads(positions)
        quantity = pos.get("quantity", 0)

        if quantity <= 0:
            await self._delete_stop(account_id, ticker)
            return

        # Send sell order via execution service
        try:
            response = await self.http.post(
                "http://execution:8003/execute",
                json={
                    "account_id": account_id,
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": quantity,
                    "price": current_price,
                    "order_type": "market"
                }
            )
            result = response.json()
            logger.info(f"📤 Stop sell executed: {result}")
        except Exception as e:
            logger.error(f"Failed to execute stop sell: {e}")

        # Publish alert
        pnl_pct = ((current_price / stop["entry_price"]) - 1) * 100
        await self.redis.xadd("stream:alerts", {
            "name": "stop_triggered",
            "severity": "warning",
            "message": f"{ticker}: Stop triggered @ {current_price:.2f} ({pnl_pct:+.1f}%)",
            "account_id": account_id,
            "ticker": ticker,
            "price": str(current_price),
            "pnl_pct": str(pnl_pct)
        }, maxlen=1000)

        # Remove stop
        await self._delete_stop(account_id, ticker)

    async def create_stop(self, config: TrailingStopConfig) -> dict:
        """Create trailing stop for position"""
        stop_key = f"{config.account_id}:{config.ticker}"

        initial_stop = config.entry_price * (1 - config.trailing_pct / 100)

        stop = {
            "ticker": config.ticker,
            "account_id": config.account_id,
            "entry_price": config.entry_price,
            "stop_price": initial_stop,
            "highest_price": config.entry_price,
            "trailing_pct": config.trailing_pct,
            "breakeven_trigger_pct": config.breakeven_trigger_pct,
            "breakeven_offset_pct": config.breakeven_offset_pct,
            "breakeven_active": False,
            "created_at": datetime.now().isoformat()
        }

        self.stops[stop_key] = stop
        await self._save_stop(stop)
        ACTIVE_STOPS.set(len(self.stops))

        logger.info(f"✅ Trailing stop created: {config.ticker} @ {config.entry_price:.2f} (stop: {initial_stop:.2f})")
        return stop

    async def get_stop(self, account_id: str, ticker: str) -> dict | None:
        """Get stop status"""
        stop_key = f"{account_id}:{ticker}"
        return self.stops.get(stop_key)

    async def get_all_stops(self, account_id: str = None) -> list:
        """Get all stops, optionally filtered by account"""
        if account_id:
            return [s for s in self.stops.values() if s["account_id"] == account_id]
        return list(self.stops.values())

    async def cancel_stop(self, account_id: str, ticker: str) -> dict:
        """Cancel trailing stop"""
        await self._delete_stop(account_id, ticker)
        logger.info(f"❌ Trailing stop cancelled: {ticker}")
        return {"cancelled": True, "ticker": ticker, "account_id": account_id}


# Initialize
svc = TrailingStopService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Trailing Stop Service",
    description="Dynamic stop-loss with breakeven",
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
    return {"status": "healthy", "active_stops": len(svc.stops)}


@app.post("/stops")
async def create_stop(config: TrailingStopConfig):
    """Create trailing stop"""
    return await svc.create_stop(config)


@app.get("/stops")
async def list_stops(account_id: str = None):
    """List all stops"""
    return await svc.get_all_stops(account_id)


@app.get("/stops/{account_id}/{ticker}")
async def get_stop(account_id: str, ticker: str):
    """Get stop status"""
    stop = await svc.get_stop(account_id, ticker.upper())
    if stop:
        return stop
    return {"error": "Stop not found"}


@app.delete("/stops/{account_id}/{ticker}")
async def cancel_stop(account_id: str, ticker: str):
    """Cancel trailing stop"""
    return await svc.cancel_stop(account_id, ticker.upper())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
