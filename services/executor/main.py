#!/usr/bin/env python3
"""Executor Service - Tinkoff REST API integration"""
import asyncio
import os
import json
import logging
import uuid
from datetime import datetime, time
from contextlib import asynccontextmanager
from enum import Enum

import httpx
import redis.asyncio as aioredis
import asyncpg
from fastapi import Response, FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("executor")

ORDERS_PLACED = Counter("orders_placed_total", "Orders", ["ticker", "side"])
PORTFOLIO_VALUE = Gauge("portfolio_value_rub", "Portfolio value")


class TinkoffAPI:
    """Direct Tinkoff Invest REST API client"""
    
    PROD_URL = "https://invest-public-api.tinkoff.ru/rest"
    SANDBOX_URL = "https://sandbox-invest-public-api.tinkoff.ru/rest"
    
    def __init__(self, token: str, sandbox: bool = True):
        self.token = token
        self.base_url = self.SANDBOX_URL if sandbox else self.PROD_URL
        self.sandbox = sandbox
        self.http = None
        self.account_id = None
        self.figi_cache = {}
    
    async def start(self):
        self.http = httpx.AsyncClient(timeout=30.0, headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
        # Get account
        await self._get_account()
    
    async def stop(self):
        if self.http:
            await self.http.aclose()
    
    async def _request(self, service: str, method: str, body: dict = None) -> dict:
        """Make REST API request"""
        url = f"{self.base_url}/tinkoff.public.invest.api.contract.v1.{service}/{method}"
        try:
            resp = await self.http.post(url, json=body or {})
            data = resp.json()
            if resp.status_code != 200:
                logger.error(f"API error: {resp.status_code} {data}")
                return {"error": data}
            return data
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e)}
    
    async def _get_account(self):
        """Get trading account ID"""
        if self.sandbox:
            # For sandbox - open account first
            result = await self._request("SandboxService", "OpenSandboxAccount", {})
            if "accountId" in result:
                self.account_id = result["accountId"]
                logger.info(f"Sandbox account: {self.account_id}")
                # Add money to sandbox
                await self._request("SandboxService", "SandboxPayIn", {
                    "accountId": self.account_id,
                    "amount": {"units": "1000000", "nano": 0, "currency": "RUB"}
                })
            else:
                # Try to get existing accounts
                result = await self._request("SandboxService", "GetSandboxAccounts", {})
                accounts = result.get("accounts", [])
                if accounts:
                    self.account_id = accounts[0]["id"]
        else:
            result = await self._request("UsersService", "GetAccounts", {})
            accounts = result.get("accounts", [])
            if accounts:
                self.account_id = accounts[0]["id"]
        
        logger.info(f"Account ID: {self.account_id}")
    
    async def get_figi(self, ticker: str) -> str | None:
        """Get FIGI by ticker"""
        if ticker in self.figi_cache:
            return self.figi_cache[ticker]
        
        result = await self._request("InstrumentsService", "FindInstrument", {
            "query": ticker
        })
        
        instruments = result.get("instruments", [])
        for inst in instruments:
            if inst.get("ticker") == ticker:
                figi = inst.get("figi")
                self.figi_cache[ticker] = figi
                logger.info(f"FIGI {ticker} = {figi}")
                return figi
        
        return None
    
    async def post_order(self, ticker: str, quantity: int, direction: str, price: float = 0) -> dict:
        """Place order"""
        figi = await self.get_figi(ticker)
        if not figi:
            return {"error": f"FIGI not found for {ticker}"}
        
        if not self.account_id:
            return {"error": "No account"}
        
        order_id = str(uuid.uuid4())
        
        body = {
            "figi": figi,
            "quantity": str(quantity),
            "direction": "ORDER_DIRECTION_BUY" if direction == "buy" else "ORDER_DIRECTION_SELL",
            "accountId": self.account_id,
            "orderType": "ORDER_TYPE_MARKET",
            "orderId": order_id
        }
        
        service = "SandboxService" if self.sandbox else "OrdersService"
        method = "PostSandboxOrder" if self.sandbox else "PostOrder"
        
        result = await self._request(service, method, body)
        
        if "error" not in result:
            result["orderId"] = result.get("orderId", order_id)
            logger.info(f"Order placed: {result.get('orderId')}")
        
        return result
    
    async def get_portfolio(self) -> dict:
        """Get portfolio"""
        if not self.account_id:
            return {"error": "No account"}
        
        service = "SandboxService" if self.sandbox else "OperationsService"
        method = "GetSandboxPortfolio" if self.sandbox else "GetPortfolio"
        
        result = await self._request(service, method, {"accountId": self.account_id})
        return result
    
    async def get_positions(self) -> dict:
        """Get positions"""
        if not self.account_id:
            return {"error": "No account"}
        
        service = "SandboxService" if self.sandbox else "OperationsService"
        method = "GetSandboxPositions" if self.sandbox else "GetPositions"
        
        result = await self._request(service, method, {"accountId": self.account_id})
        return result


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderRequest(BaseModel):
    ticker: str
    side: OrderSide
    quantity: int = 1
    price: float | None = None


class ExecutorService:
    def __init__(self):
        self.redis = None
        self.pg = None
        self.tinkoff = None
        self.running = False
        self.token = os.getenv("TINKOFF_TOKEN", "")
        self.sandbox = os.getenv("TINKOFF_SANDBOX", "true").lower() == "true"
        self.auto_execute = os.getenv("AUTO_EXECUTE", "false").lower() == "true"
        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.45"))
        self.max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", "20"))
        self.daily_trades = 0
        
        # Simulation fallback
        self.sim_portfolio = {"cash": 1000000, "positions": {}}
    
    async def start(self):
        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        
        try:
            self.pg = await asyncpg.create_pool(
                os.getenv("DATABASE_URL", "postgresql://trading:trading123@postgres:5432/trading"),
                min_size=2, max_size=10
            )
            await self._init_db()
        except Exception as e:
            logger.warning(f"PostgreSQL: {e}")
        
        # Initialize Tinkoff API
        if self.token:
            self.tinkoff = TinkoffAPI(self.token, self.sandbox)
            await self.tinkoff.start()
        
        self.running = True
        if self.auto_execute:
            asyncio.create_task(self._signal_listener())
        
        mode = "SANDBOX" if self.sandbox else "PRODUCTION"
        auto = "AUTO" if self.auto_execute else "MANUAL"
        api = "TINKOFF" if self.tinkoff and self.tinkoff.account_id else "SIMULATION"
        logger.info(f"✅ Executor started [{mode}] [{auto}] [{api}]")
    
    async def stop(self):
        self.running = False
        if self.tinkoff:
            await self.tinkoff.stop()
        if self.redis:
            await self.redis.close()
        if self.pg:
            await self.pg.close()
    
    async def _init_db(self):
        async with self.pg.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY, ticker VARCHAR(20), side VARCHAR(10),
                    quantity INT, price DECIMAL(20,4), value DECIMAL(20,2),
                    order_id VARCHAR(100), status VARCHAR(20) DEFAULT 'pending',
                    signal_confidence DECIMAL(5,3), executed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
    
    def _is_trading_hours(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        return time(10, 0) <= now.time() <= time(18, 40)
    
    async def _signal_listener(self):
        logger.info("📡 Signal listener started")
        last_id = "$"
        while self.running:
            try:
                result = await self.redis.xread({"stream:signals": last_id}, count=10, block=5000)
                if result:
                    for stream, messages in result:
                        for msg_id, data in messages:
                            last_id = msg_id
                            await self._process_signal(data)
            except Exception as e:
                logger.error(f"Listener error: {e}")
                await asyncio.sleep(5)
    
    async def _process_signal(self, signal: dict):
        ticker = signal.get("ticker")
        sig = signal.get("signal")
        confidence = float(signal.get("confidence", 0))
        price = float(signal.get("price", 0))
        
        logger.info(f"📨 Signal: {ticker} {sig.upper()} conf={confidence:.1%}")
        
        if confidence < self.min_confidence:
            return
        if not self._is_trading_hours():
            return
        if self.daily_trades >= self.max_daily_trades:
            return
        
        if sig == "buy":
            await self.execute_order(ticker, OrderSide.BUY, 1, price, confidence)
        elif sig == "sell":
            await self.execute_order(ticker, OrderSide.SELL, 1, price, confidence)
    
    async def execute_order(self, ticker: str, side: OrderSide, quantity: int, price: float, confidence: float = 0) -> dict:
        result = {"ticker": ticker, "side": side.value, "quantity": quantity, "price": price}
        
        # Try Tinkoff API first
        if self.tinkoff and self.tinkoff.account_id:
            api_result = await self.tinkoff.post_order(ticker, quantity, side.value, price)
            
            if "error" not in api_result:
                result["order_id"] = api_result.get("orderId")
                result["status"] = "submitted"
                result["execution_price"] = api_result.get("executedOrderPrice", {})
                logger.info(f"✅ Tinkoff order: {result['order_id']}")
            else:
                result["status"] = "error"
                result["error"] = api_result.get("error")
                logger.error(f"❌ Order failed: {result['error']}")
        else:
            # Simulation fallback
            result["order_id"] = f"SIM_{datetime.now().timestamp()}"
            result["status"] = "simulated"
            
            value = price * quantity
            if side == OrderSide.BUY:
                self.sim_portfolio["cash"] -= value
                if ticker not in self.sim_portfolio["positions"]:
                    self.sim_portfolio["positions"][ticker] = {"quantity": 0, "avg_price": 0}
                pos = self.sim_portfolio["positions"][ticker]
                total = pos["quantity"] + quantity
                pos["avg_price"] = (pos["avg_price"] * pos["quantity"] + price * quantity) / total if total > 0 else price
                pos["quantity"] = total
            else:
                self.sim_portfolio["cash"] += value
                if ticker in self.sim_portfolio["positions"]:
                    self.sim_portfolio["positions"][ticker]["quantity"] -= quantity
            
            logger.info(f"📝 SIM: {side.value.upper()} {quantity} {ticker} @ {price:.2f}")
        
        # Save to DB
        if self.pg:
            try:
                async with self.pg.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO trades (ticker, side, quantity, price, value, order_id, status, signal_confidence) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                        ticker, side.value, quantity, price, price * quantity, result.get("order_id"), result["status"], confidence
                    )
            except Exception as e:
                logger.error(f"DB: {e}")
        
        ORDERS_PLACED.labels(ticker=ticker, side=side.value).inc()
        self.daily_trades += 1
        
        await self.redis.xadd("stream:trades", {
            "ticker": ticker, "side": side.value, "quantity": str(quantity),
            "price": str(price), "status": result["status"]
        }, maxlen=1000)
        
        return result
    
    async def get_portfolio(self) -> dict:
        if self.tinkoff and self.tinkoff.account_id:
            return await self.tinkoff.get_portfolio()
        
        # Simulation
        total = self.sim_portfolio["cash"]
        for t, p in self.sim_portfolio["positions"].items():
            total += p["quantity"] * p["avg_price"]
        PORTFOLIO_VALUE.set(total)
        return {"cash": self.sim_portfolio["cash"], "positions": self.sim_portfolio["positions"], 
                "total_value": total, "mode": "simulation"}
    
    async def get_trades(self, limit: int = 50) -> list:
        if not self.pg:
            return []
        async with self.pg.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ticker, side, quantity, price, value, status, signal_confidence, executed_at FROM trades ORDER BY executed_at DESC LIMIT $1",
                limit
            )
            return [dict(r) for r in rows]


svc = ExecutorService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(title="Executor", lifespan=lifespan)
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
    return {
        "status": "healthy",
        "mode": "sandbox" if svc.sandbox else "production",
        "auto_execute": svc.auto_execute,
        "daily_trades": svc.daily_trades,
        "tinkoff_connected": bool(svc.tinkoff and svc.tinkoff.account_id),
        "account_id": svc.tinkoff.account_id if svc.tinkoff else None,
        "token_set": bool(svc.token)
    }


@app.post("/order")
async def order(req: OrderRequest):
    price = req.price
    if not price:
        data = await svc.redis.get(f"price:{req.ticker}")
        if data:
            price = json.loads(data).get("price", 0)
    return await svc.execute_order(req.ticker, req.side, req.quantity, price or 0)


@app.get("/portfolio")
async def portfolio():
    return await svc.get_portfolio()


@app.get("/trades")
async def trades(limit: int = 50):
    return await svc.get_trades(limit)


@app.post("/toggle-auto")
async def toggle():
    svc.auto_execute = not svc.auto_execute
    if svc.auto_execute:
        asyncio.create_task(svc._signal_listener())
    return {"auto_execute": svc.auto_execute}



# ============ PAIR TRADING EXECUTOR ============
from pair_executor import pair_executor

_pair_task = None

@app.get("/pairs/status")
async def pair_status():
    return pair_executor.get_status()

@app.post("/pairs/enable")
async def enable_pairs(capital: float = 100000, confidence: float = 0.6):
    global _pair_task
    pair_executor.capital_per_pair = capital
    pair_executor.min_confidence = confidence
    pair_executor.enabled = True
    if _pair_task is None or _pair_task.done():
        _pair_task = asyncio.create_task(pair_executor.run_loop())
    return {"status": "enabled", "capital": capital, "min_confidence": confidence}

@app.post("/pairs/disable")
async def disable_pairs():
    pair_executor.enabled = False
    return {"status": "disabled"}

@app.post("/pairs/manual")
async def open_pair_manual(long_ticker: str, short_ticker: str, capital: float = 50000):
    pair_executor.min_confidence = 0
    pair_executor.capital_per_pair = capital
    signal = {
        "pair": [long_ticker, short_ticker],
        "long": long_ticker,
        "short": short_ticker,
        "confidence": 1.0,
        "zscore": 0
    }
    result = await pair_executor.open_pair(signal)
    return {"opened": result is not None, "status": pair_executor.get_status()}


@app.post("/pairs/close/{pair_a}/{pair_b}")
async def close_pair_endpoint(pair_a: str, pair_b: str):
    result = await pair_executor.close_pair((pair_a, pair_b), "manual")
    return result or {"error": "Pair not found"}


@app.get("/positions")
async def positions():
    try:
        p = await svc.get_portfolio()
        result = []
        for tk, d in p.get("positions", {}).items():
            q = d.get("quantity", 0)
            if q != 0:
                result.append({"ticker": tk, "quantity": q, "entry_price": d.get("avg_price", 0), "current_price": d.get("current_price", 0), "unrealized_pnl": d.get("unrealized_pnl", 0)})
        return {"positions": result}
    except:
        return {"positions": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)

