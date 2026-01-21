#!/usr/bin/env python3
"""Execution Service v2.0 - With Risk Manager integration"""
import asyncio
import os
import json
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import httpx
import sys
sys.path.insert(0, '/app/shared')
try:
    from circuit_breaker import ResilientClient, CircuitOpenError
    USE_CIRCUIT_BREAKER = True
except ImportError:
    USE_CIRCUIT_BREAKER = False
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("execution")

# Config
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")
TINKOFF_SANDBOX = os.getenv("TINKOFF_SANDBOX", "true").lower() == "true"
BROKER_MODE = os.getenv("BROKER_MODE", "paper")
RISK_MANAGER_URL = os.getenv("RISK_MANAGER_URL", "http://risk-manager:8001")  # NEW

TINKOFF_BASE = "https://sandbox-invest-public-api.tbank.ru" if TINKOFF_SANDBOX else "https://invest-public-api.tbank.ru"

# Metrics
ORDERS_TOTAL = Counter("orders_total", "Total orders", ["side", "status", "broker"])
EXEC_LATENCY = Histogram("execution_latency_seconds", "Order execution latency")
ORDERS_REJECTED = Counter("orders_rejected_total", "Rejected by risk manager")  # NEW
ACTIVE_ORDERS = Gauge("active_orders", "Currently pending orders")  # NEW

# Extended ticker mapping
TICKER_TO_FIGI = {
    "SBER": "BBG004730N88", "GAZP": "BBG004730RP0", "LKOH": "BBG004731032",
    "YNDX": "BBG006L8G4H1", "TCSG": "BBG00QPYJ5H0", "ROSN": "BBG004731354",
    "NVTK": "BBG00475KKY8", "GMKN": "BBG004731489", "MGNT": "BBG004RVFCY3",
    "PLZL": "BBG000R607Y3", "VTBR": "BBG004730ZJ9", "MTSS": "BBG004S681W1",
    "ALRS": "BBG004S68B31", "CHMF": "BBG00475K6C3", "NLMK": "BBG004S681B4",
    "MOEX": "BBG004730JJ5", "TATN": "BBG004RVFFC0", "SNGS": "BBG0047315D0",
    "POLY": "BBG004PYF2N3", "PHOR": "BBG004S689R0", "IRAO": "BBG004S68473",
    "AFLT": "BBG004S683W7", "PIKK": "BBG004S68CP5", "RUAL": "BBG008F2T3T2",
}


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderRequest(BaseModel):
    account_id: str = "default"
    ticker: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    confidence: float = 0.5           # NEW: for risk manager
    risk_profile: str = "balanced"    # NEW
    skip_risk_check: bool = False     # NEW: for manual orders


class OrderResponse(BaseModel):
    order_id: str
    status: OrderStatus
    filled_quantity: int
    filled_price: float
    commission: float
    timestamp: str
    broker: str
    stop_loss: Optional[float] = None   # NEW
    take_profit: Optional[float] = None  # NEW


class ExecutionService:
    def __init__(self):
        self.pool = None
        self.redis = None
        self.http = None
        self.tinkoff_account_id = None
        self.broker_mode = BROKER_MODE
        self.pending_orders = {}
    
    async def start(self):
        self.pool = await asyncpg.create_pool(
            os.getenv("DB_DSN", os.getenv("DATABASE_URL", "postgresql://${DB_USER:-trading}:${DB_PASSWORD:-trading123}@${DB_HOST:-postgres}:5432/trading")),
            min_size=2, max_size=10
        )
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=30.0)  # Removed verify=False
        await self._init_schema()
        
        if TINKOFF_TOKEN and self.broker_mode in ("sandbox", "live"):
            await self._init_tinkoff()
        
        logger.info(f"✅ Execution started (mode: {self.broker_mode})")
        logger.info(f"🛡️ Risk Manager: {RISK_MANAGER_URL}")
    
    async def _init_tinkoff(self):
        """Initialize Tinkoff with retry logic"""
        try:
            headers = {"Authorization": f"Bearer {TINKOFF_TOKEN}"}
            
            if TINKOFF_SANDBOX:
                resp = await self.http.post(
                    f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.SandboxService/GetSandboxAccounts",
                    headers=headers, json={}
                )
                data = resp.json()
                
                if data.get("accounts"):
                    self.tinkoff_account_id = data["accounts"][0]["id"]
                else:
                    resp = await self.http.post(
                        f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.SandboxService/OpenSandboxAccount",
                        headers=headers, json={}
                    )
                    self.tinkoff_account_id = resp.json().get("accountId")
                    
                    # Add initial capital
                    await self.http.post(
                        f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.SandboxService/SandboxPayIn",
                        headers=headers,
                        json={"accountId": self.tinkoff_account_id, "amount": {"units": "10000000", "nano": 0, "currency": "rub"}}
                    )
                
                logger.info(f"📦 Tinkoff Sandbox: {self.tinkoff_account_id}")
            else:
                resp = await self.http.post(
                    f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts",
                    headers=headers, json={}
                )
                accounts = resp.json().get("accounts", [])
                if accounts:
                    self.tinkoff_account_id = accounts[0]["id"]
                    logger.info(f"💰 Tinkoff Live: {self.tinkoff_account_id}")
                    
        except Exception as e:
            logger.error(f"Tinkoff init error: {e}")
            self.broker_mode = "paper"
    
    async def stop(self):
        if self.http: await self.http.aclose()
        if self.redis: await self.redis.close()
        if self.pool: await self.pool.close()
    
    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id VARCHAR(50) PRIMARY KEY,
                    account_id VARCHAR(50) NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    figi VARCHAR(20),
                    side VARCHAR(10) NOT NULL,
                    order_type VARCHAR(20) NOT NULL,
                    quantity INT NOT NULL,
                    filled_quantity INT DEFAULT 0,
                    filled_price NUMERIC(12,4),
                    status VARCHAR(20) DEFAULT 'pending',
                    commission NUMERIC(12,4) DEFAULT 0,
                    broker VARCHAR(20) DEFAULT 'paper',
                    broker_order_id VARCHAR(100),
                    stop_loss NUMERIC(12,4),
                    take_profit NUMERIC(12,4),
                    confidence NUMERIC(5,3),
                    risk_check_passed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Index for faster queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_account_created 
                ON orders(account_id, created_at DESC)
            """)
    
    # ==================== NEW: Risk Manager Integration ====================
    
    async def _check_risk(self, order: OrderRequest, price: float) -> dict:
        """Check order with Risk Manager before execution"""
        try:
            resp = await self.http.post(
                f"{RISK_MANAGER_URL}/check",
                json={
                    "account_id": order.account_id,
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "price": price,
                    "confidence": order.confidence,
                    "risk_profile": order.risk_profile
                },
                timeout=5.0
            )
            return resp.json()
        except httpx.ConnectError:
            logger.warning("⚠️ Risk Manager unavailable, using defaults")
            return {
                "approved": True,
                "adjusted_quantity": order.quantity,
                "stop_loss": price * 0.98 if order.side == OrderSide.BUY else price * 1.02,
                "take_profit": price * 1.03 if order.side == OrderSide.BUY else price * 0.97,
                "reason": "Risk Manager unavailable"
            }
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return {"approved": False, "reason": str(e)}
    
    async def _notify_risk_manager(self, order: OrderRequest, result: dict):
        """Notify Risk Manager about executed trade"""
        try:
            await self.http.post(
                f"{RISK_MANAGER_URL}/position",
                params={
                    "ticker": order.ticker.upper(),
                    "side": "long" if order.side == OrderSide.BUY else "short",
                    "quantity": result["filled_quantity"],
                    "price": result["filled_price"],
                    "confidence": order.confidence
                },
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Could not notify Risk Manager: {e}")
    
    # ==================== Main Execute ====================
    
    async def execute_order(self, order: OrderRequest) -> OrderResponse:
        start_time = datetime.now()
        order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
        figi = TICKER_TO_FIGI.get(order.ticker.upper())
        
        # Get market price
        price_data = await self.redis.get(f"price:{order.ticker}")
        market_price = json.loads(price_data).get("close", 0) if price_data else order.price or 0
        
        if market_price <= 0:
            raise HTTPException(status_code=400, detail=f"No price for {order.ticker}")
        
        # NEW: Risk check (unless skipped)
        stop_loss = None
        take_profit = None
        
        if not order.skip_risk_check:
            risk_result = await self._check_risk(order, market_price)
            
            if not risk_result.get("approved"):
                ORDERS_REJECTED.inc()
                logger.warning(f"❌ Order rejected by Risk Manager: {risk_result.get('reason')}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Risk check failed: {risk_result.get('reason')}"
                )
            
            # Apply adjusted quantity from risk manager
            order.quantity = risk_result.get("adjusted_quantity", order.quantity)
            stop_loss = risk_result.get("stop_loss")
            take_profit = risk_result.get("take_profit")
            
            logger.info(f"✅ Risk approved: qty={order.quantity}, SL={stop_loss}, TP={take_profit}")
        
        # Execute
        ACTIVE_ORDERS.inc()
        try:
            if self.broker_mode == "paper" or not figi or not self.tinkoff_account_id:
                result = await self._paper_execute(order, market_price)
            else:
                result = await self._tinkoff_execute(order, figi, market_price)
        finally:
            ACTIVE_ORDERS.dec()
        
        # Add stops to result
        result["stop_loss"] = stop_loss
        result["take_profit"] = take_profit
        
        # Store and notify
        await self._store_order(order_id, order, figi, result)
        await self._update_positions(order, result)
        
        # Notify Risk Manager about new position
        if order.side == OrderSide.BUY and not order.skip_risk_check:
            await self._notify_risk_manager(order, result)
        
        # Publish to stream
        await self.redis.xadd("stream:executions", {
            "order_id": order_id,
            "account_id": order.account_id,
            "ticker": order.ticker,
            "side": order.side.value,
            "quantity": str(result["filled_quantity"]),
            "price": str(result["filled_price"]),
            "broker": result["broker"],
            "stop_loss": str(stop_loss or 0),
            "take_profit": str(take_profit or 0)
        }, maxlen=10000)
        
        # Also notify Risk Manager stream for position tracking
        await self.redis.xadd("stream:trades", {
            "ticker": order.ticker.upper(),
            "side": order.side.value,
            "quantity": str(result["filled_quantity"]),
            "price": str(result["filled_price"]),
            "confidence": str(order.confidence)
        }, maxlen=10000)
        
        # Metrics
        EXEC_LATENCY.observe((datetime.now() - start_time).total_seconds())
        ORDERS_TOTAL.labels(
            side=order.side.value, 
            status=result["status"].value,
            broker=result["broker"]
        ).inc()
        
        logger.info(f"📈 {order.side.value.upper()} {order.ticker} x{result['filled_quantity']} @ {result['filled_price']} [{result['broker']}]")
        
        return OrderResponse(
            order_id=order_id,
            status=result["status"],
            filled_quantity=result["filled_quantity"],
            filled_price=result["filled_price"],
            commission=result["commission"],
            timestamp=datetime.now().isoformat(),
            broker=result["broker"],
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    async def _paper_execute(self, order: OrderRequest, market_price: float) -> dict:
        """Paper trading with realistic slippage"""
        import random
        
        # Slippage: 0.01-0.05% depending on size
        base_slippage = 0.0001
        size_impact = min(0.0004, order.quantity / 10000 * 0.0001)
        slippage = random.uniform(base_slippage, base_slippage + size_impact)
        
        if order.side == OrderSide.BUY:
            filled_price = market_price * (1 + slippage)
        else:
            filled_price = market_price * (1 - slippage)
        
        # Commission: 0.05%
        commission = order.quantity * filled_price * 0.0005
        
        # Simulate partial fill for large orders (10% chance)
        filled_qty = order.quantity
        if order.quantity > 1000 and random.random() < 0.1:
            filled_qty = int(order.quantity * random.uniform(0.7, 0.95))
        
        return {
            "status": OrderStatus.FILLED if filled_qty == order.quantity else OrderStatus.PARTIAL,
            "filled_quantity": filled_qty,
            "filled_price": round(filled_price, 4),
            "commission": round(commission, 2),
            "broker": "paper"
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _tinkoff_execute(self, order: OrderRequest, figi: str, market_price: float) -> dict:
        """Execute via Tinkoff API with retry"""
        try:
            headers = {"Authorization": f"Bearer {TINKOFF_TOKEN}"}
            
            direction = 1 if order.side == OrderSide.BUY else 2
            order_type = 2 if order.order_type == OrderType.MARKET else 1
            
            endpoint = "SandboxService/PostSandboxOrder" if TINKOFF_SANDBOX else "OrdersService/PostOrder"
            
            payload = {
                "figi": figi,
                "quantity": str(order.quantity),
                "direction": direction,
                "accountId": self.tinkoff_account_id,
                "orderType": order_type,
                "orderId": str(uuid.uuid4())
            }
            
            resp = await self.http.post(
                f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.{endpoint}",
                headers=headers, json=payload
            )
            
            data = resp.json()
            
            if "code" in data:
                logger.error(f"Tinkoff error: {data}")
                return await self._paper_execute(order, market_price)
            
            filled_qty = int(data.get("lotsExecuted", order.quantity))
            exec_price = data.get("executedOrderPrice", {})
            filled_price = int(exec_price.get("units", 0)) + int(exec_price.get("nano", 0)) / 1e9
            if filled_price == 0:
                filled_price = market_price
            
            commission_data = data.get("initialCommission", {})
            commission = int(commission_data.get("units", 0)) + int(commission_data.get("nano", 0)) / 1e9
            
            logger.info(f"🏦 Tinkoff: {data.get('orderId')} executed")
            
            return {
                "status": OrderStatus.FILLED,
                "filled_quantity": filled_qty,
                "filled_price": round(filled_price, 4),
                "commission": round(commission, 2),
                "broker": "tinkoff_sandbox" if TINKOFF_SANDBOX else "tinkoff_live"
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Tinkoff HTTP error: {e}")
            raise  # Will be retried
        except Exception as e:
            logger.error(f"Tinkoff error: {e}")
            return await self._paper_execute(order, market_price)
    
    async def _store_order(self, order_id: str, order: OrderRequest, figi: str, result: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO orders (id, account_id, ticker, figi, side, order_type, quantity,
                    filled_quantity, filled_price, status, commission, broker, stop_loss, 
                    take_profit, confidence, risk_check_passed)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            """, order_id, order.account_id, order.ticker, figi, order.side.value, 
                order.order_type.value, order.quantity, result["filled_quantity"], 
                result["filled_price"], result["status"].value, result["commission"], 
                result["broker"], result.get("stop_loss"), result.get("take_profit"),
                order.confidence, not order.skip_risk_check)
    
    async def _update_positions(self, order: OrderRequest, result: dict):
        key = f"positions:{order.account_id}"
        positions = await self.redis.hgetall(key)
        current = json.loads(positions.get(order.ticker, "{}"))
        current_qty = current.get("quantity", 0)
        current_avg = current.get("avg_price", 0)
        
        if order.side == OrderSide.BUY:
            new_qty = current_qty + result["filled_quantity"]
            if new_qty > 0:
                new_avg = ((current_qty * current_avg) + (result["filled_quantity"] * result["filled_price"])) / new_qty
            else:
                new_avg = result["filled_price"]
        else:
            new_qty = current_qty - result["filled_quantity"]
            new_avg = current_avg if new_qty > 0 else 0
        
        if new_qty > 0:
            await self.redis.hset(key, order.ticker, json.dumps({
                "quantity": new_qty,
                "avg_price": round(new_avg, 4),
                "stop_loss": result.get("stop_loss"),
                "take_profit": result.get("take_profit"),
                "updated": datetime.now().isoformat()
            }))
        else:
            await self.redis.hdel(key, order.ticker)
    
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel pending order (for limit orders)"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE id = $1 AND status = 'pending'", 
                order_id
            )
            if not row:
                raise HTTPException(404, "Order not found or already filled")
            
            # If Tinkoff order, cancel there too
            if row["broker_order_id"] and self.tinkoff_account_id:
                try:
                    headers = {"Authorization": f"Bearer {TINKOFF_TOKEN}"}
                    endpoint = "SandboxService/CancelSandboxOrder" if TINKOFF_SANDBOX else "OrdersService/CancelOrder"
                    await self.http.post(
                        f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.{endpoint}",
                        headers=headers,
                        json={"accountId": self.tinkoff_account_id, "orderId": row["broker_order_id"]}
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel Tinkoff order: {e}")
            
            await conn.execute(
                "UPDATE orders SET status = 'cancelled', updated_at = NOW() WHERE id = $1",
                order_id
            )
            
            return {"order_id": order_id, "status": "cancelled"}
    
    async def get_orders(self, account_id: str, limit: int = 50) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM orders WHERE account_id = $1 ORDER BY created_at DESC LIMIT $2", 
                account_id, limit
            )
        return [dict(r) for r in rows]
    
    async def get_tinkoff_portfolio(self) -> dict:
        if not TINKOFF_TOKEN or not self.tinkoff_account_id:
            return {"error": "Tinkoff not configured"}
        try:
            headers = {"Authorization": f"Bearer {TINKOFF_TOKEN}"}
            endpoint = "SandboxService/GetSandboxPortfolio" if TINKOFF_SANDBOX else "OperationsService/GetPortfolio"
            resp = await self.http.post(
                f"{TINKOFF_BASE}/rest/tinkoff.public.invest.api.contract.v1.{endpoint}",
                headers=headers, json={"accountId": self.tinkoff_account_id}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


svc = ExecutionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()

app = FastAPI(title="Execution Service", version="2.0.0", lifespan=lifespan)
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
        "mode": svc.broker_mode,
        "tinkoff": bool(TINKOFF_TOKEN),
        "account": svc.tinkoff_account_id,
        "risk_manager": RISK_MANAGER_URL
    }


@app.post("/execute", response_model=OrderResponse)
async def execute_order(order: OrderRequest):
    return await svc.execute_order(order)


@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    return await svc.cancel_order(order_id)


@app.get("/orders/{account_id}")
async def get_account_orders(account_id: str, limit: int = 50):
    return await svc.get_orders(account_id, limit)


@app.get("/tinkoff/portfolio")
async def get_tinkoff_portfolio():
    return await svc.get_tinkoff_portfolio()


@app.get("/tickers")
async def get_tickers():
    return TICKER_TO_FIGI


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)


@app.get("/circuit-status")
async def circuit_status():
    if hasattr(execution, 'client') and hasattr(execution.client, 'status'):
        return execution.client.status()
    return {"circuit_breaker": "not_enabled"}
