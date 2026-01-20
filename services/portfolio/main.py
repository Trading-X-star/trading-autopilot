#!/usr/bin/env python3
"""Portfolio Service - Portfolio tracking and P&L calculation"""
import asyncio
import os
import json
import logging
from datetime import datetime, date
from contextlib import asynccontextmanager
from decimal import Decimal

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("portfolio")

# Metrics
PORTFOLIO_VALUE = Gauge("portfolio_value", "Total portfolio value", ["account_id"])
UNREALIZED_PNL = Gauge("unrealized_pnl", "Unrealized P&L", ["account_id"])
REALIZED_PNL = Gauge("realized_pnl_daily", "Daily realized P&L", ["account_id"])


class PortfolioService:
    def __init__(self):
        self.pool = None
        self.redis = None
        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000"))

    async def start(self):
        self.pool = await asyncpg.create_pool(
            os.getenv("DB_DSN", "postgresql://trading:trading123@localhost:5433/trading"),
            min_size=2, max_size=10
        )
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        await self._init_schema()
        asyncio.create_task(self._metrics_loop())
        logger.info(f"✅ Portfolio started (initial: {self.initial_capital:,.0f})")

    async def stop(self):
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.close()
        logger.info("🛑 Portfolio stopped")

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id SERIAL PRIMARY KEY,
                    account_id VARCHAR(50) NOT NULL,
                    date DATE NOT NULL,
                    cash NUMERIC(14,2),
                    positions_value NUMERIC(14,2),
                    total_value NUMERIC(14,2),
                    realized_pnl NUMERIC(14,2),
                    unrealized_pnl NUMERIC(14,2),
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(account_id, date)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades_history (
                    id SERIAL PRIMARY KEY,
                    account_id VARCHAR(50) NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity INT NOT NULL,
                    entry_price NUMERIC(12,4),
                    exit_price NUMERIC(12,4),
                    pnl NUMERIC(14,2),
                    pnl_pct NUMERIC(8,4),
                    opened_at TIMESTAMP,
                    closed_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def _metrics_loop(self):
        """Update Prometheus metrics periodically"""
        while True:
            try:
                # Get all accounts
                accounts = await self.redis.keys("account:*")
                for key in accounts:
                    account_id = key.split(":")[-1]
                    summary = await self.get_summary(account_id)
                    PORTFOLIO_VALUE.labels(account_id=account_id).set(summary["total_value"])
                    UNREALIZED_PNL.labels(account_id=account_id).set(summary["unrealized_pnl"])
                    REALIZED_PNL.labels(account_id=account_id).set(summary["daily_realized_pnl"])
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
            await asyncio.sleep(30)

    async def get_positions(self, account_id: str) -> list:
        """Get current positions with P&L"""
        positions_data = await self.redis.hgetall(f"positions:{account_id}")
        positions = []

        for ticker, data in positions_data.items():
            pos = json.loads(data)
            quantity = pos.get("quantity", 0)
            avg_price = pos.get("avg_price", 0)

            # Get current price
            price_data = await self.redis.get(f"price:{ticker}")
            if price_data:
                current_price = json.loads(price_data).get("close", avg_price)
            else:
                current_price = avg_price

            # Calculate P&L
            market_value = quantity * current_price
            cost_basis = quantity * avg_price
            unrealized_pnl = market_value - cost_basis
            pnl_pct = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0

            positions.append({
                "ticker": ticker,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit")
            })

        return sorted(positions, key=lambda x: x["market_value"], reverse=True)

    async def get_summary(self, account_id: str) -> dict:
        """Get portfolio summary"""
        # Get account data
        account_data = await self.redis.get(f"account:{account_id}")
        if account_data:
            account = json.loads(account_data)
            cash = account.get("balance", self.initial_capital)
        else:
            cash = self.initial_capital

        # Calculate positions value
        positions = await self.get_positions(account_id)
        positions_value = sum(p["market_value"] for p in positions)
        unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)

        total_value = cash + positions_value
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = ((total_value / self.initial_capital) - 1) * 100

        # Get daily realized P&L
        daily_pnl = await self._get_daily_realized(account_id)

        return {
            "account_id": account_id,
            "cash": round(cash, 2),
            "positions_value": round(positions_value, 2),
            "total_value": round(total_value, 2),
            "positions_count": len(positions),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "daily_realized_pnl": round(daily_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "initial_capital": self.initial_capital,
            "timestamp": datetime.now().isoformat()
        }

    async def _get_daily_realized(self, account_id: str) -> float:
        """Get today's realized P&L"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COALESCE(SUM(pnl), 0) as daily_pnl
                FROM trades_history
                WHERE account_id = $1 AND DATE(closed_at) = CURRENT_DATE
            """, account_id)
        return float(row["daily_pnl"]) if row else 0.0

    async def record_trade(self, account_id: str, ticker: str, side: str,
                          quantity: int, entry_price: float, exit_price: float) -> dict:
        """Record closed trade"""
        if side == "sell":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl_pct = ((exit_price / entry_price) - 1) * 100 if side == "sell" else ((entry_price / exit_price) - 1) * 100

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO trades_history (account_id, ticker, side, quantity, entry_price, exit_price, pnl, pnl_pct)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, account_id, ticker, side, quantity, entry_price, exit_price, pnl, pnl_pct)

        logger.info(f"📝 Trade recorded: {ticker} {side} P&L: {pnl:+.2f}")
        return {"ticker": ticker, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2)}

    async def get_history(self, account_id: str, days: int = 30) -> list:
        """Get portfolio history"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT date, cash, positions_value, total_value, realized_pnl, unrealized_pnl
                FROM portfolio_snapshots
                WHERE account_id = $1
                ORDER BY date DESC
                LIMIT $2
            """, account_id, days)
        return [dict(r) for r in rows]

    async def save_snapshot(self, account_id: str) -> dict:
        """Save daily portfolio snapshot"""
        summary = await self.get_summary(account_id)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO portfolio_snapshots (account_id, date, cash, positions_value, 
                                                total_value, realized_pnl, unrealized_pnl)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (account_id, date) DO UPDATE SET
                    cash = EXCLUDED.cash,
                    positions_value = EXCLUDED.positions_value,
                    total_value = EXCLUDED.total_value,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl
            """, account_id, date.today(), summary["cash"], summary["positions_value"],
                summary["total_value"], summary["daily_realized_pnl"], summary["unrealized_pnl"])

        return {"saved": True, "date": str(date.today()), "total_value": summary["total_value"]}

    async def get_trades(self, account_id: str, limit: int = 50) -> list:
        """Get trade history"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM trades_history
                WHERE account_id = $1
                ORDER BY closed_at DESC
                LIMIT $2
            """, account_id, limit)
        return [dict(r) for r in rows]


# Initialize
svc = PortfolioService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Portfolio Service",
    description="Portfolio tracking and P&L calculation",
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
    return {"status": "healthy"}


@app.get("/positions/{account_id}")
async def get_positions(account_id: str):
    """Get current positions"""
    return await svc.get_positions(account_id)


@app.get("/summary/{account_id}")
async def get_summary(account_id: str):
    """Get portfolio summary"""
    return await svc.get_summary(account_id)


@app.get("/history/{account_id}")
async def get_history(account_id: str, days: int = 30):
    """Get portfolio history"""
    return await svc.get_history(account_id, days)


@app.get("/trades/{account_id}")
async def get_trades(account_id: str, limit: int = 50):
    """Get trade history"""
    return await svc.get_trades(account_id, limit)


@app.post("/snapshot/{account_id}")
async def save_snapshot(account_id: str):
    """Save portfolio snapshot"""
    return await svc.save_snapshot(account_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
