#!/usr/bin/env python3
"""Profit Distribution - Auto-transfer 10% profit to main account"""
import asyncio
import os
import json
import logging
from datetime import datetime, time
from contextlib import asynccontextmanager

import httpx
import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("profit-distribution")

# Config
DISTRIBUTION_PCT = float(os.getenv("DISTRIBUTION_PCT", "10.0"))  # 10% of profits
MIN_PROFIT_THRESHOLD = float(os.getenv("MIN_PROFIT_THRESHOLD", "1000"))  # Min 1000 RUB profit
DISTRIBUTION_TIME = os.getenv("DISTRIBUTION_TIME", "23:00")  # Daily at 23:00

# Metrics
DISTRIBUTIONS_TOTAL = Counter("profit_distributions_total", "Total distributions", ["from_account"])
DISTRIBUTED_AMOUNT = Counter("distributed_amount_total", "Total distributed amount")
PENDING_PROFIT = Gauge("pending_profit", "Pending profit to distribute", ["account_id"])


class DistributionResult(BaseModel):
    from_account: str
    to_account: str
    profit: float
    distributed: float
    timestamp: str


class ProfitDistributionService:
    def __init__(self):
        self.pool = None
        self.redis = None
        self.http = None
        self.running = False
        self.account_manager_url = os.getenv("ACCOUNT_MANAGER_URL", "http://account-manager:8020")

    async def start(self):
        self.pool = await asyncpg.create_pool(
            os.getenv("DB_DSN", "postgresql://trading:trading123@localhost:5433/trading"),
            min_size=2, max_size=5
        )
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)

        await self._init_schema()

        self.running = True
        asyncio.create_task(self._distribution_scheduler())

        logger.info(f"✅ Profit Distribution started ({DISTRIBUTION_PCT}% at {DISTRIBUTION_TIME})")

    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.close()
        logger.info("🛑 Profit Distribution stopped")

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS profit_distributions (
                    id SERIAL PRIMARY KEY,
                    from_account_id VARCHAR(50) NOT NULL,
                    to_account_id VARCHAR(50) NOT NULL,
                    realized_profit NUMERIC(14,2),
                    distributed_amount NUMERIC(14,2),
                    distribution_pct NUMERIC(5,2),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def _distribution_scheduler(self):
        """Run distribution at scheduled time"""
        target_hour, target_minute = map(int, DISTRIBUTION_TIME.split(":"))

        while self.running:
            try:
                now = datetime.now()

                # Check if it's distribution time
                if now.hour == target_hour and now.minute == target_minute:
                    logger.info("⏰ Starting scheduled profit distribution...")
                    await self.distribute_all()
                    await asyncio.sleep(60)  # Prevent double run

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    async def _get_accounts(self) -> list:
        """Get all accounts from account-manager"""
        try:
            response = await self.http.get(f"{self.account_manager_url}/accounts")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            return []

    async def _get_main_account(self) -> dict | None:
        """Get main account"""
        try:
            response = await self.http.get(f"{self.account_manager_url}/main")
            data = response.json()
            return data if data and "id" in data else None
        except Exception as e:
            logger.error(f"Failed to get main account: {e}")
            return None

    async def _get_daily_profit(self, account_id: str) -> float:
        """Get today's realized profit for account"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COALESCE(SUM(pnl), 0) as profit
                FROM trades_history
                WHERE account_id = $1 
                AND DATE(closed_at) = CURRENT_DATE
                AND pnl > 0
            """, account_id)
        return float(row["profit"]) if row else 0.0

    async def _transfer_funds(self, from_id: str, to_id: str, amount: float, description: str) -> bool:
        """Transfer funds between accounts"""
        try:
            # Deduct from source
            await self.http.post(
                f"{self.account_manager_url}/accounts/{from_id}/balance",
                params={"amount": -amount, "tx_type": "profit_distribution", "description": description}
            )

            # Add to destination
            await self.http.post(
                f"{self.account_manager_url}/accounts/{to_id}/balance",
                params={"amount": amount, "tx_type": "profit_received", "description": description}
            )

            return True
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return False

    async def distribute_account(self, account_id: str) -> DistributionResult | None:
        """Distribute profit from one account to main"""
        main = await self._get_main_account()
        if not main:
            logger.warning("No main account configured")
            return None

        if account_id == main["id"]:
            logger.debug(f"Skipping main account {account_id}")
            return None

        # Get today's profit
        profit = await self._get_daily_profit(account_id)
        PENDING_PROFIT.labels(account_id=account_id).set(profit)

        if profit < MIN_PROFIT_THRESHOLD:
            logger.debug(f"Account {account_id}: profit {profit:.2f} below threshold")
            return None

        # Calculate distribution
        distribution_amount = profit * (DISTRIBUTION_PCT / 100)

        # Transfer
        description = f"Daily {DISTRIBUTION_PCT}% profit distribution"
        success = await self._transfer_funds(account_id, main["id"], distribution_amount, description)

        if not success:
            return None

        # Record distribution
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO profit_distributions (from_account_id, to_account_id, realized_profit, distributed_amount, distribution_pct)
                VALUES ($1, $2, $3, $4, $5)
            """, account_id, main["id"], profit, distribution_amount, DISTRIBUTION_PCT)

        # Update metrics
        DISTRIBUTIONS_TOTAL.labels(from_account=account_id).inc()
        DISTRIBUTED_AMOUNT.inc(distribution_amount)
        PENDING_PROFIT.labels(account_id=account_id).set(0)

        result = DistributionResult(
            from_account=account_id,
            to_account=main["id"],
            profit=round(profit, 2),
            distributed=round(distribution_amount, 2),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"💰 Distributed {distribution_amount:.2f} from {account_id} (profit: {profit:.2f})")

        # Send alert
        await self.redis.xadd("stream:alerts", {
            "name": "profit_distributed",
            "severity": "info",
            "message": f"Transferred {distribution_amount:.2f} RUB ({DISTRIBUTION_PCT}% of {profit:.2f} profit)",
            "from_account": account_id,
            "to_account": main["id"]
        }, maxlen=1000)

        return result

    async def distribute_all(self) -> list[DistributionResult]:
        """Distribute profits from all accounts"""
        accounts = await self._get_accounts()
        results = []

        for account in accounts:
            if not account.get("is_active", True):
                continue

            result = await self.distribute_account(account["id"])
            if result:
                results.append(result)

        logger.info(f"📊 Distribution complete: {len(results)} transfers")
        return results

    async def get_history(self, account_id: str = None, limit: int = 50) -> list:
        """Get distribution history"""
        async with self.pool.acquire() as conn:
            if account_id:
                rows = await conn.fetch("""
                    SELECT * FROM profit_distributions
                    WHERE from_account_id = $1 OR to_account_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, account_id, limit)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM profit_distributions
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)

        return [dict(r) for r in rows]

    async def get_stats(self) -> dict:
        """Get distribution statistics"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_distributions,
                    COALESCE(SUM(distributed_amount), 0) as total_distributed,
                    COALESCE(SUM(realized_profit), 0) as total_profit
                FROM profit_distributions
            """)

        return {
            "total_distributions": row["total_distributions"],
            "total_distributed": float(row["total_distributed"]),
            "total_profit_processed": float(row["total_profit"]),
            "distribution_pct": DISTRIBUTION_PCT,
            "min_threshold": MIN_PROFIT_THRESHOLD,
            "schedule_time": DISTRIBUTION_TIME
        }


# Initialize
svc = ProfitDistributionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Profit Distribution",
    description="Auto-transfer profits to main account",
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
    return {"status": "healthy", "distribution_pct": DISTRIBUTION_PCT}


@app.post("/distribute")
async def distribute_all():
    """Manually trigger distribution for all accounts"""
    return await svc.distribute_all()


@app.post("/distribute/{account_id}")
async def distribute_account(account_id: str):
    """Distribute from specific account"""
    result = await svc.distribute_account(account_id)
    if result:
        return result
    return {"message": "No distribution needed", "account_id": account_id}


@app.get("/history")
async def get_history(account_id: str = None, limit: int = 50):
    """Get distribution history"""
    return await svc.get_history(account_id, limit)


@app.get("/stats")
async def get_stats():
    """Get distribution statistics"""
    return await svc.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
