#!/usr/bin/env python3
"""Account Manager - Multi-account management (up to 3 accounts)"""
import asyncio
import os
import json
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Gauge, Counter, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("account-manager")

MAX_ACCOUNTS = 3

# Metrics
ACCOUNTS_TOTAL = Gauge("accounts_total", "Total accounts")
ACCOUNT_BALANCE = Gauge("account_balance", "Account balance", ["account_id", "name"])


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class AccountCreate(BaseModel):
    name: str
    risk_profile: RiskProfile = RiskProfile.BALANCED
    initial_balance: float = 1_000_000
    description: str = ""


class AccountUpdate(BaseModel):
    name: str | None = None
    risk_profile: RiskProfile | None = None
    description: str | None = None
    is_active: bool | None = None


class AccountManager:
    def __init__(self):
        self.pool = None
        self.redis = None

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
        await self._sync_to_redis()
        asyncio.create_task(self._metrics_loop())
        logger.info(f"✅ Account Manager started (max: {MAX_ACCOUNTS})")

    async def stop(self):
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.close()
        logger.info("🛑 Account Manager stopped")

    async def _init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    risk_profile VARCHAR(20) DEFAULT 'balanced',
                    initial_balance NUMERIC(14,2) DEFAULT 1000000,
                    current_balance NUMERIC(14,2) DEFAULT 1000000,
                    description TEXT,
                    is_active BOOLEAN DEFAULT true,
                    is_main BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS account_transactions (
                    id SERIAL PRIMARY KEY,
                    account_id VARCHAR(50) REFERENCES accounts(id),
                    type VARCHAR(20) NOT NULL,
                    amount NUMERIC(14,2) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def _sync_to_redis(self):
        """Sync accounts to Redis for fast access"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM accounts")

        for row in rows:
            account = dict(row)
            account["initial_balance"] = float(account["initial_balance"])
            account["current_balance"] = float(account["current_balance"])
            account["balance"] = account["current_balance"]
            account["created_at"] = account["created_at"].isoformat()
            account["updated_at"] = account["updated_at"].isoformat()
            await self.redis.set(f"account:{row['id']}", json.dumps(account, default=str))

        ACCOUNTS_TOTAL.set(len(rows))

    async def _metrics_loop(self):
        while True:
            try:
                accounts = await self.list_accounts()
                ACCOUNTS_TOTAL.set(len(accounts))
                for acc in accounts:
                    ACCOUNT_BALANCE.labels(
                        account_id=acc["id"],
                        name=acc["name"]
                    ).set(acc["current_balance"])
            except Exception as e:
                logger.error(f"Metrics error: {e}")
            await asyncio.sleep(30)

    async def create_account(self, data: AccountCreate) -> dict:
        """Create new account (max 3)"""
        # Check limit
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM accounts")

        if count >= MAX_ACCOUNTS:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_ACCOUNTS} accounts allowed")

        account_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"
        is_main = count == 0  # First account is main

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO accounts (id, name, risk_profile, initial_balance, current_balance, description, is_main)
                VALUES ($1, $2, $3, $4, $4, $5, $6)
            """, account_id, data.name, data.risk_profile.value, data.initial_balance, data.description, is_main)

        account = {
            "id": account_id,
            "name": data.name,
            "risk_profile": data.risk_profile.value,
            "initial_balance": data.initial_balance,
            "current_balance": data.initial_balance,
            "balance": data.initial_balance,
            "description": data.description,
            "is_active": True,
            "is_main": is_main,
            "created_at": datetime.now().isoformat()
        }

        await self.redis.set(f"account:{account_id}", json.dumps(account))
        ACCOUNTS_TOTAL.inc()

        logger.info(f"✅ Account created: {data.name} ({account_id})")
        return account

    async def get_account(self, account_id: str) -> dict:
        """Get account by ID"""
        # Try Redis first
        data = await self.redis.get(f"account:{account_id}")
        if data:
            return json.loads(data)

        # Fallback to DB
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM accounts WHERE id = $1", account_id)

        if not row:
            raise HTTPException(status_code=404, detail="Account not found")

        return dict(row)

    async def list_accounts(self) -> list:
        """List all accounts"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM accounts ORDER BY created_at")

        accounts = []
        for row in rows:
            acc = dict(row)
            acc["initial_balance"] = float(acc["initial_balance"])
            acc["current_balance"] = float(acc["current_balance"])
            acc["created_at"] = acc["created_at"].isoformat()
            acc["updated_at"] = acc["updated_at"].isoformat()
            accounts.append(acc)

        return accounts

    async def update_account(self, account_id: str, data: AccountUpdate) -> dict:
        """Update account"""
        updates = []
        values = [account_id]
        idx = 2

        if data.name is not None:
            updates.append(f"name = ${idx}")
            values.append(data.name)
            idx += 1

        if data.risk_profile is not None:
            updates.append(f"risk_profile = ${idx}")
            values.append(data.risk_profile.value)
            idx += 1

        if data.description is not None:
            updates.append(f"description = ${idx}")
            values.append(data.description)
            idx += 1

        if data.is_active is not None:
            updates.append(f"is_active = ${idx}")
            values.append(data.is_active)
            idx += 1

        if not updates:
            return await self.get_account(account_id)

        updates.append("updated_at = NOW()")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE accounts SET {', '.join(updates)} WHERE id = $1",
                *values
            )

        await self._sync_to_redis()
        logger.info(f"📝 Account updated: {account_id}")
        return await self.get_account(account_id)

    async def update_balance(self, account_id: str, amount: float, tx_type: str, description: str = "") -> dict:
        """Update account balance"""
        async with self.pool.acquire() as conn:
            # Update balance
            await conn.execute("""
                UPDATE accounts SET current_balance = current_balance + $1, updated_at = NOW()
                WHERE id = $2
            """, amount, account_id)

            # Record transaction
            await conn.execute("""
                INSERT INTO account_transactions (account_id, type, amount, description)
                VALUES ($1, $2, $3, $4)
            """, account_id, tx_type, amount, description)

        await self._sync_to_redis()
        return await self.get_account(account_id)

    async def get_main_account(self) -> dict | None:
        """Get main account for profit distribution"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM accounts WHERE is_main = true")

        if row:
            acc = dict(row)
            acc["initial_balance"] = float(acc["initial_balance"])
            acc["current_balance"] = float(acc["current_balance"])
            return acc
        return None

    async def set_main_account(self, account_id: str) -> dict:
        """Set account as main"""
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE accounts SET is_main = false")
            await conn.execute("UPDATE accounts SET is_main = true WHERE id = $1", account_id)

        await self._sync_to_redis()
        logger.info(f"⭐ Main account set: {account_id}")
        return await self.get_account(account_id)

    async def get_positions(self, account_id: str) -> list:
        """Get account positions from Redis"""
        positions_data = await self.redis.hgetall(f"positions:{account_id}")
        positions = []

        for ticker, data in positions_data.items():
            pos = json.loads(data)
            positions.append({"ticker": ticker, **pos})

        return positions

    async def get_transactions(self, account_id: str, limit: int = 50) -> list:
        """Get account transactions"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM account_transactions
                WHERE account_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, account_id, limit)

        return [dict(r) for r in rows]

    async def delete_account(self, account_id: str) -> dict:
        """Delete account (only if no positions)"""
        positions = await self.get_positions(account_id)
        if positions:
            raise HTTPException(status_code=400, detail="Cannot delete account with open positions")

        account = await self.get_account(account_id)
        if account.get("is_main"):
            raise HTTPException(status_code=400, detail="Cannot delete main account")

        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM account_transactions WHERE account_id = $1", account_id)
            await conn.execute("DELETE FROM accounts WHERE id = $1", account_id)

        await self.redis.delete(f"account:{account_id}")
        ACCOUNTS_TOTAL.dec()

        logger.info(f"🗑️ Account deleted: {account_id}")
        return {"deleted": account_id}


# Initialize
svc = AccountManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Account Manager",
    description="Multi-account management (up to 3)
# ============================================================
# METRICS ENDPOINT (fixed - no 307 redirects)
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
",
    version="1.0.0",
    lifespan=lifespan
)

# OLD: metrics mount removed


@app.get("/health")
async def health():
    accounts = await svc.list_accounts()
    return {"status": "healthy", "accounts": len(accounts), "max": MAX_ACCOUNTS}


@app.post("/accounts")
async def create_account(data: AccountCreate):
    """Create new account"""
    return await svc.create_account(data)


@app.get("/accounts")
async def list_accounts():
    """List all accounts"""
    return await svc.list_accounts()


@app.get("/accounts/{account_id}")
async def get_account(account_id: str):
    """Get account by ID"""
    return await svc.get_account(account_id)


@app.patch("/accounts/{account_id}")
async def update_account(account_id: str, data: AccountUpdate):
    """Update account"""
    return await svc.update_account(account_id, data)


@app.delete("/accounts/{account_id}")
async def delete_account(account_id: str):
    """Delete account"""
    return await svc.delete_account(account_id)


@app.get("/accounts/{account_id}/positions")
async def get_positions(account_id: str):
    """Get account positions"""
    return await svc.get_positions(account_id)


@app.get("/accounts/{account_id}/transactions")
async def get_transactions(account_id: str, limit: int = 50):
    """Get account transactions"""
    return await svc.get_transactions(account_id, limit)


@app.post("/accounts/{account_id}/balance")
async def update_balance(account_id: str, amount: float, tx_type: str = "adjustment", description: str = ""):
    """Adjust account balance"""
    return await svc.update_balance(account_id, amount, tx_type, description)


@app.get("/main")
async def get_main_account():
    """Get main account"""
    return await svc.get_main_account()


@app.post("/accounts/{account_id}/set-main")
async def set_main_account(account_id: str):
    """Set as main account"""
    return await svc.set_main_account(account_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
