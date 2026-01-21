"""
Connection Pool Manager for PostgreSQL and Redis.
Ensures connection reuse and proper resource management.
"""
import asyncio
import os
import logging
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    min_size: int = 5
    max_size: int = 20
    max_idle_time: float = 300.0  # 5 minutes
    command_timeout: float = 30.0

class DatabasePool:
    """PostgreSQL connection pool singleton."""

    _instance: Optional['DatabasePool'] = None
    _pool: Optional[asyncpg.Pool] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self, dsn: Optional[str] = None, config: Optional[PoolConfig] = None):
        async with self._lock:
            if self._pool is not None:
                return

            cfg = config or PoolConfig()
            dsn = dsn or os.getenv("DATABASE_URL")

            if not dsn:
                raise ValueError("DATABASE_URL not set")

            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=cfg.min_size,
                max_size=cfg.max_size,
                max_inactive_connection_lifetime=cfg.max_idle_time,
                command_timeout=cfg.command_timeout
            )
            logger.info(f"PostgreSQL pool initialized: {cfg.min_size}-{cfg.max_size} connections")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            await self.initialize()
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args):
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args):
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def close(self):
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                logger.info("PostgreSQL pool closed")

    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        return self._pool

class RedisPool:
    """Redis connection pool singleton."""

    _instance: Optional['RedisPool'] = None
    _pool: Optional[aioredis.Redis] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self, url: Optional[str] = None, max_connections: int = 50):
        async with self._lock:
            if self._pool is not None:
                return

            url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")

            self._pool = aioredis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=max_connections
            )
            # Test connection
            await self._pool.ping()
            logger.info(f"Redis pool initialized: max {max_connections} connections")

    @property
    def client(self) -> aioredis.Redis:
        if self._pool is None:
            raise RuntimeError("Redis pool not initialized. Call initialize() first.")
        return self._pool

    async def get(self, key: str) -> Optional[str]:
        return await self.client.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        return await self.client.set(key, value, ex=ex)

    async def hget(self, name: str, key: str) -> Optional[str]:
        return await self.client.hget(name, key)

    async def hset(self, name: str, key: str, value: str):
        return await self.client.hset(name, key, value)

    async def publish(self, channel: str, message: str):
        return await self.client.publish(channel, message)

    async def close(self):
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                logger.info("Redis pool closed")

# Global singletons
db_pool = DatabasePool()
redis_pool = RedisPool()

async def init_pools():
    """Initialize all connection pools."""
    await db_pool.initialize()
    await redis_pool.initialize()

async def close_pools():
    """Close all connection pools."""
    await db_pool.close()
    await redis_pool.close()

__all__ = ['db_pool', 'redis_pool', 'init_pools', 'close_pools', 'DatabasePool', 'RedisPool', 'PoolConfig']
