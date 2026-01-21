#!/bin/bash
# =============================================================================
# TRADING AUTOPILOT - COMPLETE IMPROVEMENT SCRIPT
# Применяет все рекомендации по улучшению одной командой
# =============================================================================

set -e
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     TRADING AUTOPILOT - APPLYING ALL IMPROVEMENTS                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# =============================================================================
# 1. СЕКРЕТЫ И БЕЗОПАСНОСТЬ
# =============================================================================
log "1/12 Настройка безопасности и секретов..."

mkdir -p secrets .env.d

# Генерация безопасных паролей
generate_password() { openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32; }

POSTGRES_PASS=$(generate_password)
REDIS_PASS=$(generate_password)
GRAFANA_PASS=$(generate_password)
JWT_SECRET=$(generate_password)
API_KEY=$(generate_password)

# Сохранение секретов
echo -n "$POSTGRES_PASS" > secrets/postgres_password
echo -n "$REDIS_PASS" > secrets/redis_password
echo -n "$GRAFANA_PASS" > secrets/grafana_password
echo -n "$JWT_SECRET" > secrets/jwt_secret
echo -n "$API_KEY" > secrets/api_key
chmod 600 secrets/*

# Создание .env.secure
cat > .env.secure << ENVEOF
# Auto-generated secure environment - $(date)
POSTGRES_PASSWORD=$POSTGRES_PASS
REDIS_PASSWORD=$REDIS_PASS
GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASS
JWT_SECRET=$JWT_SECRET
API_KEY=$API_KEY
DATABASE_URL=postgresql://trading:\${POSTGRES_PASSWORD}@postgres:5432/trading
REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379/0
ENVEOF
chmod 600 .env.secure

success "Секреты сгенерированы в secrets/ и .env.secure"

# =============================================================================
# 2. ИСПРАВЛЕНИЕ DOCKER-COMPOSE С СЕКРЕТАМИ
# =============================================================================
log "2/12 Обновление docker-compose.yml..."

cat > docker-compose.secure.yml << 'DCEOF'
version: "3.8"

x-common: &common
  restart: unless-stopped
  networks:
    - trading-net
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"

services:
  postgres:
    image: postgres:16-alpine
    <<: *common
    container_name: postgres
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trading
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./configs/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading -d trading"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    <<: *common
    container_name: redis
    command: >
      sh -c "redis-server 
      --requirepass $$(cat /run/secrets/redis_password)
      --appendonly yes 
      --maxmemory 512mb 
      --maxmemory-policy allkeys-lru"
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "$$(cat /run/secrets/redis_password)", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

secrets:
  postgres_password:
    file: ./secrets/postgres_password
  redis_password:
    file: ./secrets/redis_password
  grafana_password:
    file: ./secrets/grafana_password
  tinkoff_token:
    file: ./secrets/tinkoff_token.txt

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  trading-net:
    driver: bridge
DCEOF

success "docker-compose.secure.yml создан"

# =============================================================================
# 3. ИСПРАВЛЕНИЕ DECIMAL VS FLOAT
# =============================================================================
log "3/12 Создание модуля для работы с Decimal..."

mkdir -p services/shared/utils

cat > services/shared/utils/decimal_utils.py << 'PYEOF'
"""
Decimal utilities for financial calculations.
All monetary values MUST use Decimal, not float.
"""
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Union, Optional
import functools

# Precision for different types
PRICE_PRECISION = Decimal("0.01")      # 2 decimal places for prices
QUANTITY_PRECISION = Decimal("1")       # Whole numbers for shares
PERCENT_PRECISION = Decimal("0.0001")   # 4 decimal places for percentages
PNL_PRECISION = Decimal("0.01")         # 2 decimal places for P&L

def to_decimal(value: Union[str, int, float, Decimal, None], default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert any value to Decimal."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        if isinstance(value, float):
            # Avoid float precision issues
            return Decimal(str(value))
        return Decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        return default

def round_price(value: Union[Decimal, float, str]) -> Decimal:
    """Round to price precision (2 decimal places)."""
    return to_decimal(value).quantize(PRICE_PRECISION, rounding=ROUND_HALF_UP)

def round_percent(value: Union[Decimal, float, str]) -> Decimal:
    """Round to percentage precision (4 decimal places)."""
    return to_decimal(value).quantize(PERCENT_PRECISION, rounding=ROUND_HALF_UP)

def round_quantity(value: Union[Decimal, float, str]) -> int:
    """Round to whole shares."""
    return int(to_decimal(value).quantize(QUANTITY_PRECISION, rounding=ROUND_HALF_UP))

def calculate_pnl(entry_price: Decimal, exit_price: Decimal, quantity: int, side: str = "buy") -> Decimal:
    """Calculate P&L with proper precision."""
    entry = to_decimal(entry_price)
    exit = to_decimal(exit_price)
    qty = Decimal(quantity)

    if side.lower() == "buy":
        pnl = (exit - entry) * qty
    else:
        pnl = (entry - exit) * qty

    return pnl.quantize(PNL_PRECISION, rounding=ROUND_HALF_UP)

def calculate_percent_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """Calculate percentage change with proper precision."""
    old = to_decimal(old_value)
    new = to_decimal(new_value)

    if old == 0:
        return Decimal("0")

    change = ((new - old) / old) * Decimal("100")
    return change.quantize(PERCENT_PRECISION, rounding=ROUND_HALF_UP)

def decimal_safe(func):
    """Decorator to ensure function returns Decimal-safe values."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, float):
            return to_decimal(result)
        if isinstance(result, dict):
            return {k: to_decimal(v) if isinstance(v, float) else v for k, v in result.items()}
        return result
    return wrapper

# Export commonly used
__all__ = [
    'Decimal', 'to_decimal', 'round_price', 'round_percent', 
    'round_quantity', 'calculate_pnl', 'calculate_percent_change',
    'decimal_safe', 'PRICE_PRECISION', 'PERCENT_PRECISION'
]
PYEOF

success "Модуль decimal_utils.py создан"

# =============================================================================
# 4. CIRCUIT BREAKER И RETRY ЛОГИКА
# =============================================================================
log "4/12 Создание Circuit Breaker и Retry модулей..."

cat > services/shared/utils/resilience.py << 'PYEOF'
"""
Resilience patterns: Circuit Breaker, Retry, Timeout
"""
import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Optional, Any, TypeVar, Coroutine
from functools import wraps
from dataclasses import dataclass, field
import httpx

logger = logging.getLogger(__name__)
T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

@dataclass 
class CircuitBreaker:
    """Circuit Breaker pattern implementation."""
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        return False

    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (failures: {self.failure_count})")

# Global circuit breakers registry
_circuits: dict[str, CircuitBreaker] = {}

def get_circuit(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    if name not in _circuits:
        _circuits[name] = CircuitBreaker(name=name, config=config or CircuitBreakerConfig())
    return _circuits[name]

def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker pattern."""
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        circuit = get_circuit(name, config)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not circuit.can_execute():
                raise CircuitOpenError(f"Circuit {name} is OPEN")

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure()
                raise
        return wrapper
    return decorator

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (httpx.TimeoutException, httpx.ConnectError, ConnectionError)

def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for retry with exponential backoff."""
    cfg = config or RetryConfig()

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(cfg.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    if attempt < cfg.max_attempts - 1:
                        delay = min(
                            cfg.base_delay * (cfg.exponential_base ** attempt),
                            cfg.max_delay
                        )
                        logger.warning(f"Retry {attempt + 1}/{cfg.max_attempts} after {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)

            raise last_exception
        return wrapper
    return decorator

class ResilientHTTPClient:
    """HTTP client with built-in resilience patterns."""

    def __init__(self, base_url: str, timeout: float = 10.0, circuit_name: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.circuit = get_circuit(circuit_name or base_url) if circuit_name else None
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @with_retry()
    async def get(self, path: str, **kwargs) -> httpx.Response:
        if self.circuit and not self.circuit.can_execute():
            raise CircuitOpenError(f"Circuit {self.circuit.name} is OPEN")

        try:
            response = await self._client.get(path, **kwargs)
            response.raise_for_status()
            if self.circuit:
                self.circuit.record_success()
            return response
        except Exception as e:
            if self.circuit:
                self.circuit.record_failure()
            raise

    @with_retry()
    async def post(self, path: str, **kwargs) -> httpx.Response:
        if self.circuit and not self.circuit.can_execute():
            raise CircuitOpenError(f"Circuit {self.circuit.name} is OPEN")

        try:
            response = await self._client.post(path, **kwargs)
            response.raise_for_status()
            if self.circuit:
                self.circuit.record_success()
            return response
        except Exception as e:
            if self.circuit:
                self.circuit.record_failure()
            raise

__all__ = [
    'CircuitBreaker', 'CircuitBreakerConfig', 'CircuitState', 'CircuitOpenError',
    'RetryConfig', 'with_circuit_breaker', 'with_retry', 'get_circuit',
    'ResilientHTTPClient'
]
PYEOF

success "Модуль resilience.py создан"

# =============================================================================
# 5. CONNECTION POOLING
# =============================================================================
log "5/12 Создание Connection Pool Manager..."

cat > services/shared/utils/connections.py << 'PYEOF'
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
PYEOF

success "Модуль connections.py создан"

# =============================================================================
# 6. BATCH PROCESSING ДЛЯ СИГНАЛОВ
# =============================================================================
log "6/12 Создание Batch Signal Processor..."

cat > services/shared/utils/batch_processor.py << 'PYEOF'
"""
Batch processing utilities for efficient signal generation.
"""
import asyncio
import logging
from typing import List, TypeVar, Callable, Coroutine, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)
T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchConfig:
    batch_size: int = 50
    max_concurrent: int = 10
    timeout_per_batch: float = 30.0
    retry_failed: bool = True

@dataclass
class BatchResult:
    successful: List[Any]
    failed: List[tuple]  # (item, error)
    total_time: float
    batches_processed: int

class BatchProcessor:
    """Process items in batches with concurrency control."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process items in batches with progress tracking."""
        start_time = time.time()
        successful = []
        failed = []

        # Split into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]

        total_items = len(items)
        processed = 0

        for batch_idx, batch in enumerate(batches):
            batch_results = await self._process_batch(batch, processor)

            for item, result, error in batch_results:
                if error is None:
                    successful.append(result)
                else:
                    failed.append((item, error))
                    if self.config.retry_failed:
                        # Single retry
                        try:
                            retry_result = await asyncio.wait_for(
                                processor(item),
                                timeout=self.config.timeout_per_batch / len(batch)
                            )
                            successful.append(retry_result)
                            failed.pop()  # Remove from failed
                        except Exception as e:
                            pass  # Keep in failed

            processed += len(batch)
            if on_progress:
                on_progress(processed, total_items)

            logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: "
                        f"{len([r for _, r, e in batch_results if e is None])} success")

        return BatchResult(
            successful=successful,
            failed=failed,
            total_time=time.time() - start_time,
            batches_processed=len(batches)
        )

    async def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]]
    ) -> List[tuple]:
        """Process a single batch concurrently."""

        async def process_with_semaphore(item: T) -> tuple:
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        processor(item),
                        timeout=self.config.timeout_per_batch
                    )
                    return (item, result, None)
                except Exception as e:
                    logger.warning(f"Failed to process {item}: {e}")
                    return (item, None, e)

        tasks = [process_with_semaphore(item) for item in batch]
        return await asyncio.gather(*tasks)

class SignalBatchGenerator:
    """Specialized batch processor for trading signals."""

    def __init__(self, tickers: List[str], batch_size: int = 50):
        self.tickers = tickers
        self.processor = BatchProcessor(BatchConfig(
            batch_size=batch_size,
            max_concurrent=10,
            timeout_per_batch=60.0
        ))

    async def generate_all_signals(
        self,
        signal_generator: Callable[[str], Coroutine[Any, Any, dict]]
    ) -> dict:
        """Generate signals for all tickers in batches."""

        start = datetime.now()
        logger.info(f"Starting batch signal generation for {len(self.tickers)} tickers")

        def on_progress(done: int, total: int):
            pct = (done / total) * 100
            logger.info(f"Signal generation progress: {done}/{total} ({pct:.1f}%)")

        result = await self.processor.process(
            self.tickers,
            signal_generator,
            on_progress
        )

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            f"Signal generation completed: {len(result.successful)} success, "
            f"{len(result.failed)} failed in {elapsed:.2f}s"
        )

        return {
            "signals": result.successful,
            "failed_tickers": [t for t, _ in result.failed],
            "total_time": result.total_time,
            "generated_at": datetime.now().isoformat()
        }

__all__ = ['BatchProcessor', 'BatchConfig', 'BatchResult', 'SignalBatchGenerator']
PYEOF

success "Модуль batch_processor.py создан"

# =============================================================================
# 7. REDIS LUA SCRIPTS ДЛЯ TRAILING STOPS
# =============================================================================
log "7/12 Создание Redis Lua scripts..."

mkdir -p services/shared/lua

cat > services/shared/lua/trailing_stop.lua << 'LUAEOF'
-- Atomic trailing stop update
-- KEYS[1] = stop key
-- ARGV[1] = current_price
-- ARGV[2] = trailing_pct
-- ARGV[3] = breakeven_trigger_pct
-- ARGV[4] = breakeven_offset_pct

local stop_data = redis.call('GET', KEYS[1])
if not stop_data then
    return nil
end

local stop = cjson.decode(stop_data)
local current_price = tonumber(ARGV[1])
local trailing_pct = tonumber(ARGV[2])
local breakeven_trigger = tonumber(ARGV[3])
local breakeven_offset = tonumber(ARGV[4])

local entry_price = tonumber(stop.entry_price)
local highest_price = tonumber(stop.highest_price) or entry_price
local stop_price = tonumber(stop.stop_price)
local breakeven_active = stop.breakeven_active or false

local updated = false
local triggered = false
local trigger_type = nil

-- Update highest price
if current_price > highest_price then
    highest_price = current_price
    stop.highest_price = highest_price
    updated = true
end

-- Calculate new trailing stop
local new_stop = highest_price * (1 - trailing_pct / 100)
if new_stop > stop_price then
    stop.stop_price = new_stop
    stop_price = new_stop
    updated = true
end

-- Check breakeven activation
local profit_pct = ((current_price / entry_price) - 1) * 100
if not breakeven_active and profit_pct >= breakeven_trigger then
    local breakeven_price = entry_price * (1 + breakeven_offset / 100)
    if breakeven_price > stop_price then
        stop.stop_price = breakeven_price
        stop_price = breakeven_price
        stop.breakeven_active = true
        updated = true
    end
end

-- Check if stop triggered
if current_price <= stop_price then
    triggered = true
    trigger_type = stop.breakeven_active and 'breakeven' or 'trailing'
end

if updated then
    stop.updated_at = redis.call('TIME')[1]
    redis.call('SET', KEYS[1], cjson.encode(stop))
end

return cjson.encode({
    updated = updated,
    triggered = triggered,
    trigger_type = trigger_type,
    stop_price = stop_price,
    highest_price = highest_price,
    profit_pct = profit_pct
})
LUAEOF

cat > services/shared/lua/batch_price_update.lua << 'LUAEOF'
-- Batch price update for multiple tickers
-- KEYS = ticker keys (price:TICKER)
-- ARGV = prices (JSON array)

local prices = cjson.decode(ARGV[1])
local updated = 0

for i, key in ipairs(KEYS) do
    local price = prices[i]
    if price then
        redis.call('SET', key, cjson.encode(price))
        redis.call('EXPIRE', key, 60)  -- 1 minute TTL
        updated = updated + 1
    end
end

return updated
LUAEOF

success "Lua scripts созданы"

# =============================================================================
# 8. ТЕСТЫ
# =============================================================================
log "8/12 Создание структуры тестов..."

mkdir -p tests/{unit,integration,fixtures}

cat > tests/conftest.py << 'PYEOF'
"""
Pytest configuration and fixtures.
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from decimal import Decimal

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    redis.hget.return_value = None
    redis.hset.return_value = True
    return redis

@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Mock database pool."""
    pool = AsyncMock()
    pool.fetch.return_value = []
    pool.fetchrow.return_value = None
    pool.execute.return_value = "OK"
    return pool

@pytest.fixture
def sample_trade() -> dict:
    """Sample trade data."""
    return {
        "ticker": "SBER",
        "side": "buy",
        "quantity": 100,
        "price": Decimal("295.50"),
        "account_id": "test_account"
    }

@pytest.fixture
def sample_signal() -> dict:
    """Sample trading signal."""
    return {
        "ticker": "GAZP",
        "signal": "buy",
        "confidence": 0.75,
        "entry": Decimal("180.25"),
        "target": Decimal("195.00"),
        "stop": Decimal("172.00"),
        "reasoning": "Bullish momentum"
    }

@pytest.fixture
def sample_ohlcv() -> list:
    """Sample OHLCV data."""
    return [
        {"date": "2026-01-20", "open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000},
        {"date": "2026-01-21", "open": 103, "high": 108, "low": 102, "close": 107, "volume": 1200000},
    ]
PYEOF

cat > tests/unit/test_decimal_utils.py << 'PYEOF'
"""Tests for decimal utilities."""
import pytest
from decimal import Decimal
import sys
sys.path.insert(0, 'services/shared/utils')
from decimal_utils import (
    to_decimal, round_price, round_percent, round_quantity,
    calculate_pnl, calculate_percent_change
)

class TestToDecimal:
    def test_from_float(self):
        assert to_decimal(10.5) == Decimal("10.5")

    def test_from_string(self):
        assert to_decimal("100.25") == Decimal("100.25")

    def test_from_int(self):
        assert to_decimal(100) == Decimal("100")

    def test_from_none(self):
        assert to_decimal(None) == Decimal("0")

    def test_from_invalid(self):
        assert to_decimal("invalid") == Decimal("0")

class TestRounding:
    def test_round_price(self):
        assert round_price(100.125) == Decimal("100.13")
        assert round_price(100.124) == Decimal("100.12")

    def test_round_percent(self):
        assert round_percent(0.12345) == Decimal("0.1235")

    def test_round_quantity(self):
        assert round_quantity(10.6) == 11
        assert round_quantity(10.4) == 10

class TestPnLCalculation:
    def test_long_profit(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("110"), 10, "buy")
        assert pnl == Decimal("100.00")

    def test_long_loss(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("90"), 10, "buy")
        assert pnl == Decimal("-100.00")

    def test_short_profit(self):
        pnl = calculate_pnl(Decimal("100"), Decimal("90"), 10, "sell")
        assert pnl == Decimal("100.00")

class TestPercentChange:
    def test_positive_change(self):
        change = calculate_percent_change(Decimal("100"), Decimal("110"))
        assert change == Decimal("10.0000")

    def test_negative_change(self):
        change = calculate_percent_change(Decimal("100"), Decimal("90"))
        assert change == Decimal("-10.0000")

    def test_zero_base(self):
        change = calculate_percent_change(Decimal("0"), Decimal("100"))
        assert change == Decimal("0")
PYEOF

cat > tests/unit/test_resilience.py << 'PYEOF'
"""Tests for resilience patterns."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import sys
sys.path.insert(0, 'services/shared/utils')
from resilience import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    with_circuit_breaker, with_retry, CircuitOpenError
)

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_failures(self):
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_count(self):
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
class TestRetryDecorator:
    async def test_succeeds_first_try(self):
        mock_func = AsyncMock(return_value="success")

        @with_retry()
        async def test_func():
            return await mock_func()

        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    async def test_retries_on_failure(self):
        mock_func = AsyncMock(side_effect=[ConnectionError(), ConnectionError(), "success"])

        @with_retry()
        async def test_func():
            return await mock_func()

        result = await test_func()
        assert result == "success"
        assert mock_func.call_count == 3
PYEOF

cat > pytest.ini << 'PYEOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short --cov=services --cov-report=term-missing --cov-fail-under=70
PYEOF

cat > requirements-test.txt << 'PYEOF'
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
httpx>=0.26.0
aioresponses>=0.7.6
PYEOF

success "Структура тестов создана"

# =============================================================================
# 9. ТИПИЗАЦИЯ И MYPY
# =============================================================================
log "9/12 Настройка mypy и типизации..."

cat > mypy.ini << 'PYEOF'
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True

[mypy-asyncpg.*]
ignore_missing_imports = True

[mypy-redis.*]
ignore_missing_imports = True

[mypy-prometheus_client.*]
ignore_missing_imports = True

[mypy-uvicorn.*]
ignore_missing_imports = True
PYEOF

success "mypy.ini создан"

# =============================================================================
# 10. СТРУКТУРИРОВАННОЕ ЛОГИРОВАНИЕ
# =============================================================================
log "10/12 Настройка структурированного логирования..."

cat > services/shared/utils/logging_config.py << 'PYEOF'
"""
Structured JSON logging with correlation IDs.
"""
import logging
import json
import sys
import uuid
from datetime import datetime
from typing import Optional
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id.set(cid)
    return cid

def set_correlation_id(cid: str):
    """Set correlation ID (e.g., from incoming request header)."""
    correlation_id.set(cid)

class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add location info
        log_obj["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }

        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging(
    service_name: str,
    level: str = "INFO",
    json_format: bool = True
) -> logging.Logger:
    """Setup logging for a service."""

    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    logger.addHandler(handler)
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra fields to log records."""

    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        extra['extra_fields'] = {
            **self.extra,
            **extra.get('extra_fields', {})
        }
        kwargs['extra'] = extra
        return msg, kwargs

def get_logger(name: str, **extra_fields) -> LoggerAdapter:
    """Get a logger with extra fields."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, extra_fields)

__all__ = [
    'setup_logging', 'get_logger', 'get_correlation_id', 
    'set_correlation_id', 'JSONFormatter', 'LoggerAdapter'
]
PYEOF

success "Модуль logging_config.py создан"

# =============================================================================
# 11. КЭШИРОВАНИЕ
# =============================================================================
log "11/12 Создание модуля кэширования..."

cat > services/shared/utils/caching.py << 'PYEOF'
"""
Caching utilities with TTL support.
"""
import asyncio
import json
import hashlib
import logging
from typing import Optional, Any, Callable, TypeVar
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)
T = TypeVar('T')

@dataclass
class CacheConfig:
    default_ttl: int = 300  # 5 minutes
    max_size: int = 10000
    prefix: str = "cache:"

class RedisCache:
    """Redis-backed cache with TTL."""

    def __init__(self, redis_client, config: Optional[CacheConfig] = None):
        self.redis = redis_client
        self.config = config or CacheConfig()

    def _make_key(self, key: str) -> str:
        return f"{self.config.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = await self.redis.get(self._make_key(key))
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        try:
            ttl = ttl or self.config.default_ttl
            await self.redis.set(
                self._make_key(key),
                json.dumps(value, default=str),
                ex=ttl
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    async def delete(self, key: str):
        """Delete key from cache."""
        try:
            await self.redis.delete(self._make_key(key))
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")

    async def get_or_set(
        self, 
        key: str, 
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and store."""
        value = await self.get(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

def cached(
    ttl: int = 300,
    key_builder: Optional[Callable[..., str]] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key from function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache (requires cache to be set on wrapper)
            if hasattr(wrapper, '_cache'):
                cached_value = await wrapper._cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {func.__name__}")
                    return cached_value

            # Compute value
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Store in cache
            if hasattr(wrapper, '_cache'):
                await wrapper._cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator

class FeatureCache:
    """Specialized cache for trading features."""

    def __init__(self, redis_client):
        self.cache = RedisCache(redis_client, CacheConfig(
            default_ttl=60,  # 1 minute for features
            prefix="features:"
        ))

    async def get_features(self, ticker: str, date: str) -> Optional[dict]:
        key = f"{ticker}:{date}"
        return await self.cache.get(key)

    async def set_features(self, ticker: str, date: str, features: dict):
        key = f"{ticker}:{date}"
        await self.cache.set(key, features, ttl=300)

    async def get_batch_features(self, tickers: list, date: str) -> dict:
        """Get features for multiple tickers."""
        results = {}
        missing = []

        for ticker in tickers:
            cached = await self.get_features(ticker, date)
            if cached:
                results[ticker] = cached
            else:
                missing.append(ticker)

        return results, missing

__all__ = ['RedisCache', 'CacheConfig', 'cached', 'FeatureCache']
PYEOF

success "Модуль caching.py создан"

# =============================================================================
# 12. HEALTHCHECK FIXES
# =============================================================================
log "12/12 Исправление healthchecks..."

cat > services/shared/utils/healthcheck.py << 'PYEOF'
"""
Unified health check utilities.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float
    message: Optional[str] = None

@dataclass
class ServiceHealth:
    service: str
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    timestamp: str

class HealthChecker:
    """Unified health checker for services."""

    def __init__(self, service_name: str, version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.utcnow()
        self._checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self._checks[name] = check_func

    async def check_component(self, name: str, check_func: callable) -> ComponentHealth:
        """Run a single component health check."""
        start = datetime.utcnow()
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=5.0)
            else:
                result = check_func()

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            if result is True or result == "ok":
                return ComponentHealth(name, HealthStatus.HEALTHY, latency)
            elif result:
                return ComponentHealth(name, HealthStatus.HEALTHY, latency, str(result))
            else:
                return ComponentHealth(name, HealthStatus.UNHEALTHY, latency, "Check returned False")

        except asyncio.TimeoutError:
            latency = 5000.0
            return ComponentHealth(name, HealthStatus.UNHEALTHY, latency, "Timeout")
        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return ComponentHealth(name, HealthStatus.UNHEALTHY, latency, str(e))

    async def get_health(self) -> ServiceHealth:
        """Get overall service health."""
        components = {}

        # Run all checks concurrently
        if self._checks:
            tasks = [
                self.check_component(name, func) 
                for name, func in self._checks.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, ComponentHealth):
                    components[result.name] = result
                elif isinstance(result, Exception):
                    logger.error(f"Health check error: {result}")

        # Determine overall status
        statuses = [c.status for c in components.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return ServiceHealth(
            service=self.service_name,
            status=overall,
            version=self.version,
            uptime_seconds=uptime,
            components={k: asdict(v) for k, v in components.items()},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def to_dict(self, health: ServiceHealth) -> dict:
        """Convert health to dict for JSON response."""
        return {
            "service": health.service,
            "status": health.status.value,
            "version": health.version,
            "uptime_seconds": round(health.uptime_seconds, 2),
            "components": {
                k: {
                    "status": v["status"].value if isinstance(v["status"], HealthStatus) else v["status"],
                    "latency_ms": round(v["latency_ms"], 2),
                    "message": v.get("message")
                }
                for k, v in health.components.items()
            },
            "timestamp": health.timestamp
        }

# Common health check functions
async def check_postgres(pool) -> bool:
    """Check PostgreSQL connection."""
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        raise Exception(f"PostgreSQL: {e}")

async def check_redis(redis_client) -> bool:
    """Check Redis connection."""
    try:
        await redis_client.ping()
        return True
    except Exception as e:
        raise Exception(f"Redis: {e}")

__all__ = [
    'HealthChecker', 'HealthStatus', 'ServiceHealth', 'ComponentHealth',
    'check_postgres', 'check_redis'
]
PYEOF

success "Модуль healthcheck.py создан"

# =============================================================================
# ФИНАЛИЗАЦИЯ
# =============================================================================
log "Создание __init__.py файлов..."

cat > services/shared/__init__.py << 'PYEOF'
"""Shared utilities for trading-autopilot services."""
PYEOF

cat > services/shared/utils/__init__.py << 'PYEOF'
"""Utility modules."""
from .decimal_utils import *
from .resilience import *
from .connections import *
from .batch_processor import *
from .caching import *
from .healthcheck import *
from .logging_config import *
PYEOF

# =============================================================================
# MAKEFILE
# =============================================================================
log "Создание Makefile..."

cat > Makefile << 'MKEOF'
.PHONY: all build up down test lint type-check security clean logs

# Default target
all: lint type-check test build

# Build all services
build:
	docker compose build

# Start with secure config
up:
	@if [ ! -f .env.secure ]; then ./apply_improvements.sh; fi
	docker compose -f docker-compose.yml -f docker-compose.secure.yml up -d

# Stop all services
down:
	docker compose down

# Run tests
test:
	pip install -r requirements-test.txt
	pytest tests/ -v --cov=services --cov-report=html

# Lint code
lint:
	pip install ruff
	ruff check services/ --fix

# Type checking
type-check:
	pip install mypy
	mypy services/ --ignore-missing-imports

# Security scan
security:
	pip install bandit safety
	bandit -r services/ -ll
	safety check

# Clean up
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

# View logs
logs:
	docker compose logs -f --tail=100

# Health check
health:
	@echo "Checking services..."
	@curl -s http://localhost:8000/health | jq . || echo "Orchestrator: DOWN"
	@curl -s http://localhost:8009/health | jq . || echo "Scheduler: DOWN"
	@curl -s http://localhost:8006/health | jq . || echo "Datafeed: DOWN"
	@curl -s http://localhost:8005/health | jq . || echo "Strategy: DOWN"
	@curl -s http://localhost:8007/health | jq . || echo "Executor: DOWN"
	@curl -s http://localhost:8001/health | jq . || echo "Risk Manager: DOWN"
MKEOF

success "Makefile создан"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    УЛУЧШЕНИЯ ПРИМЕНЕНЫ!                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Созданные файлы и модули:"
echo "  ✓ secrets/              - Безопасные пароли"
echo "  ✓ .env.secure           - Переменные окружения"
echo "  ✓ docker-compose.secure.yml"
echo "  ✓ services/shared/utils/decimal_utils.py"
echo "  ✓ services/shared/utils/resilience.py"
echo "  ✓ services/shared/utils/connections.py"
echo "  ✓ services/shared/utils/batch_processor.py"
echo "  ✓ services/shared/utils/caching.py"
echo "  ✓ services/shared/utils/healthcheck.py"
echo "  ✓ services/shared/utils/logging_config.py"
echo "  ✓ services/shared/lua/trailing_stop.lua"
echo "  ✓ tests/                - Структура тестов"
echo "  ✓ mypy.ini              - Типизация"
echo "  ✓ pytest.ini            - Конфигурация тестов"
echo "  ✓ Makefile              - Автоматизация"
echo ""
echo "Следующие шаги:"
echo "  1. make build           - Пересобрать контейнеры"
echo "  2. make up              - Запустить с новыми настройками"
echo "  3. make test            - Запустить тесты"
echo "  4. make health          - Проверить здоровье сервисов"
echo ""
