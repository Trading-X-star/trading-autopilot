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
