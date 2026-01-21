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
