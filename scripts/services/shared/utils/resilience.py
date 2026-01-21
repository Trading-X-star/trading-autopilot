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
