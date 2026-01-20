"""Circuit Breaker для защиты от каскадных сбоев"""
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any
import httpx

class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max: int = 3
    
    state: State = field(default=State.CLOSED, init=False)
    failures: int = field(default=0, init=False)
    last_failure: float = field(default=0, init=False)
    half_open_calls: int = field(default=0, init=False)
    
    def _should_try(self) -> bool:
        if self.state == State.CLOSED:
            return True
        if self.state == State.OPEN:
            if time.time() - self.last_failure >= self.recovery_timeout:
                self.state = State.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        return self.half_open_calls < self.half_open_max
    
    def _on_success(self):
        if self.state == State.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max:
                self.state = State.CLOSED
                self.failures = 0
        else:
            self.failures = 0
    
    def _on_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.state == State.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = State.OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if not self._should_try():
            raise CircuitOpenError(f"Circuit OPEN, retry after {self.recovery_timeout}s")
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

class CircuitOpenError(Exception):
    pass

class ResilientClient:
    """HTTP клиент с Circuit Breaker для каждого сервиса"""
    def __init__(self, timeout: float = 10.0):
        self.http = httpx.AsyncClient(timeout=timeout)
        self.breakers: dict[str, CircuitBreaker] = {}
    
    def _get_breaker(self, service: str) -> CircuitBreaker:
        if service not in self.breakers:
            self.breakers[service] = CircuitBreaker()
        return self.breakers[service]
    
    async def get(self, service: str, url: str, **kwargs):
        breaker = self._get_breaker(service)
        return await breaker.call(self.http.get, url, **kwargs)
    
    async def post(self, service: str, url: str, **kwargs):
        breaker = self._get_breaker(service)
        return await breaker.call(self.http.post, url, **kwargs)
    
    def status(self) -> dict:
        return {name: {"state": b.state.value, "failures": b.failures} 
                for name, b in self.breakers.items()}
    
    async def close(self):
        await self.http.aclose()
