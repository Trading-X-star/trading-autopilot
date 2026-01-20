import asyncio
import aiohttp
from circuitbreaker import circuit
from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)

class CircuitBreakerConfig:
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 60
    EXPECTED_EXCEPTION = Exception

class MOEXCircuitBreaker:
    def __init__(self):
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def call_api(self, endpoint: str, timeout: int = 5):
        """Protected MOEX API call with circuit breaker"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://iss.moex.com/iss/{endpoint}.json",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    logger.error(f"MOEX API error: {resp.status}")
                    raise Exception(f"MOEX API returned {resp.status}")
                return await resp.json()

class RateLimiter:
    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.tokens = calls_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.calls_per_second,
                self.tokens + time_passed * self.calls_per_second
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.calls_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

moex_breaker = MOEXCircuitBreaker()
rate_limiter = RateLimiter(calls_per_second=10)
