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
