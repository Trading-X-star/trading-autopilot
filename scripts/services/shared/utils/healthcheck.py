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
