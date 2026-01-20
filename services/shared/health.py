"""Расширенные Health Checks с деградацией"""
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional

class Status(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    name: str
    status: Status
    latency_ms: float = 0
    message: str = ""

@dataclass 
class ServiceHealth:
    service: str
    status: Status
    version: str
    uptime_sec: float
    components: list
    timestamp: str
    
    def dict(self):
        return {
            "service": self.service,
            "status": self.status.value,
            "version": self.version,
            "uptime_sec": round(self.uptime_sec, 1),
            "components": [
                {"name": c.name, "status": c.status.value, "latency_ms": round(c.latency_ms, 2), "message": c.message}
                for c in self.components
            ],
            "timestamp": self.timestamp
        }

class HealthChecker:
    def __init__(self, service: str, version: str = "1.0.0"):
        self.service = service
        self.version = version
        self.start_time = time.time()
        self.checks: list[tuple[str, Callable]] = []
    
    def add(self, name: str, check: Callable):
        self.checks.append((name, check))
        return self
    
    async def check(self) -> ServiceHealth:
        components = []
        overall = Status.HEALTHY
        
        for name, func in self.checks:
            start = time.time()
            try:
                result = await func() if asyncio.iscoroutinefunction(func) else func()
                latency = (time.time() - start) * 1000
                
                if result is True:
                    components.append(ComponentHealth(name, Status.HEALTHY, latency))
                elif result is False:
                    components.append(ComponentHealth(name, Status.UNHEALTHY, latency))
                    overall = Status.UNHEALTHY
                else:
                    components.append(ComponentHealth(name, Status.DEGRADED, latency, str(result)))
                    if overall == Status.HEALTHY:
                        overall = Status.DEGRADED
            except Exception as e:
                components.append(ComponentHealth(name, Status.UNHEALTHY, 0, str(e)[:100]))
                overall = Status.UNHEALTHY
        
        return ServiceHealth(
            service=self.service,
            status=overall,
            version=self.version,
            uptime_sec=time.time() - self.start_time,
            components=components,
            timestamp=datetime.now().isoformat()
        )
