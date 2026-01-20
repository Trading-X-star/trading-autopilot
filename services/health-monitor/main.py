#!/usr/bin/env python3
"""Health Monitor - Service health checks and system monitoring"""
import asyncio
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from prometheus_client import Gauge, Counter, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("health-monitor")

# Service endpoints to monitor
SERVICES = {
    "orchestrator": "http://orchestrator:8000/health",
    "data-ingestion": "http://data-ingestion:8002/health",
    "risk-manager": "http://risk-manager:8001/health",
    "execution": "http://execution:8003/health",
    "portfolio": "http://portfolio:8004/health",
    "strategy": "http://strategy:8005/health",
    "account-manager": "http://account-manager:8020/health",
    "trailing-stop": "http://trailing-stop:8023/health",
    "profit-distribution": "http://profit-distribution:8024/health",
    "decision-router": "http://decision-router:8021/health",
    "multi-dashboard": "http://multi-dashboard:8022/health",
    "alert-manager": "http://alert-manager:8012/health",
}

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))  # seconds
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "3"))  # failures before alert

# Metrics
SERVICE_UP = Gauge("service_up", "Service health status", ["service"])
SERVICE_LATENCY = Gauge("service_latency_ms", "Service response latency", ["service"])
HEALTH_CHECKS = Counter("health_checks_total", "Total health checks", ["service", "status"])


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthMonitor:
    def __init__(self):
        self.redis = None
        self.http = None
        self.running = False
        self.status: dict[str, dict] = {}
        self.failure_counts: dict[str, int] = {}

    async def start(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=10.0)

        # Initialize status
        for service in SERVICES:
            self.status[service] = {
                "status": ServiceStatus.UNKNOWN,
                "last_check": None,
                "latency_ms": None,
                "details": None
            }
            self.failure_counts[service] = 0

        self.running = True
        asyncio.create_task(self._monitor_loop())

        logger.info(f"✅ Health Monitor started (checking {len(SERVICES)} services every {CHECK_INTERVAL}s)")

    async def stop(self):
        self.running = False
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("🛑 Health Monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_all_services()
                await self._update_redis_status()
                await asyncio.sleep(CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)

    async def _check_all_services(self):
        """Check all services in parallel"""
        tasks = [self._check_service(name, url) for name, url in SERVICES.items()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service(self, name: str, url: str):
        """Check single service health"""
        start = datetime.now()

        try:
            response = await self.http.get(url)
            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code == 200:
                status = ServiceStatus.HEALTHY
                details = response.json()
                self.failure_counts[name] = 0
            else:
                status = ServiceStatus.DEGRADED
                details = {"error": f"HTTP {response.status_code}"}
                self.failure_counts[name] += 1

        except httpx.ConnectError:
            latency = None
            status = ServiceStatus.UNHEALTHY
            details = {"error": "Connection refused"}
            self.failure_counts[name] += 1

        except httpx.TimeoutException:
            latency = 10000  # timeout
            status = ServiceStatus.DEGRADED
            details = {"error": "Timeout"}
            self.failure_counts[name] += 1

        except Exception as e:
            latency = None
            status = ServiceStatus.UNHEALTHY
            details = {"error": str(e)}
            self.failure_counts[name] += 1

        # Update status
        self.status[name] = {
            "status": status,
            "last_check": datetime.now().isoformat(),
            "latency_ms": round(latency, 2) if latency else None,
            "details": details
        }

        # Update metrics
        SERVICE_UP.labels(service=name).set(1 if status == ServiceStatus.HEALTHY else 0)
        if latency:
            SERVICE_LATENCY.labels(service=name).set(latency)
        HEALTH_CHECKS.labels(service=name, status=status.value).inc()

        # Alert if threshold reached
        if self.failure_counts[name] == ALERT_THRESHOLD:
            await self._send_alert(name, status, details)

        # Log status changes
        if status != ServiceStatus.HEALTHY:
            logger.warning(f"⚠️ {name}: {status.value} - {details}")

    async def _update_redis_status(self):
        """Update system status in Redis"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "services": self.status,
            "healthy_count": sum(1 for s in self.status.values() if s["status"] == ServiceStatus.HEALTHY),
            "total_count": len(SERVICES)
        }
        await self.redis.set("system:health", json.dumps(summary, default=str))

    async def _send_alert(self, service: str, status: ServiceStatus, details: dict):
        """Send alert for unhealthy service"""
        await self.redis.xadd("stream:alerts", {
            "name": "service_down",
            "severity": "error",
            "message": f"Service {service} is {status.value}: {details.get('error', 'unknown')}",
            "service": service
        }, maxlen=1000)

        logger.error(f"🚨 Alert sent: {service} is {status.value}")

    async def get_status(self) -> dict:
        """Get current health status"""
        healthy = sum(1 for s in self.status.values() if s["status"] == ServiceStatus.HEALTHY)

        overall = ServiceStatus.HEALTHY
        if healthy < len(SERVICES):
            overall = ServiceStatus.DEGRADED
        if healthy < len(SERVICES) / 2:
            overall = ServiceStatus.UNHEALTHY

        return {
            "overall": overall.value,
            "healthy": healthy,
            "total": len(SERVICES),
            "services": self.status,
            "timestamp": datetime.now().isoformat()
        }

    async def check_service(self, name: str) -> dict:
        """Check specific service"""
        if name not in SERVICES:
            return {"error": f"Unknown service: {name}"}

        await self._check_service(name, SERVICES[name])
        return {name: self.status[name]}

    async def get_metrics_summary(self) -> dict:
        """Get metrics summary"""
        healthy = []
        unhealthy = []
        latencies = {}

        for name, status in self.status.items():
            if status["status"] == ServiceStatus.HEALTHY:
                healthy.append(name)
            else:
                unhealthy.append(name)

            if status["latency_ms"]:
                latencies[name] = status["latency_ms"]

        avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0

        return {
            "healthy_services": healthy,
            "unhealthy_services": unhealthy,
            "average_latency_ms": round(avg_latency, 2),
            "slowest_service": max(latencies, key=latencies.get) if latencies else None,
            "fastest_service": min(latencies, key=latencies.get) if latencies else None
        }


# Initialize
svc = HealthMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Health Monitor",
    description="Service health monitoring",
    version="1.0.0",
    lifespan=lifespan
)
# ============================================================
# METRICS ENDPOINT (fixed - no 307 redirects)
# ============================================================
@app.get("/metrics")
@app.get("/metrics/")
async def prometheus_metrics():
    from fastapi import Response
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# OLD: metrics mount removed


@app.get("/health")
async def health():
    status = await svc.get_status()
    return {"status": status["overall"], "healthy": status["healthy"], "total": status["total"]}


@app.get("/status")
async def get_status():
    """Get full health status"""
    return await svc.get_status()


@app.get("/status/{service}")
async def get_service_status(service: str):
    """Get specific service status"""
    if service in svc.status:
        return {service: svc.status[service]}
    return {"error": "Service not found"}


@app.post("/check/{service}")
async def check_service(service: str):
    """Force check specific service"""
    return await svc.check_service(service)


@app.post("/check")
async def check_all():
    """Force check all services"""
    await svc._check_all_services()
    return await svc.get_status()


@app.get("/summary")
async def get_summary():
    """Get metrics summary"""
    return await svc.get_metrics_summary()


@app.get("/services")
async def list_services():
    """List monitored services"""
    return {"services": list(SERVICES.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
