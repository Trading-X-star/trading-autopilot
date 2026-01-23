"""Self-Healing Watchdog - автоматическое восстановление сервисов"""
import asyncio
import os
import subprocess
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("watchdog")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
MAX_RESTARTS = int(os.getenv("MAX_RESTARTS", "3"))
RESTART_COOLDOWN = int(os.getenv("RESTART_COOLDOWN", "300"))  # 5 min

SERVICES = {
    "strategy": {"url": "http://strategy:8005/health", "critical": True},
    "executor": {"url": "http://executor:8007/health", "critical": True},
    "scheduler": {"url": "http://scheduler:8009/health", "critical": True},
    "datafeed": {"url": "http://datafeed:8006/health", "critical": True},
    "risk-manager": {"url": "http://risk-manager:8001/health", "critical": True},
    "dashboard": {"url": "http://dashboard:8080/health", "critical": False},
    "orchestrator": {"url": "http://orchestrator:8000/health", "critical": False},
}

class SelfHealingWatchdog:
    def __init__(self):
        self.redis = None
        self.restart_counts = defaultdict(int)
        self.last_restart = {}
        self.service_status = {}
        self.running = False
        self.events = []
    
    async def start(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.running = True
        asyncio.create_task(self.monitor_loop())
        logger.info("🐕 Watchdog started")
    
    async def stop(self):
        self.running = False
        if self.redis:
            await self.redis.close()
        logger.info("🐕 Watchdog stopped")
    
    async def check_service(self, name: str, config: dict) -> tuple:
        """Check service health, returns (is_healthy, latency_ms, error)"""
        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(config["url"])
                latency = (datetime.now() - start).total_seconds() * 1000
                
                if resp.status_code == 200:
                    return True, latency, None
                else:
                    return False, latency, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return False, 10000, "Timeout"
        except Exception as e:
            return False, 0, str(e)
    
    async def heal_service(self, name: str, reason: str):
        """Attempt to heal a service"""
        now = datetime.now()
        
        # Check cooldown
        last = self.last_restart.get(name)
        if last and (now - last).seconds < RESTART_COOLDOWN:
            logger.warning(f"⏳ {name} in cooldown, skipping heal")
            return False
        
        # Check max restarts
        if self.restart_counts[name] >= MAX_RESTARTS:
            logger.error(f"🚨 {name} exceeded max restarts ({MAX_RESTARTS})")
            await self.alert(f"CRITICAL: {name} exceeded max restarts, manual intervention required")
            return False
        
        logger.warning(f"🔧 Healing {name}: {reason}")
        self.events.append({
            "time": now.isoformat(),
            "service": name,
            "action": "restart",
            "reason": reason
        })
        
        # Try restart
        try:
            result = subprocess.run(
                ["docker", "compose", "restart", name],
                capture_output=True,
                timeout=60,
                cwd="/app"
            )
            
            self.restart_counts[name] += 1
            self.last_restart[name] = now
            
            # Wait and verify
            await asyncio.sleep(15)
            is_healthy, _, _ = await self.check_service(name, SERVICES[name])
            
            if is_healthy:
                logger.info(f"✅ {name} healed successfully")
                await self.alert(f"✅ Service {name} recovered after restart")
                return True
            else:
                # Try recreate
                logger.warning(f"⚠️ {name} still unhealthy, trying recreate")
                subprocess.run(
                    ["docker", "compose", "up", "-d", "--force-recreate", name],
                    capture_output=True,
                    timeout=120,
                    cwd="/app"
                )
                await asyncio.sleep(20)
                
                is_healthy, _, _ = await self.check_service(name, SERVICES[name])
                if is_healthy:
                    logger.info(f"✅ {name} healed after recreate")
                    await self.alert(f"✅ Service {name} recovered after recreate")
                    return True
                else:
                    logger.error(f"❌ {name} could not be healed")
                    await self.alert(f"❌ CRITICAL: {name} could not be healed")
                    return False
                    
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Heal timeout for {name}")
            return False
        except Exception as e:
            logger.error(f"❌ Heal failed for {name}: {e}")
            return False
    
    async def alert(self, message: str):
        """Send alert via Redis pub/sub"""
        try:
            await self.redis.publish("watchdog:alerts", message)
            await self.redis.xadd(
                "stream:alerts",
                {"source": "watchdog", "message": message, "severity": "warning"},
                maxlen=1000
            )
        except Exception as e:
            logger.error(f"Alert failed: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                for name, config in SERVICES.items():
                    is_healthy, latency, error = await self.check_service(name, config)
                    
                    self.service_status[name] = {
                        "healthy": is_healthy,
                        "latency_ms": latency,
                        "error": error,
                        "checked_at": datetime.now().isoformat()
                    }
                    
                    if not is_healthy:
                        logger.warning(f"⚠️ {name} unhealthy: {error}")
                        if config["critical"]:
                            await self.heal_service(name, error)
                    else:
                        # Reset restart count on successful check
                        if name in self.restart_counts and self.restart_counts[name] > 0:
                            # Decay restart count over time
                            last = self.last_restart.get(name)
                            if last and (datetime.now() - last).seconds > RESTART_COOLDOWN * 2:
                                self.restart_counts[name] = max(0, self.restart_counts[name] - 1)
                
                await asyncio.sleep(CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(CHECK_INTERVAL)
    
    def get_status(self) -> dict:
        healthy_count = sum(1 for s in self.service_status.values() if s.get("healthy"))
        return {
            "status": "healthy" if healthy_count == len(SERVICES) else "degraded",
            "services": self.service_status,
            "restart_counts": dict(self.restart_counts),
            "recent_events": self.events[-20:]
        }

watchdog = SelfHealingWatchdog()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await watchdog.start()
    yield
    await watchdog.stop()

app = FastAPI(title="Self-Healing Watchdog", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/status")
async def status():
    return watchdog.get_status()

@app.post("/heal/{service}")
async def manual_heal(service: str):
    if service not in SERVICES:
        return {"error": f"Unknown service: {service}"}
    result = await watchdog.heal_service(service, "manual_request")
    return {"service": service, "healed": result}

@app.post("/reset-counts")
async def reset_counts():
    watchdog.restart_counts.clear()
    watchdog.last_restart.clear()
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8026)
