from fastapi import FastAPI
from pydantic import BaseModel
import redis
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kill Switch Service")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

r = redis.Redis(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD, decode_responses=True)

class EmergencyResponse(BaseModel):
    status: str
    timestamp: str
    message: str

@app.on_event("startup")
async def startup():
    if not r.exists("TRADING_ENABLED"):
        r.set("TRADING_ENABLED", "true")
    logger.info("✅ Kill Switch Service started")

@app.post("/emergency-stop")
async def emergency_stop(reason: str = "manual"):
    r.set("TRADING_ENABLED", "false")
    r.set("STOP_REASON", reason)
    r.set("STOP_TIME", datetime.utcnow().isoformat())
    logger.critical(f"🚨 EMERGENCY STOP: {reason}")
    return EmergencyResponse(status="STOPPED", timestamp=datetime.utcnow().isoformat(), message=f"Stopped: {reason}")

@app.post("/resume-trading")
async def resume_trading(confirmation: str = ""):
    if confirmation != "CONFIRM_RESUME":
        return {"error": "Invalid confirmation"}
    r.set("TRADING_ENABLED", "true")
    r.delete("STOP_REASON")
    logger.info("✅ Trading resumed")
    return EmergencyResponse(status="ACTIVE", timestamp=datetime.utcnow().isoformat(), message="Trading resumed")

@app.get("/status")
async def get_status():
    return {"trading_enabled": r.get("TRADING_ENABLED") == "true", "stop_reason": r.get("STOP_REASON"), "stop_time": r.get("STOP_TIME")}

@app.get("/health")
async def health():
    return {"status": "healthy"}
