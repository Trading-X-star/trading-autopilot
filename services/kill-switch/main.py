from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kill Switch Service")
r = redis.Redis(host='redis', port=6379, decode_responses=True)

class EmergencyResponse(BaseModel):
    status: str
    timestamp: str
    message: str

@app.on_event("startup")
async def startup():
    if not r.exists("TRADING_ENABLED"):
        r.set("TRADING_ENABLED", "true")
    logger.info("Kill Switch Service started")

@app.post("/emergency-stop", response_model=EmergencyResponse)
async def emergency_stop(reason: str = "manual"):
    """Немедленная остановка всей торговли"""
    r.set("TRADING_ENABLED", "false")
    r.set("STOP_REASON", reason)
    r.set("STOP_TIME", datetime.utcnow().isoformat())
    r.publish("emergency", "STOP_ALL")
    
    logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
    
    return EmergencyResponse(
        status="STOPPED",
        timestamp=datetime.utcnow().isoformat(),
        message=f"All trading stopped. Reason: {reason}"
    )

@app.post("/resume-trading", response_model=EmergencyResponse)
async def resume_trading(confirmation: str):
    """Возобновление торговли (требует подтверждения)"""
    if confirmation != "CONFIRM_RESUME":
        raise HTTPException(400, "Invalid confirmation code")
    
    r.set("TRADING_ENABLED", "true")
    r.delete("STOP_REASON")
    r.publish("trading", "RESUMED")
    
    logger.info("✅ Trading resumed")
    
    return EmergencyResponse(
        status="ACTIVE",
        timestamp=datetime.utcnow().isoformat(),
        message="Trading resumed"
    )

@app.get("/status")
async def get_status():
    return {
        "trading_enabled": r.get("TRADING_ENABLED") == "true",
        "stop_reason": r.get("STOP_REASON"),
        "stop_time": r.get("STOP_TIME")
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "kill-switch"}
