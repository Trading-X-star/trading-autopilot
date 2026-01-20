#!/usr/bin/env python3
"""Alert Manager - Telegram notifications and alert routing"""
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
from pydantic import BaseModel
from prometheus_client import Counter, make_asgi_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("alert-manager")

# Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Metrics
ALERTS_SENT = Counter("alerts_sent_total", "Alerts sent", ["severity", "channel"])
ALERTS_FAILED = Counter("alerts_failed_total", "Failed alerts", ["channel"])


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Alert(BaseModel):
    name: str
    message: str
    severity: Severity = Severity.INFO
    account_id: str | None = None
    ticker: str | None = None
    data: dict | None = None


# Emoji mapping
SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARNING: "⚠️",
    Severity.ERROR: "❌",
    Severity.CRITICAL: "🚨"
}


class AlertManager:
    def __init__(self):
        self.redis = None
        self.http = None
        self.running = False
        self.telegram_enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

    async def start(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True
        )
        self.http = httpx.AsyncClient(timeout=30.0)

        self.running = True
        asyncio.create_task(self._alert_consumer())

        status = "enabled" if self.telegram_enabled else "disabled (no token)"
        logger.info(f"✅ Alert Manager started (Telegram: {status})")

        if self.telegram_enabled:
            await self.send_telegram("🤖 Trading Autopilot started!")

    async def stop(self):
        self.running = False
        if self.telegram_enabled:
            await self.send_telegram("🛑 Trading Autopilot stopped")
        if self.http:
            await self.http.aclose()
        if self.redis:
            await self.redis.close()
        logger.info("🛑 Alert Manager stopped")

    async def _alert_consumer(self):
        """Consume alerts from Redis stream"""
        last_id = "$"  # Only new messages

        while self.running:
            try:
                result = await self.redis.xread(
                    {"stream:alerts": last_id},
                    count=10,
                    block=5000
                )

                if not result:
                    continue

                for stream_name, messages in result:
                    for msg_id, data in messages:
                        last_id = msg_id
                        await self._process_alert(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(5)

    async def _process_alert(self, data: dict):
        """Process and route alert"""
        name = data.get("name", "unknown")
        message = data.get("message", "")
        severity = data.get("severity", "info")

        logger.info(f"🔔 Alert [{severity}]: {name} - {message}")

        # Format message
        emoji = SEVERITY_EMOJI.get(Severity(severity), "ℹ️")

        # Build Telegram message
        lines = [f"{emoji} *{name}*", message]

        if data.get("account_id"):
            lines.append(f"Account: `{data['account_id']}`")
        if data.get("ticker"):
            lines.append(f"Ticker: {data['ticker']}")
        if data.get("price"):
            lines.append(f"Price: {data['price']}")
        if data.get("pnl_pct"):
            pnl = float(data['pnl_pct'])
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            lines.append(f"{pnl_emoji} P&L: {pnl:+.1f}%")

        text = "\n".join(lines)

        # Send to Telegram
        if self.telegram_enabled:
            success = await self.send_telegram(text)
            if success:
                ALERTS_SENT.labels(severity=severity, channel="telegram").inc()
            else:
                ALERTS_FAILED.labels(channel="telegram").inc()

        # Store in history
        await self._store_alert(data)

    async def send_telegram(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message to Telegram"""
        if not self.telegram_enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            response = await self.http.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode
            })

            if response.status_code == 200:
                logger.debug(f"📤 Telegram sent: {text[:50]}...")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def _store_alert(self, data: dict):
        """Store alert in Redis list (last 1000)"""
        alert = {
            **data,
            "timestamp": datetime.now().isoformat()
        }
        await self.redis.lpush("alerts:history", json.dumps(alert))
        await self.redis.ltrim("alerts:history", 0, 999)

    async def send_alert(self, alert: Alert) -> dict:
        """Send alert manually"""
        data = {
            "name": alert.name,
            "message": alert.message,
            "severity": alert.severity.value,
            "account_id": alert.account_id,
            "ticker": alert.ticker
        }

        if alert.data:
            data.update(alert.data)

        # Add to stream
        await self.redis.xadd("stream:alerts", data, maxlen=1000)

        return {"sent": True, "alert": alert.name}

    async def get_history(self, limit: int = 50) -> list:
        """Get alert history"""
        alerts = await self.redis.lrange("alerts:history", 0, limit - 1)
        return [json.loads(a) for a in alerts]

    async def test_telegram(self) -> dict:
        """Test Telegram connection"""
        if not self.telegram_enabled:
            return {"success": False, "error": "Telegram not configured"}

        success = await self.send_telegram("🧪 Test message from Trading Autopilot")
        return {"success": success}

    async def send_daily_report(self) -> dict:
        """Send daily performance report"""
        # This would be called by scheduler
        # Gather stats from other services

        lines = [
            "📊 *Daily Report*",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "Performance summary will be here...",
        ]

        text = "\n".join(lines)
        success = await self.send_telegram(text)
        return {"sent": success}


# Initialize
svc = AlertManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.start()
    yield
    await svc.stop()


app = FastAPI(
    title="Alert Manager",
    description="Telegram notifications",
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
    return {
        "status": "healthy",
        "telegram_enabled": svc.telegram_enabled
    }


@app.post("/send")
async def send_alert(alert: Alert):
    """Send alert"""
    return await svc.send_alert(alert)


@app.get("/history")
async def get_history(limit: int = 50):
    """Get alert history"""
    return await svc.get_history(limit)


@app.post("/test")
async def test_telegram():
    """Test Telegram connection"""
    return await svc.test_telegram()


@app.post("/telegram")
async def send_telegram_message(text: str):
    """Send custom Telegram message"""
    success = await svc.send_telegram(text)
    return {"success": success}


@app.post("/report/daily")
async def send_daily_report():
    """Send daily report"""
    return await svc.send_daily_report()


# Quick alert endpoints
@app.post("/alert/info")
async def alert_info(name: str, message: str):
    return await svc.send_alert(Alert(name=name, message=message, severity=Severity.INFO))


@app.post("/alert/warning")
async def alert_warning(name: str, message: str):
    return await svc.send_alert(Alert(name=name, message=message, severity=Severity.WARNING))


@app.post("/alert/error")
async def alert_error(name: str, message: str):
    return await svc.send_alert(Alert(name=name, message=message, severity=Severity.ERROR))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
