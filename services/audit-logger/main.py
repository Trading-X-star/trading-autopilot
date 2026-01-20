from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import asyncpg
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audit Logger")
db_pool = None

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "trading")
DB_USER = os.getenv("DB_USER", "trading")
DB_PASS = os.getenv("DB_PASSWORD", "trading123")

class AuditEvent(BaseModel):
    event_type: str
    actor: str
    action: str
    resource: str
    details: dict = {}

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host=DB_HOST, database=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    await db_pool.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            event_type VARCHAR(50),
            actor VARCHAR(100),
            action VARCHAR(100),
            resource VARCHAR(200),
            details JSONB,
            ip_address VARCHAR(50)
        )
    ''')
    logger.info("✅ Audit Logger started")

@app.post("/log")
async def log_event(event: AuditEvent, request: Request):
    await db_pool.execute(
        'INSERT INTO audit_log (event_type, actor, action, resource, details, ip_address) VALUES ($1,$2,$3,$4,$5,$6)',
        event.event_type, event.actor, event.action, event.resource, json.dumps(event.details), request.client.host
    )
    return {"status": "logged"}

@app.get("/logs")
async def get_logs(limit: int = 100):
    rows = await db_pool.fetch('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT $1', limit)
    return [dict(r) for r in rows]

@app.get("/health")
async def health():
    return {"status": "healthy"}
