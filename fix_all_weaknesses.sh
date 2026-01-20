#!/bin/bash
set -e
echo "🔧 TRADING-AUTOPILOT: Исправление всех слабых сторон"
echo "===================================================="

# ============================================
# 1. KILL SWITCH SERVICE
# ============================================
echo "[1/10] 🛑 Создание Kill Switch..."
mkdir -p services/kill-switch

cat > services/kill-switch/main.py << 'EOF'
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
EOF

cat > services/kill-switch/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
redis==5.0.1
pydantic==2.5.3
EOF

cat > services/kill-switch/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8020/health || exit 1
EXPOSE 8020
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]
EOF

# ============================================
# 2. CIRCUIT BREAKER LIBRARY
# ============================================
echo "[2/10] ⚡ Создание Circuit Breaker..."
mkdir -p services/shared

cat > services/shared/circuit_breaker.py << 'EOF'
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
EOF

cat > services/shared/requirements.txt << 'EOF'
circuitbreaker==2.0.0
aiohttp==3.9.1
tenacity==8.2.3
EOF

# ============================================
# 3. ADVANCED RISK MANAGER
# ============================================
echo "[3/10] 📊 Улучшение Risk Manager..."

cat > services/risk-manager/advanced_risk.py << 'EOF'
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    account_id: str

@dataclass
class RiskDecision:
    allowed: bool
    reason: Optional[str] = None
    adjusted_quantity: Optional[float] = None

class AdvancedRiskManager:
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        
        # Конфигурируемые параметры
        self.max_drawdown_pct = float(self.r.get("RISK:MAX_DRAWDOWN") or 5.0)
        self.max_position_pct = float(self.r.get("RISK:MAX_POSITION") or 10.0)
        self.daily_loss_limit_pct = float(self.r.get("RISK:DAILY_LOSS") or 2.0)
        self.max_correlation = float(self.r.get("RISK:MAX_CORRELATION") or 0.7)
        self.volatility_threshold = float(self.r.get("RISK:VOL_THRESHOLD") or 30.0)
    
    async def check_trade(self, trade: Trade) -> RiskDecision:
        """Комплексная проверка сделки"""
        
        # Проверка Kill Switch
        if self.r.get("TRADING_ENABLED") != "true":
            return RiskDecision(False, "Trading is disabled by kill switch")
        
        checks = await asyncio.gather(
            self.check_position_size(trade),
            self.check_drawdown(trade.account_id),
            self.check_daily_loss(trade.account_id),
            self.check_volatility_regime(trade.symbol),
            self.check_concentration_risk(trade),
            return_exceptions=True
        )
        
        for i, result in enumerate(checks):
            if isinstance(result, Exception):
                logger.error(f"Risk check {i} failed: {result}")
                return RiskDecision(False, f"Risk check error: {result}")
            if not result[0]:
                return RiskDecision(False, result[1])
        
        return RiskDecision(True)
    
    async def check_position_size(self, trade: Trade) -> tuple:
        portfolio_value = float(self.r.get(f"PORTFOLIO:{trade.account_id}:VALUE") or 0)
        if portfolio_value == 0:
            return (False, "Portfolio value unknown")
        
        trade_value = trade.quantity * trade.price
        position_pct = (trade_value / portfolio_value) * 100
        
        if position_pct > self.max_position_pct:
            return (False, f"Position size {position_pct:.1f}% exceeds limit {self.max_position_pct}%")
        return (True, None)
    
    async def check_drawdown(self, account_id: str) -> tuple:
        peak = float(self.r.get(f"PORTFOLIO:{account_id}:PEAK") or 0)
        current = float(self.r.get(f"PORTFOLIO:{account_id}:VALUE") or 0)
        
        if peak == 0:
            return (True, None)
        
        drawdown = ((peak - current) / peak) * 100
        if drawdown > self.max_drawdown_pct:
            return (False, f"Drawdown {drawdown:.1f}% exceeds limit {self.max_drawdown_pct}%")
        return (True, None)
    
    async def check_daily_loss(self, account_id: str) -> tuple:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_pnl = float(self.r.get(f"PNL:{account_id}:{today}") or 0)
        portfolio_value = float(self.r.get(f"PORTFOLIO:{account_id}:VALUE") or 1)
        
        daily_loss_pct = abs(min(0, daily_pnl)) / portfolio_value * 100
        if daily_loss_pct > self.daily_loss_limit_pct:
            return (False, f"Daily loss {daily_loss_pct:.1f}% exceeds limit {self.daily_loss_limit_pct}%")
        return (True, None)
    
    async def check_volatility_regime(self, symbol: str) -> tuple:
        rvi = float(self.r.get("MARKET:RVI") or 20)
        if rvi > self.volatility_threshold:
            return (False, f"High volatility regime (RVI={rvi:.1f}), trading paused")
        return (True, None)
    
    async def check_concentration_risk(self, trade: Trade) -> tuple:
        """Проверка концентрации по секторам"""
        sector = self.r.get(f"SYMBOL:{trade.symbol}:SECTOR") or "unknown"
        sector_exposure = float(self.r.get(f"EXPOSURE:{trade.account_id}:{sector}") or 0)
        
        if sector_exposure > 30:  # 30% max на сектор
            return (False, f"Sector {sector} exposure {sector_exposure:.1f}% too high")
        return (True, None)
EOF

# ============================================
# 4. CONFIG MANAGEMENT
# ============================================
echo "[4/10] ⚙️ Создание системы конфигурации..."
mkdir -p config/{dev,staging,prod}

cat > config/prod/settings.yaml << 'EOF'
# Production Configuration
app:
  name: trading-autopilot
  environment: production
  debug: false

trading:
  enabled: true
  max_accounts: 3
  emergency_stop_enabled: true
  paper_trading: false

risk:
  max_drawdown_pct: 5.0
  max_position_pct: 10.0
  daily_loss_limit_pct: 2.0
  profit_distribution_pct: 10.0
  volatility_threshold: 30.0
  max_sector_exposure_pct: 30.0

moex:
  base_url: "https://iss.moex.com/iss"
  api_timeout_sec: 5
  rate_limit_per_sec: 10
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout_sec: 60
    expected_exceptions:
      - ConnectionError
      - TimeoutError

redis:
  host: redis
  port: 6379
  sentinel_enabled: true
  master_name: "mymaster"
  password_secret: "vault:secret/redis#password"

postgres:
  host: postgres
  port: 5432
  database: trading
  replication_enabled: true
  max_connections: 100
  ssl_mode: require

monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  tracing_enabled: true
  log_level: INFO

alerts:
  telegram_enabled: true
  email_enabled: false
  slack_enabled: false
  drawdown_alert_threshold_pct: 3.0
  daily_loss_alert_threshold_pct: 1.5
EOF

cat > config/dev/settings.yaml << 'EOF'
# Development Configuration
app:
  name: trading-autopilot
  environment: development
  debug: true

trading:
  enabled: true
  max_accounts: 3
  emergency_stop_enabled: true
  paper_trading: true  # Всегда paper в dev!

risk:
  max_drawdown_pct: 10.0
  max_position_pct: 20.0
  daily_loss_limit_pct: 5.0
  profit_distribution_pct: 10.0

moex:
  base_url: "https://iss.moex.com/iss"
  api_timeout_sec: 10
  rate_limit_per_sec: 5
EOF

# ============================================
# 5. API GATEWAY (TRAEFIK)
# ============================================
echo "[5/10] 🚪 Настройка API Gateway..."
mkdir -p config/traefik

cat > config/traefik/traefik.yaml << 'EOF'
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"
  metrics:
    address: ":8082"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: trading-net

metrics:
  prometheus:
    entryPoint: metrics

accessLog:
  filePath: "/var/log/traefik/access.log"
  format: json

log:
  level: INFO
EOF

cat > config/traefik/dynamic.yaml << 'EOF'
http:
  middlewares:
    rate-limit:
      rateLimit:
        average: 100
        burst: 50
    
    auth-headers:
      headers:
        customRequestHeaders:
          X-Forwarded-Proto: "https"
    
    circuit-breaker:
      circuitBreaker:
        expression: "ResponseCodeRatio(500, 600, 0, 600) > 0.30"
EOF

# ============================================
# 6. REDIS SENTINEL
# ============================================
echo "[6/10] 🔄 Настройка Redis Sentinel..."
mkdir -p config/redis

cat > config/redis/sentinel.conf << 'EOF'
port 26379
sentinel monitor mymaster redis 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 60000
sentinel parallel-syncs mymaster 1
EOF

cat > config/redis/redis.conf << 'EOF'
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
EOF

# ============================================
# 7. AUDIT LOGGING
# ============================================
echo "[7/10] 📝 Создание Audit Logger..."
mkdir -p services/audit-logger

cat > services/audit-logger/main.py << 'EOF'
from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import asyncpg
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audit Logger")
db_pool = None

class AuditEvent(BaseModel):
    event_type: str
    actor: str
    action: str
    resource: str
    details: dict
    ip_address: str = None
    timestamp: datetime = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host='postgres', database='trading',
        user='trading', password='trading'
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
            ip_address INET
        )
    ''')
    await db_pool.execute('''
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)
    ''')
    logger.info("Audit Logger started")

@app.post("/log")
async def log_event(event: AuditEvent, request: Request):
    event.ip_address = request.client.host
    event.timestamp = datetime.utcnow()
    
    await db_pool.execute('''
        INSERT INTO audit_log (event_type, actor, action, resource, details, ip_address)
        VALUES ($1, $2, $3, $4, $5, $6)
    ''', event.event_type, event.actor, event.action, 
        event.resource, json.dumps(event.details), event.ip_address)
    
    logger.info(f"AUDIT: {event.actor} {event.action} {event.resource}")
    return {"status": "logged"}

@app.get("/logs")
async def get_logs(limit: int = 100, event_type: str = None):
    if event_type:
        rows = await db_pool.fetch(
            'SELECT * FROM audit_log WHERE event_type = $1 ORDER BY timestamp DESC LIMIT $2',
            event_type, limit
        )
    else:
        rows = await db_pool.fetch(
            'SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT $1', limit
        )
    return [dict(r) for r in rows]

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF

cat > services/audit-logger/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
asyncpg==0.29.0
pydantic==2.5.3
EOF

cat > services/audit-logger/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8021/health || exit 1
EXPOSE 8021
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8021"]
EOF

# ============================================
# 8. ОБНОВЛЕНИЕ DOCKER-COMPOSE
# ============================================
echo "[8/10] 🐳 Обновление docker-compose.yml..."

cat > docker-compose.override.yml << 'EOF'
version: '3.8'

services:
  # API Gateway
  traefik:
    image: traefik:v3.0
    container_name: traefik
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
      - "8081:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./config/traefik:/etc/traefik
    networks:
      - trading-net
    labels:
      - "traefik.enable=true"

  # Kill Switch
  kill-switch:
    build: ./services/kill-switch
    container_name: kill-switch
    restart: unless-stopped
    ports:
      - "8020:8020"
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - trading-net
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.killswitch.rule=PathPrefix(`/api/emergency`)"

  # Audit Logger
  audit-logger:
    build: ./services/audit-logger
    container_name: audit-logger
    restart: unless-stopped
    ports:
      - "8021:8021"
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - trading-net

  # Redis Sentinel
  redis-sentinel:
    image: redis:7-alpine
    container_name: redis-sentinel
    restart: unless-stopped
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./config/redis/sentinel.conf:/etc/redis/sentinel.conf
    depends_on:
      - redis
    networks:
      - trading-net

  # Обновляем существующие сервисы с labels для Traefik
  dashboard:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=PathPrefix(`/dashboard`)"
      - "traefik.http.services.dashboard.loadbalancer.server.port=8080"

  orchestrator:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.orchestrator.rule=PathPrefix(`/api/v1`)"
      - "traefik.http.middlewares.orchestrator-ratelimit.ratelimit.average=100"

networks:
  trading-net:
    external: true
EOF

# ============================================
# 9. GRAFANA ALERTS
# ============================================
echo "[9/10] 🚨 Создание Grafana алертов..."
mkdir -p config/grafana/provisioning/alerting

cat > config/grafana/provisioning/alerting/alerts.yaml << 'EOF'
apiVersion: 1
groups:
  - orgId: 1
    name: Trading Alerts
    folder: Trading
    interval: 1m
    rules:
      - uid: drawdown-alert
        title: High Drawdown Alert
        condition: C
        data:
          - refId: A
            queryType: ''
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: trading_drawdown_percent > 3
        for: 5m
        annotations:
          summary: "Drawdown превысил 3%"
        labels:
          severity: warning

      - uid: daily-loss-alert
        title: Daily Loss Limit Alert
        condition: C
        data:
          - refId: A
            queryType: ''
            relativeTimeRange:
              from: 300
              to: 0
            datasourceUid: prometheus
            model:
              expr: trading_daily_pnl_percent < -2
        for: 1m
        annotations:
          summary: "Дневной убыток превысил 2%"
        labels:
          severity: critical

      - uid: service-down-alert
        title: Service Down Alert
        condition: C
        data:
          - refId: A
            datasourceUid: prometheus
            model:
              expr: up{job=~"trading-.*"} == 0
        for: 2m
        annotations:
          summary: "Сервис недоступен"
        labels:
          severity: critical
EOF

# ============================================
# 10. ФИНАЛЬНАЯ СБОРКА
# ============================================
echo "[10/10] 🏗️ Сборка и запуск..."

# Создать сеть если не существует
docker network create trading-net 2>/dev/null || true

# Собрать новые сервисы
docker compose -f docker-compose.yml -f docker-compose.override.yml build kill-switch audit-logger

# Запустить всё
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo ""
echo "=============================================="
echo "✅ ВСЕ УЛУЧШЕНИЯ ПРИМЕНЕНЫ!"
echo "=============================================="
echo ""
echo "Новые сервисы:"
echo "  • Kill Switch:    http://localhost:8020"
echo "  • Audit Logger:   http://localhost:8021"
echo "  • API Gateway:    http://localhost:80"
echo "  • Traefik UI:     http://localhost:8081"
echo "  • Redis Sentinel: порт 26379"
echo ""
echo "Команды управления:"
echo "  • Остановить торговлю: curl -X POST localhost:8020/emergency-stop"
echo "  • Возобновить:         curl -X POST localhost:8020/resume-trading?confirmation=CONFIRM_RESUME"
echo "  • Статус:              curl localhost:8020/status"
echo "  • Audit logs:          curl localhost:8021/logs"
echo ""
