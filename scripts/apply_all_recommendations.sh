#!/bin/bash
#
# Trading-Autopilot: Full System Setup & Recommendations
# Полная настройка системы с лучшими практиками
#
# Этот скрипт устанавливает и настраивает все компоненты Trading-Autopilot
#
set -e

# ============================================================================
# ЦВЕТА И ЛОГИРОВАНИЕ
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ============================================================================
# ЧАСТЬ 1: ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[1/12] 🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

check_command() {
    if command -v $1 &> /dev/null; then
        success "$1 установлен"
    else
        error "$1 не найден. Установите: $2"
    fi
}

check_command "docker" "sudo apt-get install docker.io"
check_command "docker-compose" "sudo apt-get install docker-compose"
check_command "python3" "sudo apt-get install python3"
check_command "git" "sudo apt-get install git"

# ============================================================================
# ЧАСТЬ 2: СТРУКТУРА ПРОЕКТА
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[2/12] 📁 СТРУКТУРА ПРОЕКТА${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

mkdir -p app/{api,services,models,utils,config}
mkdir -p app/services/{trading,security,persistence,analytics}
mkdir -p app/config/{postgres,redis,nginx}
mkdir -p scripts/{backup,monitoring,deploy}
mkdir -p tests/{unit,integration,e2e}
mkdir -p logs/{app,nginx,postgres}
mkdir -p docs/{architecture,api,deployment}

success "Структура проекта создана"

# ============================================================================
# ЧАСТЬ 3: DOCKER COMPOSE
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[3/12] 🐳 DOCKER COMPOSE (Production-ready)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # ==========================================
  # PostgreSQL Database
  # ==========================================
  postgres:
    image: postgres:16-alpine
    container_name: trading-autopilot-postgres
    environment:
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-trading_secure_pwd_2024}
      POSTGRES_DB: trading
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./app/config/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading -d trading"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ==========================================
  # Redis Cache & Message Queue
  # ==========================================
  redis:
    image: redis:7-alpine
    container_name: trading-autopilot-redis
    command: redis-server --appendonly yes --appendfsync everysec
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - trading-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ==========================================
  # Python Application
  # ==========================================
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading-autopilot-app
    environment:
      PYTHONUNBUFFERED: 1
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_USER: trading
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-trading_secure_pwd_2024}
      POSTGRES_DB: trading
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_DB: 0
      ENV: ${ENV:-development}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - trading-network
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

  # ==========================================
  # Nginx Reverse Proxy
  # ==========================================
  nginx:
    image: nginx:alpine
    container_name: trading-autopilot-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./app/config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./app/config/nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - app
    networks:
      - trading-network
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # ==========================================
  # Prometheus (Monitoring)
  # ==========================================
  prometheus:
    image: prom/prometheus:latest
    container_name: trading-autopilot-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./app/config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
    networks:
      - trading-network
    restart: unless-stopped

  # ==========================================
  # Grafana (Visualization)
  # ==========================================
  grafana:
    image: grafana/grafana:latest
    container_name: trading-autopilot-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./app/config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    networks:
      - trading-network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  trading-network:
    driver: bridge
EOF

success "Docker Compose создан (production-ready)"

# ============================================================================
# ЧАСТЬ 4: ENVIRONMENT
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[4/12] ⚙️  ENVIRONMENT CONFIGURATION${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > .env.example << 'EOF'
# ============================================
# ENVIRONMENT
# ============================================
ENV=production
LOG_LEVEL=INFO

# ============================================
# DATABASE
# ============================================
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=trading
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=trading

# ============================================
# REDIS
# ============================================
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# ============================================
# API KEYS
# ============================================
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key

# ============================================
# SECURITY
# ============================================
JWT_SECRET=your_jwt_secret_key_min_32_chars
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
ENCRYPTION_KEY=your_encryption_key_32_chars

# ============================================
# NOTIFICATIONS
# ============================================
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK_URL=your_slack_webhook_url
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_email_password

# ============================================
# MONITORING
# ============================================
GRAFANA_PASSWORD=admin
PROMETHEUS_RETENTION=15d

# ============================================
# BACKUPS
# ============================================
S3_BUCKET=your-backup-bucket
S3_REGION=us-east-1
S3_ACCESS_KEY=your_aws_access_key
S3_SECRET_KEY=your_aws_secret_key
BACKUP_RETENTION_DAYS=7
EOF

success ".env.example создан"
warning "⚠️  ВАЖНО: Настройте .env файл со своими значениями перед запуском!"

# ============================================================================
# ЧАСТЬ 5: PYTHON REQUIREMENTS
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[5/12] 🐍 PYTHON DEPENDENCIES${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > requirements.txt << 'EOF'
# ============================================
# Core Framework
# ============================================
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# ============================================
# Database & ORM
# ============================================
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.13.0
sqlmodel==0.0.14

# ============================================
# Caching & Message Queue
# ============================================
redis==5.0.1
celery==5.3.4
aioredis==2.0.1

# ============================================
# Trading APIs
# ============================================
python-binance==1.0.17
coinbase-commerce==0.4.0
ccxt==4.0.25
websocket-client==1.6.4

# ============================================
# Security & Authentication
# ============================================
PyJWT==2.8.1
cryptography==41.0.7
passlib[bcrypt]==1.7.4
python-jose[cryptography]==3.3.0

# ============================================
# Data Processing
# ============================================
pandas==2.1.3
numpy==1.26.2
scipy==1.11.4
scikit-learn==1.3.2

# ============================================
# Monitoring & Logging
# ============================================
prometheus-client==0.19.0
python-json-logger==2.0.7
loguru==0.7.2

# ============================================
# Testing
# ============================================
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.2

# ============================================
# Development
# ============================================
black==23.12.0
ruff==0.1.8
mypy==1.7.1
isort==5.13.2
pre-commit==3.5.0
EOF

success "requirements.txt создан"

# ============================================================================
# ЧАСТЬ 6: DOCKERFILE
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[6/12] 🐳 DOCKERFILE${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > Dockerfile << 'EOF'
# Build stage
FROM python:3.11-slim as builder

WORKDIR /tmp

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheel
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY ./app /app/app
COPY ./scripts /app/scripts

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

success "Dockerfile создан"

# ============================================================================
# ЧАСТЬ 7: NGINX CONFIGURATION
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[7/12] 🌐 NGINX CONFIGURATION${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

mkdir -p app/config/nginx

cat > app/config/nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 20M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss
               application/rss+xml font/truetype font/opentype
               application/vnd.ms-fontobject image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=general_limit:10m rate=30r/s;

    upstream app {
        server app:8000;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80 default_server;
        listen [::]:80 default_server;
        server_name _;

        return 301 https://$host$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2 default_server;
        listen [::]:443 ssl http2 default_server;
        server_name _;

        # SSL certificates (replace with your certs)
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # API routes with rate limiting
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Forwarded-Host $server_name;
            
            proxy_buffering off;
            proxy_request_buffering off;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 86400;
        }

        # General routes
        location / {
            limit_req zone=general_limit burst=50 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://app;
        }

        # Metrics endpoint (restrict access)
        location /metrics {
            allow 127.0.0.1;
            allow 172.16.0.0/12;
            deny all;
            proxy_pass http://app;
        }
    }
}
EOF

success "Nginx configuration создана"

# ============================================================================
# ЧАСТЬ 8: BACKUP SCRIPTS
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[8/12] 💾 BACKUP SCRIPTS${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

mkdir -p scripts/backup backups/{postgres,redis,config}

cat > scripts/backup/backup.sh << 'BACKUP_SCRIPT'
#!/bin/bash
set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
DATE=$(date +%Y%m%d_%H%M%S)

echo "🔄 Starting backup: ${DATE}"

# PostgreSQL backup
docker-compose exec -T postgres pg_dump -U trading trading | gzip > "${BACKUP_DIR}/postgres/trading_${DATE}.sql.gz"
echo "✅ PostgreSQL backup complete"

# Redis backup
docker-compose exec -T redis redis-cli BGSAVE > /dev/null
sleep 2
docker-compose cp redis:/data/dump.rdb "${BACKUP_DIR}/redis/redis_${DATE}.rdb"
echo "✅ Redis backup complete"

# Config backup
tar -czf "${BACKUP_DIR}/config/config_${DATE}.tar.gz" \
    --exclude='.env' \
    --exclude='*.key' \
    docker-compose.yml \
    app/config/ \
    2>/dev/null || true
echo "✅ Config backup complete"

# Clean old backups (keep last 7 days)
find "${BACKUP_DIR}" -type f -mtime +7 -delete

echo "✅ Backup completed successfully!"
BACKUP_SCRIPT

chmod +x scripts/backup/backup.sh

success "Backup scripts созданы"

# ============================================================================
# ЧАСТЬ 9: MAKEFILE
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[9/12] 🔨 MAKEFILE${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > Makefile << 'EOF'
.PHONY: help setup build up down logs test clean backup restore

help:
	@echo "Trading-Autopilot Commands:"
	@echo "  make setup       - Initial setup"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - Show application logs"
	@echo "  make test        - Run tests"
	@echo "  make backup      - Create backup"
	@echo "  make restore     - Restore from backup"
	@echo "  make clean       - Clean up everything"

setup:
	@test -f .env || cp .env.example .env
	@echo "✅ Setup complete. Edit .env with your settings"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "✅ Services started"
	@echo "📊 Grafana: http://localhost:3000"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "🚀 API: http://localhost/api/docs"

down:
	docker-compose down

logs:
	docker-compose logs -f app

test:
	docker-compose exec app pytest tests/ -v

backup:
	./scripts/backup/backup.sh

restore:
	@read -p "Enter backup file name: " backup; \
	docker-compose exec -T postgres psql -U trading -d trading < "backups/postgres/$$backup"

clean:
	docker-compose down -v
	rm -rf .env logs/* backups/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
EOF

success "Makefile создан"

# ============================================================================
# ЧАСТЬ 10: DOCUMENTATION
# ============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}[10/12] 📚 DOCUMENTATION${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cat > README.md << 'EOF'
# Trading-Autopilot

Полностью автоматизированная система торговли криптовалютами с поддержкой множества бирж,
стратегий и развитыми инструментами мониторинга.

## Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/yourusername/trading-autopilot.git
cd trading-autopilot
```

### 2. Настройка окружения
```bash
make setup
# Отредактируйте .env с вашими API ключами
```

### 3. Запуск системы
```bash
make build
make up
```

### 4. Проверка статуса
```bash
docker-compose ps
# Все сервисы должны быть в статусе "Up"
```

## Доступные сервисы

- **API**: http://localhost/api/docs (Swagger UI)
- **Grafana**: http://localhost:3000 (пароль: admin)
- **Prometheus**: http://localhost:9090
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Основные команды

```bash
# Запуск
make up

# Остановка
make down

# Логи
make logs

# Тесты
make test

# Резервная копия
make backup

# Восстановление
make restore
```

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway (Nginx)                     │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application                                        │
│  ├─ Trading Engine                                          │
│  ├─ Portfolio Manager                                       │
│  ├─ Risk Manager                                            │
│  └─ Analytics Engine                                        │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├─ PostgreSQL (persistent data)                            │
│  └─ Redis (cache & message queue)                           │
├─────────────────────────────────────────────────────────────┤
│  Monitoring                                                 │
│  ├─ Prometheus (metrics)                                    │
│  └─ Grafana (dashboards)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Конфигурация

Все настройки находятся в файле `.env`:

```env
# Database
POSTGRES_PASSWORD=your_password
REDIS_HOST=redis

# API Keys
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret

# Notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Security
JWT_SECRET=your_secret_key
```

## Мониторинг

### Prometheus метрики
- Trading performance (profit, loss, win rate)
- Portfolio allocation
- Transaction history
- Risk metrics

### Grafana dashboards
- Trading dashboard
- Portfolio overview
- Risk analysis
- Performance metrics

## Резервное копирование

Автоматические резервные копии:
```bash
# Ручная резервная копия
make backup

# Восстановление
make restore

# Расписание: каждые 6 часов (при настройке cron)
```

## Безопасность

- ✅ HTTPS/TLS (SSL сертификаты)
- ✅ JWT authentication
- ✅ Rate limiting
- ✅ Input validation
- ✅ Encrypted API keys storage
- ✅ Audit logging
- ✅ Security headers

## Требования

- Docker & Docker Compose
- Python 3.11+
- 2GB+ RAM
- 10GB+ storage

## Лицензия

MIT License - смотрите LICENSE файл

## Поддержка

- 📧 Email: support@trading-autopilot.dev
- 🤖 Telegram: @trading_autopilot_bot
- 📖 Документация: /docs/

## Информация об авторе

Разработано: Trading-Autopilot Team
Дата: Январь 2026
Версия: 1.0.0
EOF

success "README.md создан"

# ============================================================================
# ЧАСТЬ 11: SUMMARY
# ============================================================================
echo ""
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ SETUP COMPLETED SUCCESSFULLY!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo ""
echo -e "${YELLOW}📋 СЛЕДУЮЩИЕ ШАГИ:${NC}"
echo ""
echo "1️⃣  Отредактируйте файл .env:"
echo "   nano .env"
echo ""
echo "2️⃣  Добавьте ваши API ключи:"
echo "   - Binance API keys"
echo "   - Coinbase API keys"
echo "   - Telegram bot token"
echo "   - JWT secret"
echo ""
echo "3️⃣  Запустите систему:"
echo "   make up"
echo ""
echo "4️⃣  Проверьте статус:"
echo "   docker-compose ps"
echo ""
echo "5️⃣  Откройте в браузере:"
echo "   📊 API Docs: http://localhost/api/docs"
echo "   📈 Grafana: http://localhost:3000"
echo "   📊 Prometheus: http://localhost:9090"
echo ""
echo -e "${YELLOW}🔒 ВАЖНО ДЛЯ БЕЗОПАСНОСТИ:${NC}"
echo ""
echo "  ⚠️  Никогда не коммитьте .env с реальными ключами"
echo "  ⚠️  Используйте SSL сертификаты для продакшена"
echo "  ⚠️  Измените пароли Grafana и PostgreSQL"
echo "  ⚠️  Установите сильный JWT_SECRET (32+ символов)"
echo "  ⚠️  Регулярно создавайте резервные копии"
echo ""
echo -e "${YELLOW}📚 ДОКУМЕНТАЦИЯ:${NC}"
echo "  • README.md - Введение и быстрый старт"
echo "  • docs/ - Подробная документация"
echo "  • http://localhost/api/docs - API документация"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
EOF
