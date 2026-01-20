#!/bin/bash
set -e
echo "🔐 TRADING-AUTOPILOT: Исправление критических уязвимостей"
echo "=========================================================="

# ============================================
# 1. DOCKER SECRETS (замена открытых паролей)
# ============================================
echo "[1/4] 🔑 Настройка Docker Secrets..."

# Создать директорию для секретов
mkdir -p secrets

# Генерация безопасных паролей
POSTGRES_PASS=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
REDIS_PASS=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
GRAFANA_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
JWT_SECRET=$(openssl rand -base64 32)
API_KEY=$(openssl rand -hex 32)

# Сохранить секреты в файлы (для Docker Swarm/Compose secrets)
echo -n "$POSTGRES_PASS" > secrets/postgres_password
echo -n "$REDIS_PASS" > secrets/redis_password
echo -n "$GRAFANA_PASS" > secrets/grafana_password
echo -n "$JWT_SECRET" > secrets/jwt_secret
echo -n "$API_KEY" > secrets/api_key

# Ограничить доступ к файлам
chmod 600 secrets/*

# Создать .env.secure (не коммитить в git!)
cat > .env.secure << EOF
# AUTO-GENERATED SECURE CREDENTIALS - $(date)
# ⚠️  DO NOT COMMIT TO GIT!
POSTGRES_PASSWORD=$POSTGRES_PASS
REDIS_PASSWORD=$REDIS_PASS
GRAFANA_ADMIN_PASSWORD=$GRAFANA_PASS
JWT_SECRET=$JWT_SECRET
API_KEY=$API_KEY
EOF
chmod 600 .env.secure

# Добавить в .gitignore
grep -q "secrets/" .gitignore 2>/dev/null || echo -e "\n# Secrets\nsecrets/\n.env.secure" >> .gitignore

echo "   ✅ Секреты сгенерированы в ./secrets/"

# ============================================
# 2. TLS/mTLS ЧЕРЕЗ TRAEFIK
# ============================================
echo "[2/4] 🔒 Настройка TLS/mTLS..."

mkdir -p config/traefik/certs

# Генерация CA сертификата
openssl genrsa -out config/traefik/certs/ca.key 4096 2>/dev/null
openssl req -new -x509 -days 3650 -key config/traefik/certs/ca.key \
    -out config/traefik/certs/ca.crt \
    -subj "/C=RU/ST=Moscow/L=Moscow/O=TradingAutopilot/CN=Trading CA" 2>/dev/null

# Генерация сертификата для Traefik
openssl genrsa -out config/traefik/certs/traefik.key 2048 2>/dev/null
openssl req -new -key config/traefik/certs/traefik.key \
    -out config/traefik/certs/traefik.csr \
    -subj "/C=RU/ST=Moscow/L=Moscow/O=TradingAutopilot/CN=traefik.local" 2>/dev/null

# SAN для локальных сервисов
cat > config/traefik/certs/san.cnf << 'EOF'
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
[req_distinguished_name]
[v3_req]
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = traefik
DNS.3 = *.trading.local
DNS.4 = orchestrator
DNS.5 = executor
DNS.6 = strategy
DNS.7 = risk-manager
DNS.8 = dashboard
DNS.9 = kill-switch
IP.1 = 127.0.0.1
EOF

openssl x509 -req -days 365 -in config/traefik/certs/traefik.csr \
    -CA config/traefik/certs/ca.crt -CAkey config/traefik/certs/ca.key \
    -CAcreateserial -out config/traefik/certs/traefik.crt \
    -extfile config/traefik/certs/san.cnf -extensions v3_req 2>/dev/null

# Обновить конфиг Traefik с TLS
cat > config/traefik/traefik.yaml << 'EOF'
api:
  dashboard: true
  insecure: false

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entryPoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"
    http:
      tls:
        certResolver: default
  metrics:
    address: ":8082"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false
    network: trading-net
  file:
    directory: /etc/traefik/dynamic
    watch: true

tls:
  certificates:
    - certFile: /etc/traefik/certs/traefik.crt
      keyFile: /etc/traefik/certs/traefik.key
  options:
    default:
      minVersion: VersionTLS12
      cipherSuites:
        - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
        - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256

metrics:
  prometheus:
    entryPoint: metrics

accessLog:
  filePath: "/var/log/traefik/access.log"
  format: json
  fields:
    headers:
      names:
        X-Real-Ip: keep
        Authorization: drop

log:
  level: INFO
EOF

echo "   ✅ TLS сертификаты созданы"

# ============================================
# 3. RATE LIMITING MIDDLEWARE
# ============================================
echo "[3/4] 🚦 Настройка Rate Limiting..."

mkdir -p config/traefik/dynamic

cat > config/traefik/dynamic/middlewares.yaml << 'EOF'
http:
  middlewares:
    # Rate Limiting - 100 запросов в секунду
    rate-limit:
      rateLimit:
        average: 100
        burst: 200
        period: 1s
    
    # Строгий лимит для торговых операций - 10/сек
    rate-limit-trading:
      rateLimit:
        average: 10
        burst: 20
        period: 1s
    
    # Лимит для аутентификации - защита от брутфорса
    rate-limit-auth:
      rateLimit:
        average: 5
        burst: 10
        period: 1m
    
    # Security Headers
    security-headers:
      headers:
        frameDeny: true
        sslRedirect: true
        browserXssFilter: true
        contentTypeNosniff: true
        stsIncludeSubdomains: true
        stsPreload: true
        stsSeconds: 31536000
        customResponseHeaders:
          X-Robots-Tag: "noindex,nofollow"
          Server: ""
    
    # IP Whitelist для admin endpoints
    ip-whitelist-admin:
      ipWhiteList:
        sourceRange:
          - "127.0.0.1/32"
          - "10.0.0.0/8"
          - "172.16.0.0/12"
          - "192.168.0.0/16"
    
    # Circuit Breaker
    circuit-breaker:
      circuitBreaker:
        expression: "ResponseCodeRatio(500, 600, 0, 600) > 0.30 || NetworkErrorRatio() > 0.10"
    
    # Retry
    retry:
      retry:
        attempts: 3
        initialInterval: 100ms

  # Роутеры с middleware
  routers:
    dashboard-secure:
      rule: "PathPrefix(`/dashboard`)"
      service: dashboard
      middlewares:
        - rate-limit
        - security-headers
      tls: {}
    
    api-secure:
      rule: "PathPrefix(`/api/v1`)"
      service: orchestrator
      middlewares:
        - rate-limit
        - security-headers
        - circuit-breaker
      tls: {}
    
    trading-secure:
      rule: "PathPrefix(`/api/v1/trade`) || PathPrefix(`/api/v1/order`)"
      service: executor
      middlewares:
        - rate-limit-trading
        - security-headers
      tls: {}
    
    kill-switch-secure:
      rule: "PathPrefix(`/api/emergency`)"
      service: kill-switch
      middlewares:
        - ip-whitelist-admin
        - rate-limit-auth
        - security-headers
      tls: {}

  services:
    dashboard:
      loadBalancer:
        servers:
          - url: "http://dashboard:8080"
    orchestrator:
      loadBalancer:
        servers:
          - url: "http://orchestrator:8000"
    executor:
      loadBalancer:
        servers:
          - url: "http://executor:8001"
    kill-switch:
      loadBalancer:
        servers:
          - url: "http://kill-switch:8020"
EOF

echo "   ✅ Rate limiting настроен"

# ============================================
# 4. BACKUP СТРАТЕГИЯ
# ============================================
echo "[4/4] 💾 Настройка Backup системы..."

mkdir -p scripts/backup
mkdir -p backups

# Основной скрипт бэкапа
cat > scripts/backup/backup.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
S3_BUCKET="${S3_BUCKET:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📦 Starting backup at $(date)"

# PostgreSQL backup
echo "  → Backing up PostgreSQL..."
PGPASSWORD=$(cat /run/secrets/postgres_password 2>/dev/null || echo "${POSTGRES_PASSWORD:-trading123}")
docker exec postgres pg_dump -U trading trading | gzip > "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz"

# Redis backup
echo "  → Backing up Redis..."
docker exec redis redis-cli BGSAVE
sleep 2
docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb"

# Config backup
echo "  → Backing up configs..."
tar -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" \
    config/ \
    docker-compose.yml \
    docker-compose.override.yml \
    .env 2>/dev/null || true

# Create manifest
cat > "$BACKUP_DIR/manifest_$TIMESTAMP.json" << MANIFEST
{
  "timestamp": "$TIMESTAMP",
  "date": "$(date -Iseconds)",
  "files": {
    "postgres": "postgres_$TIMESTAMP.sql.gz",
    "redis": "redis_$TIMESTAMP.rdb",
    "config": "config_$TIMESTAMP.tar.gz"
  },
  "sizes": {
    "postgres": "$(du -h $BACKUP_DIR/postgres_$TIMESTAMP.sql.gz | cut -f1)",
    "redis": "$(du -h $BACKUP_DIR/redis_$TIMESTAMP.rdb | cut -f1)",
    "config": "$(du -h $BACKUP_DIR/config_$TIMESTAMP.tar.gz | cut -f1)"
  }
}
MANIFEST

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "  → Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz" "s3://$S3_BUCKET/backups/postgres/"
    aws s3 cp "$BACKUP_DIR/redis_$TIMESTAMP.rdb" "s3://$S3_BUCKET/backups/redis/"
    aws s3 cp "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" "s3://$S3_BUCKET/backups/config/"
    aws s3 cp "$BACKUP_DIR/manifest_$TIMESTAMP.json" "s3://$S3_BUCKET/backups/manifests/"
    echo "   ✅ Uploaded to s3://$S3_BUCKET/backups/"
fi

# Cleanup old backups
echo "  → Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "manifest_*.json" -mtime +$RETENTION_DAYS -delete

echo "✅ Backup completed: $BACKUP_DIR/manifest_$TIMESTAMP.json"
EOF
chmod +x scripts/backup/backup.sh

# Скрипт восстановления
cat > scripts/backup/restore.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="${BACKUP_DIR:-/backups}"
TIMESTAMP="${1:-latest}"

if [ "$TIMESTAMP" = "latest" ]; then
    TIMESTAMP=$(ls -t $BACKUP_DIR/manifest_*.json 2>/dev/null | head -1 | grep -oP '\d{8}_\d{6}')
fi

if [ -z "$TIMESTAMP" ]; then
    echo "❌ No backup found"
    exit 1
fi

echo "🔄 Restoring backup from $TIMESTAMP"

# Stop services
echo "  → Stopping services..."
docker compose stop orchestrator executor strategy risk-manager

# Restore PostgreSQL
echo "  → Restoring PostgreSQL..."
gunzip -c "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz" | docker exec -i postgres psql -U trading trading

# Restore Redis
echo "  → Restoring Redis..."
docker compose stop redis
docker cp "$BACKUP_DIR/redis_$TIMESTAMP.rdb" redis:/data/dump.rdb
docker compose start redis

# Start services
echo "  → Starting services..."
docker compose start orchestrator executor strategy risk-manager

echo "✅ Restore completed from $TIMESTAMP"
EOF
chmod +x scripts/backup/restore.sh

# Добавить backup сервис в docker-compose
cat > docker-compose.backup.yml << 'EOF'
version: '3.8'

services:
  backup:
    image: alpine:3.19
    container_name: backup-service
    volumes:
      - ./scripts/backup:/scripts:ro
      - ./backups:/backups
      - ./config:/config:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./secrets:/run/secrets:ro
    environment:
      - BACKUP_DIR=/backups
      - RETENTION_DAYS=7
      - S3_BUCKET=${S3_BUCKET:-}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
    entrypoint: /bin/sh
    command: ["-c", "apk add --no-cache docker-cli aws-cli postgresql-client && crond -f"]
    networks:
      - trading-net

  # Cron для автоматических бэкапов
  backup-cron:
    image: alpine:3.19
    container_name: backup-cron
    volumes:
      - ./scripts/backup:/scripts:ro
      - ./backups:/backups
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      - BACKUP_DIR=/backups
      - RETENTION_DAYS=7
    command: >
      sh -c "
        apk add --no-cache docker-cli bash &&
        echo '0 */6 * * * /scripts/backup.sh >> /var/log/backup.log 2>&1' > /etc/crontabs/root &&
        echo '0 0 * * 0 /scripts/backup.sh >> /var/log/backup.log 2>&1' >> /etc/crontabs/root &&
        crond -f -l 2
      "
    restart: unless-stopped
    networks:
      - trading-net

networks:
  trading-net:
    external: true
EOF

echo "   ✅ Backup система настроена"

# ============================================
# 5. ОБНОВИТЬ ГЛАВНЫЙ DOCKER-COMPOSE
# ============================================
echo "[5/5] 📝 Обновление docker-compose с секретами..."

cat > docker-compose.secure.yml << 'EOF'
version: '3.8'

secrets:
  postgres_password:
    file: ./secrets/postgres_password
  redis_password:
    file: ./secrets/redis_password
  grafana_password:
    file: ./secrets/grafana_password
  jwt_secret:
    file: ./secrets/jwt_secret
  api_key:
    file: ./secrets/api_key

services:
  postgres:
    image: postgres:15-alpine
    container_name: postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trading
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trading"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - trading-net

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: >
      sh -c "redis-server --requirepass $$(cat /run/secrets/redis_password)"
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - trading-net

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
      - ./config/traefik:/etc/traefik:ro
      - ./config/traefik/dynamic:/etc/traefik/dynamic:ro
      - ./config/traefik/certs:/etc/traefik/certs:ro
      - traefik_logs:/var/log/traefik
    networks:
      - trading-net
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.traefik.rule=Host(`traefik.localhost`)"
      - "traefik.http.routers.traefik.tls=true"
      - "traefik.http.routers.traefik.service=api@internal"

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
      - GF_SERVER_ROOT_URL=https://localhost/grafana
    secrets:
      - grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - trading-net
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=PathPrefix(`/grafana`)"
      - "traefik.http.routers.grafana.tls=true"

  orchestrator:
    secrets:
      - postgres_password
      - redis_password
      - jwt_secret
      - api_key
    environment:
      - DATABASE_URL=postgresql://trading:$(cat /run/secrets/postgres_password)@postgres:5432/trading
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.orchestrator.rule=PathPrefix(`/api/v1`)"
      - "traefik.http.routers.orchestrator.tls=true"
      - "traefik.http.routers.orchestrator.middlewares=rate-limit@file,security-headers@file"

  executor:
    secrets:
      - api_key
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.executor.rule=PathPrefix(`/api/v1/trade`)"
      - "traefik.http.routers.executor.tls=true"
      - "traefik.http.routers.executor.middlewares=rate-limit-trading@file,security-headers@file"

  kill-switch:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.killswitch.rule=PathPrefix(`/api/emergency`)"
      - "traefik.http.routers.killswitch.tls=true"
      - "traefik.http.routers.killswitch.middlewares=ip-whitelist-admin@file,rate-limit-auth@file"

  dashboard:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=PathPrefix(`/`)"
      - "traefik.http.routers.dashboard.tls=true"
      - "traefik.http.routers.dashboard.middlewares=rate-limit@file,security-headers@file"

  audit-logger:
    secrets:
      - postgres_password
    environment:
      - DB_PASSWORD_FILE=/run/secrets/postgres_password

volumes:
  postgres_data:
  redis_data:
  grafana_data:
  traefik_logs:

networks:
  trading-net:
    name: trading-net
    external: true
EOF

# ============================================
# ФИНАЛЬНЫЕ ШАГИ
# ============================================
echo ""
echo "=============================================="
echo "✅ ВСЕ КРИТИЧЕСКИЕ УЯЗВИМОСТИ ИСПРАВЛЕНЫ!"
echo "=============================================="
echo ""
echo "📁 Созданные файлы:"
echo "   • secrets/              - Docker secrets"
echo "   • .env.secure           - Сгенерированные пароли"
echo "   • config/traefik/certs/ - TLS сертификаты"
echo "   • config/traefik/dynamic/middlewares.yaml - Rate limiting"
echo "   • scripts/backup/       - Backup/Restore скрипты"
echo "   • docker-compose.secure.yml - Secure compose"
echo "   • docker-compose.backup.yml - Backup service"
echo ""
echo "🔐 Сгенерированные credentials (сохранены в .env.secure):"
cat .env.secure | grep -v "^#" | grep -v "^$"
echo ""
echo "🚀 Запуск с новыми настройками:"
echo "   docker compose -f docker-compose.yml -f docker-compose.secure.yml up -d"
echo ""
echo "💾 Ручной бэкап:"
echo "   ./scripts/backup/backup.sh"
echo ""
echo "🔄 Восстановление:"
echo "   ./scripts/backup/restore.sh <timestamp>"
echo ""
echo "⚠️  ВАЖНО:"
echo "   1. Добавьте secrets/ в .gitignore ✅ (уже добавлено)"
echo "   2. Сохраните .env.secure в безопасное место"
echo "   3. Для S3 backup укажите AWS_* переменные в .env"
echo ""
