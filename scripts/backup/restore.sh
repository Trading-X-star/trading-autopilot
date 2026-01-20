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
