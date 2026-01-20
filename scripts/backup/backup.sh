#!/bin/bash
set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📦 Starting backup at $(date)"

# PostgreSQL backup
echo "  → Backing up PostgreSQL..."
docker exec postgres pg_dump -U trading trading 2>/dev/null | gzip > "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz"
echo "    ✅ postgres_$TIMESTAMP.sql.gz ($(du -h $BACKUP_DIR/postgres_$TIMESTAMP.sql.gz | cut -f1))"

# Redis backup
echo "  → Backing up Redis..."
docker exec redis redis-cli -a "${REDIS_PASSWORD:-}" BGSAVE 2>/dev/null || docker exec redis redis-cli BGSAVE 2>/dev/null
sleep 2
docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb" 2>/dev/null || echo "    ⚠️ Redis RDB not found"

# Config backup
echo "  → Backing up configs..."
tar -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" config/ docker-compose*.yml .env 2>/dev/null

echo ""
echo "✅ Backup completed!"
ls -lh "$BACKUP_DIR"/*$TIMESTAMP* 2>/dev/null
