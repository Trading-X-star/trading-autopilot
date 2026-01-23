#!/bin/bash
# Daily automation tasks
set -e
cd /home/omen/trading-autopilot

LOG_DIR="logs/cron"
mkdir -p $LOG_DIR
DATE=$(date +%Y-%m-%d)

echo "[$DATE] Starting daily tasks..."

# 1. Check if retraining needed
echo "[$DATE] Checking retrain status..."
curl -s http://localhost:8025/check-retrain | tee -a $LOG_DIR/retrain_$DATE.log

# 2. Backup databases
echo "[$DATE] Running backup..."
docker exec postgres pg_dump -U trading trading | gzip > backups/postgres_$DATE.sql.gz

# 3. Cleanup old logs
find logs -name "*.log" -mtime +7 -delete
find backups -name "*.gz" -mtime +30 -delete

# 4. Health report
echo "[$DATE] Generating health report..."
curl -s http://localhost:8026/status > $LOG_DIR/health_$DATE.json

echo "[$DATE] Daily tasks completed"
