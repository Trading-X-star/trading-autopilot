#!/bin/bash
# Weekly model retraining
set -e
cd /home/omen/trading-autopilot

DATE=$(date +%Y-%m-%d)
LOG_FILE="logs/cron/retrain_$DATE.log"

echo "[$DATE] Starting weekly retrain..." | tee $LOG_FILE

# Trigger training
curl -X POST http://localhost:8025/train \
  -H "Content-Type: application/json" \
  -d '{"min_accuracy": 0.52, "optimize": true, "n_trials": 50}' \
  | tee -a $LOG_FILE

echo "[$DATE] Retrain completed" | tee -a $LOG_FILE
