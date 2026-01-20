#!/bin/bash
echo "🤖 Trading Autopilot"
echo "════════════════════════════════════════════"

echo "📦 Starting infrastructure..."
docker-compose up -d postgres redis
sleep 5

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ All services started!"
echo "════════════════════════════════════════════"
echo "📊 Dashboard:    http://localhost:8022"
echo "📈 Grafana:      http://localhost:3000 (admin/admin123)"
echo "📡 Prometheus:   http://localhost:9090"
echo "🔧 API:          http://localhost:8020/docs"
echo "════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  docker-compose logs -f    # Logs"
echo "  docker-compose ps         # Status"
echo "  docker-compose down       # Stop"
