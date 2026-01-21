.PHONY: build up down logs health status

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

health:
	@echo "=== Service Health ==="
	@docker exec postgres pg_isready -U trading 2>/dev/null && echo "PostgreSQL: ✓" || echo "PostgreSQL: ✗"
	@docker exec redis redis-cli ping 2>/dev/null | grep -q PONG && echo "Redis: ✓" || echo "Redis: ✗"
	@curl -sf localhost:8009/health >/dev/null && echo "Scheduler: ✓" || echo "Scheduler: ✗"
	@curl -sf localhost:8000/health >/dev/null && echo "Orchestrator: ✓" || echo "Orchestrator: ✗"

status:
	docker compose ps
