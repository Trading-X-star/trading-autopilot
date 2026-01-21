
# AI Stack commands
ai-up:
	docker compose -f docker-compose.yml -f docker-compose.ai-stack.yml up -d

ai-down:
	docker compose -f docker-compose.ai-stack.yml down

ai-status:
	docker compose -f docker-compose.ai-stack.yml ps
