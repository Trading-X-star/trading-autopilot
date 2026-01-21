#!/bin/bash
# Полный скрипт слишком большой для одного сообщения
# Создайте файлы по отдельности:

cd ~/trading-autopilot

# 1. CI/CD
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff black
      - run: ruff check .

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip
      - run: pip install -r requirements.txt pytest pytest-asyncio pytest-cov
      - run: pytest tests/ -v --cov

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          severity: CRITICAL,HIGH
          exit-code: 1

  build:
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      packages: write
    strategy:
      matrix:
        service: [orchestrator, strategy, executor, risk-manager, datafeed]
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: ./services/${{ matrix.service }}
          push: true
          tags: ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest
EOF

# 2. Dependabot
cat > .github/dependabot.yml << 'EOF'
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: docker
    directory: "/services/orchestrator"
    schedule:
      interval: weekly
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
EOF

# 3. CODEOWNERS
cat > .github/CODEOWNERS << 'EOF'
* @Trading-X-star
/services/executor/ @Trading-X-star
/services/risk-manager/ @Trading-X-star
/.github/ @Trading-X-star
EOF

# 4. Tests
mkdir -p tests/unit tests/integration
cat > tests/conftest.py << 'EOF'
import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_signal():
    return {
        "ticker": "SBER",
        "action": "buy",
        "quantity": 10,
        "price": 280.50,
        "confidence": 0.75
    }
EOF

# 5. SECURITY.md
cat > SECURITY.md << 'EOF'
# Security Policy

## Reporting a Vulnerability
Email: security@trading-autopilot.dev

## Supported Versions
| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅        |
EOF

# 6. Commit & Push
git add -A
git commit -m "chore: add CI/CD, security, tests, docs"
git push origin main

# 7. Create release
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0

echo "✅ Done!"
