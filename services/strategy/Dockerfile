# Dockerfile for Trainer v7 Ultimate
FROM python:3.9-slim-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL python files
COPY *.py .

RUN mkdir -p /app/models /app/data /app/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "main.py"]

