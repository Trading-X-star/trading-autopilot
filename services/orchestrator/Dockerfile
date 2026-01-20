FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn asyncpg redis httpx prometheus-client
COPY main.py .
CMD ["python", "main.py"]
