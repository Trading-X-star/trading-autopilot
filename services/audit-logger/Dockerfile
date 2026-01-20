FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8021/health || exit 1
EXPOSE 8021
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8021"]
