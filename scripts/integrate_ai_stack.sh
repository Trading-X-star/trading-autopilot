#!/bin/bash
# =============================================================================
# TRADING AUTOPILOT - AI STACK INTEGRATION
# Добавляет Qdrant, Langfuse, SearXNG, Neo4j, n8n, Open WebUI
# =============================================================================

set -e
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }

cd ~/trading-autopilot

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     TRADING AUTOPILOT - AI STACK INTEGRATION                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# =============================================================================
# 1. ГЕНЕРАЦИЯ СЕКРЕТОВ
# =============================================================================
log "1/5 Генерация секретов..."

generate_secret() { openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32; }

LANGFUSE_SECRET=$(generate_secret)
LANGFUSE_SALT=$(generate_secret)
N8N_ENCRYPTION_KEY=$(generate_secret)
WEBUI_SECRET=$(generate_secret)
NEO4J_PASSWORD=$(generate_secret)

# Сохраняем секреты
mkdir -p secrets
echo -n "$LANGFUSE_SECRET" > secrets/langfuse_secret
echo -n "$LANGFUSE_SALT" > secrets/langfuse_salt
echo -n "$N8N_ENCRYPTION_KEY" > secrets/n8n_encryption_key
echo -n "$WEBUI_SECRET" > secrets/webui_secret
echo -n "$NEO4J_PASSWORD" > secrets/neo4j_password
chmod 600 secrets/*

success "Секреты сгенерированы"

# =============================================================================
# 2. КОНФИГУРАЦИЯ SEARXNG
# =============================================================================
log "2/5 Настройка SearXNG..."

mkdir -p configs/searxng

cat > configs/searxng/settings.yml << 'YAML'
use_default_settings: true

general:
  instance_name: "Trading News Search"
  privacypolicy_url: false
  donation_url: false
  contact_url: false
  enable_metrics: true

search:
  safe_search: 0
  autocomplete: "google"
  default_lang: "ru-RU"

server:
  secret_key: "CHANGE_ME_RANDOM_STRING"
  limiter: false
  image_proxy: true

ui:
  static_use_hash: true
  default_theme: simple
  theme_args:
    simple_style: dark

engines:
  # Финансовые новости
  - name: google news
    engine: google_news
    shortcut: gn
    disabled: false

  - name: bing news
    engine: bing_news
    shortcut: bn
    disabled: false

  - name: yahoo news
    engine: yahoo_news
    shortcut: yn
    disabled: false

  # Русскоязычные источники
  - name: yandex
    engine: yandex
    shortcut: ya
    disabled: false

  # Финансовые данные  
  - name: currency
    engine: currency_convert
    shortcut: cc
    disabled: false

  - name: wikidata
    engine: wikidata
    shortcut: wd
    disabled: false

  # Отключаем ненужное
  - name: pinterest
    disabled: true
  - name: piped
    disabled: true
YAML

success "SearXNG настроен"

# =============================================================================
# 3. DOCKER COMPOSE ДЛЯ AI STACK
# =============================================================================
log "3/5 Создание docker-compose.ai-stack.yml..."

cat > docker-compose.ai-stack.yml << 'YAML'
version: "3.8"

x-common: &common
  restart: unless-stopped
  networks:
    - trading-net

services:
  # =========================================================================
  # QDRANT - Vector Database для RAG
  # =========================================================================
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    <<: *common
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
      - ./configs/qdrant:/qdrant/config:ro
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G

  # =========================================================================
  # LANGFUSE - LLM Observability
  # =========================================================================
  langfuse:
    image: langfuse/langfuse:latest
    container_name: langfuse
    <<: *common
    ports:
      - "3001:3000"
    environment:
      - DATABASE_URL=postgresql://trading:${POSTGRES_PASSWORD:-trading}@postgres:5432/langfuse
      - NEXTAUTH_URL=http://localhost:3001
      - NEXTAUTH_SECRET=${LANGFUSE_SECRET:-changeme}
      - SALT=${LANGFUSE_SALT:-changeme}
      - TELEMETRY_ENABLED=false
      - LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=true
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:3000/api/public/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # =========================================================================
  # SEARXNG - Private Search Engine
  # =========================================================================
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    <<: *common
    ports:
      - "8888:8080"
    volumes:
      - ./configs/searxng:/etc/searxng:rw
      - searxng_data:/var/cache/searxng
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
      - UWSGI_WORKERS=4
      - UWSGI_THREADS=4
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  # =========================================================================
  # NEO4J - Graph Database для связей
  # =========================================================================
  neo4j:
    image: neo4j:5-community
    container_name: neo4j
    <<: *common
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-trading123}
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=1G
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # =========================================================================
  # N8N - Workflow Automation
  # =========================================================================
  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    <<: *common
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n
      - ./configs/n8n:/home/node/configs:ro
    environment:
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - NODE_ENV=production
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=Asia/Yekaterinburg
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY:-changeme}
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=trading
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD:-trading}
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:5678/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  # =========================================================================
  # OPEN WEBUI - Chat Interface
  # =========================================================================
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    <<: *common
    ports:
      - "3003:8080"
    volumes:
      - openwebui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - OPENAI_API_BASE_URL=http://orchestrator:8000/v1
      - WEBUI_SECRET_KEY=${WEBUI_SECRET:-changeme}
      - WEBUI_AUTH=false
      - ENABLE_SIGNUP=false
      - DEFAULT_MODELS=gpt-4
      - ENABLE_RAG_WEB_SEARCH=true
      - RAG_WEB_SEARCH_ENGINE=searxng
      - SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  qdrant_data:
  searxng_data:
  neo4j_data:
  neo4j_logs:
  n8n_data:
  openwebui_data:

networks:
  trading-net:
    external: true
YAML

success "docker-compose.ai-stack.yml создан"

# =============================================================================
# 4. PYTHON МОДУЛИ ДЛЯ ИНТЕГРАЦИИ
# =============================================================================
log "4/5 Создание Python модулей..."

mkdir -p services/shared/integrations

# --- Qdrant Vector Store ---
cat > services/shared/integrations/vector_store.py << 'PYTHON'
"""
Qdrant Vector Store for semantic search.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    UpdateStatus
)

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    id: str
    score: float
    payload: Dict[str, Any]

class TradingVectorStore:
    """Vector store for trading-related embeddings."""

    COLLECTIONS = {
        "news": {"size": 384, "distance": Distance.COSINE},
        "patterns": {"size": 256, "distance": Distance.COSINE},
        "signals": {"size": 128, "distance": Distance.EUCLID},
    }

    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.sync_client = QdrantClient(host=host, port=port)
        self.async_client = AsyncQdrantClient(host=host, port=port)
        self._init_collections()

    def _init_collections(self):
        """Initialize all required collections."""
        for name, config in self.COLLECTIONS.items():
            try:
                if not self.sync_client.collection_exists(name):
                    self.sync_client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=config["size"],
                            distance=config["distance"]
                        )
                    )
                    logger.info(f"Created collection: {name}")
            except Exception as e:
                logger.error(f"Failed to create collection {name}: {e}")

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def add_news(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add news article to vector store."""
        try:
            point_id = self._generate_id(text)
            payload = {
                **metadata,
                "text": text[:1000],  # Truncate for storage
                "indexed_at": datetime.utcnow().isoformat()
            }

            result = await self.async_client.upsert(
                collection_name="news",
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
            )
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            logger.error(f"Failed to add news: {e}")
            return False

    async def search_similar_news(
        self,
        embedding: List[float],
        ticker: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search for similar news articles."""
        try:
            query_filter = None
            if ticker:
                query_filter = Filter(
                    must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
                )

            results = await self.async_client.search(
                collection_name="news",
                query_vector=embedding,
                query_filter=query_filter,
                limit=limit
            )

            return [
                SearchResult(id=str(r.id), score=r.score, payload=r.payload)
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def add_pattern(
        self,
        pattern_name: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add trading pattern to vector store."""
        try:
            point_id = self._generate_id(pattern_name)
            payload = {
                **metadata,
                "pattern_name": pattern_name,
                "indexed_at": datetime.utcnow().isoformat()
            }

            result = await self.async_client.upsert(
                collection_name="patterns",
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
            )
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            return False

    async def find_similar_patterns(
        self,
        embedding: List[float],
        limit: int = 5
    ) -> List[SearchResult]:
        """Find similar historical patterns."""
        try:
            results = await self.async_client.search(
                collection_name="patterns",
                query_vector=embedding,
                limit=limit
            )
            return [
                SearchResult(id=str(r.id), score=r.score, payload=r.payload)
                for r in results
            ]
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {}
        for name in self.COLLECTIONS:
            try:
                info = await self.async_client.get_collection(name)
                stats[name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.value
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        return stats

# Singleton instance
_vector_store: Optional[TradingVectorStore] = None

def get_vector_store() -> TradingVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = TradingVectorStore()
    return _vector_store

__all__ = ['TradingVectorStore', 'SearchResult', 'get_vector_store']
PYTHON

# --- Langfuse Integration ---
cat > services/shared/integrations/llm_monitoring.py << 'PYTHON'
"""
Langfuse integration for LLM observability.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import asyncio

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

logger = logging.getLogger(__name__)

class LLMMonitor:
    """Monitor LLM calls with Langfuse."""

    def __init__(self):
        self.enabled = LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        self.client: Optional[Langfuse] = None

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                    host=os.getenv("LANGFUSE_HOST", "http://langfuse:3000")
                )
                logger.info("Langfuse monitoring enabled")
            except Exception as e:
                logger.warning(f"Langfuse init failed: {e}")
                self.enabled = False

    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace for monitoring."""
        if not self.enabled or not self.client:
            yield None
            return

        trace = self.client.trace(
            name=name,
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
        try:
            yield trace
        finally:
            trace.update(end_time=datetime.utcnow())

    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an LLM generation."""
        if not self.enabled or not self.client:
            return

        try:
            self.client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=prompt,
                output=completion,
                usage=usage,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.warning(f"Failed to log generation: {e}")

    def log_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """Log a score for evaluation."""
        if not self.enabled or not self.client:
            return

        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
        except Exception as e:
            logger.warning(f"Failed to log score: {e}")

    def flush(self):
        """Flush pending events."""
        if self.client:
            self.client.flush()

def monitored_llm_call(name: str = "llm_call"):
    """Decorator to monitor LLM calls."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = get_llm_monitor()
            with monitor.trace(name, {"args": str(args)[:200]}) as trace:
                start = datetime.utcnow()
                try:
                    result = await func(*args, **kwargs)
                    if trace and hasattr(result, 'usage'):
                        monitor.log_generation(
                            trace_id=trace.id,
                            name=func.__name__,
                            model=kwargs.get('model', 'unknown'),
                            prompt=str(kwargs.get('prompt', ''))[:500],
                            completion=str(result)[:500],
                            usage=getattr(result, 'usage', None)
                        )
                    return result
                except Exception as e:
                    if trace:
                        monitor.log_score(trace.id, "error", 0, str(e))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator

# Singleton
_llm_monitor: Optional[LLMMonitor] = None

def get_llm_monitor() -> LLMMonitor:
    global _llm_monitor
    if _llm_monitor is None:
        _llm_monitor = LLMMonitor()
    return _llm_monitor

__all__ = ['LLMMonitor', 'get_llm_monitor', 'monitored_llm_call']
PYTHON

# --- SearXNG News Search ---
cat > services/shared/integrations/news_search.py << 'PYTHON'
"""
SearXNG integration for news search.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    url: str
    content: str
    source: str
    published: Optional[datetime]
    relevance_score: float = 0.0

class NewsSearcher:
    """Search news using SearXNG."""

    def __init__(self, base_url: str = "http://searxng:8080"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 30.0

    async def search(
        self,
        query: str,
        categories: List[str] = ["news"],
        language: str = "ru",
        time_range: str = "day",
        limit: int = 20
    ) -> List[NewsArticle]:
        """Search for news articles."""
        try:
            params = {
                "q": query,
                "format": "json",
                "categories": ",".join(categories),
                "language": language,
                "time_range": time_range
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()

            articles = []
            for idx, result in enumerate(data.get("results", [])[:limit]):
                articles.append(NewsArticle(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    source=result.get("engine", "unknown"),
                    published=self._parse_date(result.get("publishedDate")),
                    relevance_score=1.0 - (idx / limit)  # Simple ranking
                ))

            logger.info(f"Found {len(articles)} articles for: {query}")
            return articles

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []

    async def search_ticker(self, ticker: str, company_name: str = "") -> List[NewsArticle]:
        """Search news for a specific ticker."""
        queries = [
            f"{ticker} акции новости",
            f"{company_name} финансы" if company_name else f"{ticker} компания"
        ]

        all_articles = []
        for query in queries:
            articles = await self.search(query, time_range="week")
            all_articles.extend(articles)

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        return sorted(unique_articles, key=lambda x: x.relevance_score, reverse=True)

    async def search_market_news(self) -> List[NewsArticle]:
        """Get general market news."""
        queries = [
            "Московская биржа MOEX новости",
            "российский фондовый рынок",
            "ЦБ РФ ставка экономика"
        ]

        all_articles = []
        for query in queries:
            articles = await self.search(query, time_range="day", limit=10)
            all_articles.extend(articles)

        return all_articles[:30]

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_str:
            return None
        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            try:
                # Try common format
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except:
                return None

    async def health_check(self) -> bool:
        """Check if SearXNG is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/healthz")
                return response.status_code == 200
        except:
            return False

# Singleton
_news_searcher: Optional[NewsSearcher] = None

def get_news_searcher() -> NewsSearcher:
    global _news_searcher
    if _news_searcher is None:
        _news_searcher = NewsSearcher()
    return _news_searcher

__all__ = ['NewsSearcher', 'NewsArticle', 'get_news_searcher']
PYTHON

# --- Neo4j Graph Store ---
cat > services/shared/integrations/graph_store.py << 'PYTHON'
"""
Neo4j integration for company/sector relationships.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None

logger = logging.getLogger(__name__)

@dataclass
class CompanyNode:
    ticker: str
    name: str
    sector: str
    properties: Dict[str, Any]

@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str
    weight: float

class TradingGraphStore:
    """Graph store for company relationships."""

    def __init__(self):
        self.enabled = NEO4J_AVAILABLE
        self.driver = None

        if self.enabled:
            try:
                uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "trading123")

                self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
                logger.info("Neo4j connection established")
            except Exception as e:
                logger.warning(f"Neo4j init failed: {e}")
                self.enabled = False

    async def add_company(self, company: CompanyNode) -> bool:
        """Add or update a company node."""
        if not self.enabled:
            return False

        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (c:Company {ticker: $ticker})
                    SET c.name = $name, c.sector = $sector, c += $properties
                    """,
                    ticker=company.ticker,
                    name=company.name,
                    sector=company.sector,
                    properties=company.properties
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add company: {e}")
            return False

    async def add_relationship(
        self,
        source_ticker: str,
        target_ticker: str,
        rel_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add relationship between companies."""
        if not self.enabled:
            return False

        try:
            async with self.driver.session() as session:
                await session.run(
                    f"""
                    MATCH (a:Company {{ticker: $source}})
                    MATCH (b:Company {{ticker: $target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.weight = $weight, r += $properties
                    """,
                    source=source_ticker,
                    target=target_ticker,
                    weight=weight,
                    properties=properties or {}
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False

    async def get_related_companies(
        self,
        ticker: str,
        rel_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get companies related to a ticker."""
        if not self.enabled:
            return []

        try:
            rel_filter = f":{rel_type}" if rel_type else ""
            async with self.driver.session() as session:
                result = await session.run(
                    f"""
                    MATCH (c:Company {{ticker: $ticker}})-[r{rel_filter}]-(related:Company)
                    RETURN related.ticker as ticker, related.name as name,
                           type(r) as relationship, r.weight as weight
                    ORDER BY r.weight DESC
                    LIMIT $limit
                    """,
                    ticker=ticker,
                    limit=limit
                )
                records = await result.data()
            return records
        except Exception as e:
            logger.error(f"Failed to get related companies: {e}")
            return []

    async def get_sector_companies(self, sector: str) -> List[Dict[str, Any]]:
        """Get all companies in a sector."""
        if not self.enabled:
            return []

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (c:Company {sector: $sector})
                    RETURN c.ticker as ticker, c.name as name
                    ORDER BY c.ticker
                    """,
                    sector=sector
                )
                records = await result.data()
            return records
        except Exception as e:
            logger.error(f"Failed to get sector companies: {e}")
            return []

    async def find_path(
        self,
        source_ticker: str,
        target_ticker: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Find relationship path between two companies."""
        if not self.enabled:
            return []

        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH path = shortestPath(
                        (a:Company {ticker: $source})-[*1..$depth]-(b:Company {ticker: $target})
                    )
                    RETURN [n in nodes(path) | n.ticker] as path,
                           [r in relationships(path) | type(r)] as relationships
                    """,
                    source=source_ticker,
                    target=target_ticker,
                    depth=max_depth
                )
                records = await result.data()
            return records
        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return []

    async def close(self):
        """Close the driver connection."""
        if self.driver:
            await self.driver.close()

# Singleton
_graph_store: Optional[TradingGraphStore] = None

def get_graph_store() -> TradingGraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = TradingGraphStore()
    return _graph_store

__all__ = ['TradingGraphStore', 'CompanyNode', 'Relationship', 'get_graph_store']
PYTHON

# --- __init__.py ---
cat > services/shared/integrations/__init__.py << 'PYTHON'
"""AI Stack integrations."""
from .vector_store import *
from .llm_monitoring import *
from .news_search import *
from .graph_store import *
PYTHON

success "Python модули созданы"

# =============================================================================
# 5. ОБНОВЛЕНИЕ REQUIREMENTS
# =============================================================================
log "5/5 Обновление зависимостей..."

cat >> requirements.txt << 'EOF'

# AI Stack integrations
qdrant-client>=1.7.0
langfuse>=2.0.0
neo4j>=5.0.0
httpx>=0.26.0
EOF

success "requirements.txt обновлён"

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ БАЗ ДАННЫХ
# =============================================================================
log "Создание баз данных для Langfuse и n8n..."

cat > configs/init-ai-stack.sql << 'SQL'
-- Create databases for AI Stack components
CREATE DATABASE IF NOT EXISTS langfuse;
CREATE DATABASE IF NOT EXISTS n8n;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE langfuse TO trading;
GRANT ALL PRIVILEGES ON DATABASE n8n TO trading;
SQL

success "SQL скрипт создан"

# =============================================================================
# ENVIRONMENT FILE
# =============================================================================
log "Создание .env.ai-stack..."

cat > .env.ai-stack << EOF
# AI Stack Environment Variables
# Generated: $(date)

# Langfuse
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=pk-lf-trading
LANGFUSE_SECRET_KEY=$(generate_secret)

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=$NEO4J_PASSWORD

# SearXNG
SEARXNG_URL=http://searxng:8080

# n8n
N8N_ENCRYPTION_KEY=$N8N_ENCRYPTION_KEY

# Open WebUI
WEBUI_SECRET_KEY=$WEBUI_SECRET
EOF

chmod 600 .env.ai-stack

success ".env.ai-stack создан"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   AI STACK ИНТЕГРИРОВАН!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Компоненты:"
echo "  ✓ Qdrant      - Vector DB      -> localhost:6333"
echo "  ✓ Langfuse    - LLM Monitoring -> localhost:3001"
echo "  ✓ SearXNG     - News Search    -> localhost:8888"
echo "  ✓ Neo4j       - Graph DB       -> localhost:7474"
echo "  ✓ n8n         - Workflows      -> localhost:5678"
echo "  ✓ Open WebUI  - Chat Interface -> localhost:3003"
echo ""
echo "Запуск:"
echo "  docker compose -f docker-compose.yml -f docker-compose.ai-stack.yml up -d"
echo ""
echo "Или:"
echo "  make ai-up"
echo ""
