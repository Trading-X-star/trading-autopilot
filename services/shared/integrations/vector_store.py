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
