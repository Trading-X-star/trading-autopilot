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
