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
