"""
Batch processing utilities for efficient signal generation.
"""
import asyncio
import logging
from typing import List, TypeVar, Callable, Coroutine, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)
T = TypeVar('T')
R = TypeVar('R')

@dataclass
class BatchConfig:
    batch_size: int = 50
    max_concurrent: int = 10
    timeout_per_batch: float = 30.0
    retry_failed: bool = True

@dataclass
class BatchResult:
    successful: List[Any]
    failed: List[tuple]  # (item, error)
    total_time: float
    batches_processed: int

class BatchProcessor:
    """Process items in batches with concurrency control."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """Process items in batches with progress tracking."""
        start_time = time.time()
        successful = []
        failed = []

        # Split into batches
        batches = [
            items[i:i + self.config.batch_size]
            for i in range(0, len(items), self.config.batch_size)
        ]

        total_items = len(items)
        processed = 0

        for batch_idx, batch in enumerate(batches):
            batch_results = await self._process_batch(batch, processor)

            for item, result, error in batch_results:
                if error is None:
                    successful.append(result)
                else:
                    failed.append((item, error))
                    if self.config.retry_failed:
                        # Single retry
                        try:
                            retry_result = await asyncio.wait_for(
                                processor(item),
                                timeout=self.config.timeout_per_batch / len(batch)
                            )
                            successful.append(retry_result)
                            failed.pop()  # Remove from failed
                        except Exception as e:
                            pass  # Keep in failed

            processed += len(batch)
            if on_progress:
                on_progress(processed, total_items)

            logger.debug(f"Batch {batch_idx + 1}/{len(batches)} completed: "
                        f"{len([r for _, r, e in batch_results if e is None])} success")

        return BatchResult(
            successful=successful,
            failed=failed,
            total_time=time.time() - start_time,
            batches_processed=len(batches)
        )

    async def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]]
    ) -> List[tuple]:
        """Process a single batch concurrently."""

        async def process_with_semaphore(item: T) -> tuple:
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        processor(item),
                        timeout=self.config.timeout_per_batch
                    )
                    return (item, result, None)
                except Exception as e:
                    logger.warning(f"Failed to process {item}: {e}")
                    return (item, None, e)

        tasks = [process_with_semaphore(item) for item in batch]
        return await asyncio.gather(*tasks)

class SignalBatchGenerator:
    """Specialized batch processor for trading signals."""

    def __init__(self, tickers: List[str], batch_size: int = 50):
        self.tickers = tickers
        self.processor = BatchProcessor(BatchConfig(
            batch_size=batch_size,
            max_concurrent=10,
            timeout_per_batch=60.0
        ))

    async def generate_all_signals(
        self,
        signal_generator: Callable[[str], Coroutine[Any, Any, dict]]
    ) -> dict:
        """Generate signals for all tickers in batches."""

        start = datetime.now()
        logger.info(f"Starting batch signal generation for {len(self.tickers)} tickers")

        def on_progress(done: int, total: int):
            pct = (done / total) * 100
            logger.info(f"Signal generation progress: {done}/{total} ({pct:.1f}%)")

        result = await self.processor.process(
            self.tickers,
            signal_generator,
            on_progress
        )

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(
            f"Signal generation completed: {len(result.successful)} success, "
            f"{len(result.failed)} failed in {elapsed:.2f}s"
        )

        return {
            "signals": result.successful,
            "failed_tickers": [t for t, _ in result.failed],
            "total_time": result.total_time,
            "generated_at": datetime.now().isoformat()
        }

__all__ = ['BatchProcessor', 'BatchConfig', 'BatchResult', 'SignalBatchGenerator']
