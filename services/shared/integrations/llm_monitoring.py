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
