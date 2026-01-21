"""
Structured JSON logging with correlation IDs.
"""
import logging
import json
import sys
import uuid
from datetime import datetime
from typing import Optional
from contextvars import ContextVar

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id.set(cid)
    return cid

def set_correlation_id(cid: str):
    """Set correlation ID (e.g., from incoming request header)."""
    correlation_id.set(cid)

class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add location info
        log_obj["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }

        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging(
    service_name: str,
    level: str = "INFO",
    json_format: bool = True
) -> logging.Logger:
    """Setup logging for a service."""

    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    logger.addHandler(handler)
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra fields to log records."""

    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        extra['extra_fields'] = {
            **self.extra,
            **extra.get('extra_fields', {})
        }
        kwargs['extra'] = extra
        return msg, kwargs

def get_logger(name: str, **extra_fields) -> LoggerAdapter:
    """Get a logger with extra fields."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, extra_fields)

__all__ = [
    'setup_logging', 'get_logger', 'get_correlation_id', 
    'set_correlation_id', 'JSONFormatter', 'LoggerAdapter'
]
