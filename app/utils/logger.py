"""
Logging utilities.

Design Decisions:
- Structured logging: JSON format for production, text for dev
- Context-aware: Includes request ID, operation type
- Performance tracking: Logs latency for key operations
- Configurable: Level and format from settings
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Optional
from functools import wraps
import time


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(level: str = "INFO", format_type: str = "text") -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: "json" or "text"
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding extra fields to log records."""

    def __init__(self, logger: logging.Logger, **extra):
        self.logger = logger
        self.extra = extra
        self._old_factory = None

    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        extra = self.extra

        def factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra = extra
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, *args):
        logging.setLogRecordFactory(self._old_factory)


def log_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution with timing.

    Args:
        operation_name: Name of the operation for logs
        logger: Logger to use (creates one if not provided)
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Starting {operation_name}")
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.info(f"Completed {operation_name} in {elapsed:.0f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"Failed {operation_name} after {elapsed:.0f}ms: {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                logger.info(f"Completed {operation_name} in {elapsed:.0f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                logger.error(f"Failed {operation_name} after {elapsed:.0f}ms: {e}")
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_token_usage(
    logger: logging.Logger,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    model: str
) -> None:
    """Log token usage for cost tracking."""
    logger.info(
        f"Token usage for {operation}: "
        f"input={input_tokens}, output={output_tokens}, "
        f"total={input_tokens + output_tokens}, model={model}"
    )
