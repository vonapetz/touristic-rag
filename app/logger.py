"""
Structured JSON logging.
Каждый запрос логируется с query_id, таймингами, метаданными.
"""

import sys
import logging
import structlog
from app.config import settings


def setup_logging():
    """Инициализация структурированного логирования."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Файловый handler
    import os
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)


def get_logger(name: str = None):
    """Получение именованного логгера."""
    return structlog.get_logger(name or "app")