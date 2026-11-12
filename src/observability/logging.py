"""
AlphaLens - Structured Logging Module
Emits JSON-formatted structured logs with trace_id, request_id, and component tags.
"""
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON objects for high-throughput log aggregation."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom fields attached to record
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "trace_id"):
            log_obj["trace_id"] = record.trace_id
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(name: str = "alphalens", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


logger = setup_logger()
