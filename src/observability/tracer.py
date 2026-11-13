"""
AlphaLens - Distributed Tracing and Context Propagation
Provides lightweight span contexts for tracking execution across retrieval, reranking, and generation.
"""
import time
import uuid
from contextlib import contextmanager
from typing import Dict, List, Optional
from src.observability.logging import logger


class Span:
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.span_id: str = str(uuid.uuid4())[:8]
        self.name: str = name
        self.trace_id: str = trace_id
        self.parent_id: parent_id = parent_id
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.attributes: Dict[str, str] = {}
        self.error: Optional[str] = None

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def finish(self, error: Optional[Exception] = None):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        if error:
            self.error = str(error)
        logger.debug(
            f"[Trace: {self.trace_id}] Span '{self.name}' completed in {self.duration_ms:.2f}ms"
        )


class Tracer:
    """Manages distributed traces and span contexts."""
    @staticmethod
    def generate_trace_id() -> str:
        return f"trace-{uuid.uuid4().hex[:12]}"

    @contextmanager
    def start_span(self, name: str, trace_id: Optional[str] = None):
        trace_id = trace_id or self.generate_trace_id()
        span = Span(name=name, trace_id=trace_id)
        span.start()
        try:
            yield span
        except Exception as e:
            span.finish(error=e)
            raise
        else:
            span.finish()


tracer = Tracer()

