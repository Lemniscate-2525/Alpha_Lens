"""
AlphaLens - Health, Liveness & Metrics Endpoints
Surfaces operational telemetry, latency percentiles, cache hit ratios, and SLI compliance.
"""
from typing import Any, Dict
from fastapi import APIRouter
from src.observability.metrics import metrics_collector
from src.database.vector_store import vector_store
from src.config import settings

router = APIRouter(tags=["Observability"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Liveness probe verifying that the API and vector store are operational."""
    chunk_count = vector_store.count()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "indexed_chunks": chunk_count,
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Surfaces real-time system metrics, percentiles (P50, P90, P95, P99), RPS, and costs."""
    return metrics_collector.get_summary()
