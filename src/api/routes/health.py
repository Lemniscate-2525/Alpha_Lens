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
    """Liveness probe verifying that the API, Redis, and vector store are operational."""
    from src.infrastructure.redis_client import redis_manager
    redis_healthy = await redis_manager.check_health()
    chunk_count = vector_store.count()
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "redis_connected": redis_healthy,
        "indexed_chunks": chunk_count,
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Surfaces real-time system metrics, percentiles (P50, P90, P95, P99), RPS, and costs."""
    return metrics_collector.get_summary()

