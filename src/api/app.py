"""
AlphaLens - FastAPI Application Factory
Initializes FastAPI, attaches routers, middleware, CORS, and registers custom exception handlers.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.middleware import ObservabilityMiddleware
from src.api.routes import query, ingest, health
from src.core.exceptions import (
    AlphaLensException,
    RateLimitExceededError,
    CircuitBreakerOpenError,
    DocumentParsingError,
    IngestionError,
    VectorStoreError
)
from src.observability.logging import logger


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade Financial RAG System for SEC filings and Wall Street earnings transcripts.",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Observability & Admission Control Middleware
    app.add_middleware(ObservabilityMiddleware)

    # Exception Handlers
    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_handler(request: Request, exc: RateLimitExceededError):
        return JSONResponse(
            status_code=429,
            content={"error": "RateLimitExceeded", "message": exc.message, "details": exc.details},
            headers={"Retry-After": str(exc.retry_after)}
        )

    @app.exception_handler(CircuitBreakerOpenError)
    async def circuit_breaker_handler(request: Request, exc: CircuitBreakerOpenError):
        return JSONResponse(
            status_code=503,
            content={"error": "CircuitBreakerOpen", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(DocumentParsingError)
    async def parsing_error_handler(request: Request, exc: DocumentParsingError):
        return JSONResponse(
            status_code=422,
            content={"error": "DocumentParsingError", "message": exc.message, "details": exc.details}
        )

    @app.exception_handler(AlphaLensException)
    async def domain_exception_handler(request: Request, exc: AlphaLensException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details}
        )

    # Include Routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")

    # Root & Cloud Health Check Endpoints
    @app.get("/", tags=["Root"])
    @app.head("/", tags=["Root"])
    async def root_status():
        """Root status probe and API directory."""
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
            "message": "AlphaLens Production Financial RAG Engine is online.",
            "endpoints": {
                "documentation": "/docs",
                "health": "/api/v1/health",
                "metrics": "/api/v1/metrics",
                "query": "/api/v1/query",
                "ingest": "/api/v1/ingest",
                "jobs": "/api/v1/ingest/jobs"
            }
        }

    @app.get("/health", tags=["Observability"])
    @app.head("/health", tags=["Observability"])
    async def root_health():
        """Root-level health probe for cloud load balancers (Render, AWS, GCP, etc.)."""
        from src.infrastructure.redis_client import redis_manager
        from src.database.vector_store import vector_store
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

    @app.on_event("startup")
    async def startup_event():
        logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} initialized successfully.")
        from src.infrastructure.redis_client import redis_manager
        from src.workers.queue_service import ingestion_queue
        await redis_manager.check_health()
        await ingestion_queue.ensure_consumer_group()

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down AlphaLens API...")
        from src.infrastructure.redis_client import redis_manager
        await redis_manager.close()

    return app


app = create_app()

