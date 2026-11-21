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
        import asyncio
        from src.infrastructure.redis_client import redis_manager
        from src.workers.queue_service import ingestion_queue
        await redis_manager.check_health()
        await ingestion_queue.ensure_consumer_group()

        # Start in-process background worker to consume queued jobs (essential for free-tier cloud deployment)
        from src.workers.ingestion_worker import IngestionWorker
        app.state.ingestion_worker = IngestionWorker(worker_id="in-process-web-worker")
        app.state.worker_task = asyncio.create_task(app.state.ingestion_worker.start())

        # Auto-seed sample documents if vector store is empty
        async def _seed_corpus():
            await asyncio.sleep(1.0)
            from src.database.vector_store import vector_store
            from src.ingestion.pipeline import ingestion_pipeline
            from pathlib import Path
            if vector_store.count() > 0:
                logger.info(f"Vector store already contains {vector_store.count()} chunks. Skipping auto-seed.")
                return

            sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
            sample_files = [
                sample_dir / "msft_q4_2024_transcript.json",
                sample_dir / "nvda_q2_2025_transcript.json"
            ]
            for sf in sample_files:
                if sf.exists():
                    try:
                        raw_json = sf.read_text(encoding="utf-8")
                        doc = ingestion_pipeline.parser.parse_json_document(raw_json)
                        p_chunks, c_chunks = ingestion_pipeline.chunker.chunk_document(doc)
                        embeddings = ingestion_pipeline.embedder.embed_batch([c.content for c in c_chunks])
                        vector_store.add_documents(documents=c_chunks, embeddings=embeddings, parent_chunks=p_chunks)
                        logger.info(f"Auto-seeded sample filing '{doc.document_id}' ({len(c_chunks)} chunks).")
                    except Exception as e:
                        logger.warning(f"Auto-seed of {sf.name} failed: {e}")

        asyncio.create_task(_seed_corpus())

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down AlphaLens API...")
        from src.infrastructure.redis_client import redis_manager
        await redis_manager.close()

    return app


app = create_app()

