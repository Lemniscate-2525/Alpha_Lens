"""
AlphaLens - API Middleware
Handles Request ID injection, response latency header, and rate limiting admission control.
"""
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.services.rate_limiter import rate_limiter
from src.core.exceptions import RateLimitExceededError
from src.observability.logging import logger


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        # Extract or generate Request ID
        request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:10]}")
        request.state.request_id = request_id

        # Admission control / Rate limit check (bypass for health, docs, telemetry, document lists, and master keys)
        path = request.url.path
        rate_limit_res = None
        bypass_paths = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
        incoming_key = request.headers.get("X-API-Key", "")
        is_master = incoming_key in {
            "alpha-lens-master-key-prod-2025",
            "alphalens_prod_secret_key_change_in_production",
        }
        should_bypass = (
            is_master
            or path in bypass_paths
            or path.startswith("/api/v1/health")
            or path.startswith("/api/v1/metrics")
            or path.startswith("/api/v1/ingest/documents")
            or path.startswith("/api/v1/ingest/jobs")
        )
        if not should_bypass:
            client_ip = request.client.host if request.client else "unknown_client"
            client_id = request.headers.get("X-API-Key", client_ip)
            try:
                rate_limit_res = await rate_limiter.check_rate_limit_async(client_id=client_id)
            except RateLimitExceededError as e:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RateLimitExceeded",
                        "message": e.message,
                        "retry_after_seconds": e.retry_after,
                        "request_id": request_id,
                    },
                    headers={
                        "Retry-After": str(e.retry_after),
                        "X-Request-ID": request_id,
                        "X-RateLimit-Limit": str(rate_limiter.capacity),
                        "X-RateLimit-Remaining": "0"
                    }
                )

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"[Req: {request_id}] Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "InternalServerError", "message": str(e), "request_id": request_id}
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        if rate_limit_res:
            response.headers["X-RateLimit-Limit"] = str(rate_limit_res.limit)
            response.headers["X-RateLimit-Remaining"] = str(int(rate_limit_res.remaining))
            response.headers["X-RateLimit-Reset"] = str(int(rate_limit_res.retry_after))
        return response

