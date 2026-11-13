"""
AlphaLens - Rate Limiting & Admission Control
Implements Token Bucket rate limiting with burst capacity and per-client fairness.
"""
import time
from typing import Dict, Tuple
from src.config import settings
from src.core.exceptions import RateLimitExceededError
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class TokenBucket:
    def __init__(self, capacity: int, fill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.fill_rate = float(fill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Attempts to consume tokens.
        Returns: (allowed: bool, retry_after_seconds: float)
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now

        # Add newly accrued tokens up to capacity
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.fill_rate))

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0
        else:
            deficit = tokens - self.tokens
            retry_after = deficit / self.fill_rate
            return False, retry_after


class RateLimiterService:
    """Manages rate limit buckets per client identifier (API Key or IP)."""

    def __init__(self):
        self.rps = settings.RATE_LIMIT_RPS
        self.burst = settings.RATE_LIMIT_BURST
        self._buckets: Dict[str, TokenBucket] = {}

    def check_rate_limit(self, client_id: str, cost: int = 1) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return

        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(capacity=self.burst, fill_rate_per_sec=self.rps)

        bucket = self._buckets[client_id]
        allowed, retry_after = bucket.consume(cost)

        if not allowed:
            metrics_collector.record_rate_limit_drop()
            logger.warning(
                f"Rate limit exceeded for client '{client_id}'. "
                f"Bucket depleted. Retry after {retry_after:.2f}s"
            )
            raise RateLimitExceededError(
                message=f"Rate limit exceeded ({self.rps} req/sec). Please retry after {retry_after:.2f} seconds.",
                retry_after=max(1, int(retry_after))
            )


rate_limiter = RateLimiterService()

