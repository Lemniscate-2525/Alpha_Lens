"""
AlphaLens - Redis-Backed Distributed Rate Limiting & Admission Control
Implements an atomic Token Bucket rate limiter in Redis via a Lua script.
State is coordinated globally across all FastAPI instances and worker processes.
"""

import hashlib
import time
from typing import Optional, Tuple
from src.config import settings
from src.core.exceptions import RateLimitExceededError
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


TOKEN_BUCKET_LUA_SCRIPT = """
-- KEYS[1]: bucket key (e.g. alpha_lens:ratelimit:{client_id})
-- ARGV[1]: capacity (number)
-- ARGV[2]: refill_rate (tokens per second)
-- ARGV[3]: requested_tokens (cost, e.g. 1)
-- ARGV[4]: window_ttl_seconds (TTL in seconds)
-- ARGV[5]: fallback_timestamp (epoch seconds float)

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local fallback_time = tonumber(ARGV[5])

local now = fallback_time
local ok, time_res = pcall(redis.call, 'TIME')
if ok and time_res and time_res[1] then
    now = tonumber(time_res[1]) + (tonumber(time_res[2]) / 1000000.0)
end

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if not tokens or not last_refill then
    tokens = capacity
    last_refill = now
else
    local elapsed = now - last_refill
    if elapsed > 0 then
        tokens = math.min(capacity, tokens + (elapsed * refill_rate))
        last_refill = now
    end
end

local allowed = 0
local retry_after = 0.0

if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
    retry_after = 0.0
else
    local deficit = cost - tokens
    retry_after = deficit / refill_rate
    allowed = 0
end

redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
redis.call('EXPIRE', key, math.ceil(ttl))

return {allowed, tostring(tokens), tostring(retry_after), tostring(capacity)}
"""


class RateLimitResult:
    def __init__(self, allowed: bool, remaining: float, retry_after: float, limit: int):
        self.allowed = allowed
        self.remaining = max(0.0, remaining)
        self.retry_after = max(0.0, retry_after)
        self.limit = limit


class DistributedRateLimiter:
    """
    Redis-backed Token Bucket rate limiter.
    Ensures that multiple independent FastAPI instances enforce shared global rate limits.
    """

    def __init__(self, redis_client=None):
        self._custom_client = redis_client
        self.capacity = settings.RATE_LIMIT_CAPACITY
        self.refill_rate = float(settings.RATE_LIMIT_REFILL_RATE)
        self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS
        self.prefix = settings.RATE_LIMIT_KEY_PREFIX
        self.fail_closed = getattr(settings, "RATE_LIMIT_FAIL_CLOSED", True)

    @property
    def rps(self) -> float:
        return self.refill_rate

    @rps.setter
    def rps(self, val: float):
        self.refill_rate = float(val)

    @property
    def burst(self) -> int:
        return self.capacity

    @burst.setter
    def burst(self, val: int):
        self.capacity = int(val)


    def _mask_client_id(self, client_id: str) -> str:
        """Masks client_id/API key to prevent leaking sensitive credentials in logs."""
        if not client_id:
            return "unknown"
        if len(client_id) <= 8:
            return f"{client_id[:2]}***"
        sha = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:8]
        return f"{client_id[:4]}...[{sha}]"

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    async def evaluate(self, client_id: str, cost: int = 1) -> RateLimitResult:
        """
        Atomically evaluates rate limit bucket in Redis using Lua script.
        Returns: RateLimitResult
        """
        if not settings.RATE_LIMIT_ENABLED:
            return RateLimitResult(allowed=True, remaining=self.capacity, retry_after=0.0, limit=self.capacity)

        key = RedisKeys.rate_limit_key(client_id, prefix=self.prefix)
        ttl = max(self.window_seconds, int(self.capacity / max(0.1, self.refill_rate) * 2) + 60)
        now_ts = time.time()

        try:
            client = await self._get_client()
            res = await client.eval(
                TOKEN_BUCKET_LUA_SCRIPT,
                1,
                key,
                self.capacity,
                self.refill_rate,
                cost,
                ttl,
                now_ts
            )
            allowed = bool(int(res[0]))
            tokens_remaining = float(res[1])
            retry_after = float(res[2])
            capacity = int(float(res[3]))

            return RateLimitResult(
                allowed=allowed,
                remaining=tokens_remaining,
                retry_after=retry_after,
                limit=capacity
            )
        except Exception as e:
            masked = self._mask_client_id(client_id)
            logger.error(f"Redis rate limiter evaluation failed for client '{masked}': {e}")
            if self.fail_closed:
                # Fail closed on authenticated protected endpoints
                raise RateLimitExceededError(
                    message="Service temporarily unavailable due to rate limiter storage failure. Please retry.",
                    retry_after=5
                )
            # Fail open if explicitly configured
            return RateLimitResult(allowed=True, remaining=1.0, retry_after=0.0, limit=self.capacity)

    async def check_rate_limit_async(self, client_id: str, cost: int = 1) -> RateLimitResult:
        """Async check that raises RateLimitExceededError when limit is exceeded."""
        res = await self.evaluate(client_id=client_id, cost=cost)
        if not res.allowed:
            metrics_collector.record_rate_limit_drop()
            masked = self._mask_client_id(client_id)
            logger.warning(
                f"Rate limit exceeded for client '{masked}'. "
                f"Retry after {res.retry_after:.2f}s (Tokens remaining: {res.remaining:.1f}/{res.limit})"
            )
            raise RateLimitExceededError(
                message=f"Rate limit exceeded ({self.refill_rate} req/sec). Please retry after {res.retry_after:.2f} seconds.",
                retry_after=max(1, int(res.retry_after))
            )
        return res

    def check_rate_limit(self, client_id: str, cost: int = 1) -> None:
        """
        Synchronous wrapper for legacy sync callers.
        Uses synchronous Redis connection to evaluate the Lua script atomically.
        """
        if not settings.RATE_LIMIT_ENABLED:
            return

        key = RedisKeys.rate_limit_key(client_id, prefix=self.prefix)
        ttl = max(self.window_seconds, int(self.capacity / max(0.1, self.refill_rate) * 2) + 60)
        now_ts = time.time()

        try:
            sync_client = redis_manager.get_sync_client()
            res = sync_client.eval(
                TOKEN_BUCKET_LUA_SCRIPT,
                1,
                key,
                self.capacity,
                self.refill_rate,
                cost,
                ttl,
                now_ts
            )
            allowed = bool(int(res[0]))
            tokens_remaining = float(res[1])
            retry_after = float(res[2])

            if not allowed:
                metrics_collector.record_rate_limit_drop()
                masked = self._mask_client_id(client_id)
                logger.warning(
                    f"Rate limit exceeded for client '{masked}'. "
                    f"Retry after {retry_after:.2f}s (Tokens remaining: {tokens_remaining:.1f}/{self.capacity})"
                )
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded ({self.refill_rate} req/sec). Please retry after {retry_after:.2f} seconds.",
                    retry_after=max(1, int(retry_after))
                )
        except RateLimitExceededError:
            raise
        except Exception as e:
            masked = self._mask_client_id(client_id)
            logger.error(f"Redis sync rate limiter failed for client '{masked}': {e}")
            if self.fail_closed:
                raise RateLimitExceededError(
                    message="Service temporarily unavailable due to rate limiter storage failure. Please retry.",
                    retry_after=5
                )


rate_limiter = DistributedRateLimiter()
RateLimiterService = DistributedRateLimiter


class TokenBucket:
    """In-memory Token Bucket retained for isolated local testing."""
    def __init__(self, capacity: int, fill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.fill_rate = float(fill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.fill_rate))
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0
        deficit = tokens - self.tokens
        return False, deficit / self.fill_rate

