"""
Unit Tests for Redis-Backed Distributed Token Bucket Rate Limiter
Validates cross-instance coordination, atomic token consumption, refill, TTL, and 429 enforcement.
"""

import asyncio
import time
import pytest
import fakeredis.aioredis
import fakeredis

from src.services.rate_limiter import DistributedRateLimiter
from src.core.exceptions import RateLimitExceededError


@pytest.mark.asyncio
async def test_distributed_rate_limiter_two_instances_share_state():
    """
    Critical Test: Two independent Python rate limiter instances
    must share one global bucket state through Redis.
    """
    server = fakeredis.FakeServer()
    client_a = fakeredis.aioredis.FakeRedis(server=server)
    client_b = fakeredis.aioredis.FakeRedis(server=server)

    limiter_a = DistributedRateLimiter(redis_client=client_a)
    limiter_b = DistributedRateLimiter(redis_client=client_b)
    limiter_a.capacity = 5
    limiter_b.capacity = 5
    limiter_a.refill_rate = 1.0
    limiter_b.refill_rate = 1.0

    client_id = "analyst_key_123"

    # Instance A consumes 3 tokens
    for _ in range(3):
        res = await limiter_a.evaluate(client_id, cost=1)
        assert res.allowed is True

    # Instance B consumes 2 tokens -> should succeed (total 5 tokens consumed)
    for _ in range(2):
        res = await limiter_b.evaluate(client_id, cost=1)
        assert res.allowed is True

    # 6th token requested via Instance A -> MUST BE REJECTED because Instance B depleted it!
    res_rejected = await limiter_a.evaluate(client_id, cost=1)
    assert res_rejected.allowed is False
    assert res_rejected.remaining < 1.0
    assert res_rejected.retry_after > 0.0

    # Instance B must also see the bucket exhausted
    res_b_rejected = await limiter_b.evaluate(client_id, cost=1)
    assert res_b_rejected.allowed is False


@pytest.mark.asyncio
async def test_rate_limiter_refill_and_burst():
    """Verifies that tokens refill correctly according to refill_rate."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    limiter = DistributedRateLimiter(redis_client=client)
    limiter.capacity = 3
    limiter.refill_rate = 10.0  # 10 tokens per sec -> 1 token every 100ms

    client_id = "client_refill_test"

    # Consume all 3 tokens
    for _ in range(3):
        res = await limiter.evaluate(client_id, cost=1)
        assert res.allowed is True

    # Immediately rejected
    res = await limiter.evaluate(client_id, cost=1)
    assert res.allowed is False

    # Wait 250ms -> should have refilled ~2 tokens
    await asyncio.sleep(0.25)
    res_after = await limiter.evaluate(client_id, cost=1)
    assert res_after.allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_exception_raising():
    """Verifies that check_rate_limit_async raises RateLimitExceededError on exhaustion."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    limiter = DistributedRateLimiter(redis_client=client)
    limiter.capacity = 2
    limiter.refill_rate = 0.5

    client_id = "test_exc_client"
    await limiter.check_rate_limit_async(client_id)
    await limiter.check_rate_limit_async(client_id)

    # 3rd request must raise RateLimitExceededError with status 429
    with pytest.raises(RateLimitExceededError) as exc_info:
        await limiter.check_rate_limit_async(client_id)

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after >= 1
