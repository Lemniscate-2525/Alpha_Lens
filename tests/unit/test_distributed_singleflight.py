"""
Unit Tests for Redis-Backed Distributed Singleflight Mutex
Validates request coalescing across independent application instances,
leader election, waiter notification, and lock release safety.
"""

import asyncio
import time
import pytest
import fakeredis.aioredis
import fakeredis

from src.services.singleflight import DistributedSingleflight
from src.services.distributed_lock import DistributedLock
from src.core.models import QueryResponse


@pytest.mark.asyncio
async def test_singleflight_concurrent_requests_coalesce_to_one_leader():
    """
    Critical Test: Two independent service instances submit the same query concurrently.
    Generation must execute EXACTLY ONCE. Both instances must receive the result.
    """
    server = fakeredis.FakeServer()
    client_a = fakeredis.aioredis.FakeRedis(server=server)
    client_b = fakeredis.aioredis.FakeRedis(server=server)

    sf_a = DistributedSingleflight(redis_client=client_a)
    sf_b = DistributedSingleflight(redis_client=client_b)

    cache_store = {}
    generation_call_count = 0

    async def mock_cache_getter():
        return cache_store.get("test_key")

    async def expensive_llm_generation():
        nonlocal generation_call_count
        generation_call_count += 1
        # Simulate LLM generation latency
        await asyncio.sleep(0.15)
        resp = QueryResponse(
            query="What was NVIDIA Q3 gross margin?",
            answer="NVIDIA Q3 GAAP gross margin was 74.6%.",
            latency_ms=150.0,
            retrieval_latency_ms=20.0,
            rerank_latency_ms=10.0,
            generation_latency_ms=120.0
        )
        cache_store["test_key"] = resp
        return resp

    # Submit concurrently via Instance A and Instance B
    task_a = asyncio.create_task(
        sf_a.execute_or_wait("test_key", expensive_llm_generation, mock_cache_getter)
    )
    task_b = asyncio.create_task(
        sf_b.execute_or_wait("test_key", expensive_llm_generation, mock_cache_getter)
    )

    (resp_a, is_leader_a), (resp_b, is_leader_b) = await asyncio.gather(task_a, task_b)

    # Assert exactly one leader and one waiter
    assert (is_leader_a and not is_leader_b) or (is_leader_b and not is_leader_a)
    # Generation must have executed exactly ONCE
    assert generation_call_count == 1
    # Both instances must receive the identical audited answer
    assert resp_a.answer == resp_b.answer
    assert resp_a.answer == "NVIDIA Q3 GAAP gross margin was 74.6%."


@pytest.mark.asyncio
async def test_distributed_lock_atomic_release():
    """Verifies that only the true owner token can release the lock."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    lock1 = DistributedLock("alpha_lens:lock:test", lease_ms=5000, redis_client=client)
    lock2 = DistributedLock("alpha_lens:lock:test", lease_ms=5000, redis_client=client)

    acquired1 = await lock1.acquire()
    assert acquired1 is True

    # lock2 cannot acquire because lock1 holds it
    acquired2 = await lock2.acquire()
    assert acquired2 is False

    # lock2 attempts to release lock1 -> MUST RETURN FALSE
    released2 = await lock2.release()
    assert released2 is False

    # Lock must still be held in Redis
    assert await client.exists("alpha_lens:lock:test") == 1

    # lock1 releases own lock -> SUCCESS
    released1 = await lock1.release()
    assert released1 is True
    assert await client.exists("alpha_lens:lock:test") == 0


@pytest.mark.asyncio
async def test_singleflight_promotes_waiter_if_leader_fails_without_cache():
    """If leader crashes without writing to cache, a waiter must become leader."""
    server = fakeredis.FakeServer()
    client_a = fakeredis.aioredis.FakeRedis(server=server)
    client_b = fakeredis.aioredis.FakeRedis(server=server)

    sf_a = DistributedSingleflight(redis_client=client_a)
    sf_b = DistributedSingleflight(redis_client=client_b)
    sf_b.poll_interval_s = 0.05
    sf_b.lease_ms = 200

    cache_store = {}

    async def mock_cache_getter():
        return cache_store.get("fail_key")

    async def failing_leader():
        # Leader acquires lock, then crashes/exits without saving to cache
        await asyncio.sleep(0.05)
        raise RuntimeError("Simulated Leader Process Crash")

    async def successful_worker():
        await asyncio.sleep(0.05)
        resp = QueryResponse(
            query="test",
            answer="Recovered Answer",
            latency_ms=50.0,
            retrieval_latency_ms=10.0,
            rerank_latency_ms=10.0,
            generation_latency_ms=30.0
        )
        cache_store["fail_key"] = resp
        return resp

    # Instance A starts and fails
    with pytest.raises(RuntimeError):
        await sf_a.execute_or_wait("fail_key", failing_leader, mock_cache_getter)

    # Instance B should now be able to become leader and finish
    resp_b, is_leader_b = await sf_b.execute_or_wait("fail_key", successful_worker, mock_cache_getter)
    assert is_leader_b is True
    assert resp_b.answer == "Recovered Answer"
