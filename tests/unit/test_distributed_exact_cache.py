"""
Unit Tests for Redis-Backed Distributed Exact Cache
Validates cross-instance cache reads/writes, TTL expiry, metadata sensitivity, and serialization.
"""

import asyncio
import time
import pytest
import fakeredis.aioredis
import fakeredis

from src.services.cache_service import TwoTierCacheService
from src.core.models import QueryResponse, Citation, SearchResult


@pytest.mark.asyncio
async def test_exact_cache_write_instance_a_read_instance_b(monkeypatch):
    """Critical Test: Cache write via Instance A is immediately readable by Instance B."""
    monkeypatch.setattr("src.ingestion.embedder.embedding_service.embed_text", lambda q: [0.1] * 384)
    server = fakeredis.FakeServer()
    client_a = fakeredis.aioredis.FakeRedis(server=server)
    client_b = fakeredis.aioredis.FakeRedis(server=server)

    cache_a = TwoTierCacheService(redis_client=client_a)
    cache_b = TwoTierCacheService(redis_client=client_b)

    query = "What was Microsoft Q4 Azure growth?"
    filters = {"ticker": "MSFT", "fiscal_quarter": "Q4", "fiscal_year": "2024"}

    original_resp = QueryResponse(
        query=query,
        answer="Microsoft Azure cloud revenue grew 29% in constant currency.",
        citations=[Citation(chunk_id="msft-c1", ticker="MSFT", fiscal_quarter="Q4", fiscal_year="2024", quote_snippet="Azure grew 29%")],
        search_results=[SearchResult(chunk_id="msft-c1", content="Azure grew 29%", score=0.92, retrieval_type="reranked")],
        latency_ms=85.0,
        retrieval_latency_ms=15.0,
        rerank_latency_ms=10.0,
        generation_latency_ms=60.0
    )

    # Save via Instance A
    await cache_a.set(query, filters, original_resp)

    # Read via Instance B
    exact_key = cache_b.compute_exact_key(query, filters)
    cached_b = await cache_b.get_exact(exact_key)

    assert cached_b is not None
    assert cached_b.answer == original_resp.answer
    assert len(cached_b.citations) == 1
    assert cached_b.citations[0].chunk_id == "msft-c1"


@pytest.mark.asyncio
async def test_exact_cache_metadata_sensitivity():
    """Different tickers or fiscal quarters must generate distinct cache keys."""
    key_nvda_q2 = TwoTierCacheService.compute_exact_key(
        "What was revenue?", {"ticker": "NVDA", "fiscal_quarter": "Q2", "fiscal_year": "2025"}
    )
    key_nvda_q3 = TwoTierCacheService.compute_exact_key(
        "What was revenue?", {"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025"}
    )
    key_msft_q2 = TwoTierCacheService.compute_exact_key(
        "What was revenue?", {"ticker": "MSFT", "fiscal_quarter": "Q2", "fiscal_year": "2025"}
    )

    assert key_nvda_q2 != key_nvda_q3
    assert key_nvda_q2 != key_msft_q2


@pytest.mark.asyncio
async def test_exact_cache_ttl_expiry(monkeypatch):
    """Expired cache keys must return None."""
    monkeypatch.setattr("src.ingestion.embedder.embedding_service.embed_text", lambda q: [0.1] * 384)
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    cache = TwoTierCacheService(redis_client=client)
    cache.exact_ttl = 1  # 1 second TTL

    query = "Temporary query"
    filters = {"ticker": "NVDA"}
    resp = QueryResponse(
        query=query,
        answer="Ephemeral Answer",
        latency_ms=10.0,
        retrieval_latency_ms=2.0,
        rerank_latency_ms=2.0,
        generation_latency_ms=6.0
    )

    await cache.set(query, filters, resp)
    exact_key = cache.compute_exact_key(query, filters)

    # Immediate read succeeds
    assert await cache.get_exact(exact_key) is not None

    # Wait for TTL expiration
    await asyncio.sleep(1.1)
    assert await cache.get_exact(exact_key) is None
