"""
Unit Tests for Redis-Backed Distributed Semantic Cache
Validates semantic query similarity, strict financial metadata isolation,
ticker/quarter boundary protection, and threshold enforcement.
"""

import asyncio
import pytest
import fakeredis.aioredis
import fakeredis

from src.services.cache_service import TwoTierCacheService
from src.core.models import QueryResponse


@pytest.mark.asyncio
async def test_semantic_cache_equivalent_queries_hit():
    """Semantically equivalent queries with identical financial filters must hit the cache."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    cache = TwoTierCacheService(redis_client=client)
    cache.cosine_threshold = 0.85

    query_a = "What was NVIDIA's Q3 2025 gross margin?"
    query_b = "Tell me NVIDIA gross margin for the third quarter of 2025"
    filters = {"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025"}

    resp = QueryResponse(
        query=query_a,
        answer="NVIDIA GAAP gross margin was 74.6% in Q3 FY25.",
        latency_ms=100.0,
        retrieval_latency_ms=20.0,
        rerank_latency_ms=10.0,
        generation_latency_ms=70.0
    )

    await cache.set(query_a, filters, resp)

    # Query B has different wording but same semantics & filters
    hit_result = await cache.get_semantic(query_b, filters)
    assert hit_result is not None
    cached_resp, score = hit_result
    assert score >= cache.cosine_threshold
    assert cached_resp.answer == "NVIDIA GAAP gross margin was 74.6% in Q3 FY25."


@pytest.mark.asyncio
async def test_semantic_cache_financial_metadata_isolation():
    """
    CRITICAL FINANCIAL RULE:
    1. NVDA Q3 2025 must NOT match NVDA Q2 2025.
    2. NVDA must NOT match MSFT.
    """
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    cache = TwoTierCacheService(redis_client=client)

    query = "What was the total revenue?"
    nvda_q3_filters = {"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025"}
    nvda_q2_filters = {"ticker": "NVDA", "fiscal_quarter": "Q2", "fiscal_year": "2025"}
    msft_q3_filters = {"ticker": "MSFT", "fiscal_quarter": "Q3", "fiscal_year": "2025"}

    resp_nvda_q3 = QueryResponse(
        query=query,
        answer="NVIDIA Q3 revenue was $35.08 billion.",
        latency_ms=100.0,
        retrieval_latency_ms=20.0,
        rerank_latency_ms=10.0,
        generation_latency_ms=70.0
    )

    # Cache response for NVDA Q3
    await cache.set(query, nvda_q3_filters, resp_nvda_q3)

    # 1. Query for NVDA Q2 MUST NOT HIT NVDA Q3!
    hit_q2 = await cache.get_semantic(query, nvda_q2_filters)
    assert hit_q2 is None, "Cross-quarter semantic match must be rejected!"

    # 2. Query for MSFT Q3 MUST NOT HIT NVDA Q3!
    hit_msft = await cache.get_semantic(query, msft_q3_filters)
    assert hit_msft is None, "Cross-ticker semantic match must be rejected!"


@pytest.mark.asyncio
async def test_semantic_cache_threshold_rejection():
    """Unrelated queries with cosine similarity < threshold must result in a cache miss."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server)

    cache = TwoTierCacheService(redis_client=client)
    cache.cosine_threshold = 0.94

    query_a = "What was the total capital expenditure on H100 GPU clusters?"
    query_b = "Who is the corporate head of legal and compliance?"
    filters = {"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025"}

    resp = QueryResponse(
        query=query_a,
        answer="Capital expenditures were $3.2 billion.",
        latency_ms=90.0,
        retrieval_latency_ms=15.0,
        rerank_latency_ms=10.0,
        generation_latency_ms=65.0
    )

    await cache.set(query_a, filters, resp)

    hit = await cache.get_semantic(query_b, filters)
    assert hit is None, "Unrelated query must not trigger semantic cache hit!"
