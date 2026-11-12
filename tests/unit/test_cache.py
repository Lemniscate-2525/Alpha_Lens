"""
Unit tests for Two-Tier Cache (Exact Hash + Semantic Similarity + Stampede Locks).
"""
import pytest
from src.services.cache_service import TwoTierCacheService
from src.core.models import QueryResponse


def test_exact_cache_key_generation():
    key1 = TwoTierCacheService.compute_exact_key("What was Q2 revenue?", {"ticker": "NVDA", "fiscal_quarter": "Q2"})
    # Same query with uppercase and extra spaces must yield identical hash
    key2 = TwoTierCacheService.compute_exact_key("  WHAT was Q2 Revenue?  ", {"ticker": "NVDA", "fiscal_quarter": "Q2"})
    assert key1 == key2

    # Different ticker must yield different hash
    key3 = TwoTierCacheService.compute_exact_key("What was Q2 revenue?", {"ticker": "MSFT", "fiscal_quarter": "Q2"})
    assert key1 != key3


def test_exact_cache_put_and_get():
    cache = TwoTierCacheService()
    dummy_resp = QueryResponse(
        query="test query",
        answer="test answer",
        latency_ms=10.0,
        retrieval_latency_ms=5.0,
        rerank_latency_ms=2.0,
        generation_latency_ms=3.0,
    )

    exact_key = cache.compute_exact_key("test query", {"ticker": "NVDA"})
    cache.exact_cache[exact_key] = (dummy_resp, 9999999999.0)

    cached = cache.get_exact(exact_key)
    assert cached is not None
    assert cached.answer == "test answer"
