"""
Integration Tests for AlphaLens Streamlit Frontend Integration
Validates end-to-end communication between AlphaLensAPIClient and the FastAPI backend.
"""

import pytest
import httpx
from starlette.testclient import TestClient

from src.api.app import app
from frontend.api_client import (
    AlphaLensAPIClient,
    APIAuthenticationError,
    APIConnectionError,
    ClientQueryResponse
)

AUTH_KEY = "alpha-lens-master-key-prod-2025"


class MockTransportClient(AlphaLensAPIClient):
    """API Client routed directly through Starlette ASGI app for local integration testing."""
    def __init__(self, api_key: str = AUTH_KEY):
        super().__init__(base_url="http://testserver", api_key=api_key)
        self._test_client = TestClient(app)

    def _get_client(self):
        # Return test client with transport
        return self._test_client


def test_frontend_health_check_integration():
    client = MockTransportClient()
    health = client.health_check()
    assert health["status"] in ["healthy", "degraded"]
    assert "redis_connected" in health


def test_frontend_metrics_integration():
    client = MockTransportClient()
    metrics = client.get_metrics()
    assert "total_requests" in metrics
    assert "p50_latency_ms" in metrics


def test_frontend_query_integration():
    client = MockTransportClient()
    resp = client.execute_query(
        query="What is Microsoft cloud revenue?",
        ticker="MSFT",
        use_reranker=False,
        use_semantic_cache=False
    )
    assert isinstance(resp, ClientQueryResponse)
    assert resp.answer is not None
    assert len(resp.answer) > 0
    assert resp.total_latency_ms > 0


def test_frontend_invalid_api_key_raises_401():
    client = MockTransportClient(api_key="wrong-unauthorized-key")
    with pytest.raises(APIAuthenticationError):
        client.execute_query("What was revenue?")


def test_frontend_document_listing_integration():
    client = MockTransportClient()
    docs = client.list_documents()
    assert isinstance(docs, list)


def test_frontend_jobs_listing_integration():
    client = MockTransportClient()
    jobs_resp = client.list_jobs(offset=0, limit=10)
    assert "jobs" in jobs_resp
    assert "total" in jobs_resp


def test_frontend_connection_failure_handling():
    # Points to an unallocated port to simulate backend downtime
    dead_client = AlphaLensAPIClient(base_url="http://127.0.0.1:59999", api_key=AUTH_KEY)
    with pytest.raises(APIConnectionError) as exc_info:
        dead_client.health_check()
    assert "Unable to connect" in str(exc_info.value)
