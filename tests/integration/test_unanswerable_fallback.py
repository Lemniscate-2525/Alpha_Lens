"""
Integration test verifying unanswerable query fallback and hallucination defense.
"""
import pytest
from starlette.testclient import TestClient
from src.api.app import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "alpha-lens-master-key-prod-2025"}


def test_unanswerable_query_returns_structured_fallback():
    query_payload = {
        "query": "What is NVIDIA's gross margin guidance for fiscal year 2029?",
        "ticker": "NVDA",
        "top_k": 3,
        "use_reranker": False,
        "use_semantic_cache": False
    }

    resp = client.post("/api/v1/query", json=query_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_unanswerable"] is True
    assert "INFORMATION_NOT_DISCLOSED_IN_FILINGS" in data["answer"]

