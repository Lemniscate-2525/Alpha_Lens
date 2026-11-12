"""
Integration tests for FastAPI endpoints: Health, Ingestion, and Query.
"""
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient
from src.api.app import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "alpha-lens-master-key-prod-2025"}


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_chunks" in data


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "latency_e2e_ms" in data


def test_ingest_and_query_flow():
    # Load sample raw transcript
    sample_file = Path("data/raw/nvda_q2_2025_transcript.json")
    if not sample_file.exists():
        pytest.skip("Sample raw transcript not found.")

    raw_json = sample_file.read_text(encoding="utf-8")

    # Ingest document
    ingest_resp = client.post(
        "/api/v1/ingest",
        json={"json_document": raw_json},
        headers=AUTH_HEADERS
    )
    assert ingest_resp.status_code == 202
    job = ingest_resp.json()
    assert job["ticker"] == "NVDA"
    assert job["status"] in ["completed", "skipped_existing"]

    # Execute query against ingested document
    query_payload = {
        "query": "What was NVIDIA's revenue and gross margin in Q2?",
        "ticker": "NVDA",
        "fiscal_quarter": "Q2",
        "top_k": 3,
        "use_reranker": False,
        "use_semantic_cache": True
    }

    query_resp = client.post(
        "/api/v1/query",
        json=query_payload,
        headers=AUTH_HEADERS
    )
    assert query_resp.status_code == 200
    res_data = query_resp.json()
    assert "answer" in res_data
    assert len(res_data["search_results"]) > 0
    assert "latency_ms" in res_data
