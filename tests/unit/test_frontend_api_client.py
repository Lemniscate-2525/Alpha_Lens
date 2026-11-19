"""
Unit Tests for Frontend API Client (frontend/api_client.py)
Validates request formatting, error handling, timeouts, authentication, and response models.
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock

from frontend.api_client import (
    AlphaLensAPIClient,
    ClientQueryResponse,
    ClientCitation,
    ClientSearchResult,
    APIAuthenticationError,
    APIConnectionError,
    APIRateLimitError,
    APITimeoutError,
    AlphaLensAPIError
)


def test_api_client_initialization_defaults():
    client = AlphaLensAPIClient(base_url="https://test-api.onrender.com", api_key="secret-key")
    assert client.base_url == "https://test-api.onrender.com"
    assert client.api_key == "secret-key"


def test_client_query_response_properties():
    citation = ClientCitation(
        chunk_id="chunk12345",
        ticker="NVDA",
        fiscal_quarter="Q3",
        fiscal_year="2025",
        quote_snippet="Data Center revenue grew 112%",
        speaker_name="Jensen Huang",
        speaker_role="CEO",
        confidence_score=0.98
    )
    assert citation.citation_id == "chunk1"
    assert citation.document_title == "NVDA Q3 2025"
    assert citation.snippet_text == "Data Center revenue grew 112%"

    chunk = ClientSearchResult(
        chunk_id="c1",
        content="Passage text",
        score=0.95,
        metadata={"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025", "speaker_name": "Jensen"}
    )
    assert chunk.ticker == "NVDA"
    assert chunk.fiscal_period == "Q3 2025"
    assert chunk.speaker == "Jensen"

    resp = ClientQueryResponse(
        query="Revenue?",
        answer="Total revenue was $30B",
        latency_ms=125.4,
        retrieval_latency_ms=25.0,
        rerank_latency_ms=45.0,
        generation_latency_ms=55.4,
        citations=[citation],
        search_results=[chunk],
        cache_hit_type="exact",
        tokens_used=150,
        estimated_cost_usd=0.0
    )
    assert resp.total_latency_ms == 125.4
    assert resp.is_cache_hit is True
    assert len(resp.context_chunks) == 1


@patch("httpx.Client.request")
def test_execute_query_success(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "query": "What is NVDA Q3 revenue?",
        "answer": "NVDA Q3 revenue was $35.1B [chunk1].",
        "latency_ms": 210.5,
        "retrieval_latency_ms": 30.0,
        "rerank_latency_ms": 50.0,
        "generation_latency_ms": 130.5,
        "cache_hit_type": None,
        "is_unanswerable": False,
        "tokens_used": 280,
        "estimated_cost_usd": 0.00015,
        "citations": [
            {
                "chunk_id": "chunk123",
                "ticker": "NVDA",
                "fiscal_quarter": "Q3",
                "fiscal_year": "2025",
                "quote_snippet": "Revenue reached 35.1 billion",
                "speaker_name": "Colette Kress",
                "speaker_role": "CFO",
                "confidence_score": 0.99
            }
        ],
        "search_results": [
            {
                "chunk_id": "chunk123",
                "content": "Revenue reached 35.1 billion, up 94% year-over-year.",
                "score": 0.92,
                "retrieval_type": "fused",
                "metadata": {"ticker": "NVDA", "fiscal_quarter": "Q3", "fiscal_year": "2025"}
            }
        ]
    }
    mock_request.return_value = mock_resp

    client = AlphaLensAPIClient(base_url="https://api.test", api_key="test-key")
    result = client.execute_query(
        query="What is NVDA Q3 revenue?",
        ticker="NVDA",
        use_reranker=True,
        use_semantic_cache=True
    )

    assert result.answer.startswith("NVDA Q3 revenue was $35.1B")
    assert result.is_cache_hit is False
    assert len(result.citations) == 1
    assert result.citations[0].speaker_name == "Colette Kress"
    assert len(result.context_chunks) == 1
    assert result.context_chunks[0].ticker == "NVDA"

    # Verify request payload
    called_args = mock_request.call_args
    assert called_args.kwargs["method"] == "POST"
    assert called_args.kwargs["json"]["query"] == "What is NVDA Q3 revenue?"
    assert called_args.kwargs["json"]["ticker"] == "NVDA"


@patch("httpx.Client.request")
def test_authentication_failure_raises_exception(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_request.return_value = mock_resp

    client = AlphaLensAPIClient(base_url="https://api.test", api_key="invalid-key")
    with pytest.raises(APIAuthenticationError) as exc_info:
        client.health_check()
    assert "Authentication failed" in str(exc_info.value)


@patch("httpx.Client.request")
def test_rate_limit_failure_raises_exception(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "45"}
    mock_resp.text = "Too Many Requests"
    mock_request.return_value = mock_resp

    client = AlphaLensAPIClient(base_url="https://api.test", api_key="test-key")
    with pytest.raises(APIRateLimitError) as exc_info:
        client.execute_query("Any query")
    assert exc_info.value.retry_after == 45


@patch("httpx.Client.request")
def test_connection_error_raises_friendly_exception(mock_request):
    mock_request.side_effect = httpx.ConnectError("Failed to resolve host")

    client = AlphaLensAPIClient(base_url="https://bad-host.test", api_key="test-key")
    with pytest.raises(APIConnectionError) as exc_info:
        client.health_check()
    assert "Unable to connect" in str(exc_info.value)


@patch("httpx.Client.request")
def test_timeout_error_raises_friendly_exception(mock_request):
    mock_request.side_effect = httpx.ReadTimeout("Read timed out")

    client = AlphaLensAPIClient(base_url="https://api.test", api_key="test-key")
    with pytest.raises(APITimeoutError) as exc_info:
        client.execute_query("Long prompt")
    assert "timed out" in str(exc_info.value)


def test_no_heavy_backend_modules_imported():
    """Confirms no machine learning or vector storage packages are imported by frontend in an isolated process."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import frontend.app\n"
        "import frontend.api_client\n"
        "import frontend.components.query_interface\n"
        "import frontend.components.filings_browser\n"
        "import frontend.components.system_telemetry\n"
        "import frontend.components.benchmark_view\n"
        "forbidden = ['torch', 'sentence_transformers', 'chromadb', 'rank_bm25']\n"
        "loaded = [pkg for pkg in forbidden if pkg in sys.modules]\n"
        "if loaded:\n"
        "    print('FORBIDDEN:' + ','.join(loaded))\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Frontend imported heavy ML libraries: {res.stdout} {res.stderr}"
