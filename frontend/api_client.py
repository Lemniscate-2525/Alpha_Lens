"""
AlphaLens - Frontend API Client
Reusable synchronous HTTP client for communicating with the deployed AlphaLens FastAPI backend.
Reads configuration from st.secrets or environment variables.
Rule: Zero backend model imports, no local RAG fallback, strict error masking.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import httpx

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


class AlphaLensAPIError(Exception):
    """Base exception for API communication failures."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class APIConnectionError(AlphaLensAPIError):
    """Raised when the backend API is unreachable or waking up."""
    pass


class APIAuthenticationError(AlphaLensAPIError):
    """Raised on HTTP 401 Unauthorized (invalid API key)."""
    pass


class APIRateLimitError(AlphaLensAPIError):
    """Raised on HTTP 429 Rate Limit Exceeded."""
    def __init__(self, message: str, retry_after: int = 60, details: Optional[Any] = None):
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after


class APITimeoutError(AlphaLensAPIError):
    """Raised when an API request times out."""
    pass


@dataclass
class ClientCitation:
    chunk_id: str
    ticker: str
    fiscal_quarter: str
    fiscal_year: str
    quote_snippet: str
    speaker_name: Optional[str] = None
    speaker_role: Optional[str] = None
    section: Optional[str] = None
    confidence_score: float = 1.0

    @property
    def citation_id(self) -> str:
        return self.chunk_id[:6]

    @property
    def document_title(self) -> str:
        return f"{self.ticker} {self.fiscal_quarter} {self.fiscal_year}".strip()

    @property
    def snippet_text(self) -> str:
        return self.quote_snippet


@dataclass
class ClientSearchResult:
    chunk_id: str
    content: str
    score: float
    retrieval_type: str = "fused"
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def fiscal_period(self) -> str:
        q = self.metadata.get("fiscal_quarter", "")
        y = self.metadata.get("fiscal_year", "")
        return f"{q} {y}".strip()

    @property
    def doc_type(self) -> str:
        return self.metadata.get("document_type", "transcript")

    @property
    def speaker(self) -> Optional[str]:
        return self.metadata.get("speaker_name")

    @property
    def role(self) -> Optional[str]:
        return self.metadata.get("speaker_role")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "GENERAL")


@dataclass
class ClientQueryResponse:
    query: str
    answer: str
    latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    citations: List[ClientCitation] = field(default_factory=list)
    search_results: List[ClientSearchResult] = field(default_factory=list)
    cache_hit_type: Optional[str] = None
    is_unanswerable: bool = False
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    trace_id: Optional[str] = None

    @property
    def total_latency_ms(self) -> float:
        return self.latency_ms

    @property
    def is_cache_hit(self) -> bool:
        return self.cache_hit_type is not None

    @property
    def context_chunks(self) -> List[ClientSearchResult]:
        return self.search_results


class AlphaLensAPIClient:
    """Production HTTP client for the AlphaLens backend API."""

    DEFAULT_BASE_URL = "https://alpha-lens-api.onrender.com"
    DEFAULT_API_KEY = "alpha-lens-master-key-prod-2025"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 60.0
    ):
        self.base_url = (base_url or self._resolve_base_url()).rstrip("/")
        self.api_key = api_key or self._resolve_api_key()
        self.timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=10.0,
            read=timeout_seconds,
            write=15.0
        )
        self._client: Optional[httpx.Client] = None

    def _resolve_base_url(self) -> str:
        # Check Streamlit Secrets first
        if _HAS_STREAMLIT:
            try:
                if "BACKEND_API_URL" in st.secrets:
                    return str(st.secrets["BACKEND_API_URL"])
            except Exception:
                pass
        # Fall back to environment variable
        return os.getenv("BACKEND_API_URL", os.getenv("API_BASE_URL", self.DEFAULT_BASE_URL))

    def _resolve_api_key(self) -> str:
        if _HAS_STREAMLIT:
            try:
                if "API_KEY" in st.secrets:
                    return str(st.secrets["API_KEY"])
                if "MASTER_API_KEY" in st.secrets:
                    return str(st.secrets["MASTER_API_KEY"])
            except Exception:
                pass
        return os.getenv("API_KEY", os.getenv("MASTER_API_KEY", self.DEFAULT_API_KEY))

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            headers = {
                "X-API-Key": self.api_key,
                "Accept": "application/json",
                "User-Agent": "AlphaLens-Streamlit-Client/1.0"
            }
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        max_retries: int = 2
    ) -> httpx.Response:
        import time
        client = self._get_client()
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "AlphaLens-Streamlit-Client/1.0"
        }

        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params,
                    files=files,
                    data=data
                )
            except httpx.ConnectTimeout:
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                raise APIConnectionError(
                    f"Connection timed out while contacting {self.base_url}. The Render container may be starting up."
                )
            except httpx.ReadTimeout:
                raise APITimeoutError(
                    "Request timed out waiting for generation to complete on the backend."
                )
            except (httpx.ConnectError, httpx.NetworkError) as e:
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                raise APIConnectionError(
                    f"Unable to connect to AlphaLens API at {self.base_url}. Verify backend availability and network."
                )
            except Exception as e:
                raise AlphaLensAPIError(f"Unexpected network transport error: {str(e)}")

            # If gateway error (502 Bad Gateway / 503 / 504), Render may be finishing boot
            if response.status_code in (502, 503, 504) and attempt < max_retries:
                time.sleep(2.0)
                continue
            break

        if response is None:
            raise APIConnectionError(f"No response received from {self.base_url}")

        if response.status_code == 401:
            raise APIAuthenticationError(
                "Authentication failed: Invalid API key. Please check your API_KEY setting.",
                status_code=401
            )
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise APIRateLimitError(
                f"Rate limit exceeded on backend. Please wait {retry_after}s before retrying.",
                retry_after=retry_after
            )
        elif response.status_code in (502, 503, 504):
            raise APIConnectionError(
                f"Backend service is waking up or temporarily unavailable (HTTP {response.status_code}). "
                "Render free tier spins down on idle and takes ~30-50 seconds to initialize. Please wait a moment and retry.",
                status_code=response.status_code
            )
        elif response.status_code >= 400:
            try:
                err_data = response.json()
                msg = err_data.get("message") or err_data.get("detail") or response.text
            except Exception:
                msg = response.text or f"HTTP {response.status_code} error"
            raise AlphaLensAPIError(
                f"Backend request failed (HTTP {response.status_code}): {msg}",
                status_code=response.status_code
            )

        return response

    def health_check(self) -> Dict[str, Any]:
        """Queries the health endpoint to verify backend and Redis connectivity."""
        try:
            # Try /health root probe first, fallback to /api/v1/health
            resp = self._request("GET", "/health")
            return resp.json()
        except AlphaLensAPIError:
            resp = self._request("GET", "/api/v1/health")
            return resp.json()

    def get_metrics(self) -> Dict[str, Any]:
        """Retrieves real-time system metrics and latency percentiles."""
        resp = self._request("GET", "/api/v1/metrics")
        return resp.json()

    def execute_query(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_quarter: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        speaker_role: Optional[str] = None,
        top_k: int = 5,
        use_reranker: bool = True,
        use_semantic_cache: bool = True
    ) -> ClientQueryResponse:
        """Submits a financial query to the backend RAG pipeline."""
        payload = {
            "query": query,
            "ticker": ticker if ticker and ticker != "ALL" else None,
            "fiscal_quarter": fiscal_quarter,
            "fiscal_year": fiscal_year,
            "speaker_role": speaker_role,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "use_semantic_cache": use_semantic_cache,
            "stream": False
        }
        resp = self._request("POST", "/api/v1/query", json=payload)
        data = resp.json()

        # Parse citations
        citations = []
        for c in data.get("citations", []):
            citations.append(ClientCitation(
                chunk_id=c.get("chunk_id", ""),
                ticker=c.get("ticker", ""),
                fiscal_quarter=c.get("fiscal_quarter", ""),
                fiscal_year=c.get("fiscal_year", ""),
                quote_snippet=c.get("quote_snippet", ""),
                speaker_name=c.get("speaker_name"),
                speaker_role=c.get("speaker_role"),
                section=c.get("section"),
                confidence_score=float(c.get("confidence_score", 1.0))
            ))

        # Parse context chunks
        search_results = []
        for s in data.get("search_results", []):
            search_results.append(ClientSearchResult(
                chunk_id=s.get("chunk_id", ""),
                content=s.get("content", ""),
                score=float(s.get("score", 0.0)),
                retrieval_type=s.get("retrieval_type", "fused"),
                parent_chunk_id=s.get("parent_chunk_id"),
                parent_content=s.get("parent_content"),
                metadata=s.get("metadata", {})
            ))

        return ClientQueryResponse(
            query=data.get("query", query),
            answer=data.get("answer", ""),
            latency_ms=float(data.get("latency_ms", 0.0)),
            retrieval_latency_ms=float(data.get("retrieval_latency_ms", 0.0)),
            rerank_latency_ms=float(data.get("rerank_latency_ms", 0.0)),
            generation_latency_ms=float(data.get("generation_latency_ms", 0.0)),
            citations=citations,
            search_results=search_results,
            cache_hit_type=data.get("cache_hit_type"),
            is_unanswerable=bool(data.get("is_unanswerable", False)),
            tokens_used=int(data.get("tokens_used", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            trace_id=data.get("trace_id")
        )

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieves list of all indexed documents from the backend."""
        resp = self._request("GET", "/api/v1/ingest/documents")
        return resp.json()

    def list_jobs(self, offset: int = 0, limit: int = 20) -> Dict[str, Any]:
        """Lists recent background ingestion jobs with status and progress."""
        resp = self._request("GET", "/api/v1/ingest/jobs", params={"offset": offset, "limit": limit})
        return resp.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetches status of a specific ingestion job."""
        resp = self._request("GET", f"/api/v1/ingest/jobs/{job_id}")
        return resp.json()

    def upload_document_json(self, json_payload: str) -> Dict[str, Any]:
        """Submits a raw JSON document string for background ingestion."""
        resp = self._request("POST", "/api/v1/ingest", json={"json_document": json_payload})
        return resp.json()

    def upload_document_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Uploads a document file via multipart/form-data for ingestion."""
        files = {"file": (filename, file_bytes, "application/json")}
        resp = self._request("POST", "/api/v1/ingest/upload", files=files)
        return resp.json()

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Requests deletion of a document from vector index and idempotency store."""
        resp = self._request("DELETE", f"/api/v1/ingest/documents/{document_id}")
        return resp.json()

    def close(self):
        if self._client and not self._client.is_closed:
            self._client.close()


# Shared singleton client instance
api_client = AlphaLensAPIClient()
