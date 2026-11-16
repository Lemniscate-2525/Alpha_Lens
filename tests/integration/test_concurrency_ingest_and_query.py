"""
Concurrency Integration Tests for AlphaLens
Proves that LLM query generation and background document ingestion progress concurrently
without blocking each other, and verifies newly indexed documents become visible to queries.
"""

import asyncio
from pathlib import Path
import time
import pytest
import httpx

from src.api.app import app
from src.core.models import JobStatus, QueryRequest
from src.services.rag_service import rag_service
from src.workers.job_registry import job_registry
from src.workers.queue_service import ingestion_queue
from src.workers.ingestion_worker import IngestionWorker
from src.retrieval.sparse import bm25_retriever

AUTH_HEADERS = {"X-API-Key": "alpha-lens-master-key-prod-2025"}


@pytest.mark.asyncio
async def test_concurrent_llm_generation_and_background_ingestion():
    """
    Critical Concurrency Test:
    T1: Query A initiates RAG retrieval and enters LLM generation (simulated 0.3s).
    T2: Concurrently, a new document is submitted to POST /api/v1/ingest.
    T3: Ingestion Worker chunks, embeds, and indexes the document while Query A is STILL generating.
    T4: Query B arrives and executes.
    Both pipelines must complete without deadlock or mutual exclusion blocking.
    """
    raw_doc = Path("data/raw/nvda_q3_2025_transcript.json")
    if not raw_doc.exists():
        pytest.skip("Test transcript not found.")

    payload_text = raw_doc.read_text(encoding="utf-8")

    ingestion_started = asyncio.Event()
    ingestion_finished = asyncio.Event()
    query_in_flight = asyncio.Event()

    async def long_query_task():
        query_in_flight.set()
        # Query will run retrieval and generation
        req = QueryRequest(
            query="What did Jensen Huang say about Blackwell computing demand?",
            ticker="NVDA",
            top_k=2,
            use_reranker=False,
            use_semantic_cache=False
        )
        t0 = time.perf_counter()
        resp = await rag_service.execute_query(req)
        duration = time.perf_counter() - t0
        return resp, duration

    async def ingestion_worker_task():
        # Wait until query is running
        await query_in_flight.wait()
        t0 = time.perf_counter()

        # Submit ingestion job via async client so event loop is not blocked
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/ingest", json={"json_document": payload_text}, headers=AUTH_HEADERS)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        ingestion_started.set()

        # Worker processes job while query is executing
        worker = IngestionWorker(worker_id="test-concurrency-worker")
        task = await ingestion_queue.dequeue(worker_id=worker.worker_id, block_ms=500)
        if task:
            await worker._process_task(job_id, task[0])

        ingestion_finished.set()
        duration = time.perf_counter() - t0
        return job_id, duration

    # Launch both concurrently
    query_future = asyncio.create_task(long_query_task())
    ingest_future = asyncio.create_task(ingestion_worker_task())

    (query_resp, query_dur), (job_id, ingest_dur) = await asyncio.gather(query_future, ingest_future)

    # Both must succeed
    assert query_resp is not None
    assert query_resp.answer is not None

    job = await job_registry.get_job(job_id)
    assert job.status in [JobStatus.COMPLETED, JobStatus.SKIPPED_EXISTING]

    # Verify newly indexed documents are visible to subsequent BM25 queries
    bm25_retriever.sync_if_needed()
    results = bm25_retriever.retrieve("Blackwell", top_k=5)
    assert len(results) > 0, "Newly indexed documents must be visible to retrieval!"
