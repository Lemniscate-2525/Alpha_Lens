"""
Integration Tests for Redis-Backed Asynchronous Ingestion & Worker Queue
Validates non-blocking HTTP 202 response, background job consumption, status transitions,
and distributed idempotency.
"""

import asyncio
import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.app import app
from src.core.models import JobStatus
from src.workers.job_registry import job_registry
from src.workers.queue_service import ingestion_queue
from src.workers.ingestion_worker import IngestionWorker
from src.database.vector_store import vector_store

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "alpha-lens-master-key-prod-2025"}


@pytest.mark.asyncio
async def test_async_ingestion_api_returns_202_immediately_and_worker_processes():
    """
    Critical Test:
    1. POST /api/v1/ingest returns HTTP 202 immediately with status QUEUED.
    2. Background worker consumes job from Redis Stream.
    3. Job status updates to COMPLETED with parent/child chunk counts.
    """
    sample_file = Path("data/raw/msft_q4_2024_transcript.json")
    if not sample_file.exists():
        pytest.skip("Test transcript not found.")

    raw_json = sample_file.read_text(encoding="utf-8")

    # Submit ingestion request
    response = client.post(
        "/api/v1/ingest",
        json={"json_document": raw_json},
        headers=AUTH_HEADERS
    )
    assert response.status_code == 202
    data = response.json()
    job_id = data["job_id"]
    assert job_id.startswith("job-")
    assert data["status"] in ["queued", "skipped_existing"]

    # Verify job exists in Redis job registry
    job_in_redis = await job_registry.get_job(job_id)
    assert job_in_redis is not None
    assert job_in_redis.document_id == "MSFT_2024_Q4_earnings_call_transcript"

    # Now simulate independent worker consuming tasks until our job is processed
    worker = IngestionWorker(worker_id="test-worker-1")
    for _ in range(10):
        task = await ingestion_queue.dequeue(worker_id=worker.worker_id, block_ms=500)
        if not task:
            break
        msg_id, fields = task
        curr_job = fields.get("job_id")
        await worker._process_task(curr_job, msg_id)
        if curr_job == job_id:
            break

    # Re-check job status -> MUST BE COMPLETED or SKIPPED_EXISTING
    completed_job = await job_registry.get_job(job_id)
    assert completed_job.status in [JobStatus.COMPLETED, JobStatus.SKIPPED_EXISTING]
    assert completed_job.total_child_chunks > 0 or completed_job.status == JobStatus.SKIPPED_EXISTING

    # Verify via API endpoint: GET /api/v1/ingest/jobs/{job_id}
    status_resp = client.get(f"/api/v1/ingest/jobs/{job_id}", headers=AUTH_HEADERS)
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ["completed", "skipped_existing"]


@pytest.mark.asyncio
async def test_async_ingestion_idempotency_prevents_duplicate_chunks():
    """Submitting the identical document twice must yield SKIPPED_EXISTING without duplicating chunks."""
    sample_file = Path("data/raw/nvda_q2_2025_transcript.json")
    if not sample_file.exists():
        pytest.skip("Test transcript not found.")

    raw_json = sample_file.read_text(encoding="utf-8")

    # First submission
    resp1 = client.post("/api/v1/ingest", json={"json_document": raw_json}, headers=AUTH_HEADERS)
    assert resp1.status_code == 202
    job1_id = resp1.json()["job_id"]

    # Worker processes job 1
    worker = IngestionWorker(worker_id="test-worker-idempotency")
    task1 = await ingestion_queue.dequeue(worker_id=worker.worker_id, block_ms=500)
    if task1:
        await worker._process_task(job1_id, task1[0])

    initial_chunk_count = vector_store.count()

    # Second submission with exact same document
    resp2 = client.post("/api/v1/ingest", json={"json_document": raw_json}, headers=AUTH_HEADERS)
    assert resp2.status_code == 202
    job2_id = resp2.json()["job_id"]

    # Worker processes job 2
    task2 = await ingestion_queue.dequeue(worker_id=worker.worker_id, block_ms=500)
    if task2:
        await worker._process_task(job2_id, task2[0])

    job2_status = await job_registry.get_job(job2_id)
    assert job2_status.status == JobStatus.SKIPPED_EXISTING
    assert job2_status.stage == "IDEMPOTENT_SKIPPED"

    # Chunk count in vector store must NOT have increased!
    assert vector_store.count() == initial_chunk_count
