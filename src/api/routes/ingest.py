"""
AlphaLens - Document Ingestion Endpoints
Receives raw financial document payloads and enqueues tasks onto the Redis background stream queue.
Returns HTTP 202 immediately. Work is consumed by independent background ingestion workers.
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from src.core.models import IngestionJob, JobStatus
from src.ingestion.parsers import FinancialDocumentParser
from src.workers.job_registry import job_registry
from src.workers.queue_service import ingestion_queue
from src.api.dependencies import verify_api_key
from src.observability.logging import logger
from src.config import settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestPayloadRequest(BaseModel):
    json_document: str = Field(..., description="Raw JSON financial document payload")


class ListJobsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    jobs: List[IngestionJob]


@router.post("", response_model=IngestionJob, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    request: IngestPayloadRequest,
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    """
    Submits a financial document for asynchronous ingestion.
    Validates payload structure, creates job in Redis, enqueues onto Redis Stream,
    and returns HTTP 202 Accepted immediately without waiting for chunking/embedding.
    """
    try:
        doc = FinancialDocumentParser.parse_json_document(request.json_document)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document JSON structure: {str(e)}"
        )

    job_id = f"job-{uuid.uuid4().hex[:10]}"

    # Register job state in Redis
    job = await job_registry.register_job(
        job_id=job_id,
        document_id=doc.document_id,
        ticker=doc.ticker,
        raw_payload=request.json_document
    )

    # Enqueue task in Redis Stream
    await ingestion_queue.enqueue(
        job_id=job_id,
        document_id=doc.document_id,
        ticker=doc.ticker
    )

    logger.info(f"Accepted document '{doc.document_id}' for async processing. Job ID: {job_id}")
    return job


@router.get("/jobs/{job_id}", response_model=IngestionJob)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    """Retrieves current processing status, stage, progress, and error details for a job."""
    job = await job_registry.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found in registry."
        )
    return job


@router.get("/jobs", response_model=ListJobsResponse)
async def list_jobs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    api_key: str = Depends(verify_api_key)
) -> ListJobsResponse:
    """Lists submitted ingestion jobs with pagination."""
    jobs, total = await job_registry.list_jobs(offset=offset, limit=limit)
    return ListJobsResponse(total=total, offset=offset, limit=limit, jobs=jobs)


@router.post("/upload", response_model=IngestionJob, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_file(
    file: UploadFile = File(..., description="Financial document file (.json, .txt)"),
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    """Accepts multipart/form-data document file upload and enqueues for async ingestion."""
    content_bytes = await file.read()
    try:
        content_str = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File content must be valid UTF-8 text or JSON: {str(e)}"
        )
    return await ingest_document(IngestPayloadRequest(json_document=content_str), api_key=api_key)


@router.get("/documents")
async def list_documents(
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """Returns all unique indexed documents and their summary metadata."""
    jobs, _ = await job_registry.list_jobs(offset=0, limit=100)
    seen_docs: Dict[str, Dict[str, Any]] = {}
    for j in jobs:
        if j.document_id and j.document_id not in seen_docs:
            seen_docs[j.document_id] = {
                "document_id": j.document_id,
                "ticker": j.ticker,
                "status": j.status.value,
                "total_parent_chunks": j.total_parent_chunks,
                "total_child_chunks": j.total_child_chunks,
                "processing_time_ms": j.processing_time_ms,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "job_id": j.job_id
            }
    return list(seen_docs.values())


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Deletes an indexed document from the vector store and removes its idempotency lock."""
    from src.database.vector_store import vector_store
    from src.infrastructure.redis_client import redis_manager
    from src.infrastructure.redis_keys import RedisKeys

    chunks_deleted = vector_store.delete_document(document_id)
    client = await redis_manager.get_async_client()
    hashes_key = RedisKeys.ingested_hashes_key(prefix=settings.REDIS_KEY_PREFIX)
    await client.hdel(hashes_key, document_id)

    logger.info(f"Deleted document '{document_id}': {chunks_deleted} chunks removed.")
    return {
        "document_id": document_id,
        "chunks_deleted": chunks_deleted,
        "status": "deleted"
    }


@router.get("/{job_id}", response_model=IngestionJob, include_in_schema=False)
async def get_job_status_legacy(
    job_id: str,
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    """Legacy alias for backward compatibility."""
    return await get_job_status(job_id=job_id, api_key=api_key)
