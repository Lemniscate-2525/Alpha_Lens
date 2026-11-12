"""
AlphaLens - Document Ingestion Endpoints
Receives raw financial document payloads and initiates asynchronous chunking and indexing.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from src.core.models import IngestionJob
from src.ingestion.pipeline import ingestion_pipeline
from src.api.dependencies import verify_api_key

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestPayloadRequest(BaseModel):
    json_document: str


@router.post("", response_model=IngestionJob, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    request: IngestPayloadRequest,
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    job = ingestion_pipeline.ingest_document_payload(request.json_document)
    return job


@router.get("/{job_id}", response_model=IngestionJob)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
) -> IngestionJob:
    job = ingestion_pipeline.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return job
