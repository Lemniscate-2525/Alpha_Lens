"""
AlphaLens - Document Ingestion Pipeline
Orchestrates document parsing, hierarchical chunking, batched embedding, and deduplication.
"""
import time
import uuid
from typing import Dict, List, Optional
from src.core.models import Document, IngestionJob, JobStatus
from src.core.exceptions import IngestionError
from src.ingestion.parsers import FinancialDocumentParser
from src.ingestion.chunker import HierarchicalFinancialChunker
from src.ingestion.embedder import embedding_service
from src.database.vector_store import vector_store
from src.observability.logging import logger


class IngestionPipeline:
    """Manages idempotent document ingestion, deduplication, and vector indexing."""

    def __init__(self):
        # In-memory document hash registry for deduplication (in production backed by DB)
        self._ingested_doc_hashes: Dict[str, str] = {}  # doc_id -> content_hash
        self._jobs: Dict[str, IngestionJob] = {}

    def ingest_document_payload(self, json_payload: str) -> IngestionJob:
        """Parses, deduplicates, chunks, embeds, and indexes a financial document."""
        start_time = time.perf_counter()
        doc = FinancialDocumentParser.parse_json_document(json_payload)

        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = IngestionJob(
            job_id=job_id,
            document_id=doc.document_id,
            ticker=doc.ticker,
            status=JobStatus.PROCESSING
        )
        self._jobs[job_id] = job

        # Deduplication check: Has this exact content hash already been indexed?
        if doc.document_id in self._ingested_doc_hashes:
            if self._ingested_doc_hashes[doc.document_id] == doc.content_hash:
                logger.info(f"Idempotency hit: Document '{doc.document_id}' unchanged. Skipping re-indexing.")
                job.status = JobStatus.SKIPPED_EXISTING
                job.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
                return job

        try:
            logger.info(f"Chunking document: {doc.document_id}")
            parent_chunks, child_chunks = HierarchicalFinancialChunker.chunk_document(doc)

            all_chunks = parent_chunks + child_chunks
            child_texts = [c.content for c in child_chunks]

            logger.info(f"Generating dense embeddings for {len(child_chunks)} child chunks...")
            embeddings = embedding_service.embed_batch(child_texts)

            # Store in vector database
            vector_store.add_chunks(all_chunks, embeddings)

            # Update BM25 index
            from src.retrieval.sparse import bm25_retriever
            bm25_retriever.index_chunks(child_chunks)

            # Record hash
            self._ingested_doc_hashes[doc.document_id] = doc.content_hash

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            job.status = JobStatus.COMPLETED
            job.total_parent_chunks = len(parent_chunks)
            job.total_child_chunks = len(child_chunks)
            job.processing_time_ms = round(duration_ms, 2)

            logger.info(
                f"Successfully ingested {doc.document_id}: "
                f"{len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks in {duration_ms:.2f}ms"
            )
            return job

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.processing_time_ms = round(duration_ms, 2)
            logger.error(f"Failed to ingest document {doc.document_id}: {e}")
            raise IngestionError(f"Ingestion failed for {doc.document_id}: {str(e)}")

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        return self._jobs.get(job_id)


# Global singleton ingestion pipeline
ingestion_pipeline = IngestionPipeline()

