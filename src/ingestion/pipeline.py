"""
AlphaLens - Document Ingestion Pipeline
Orchestrates document parsing, hierarchical chunking, batched embedding, deduplication,
and shared index synchronization across distributed workers.
"""

import asyncio
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import uuid

from src.core.models import Document, IngestionJob, JobStatus
from src.core.exceptions import IngestionError
from src.config import settings
from src.ingestion.parsers import FinancialDocumentParser
from src.ingestion.chunker import HierarchicalFinancialChunker
from src.ingestion.embedder import embedding_service
from src.database.vector_store import vector_store
from src.retrieval.sparse import bm25_retriever
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.workers.job_registry import job_registry
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class IngestionPipeline:
    """Manages idempotent document ingestion, deduplication, and vector indexing."""

    def __init__(self):
        # In-memory document hash registry mirror for offline/mock fallback
        self._ingested_doc_hashes: Dict[str, str] = {}
        self._jobs: Dict[str, IngestionJob] = {}

    async def execute_job(
        self,
        job_id: str,
        raw_payload: str,
        worker_id: str = "worker-1"
    ) -> IngestionJob:
        """
        Executes full document ingestion lifecycle asynchronously from a worker process.
        Updates job progress in Redis, preserves idempotency, and commits to shared indexes.
        """
        start_time = time.perf_counter()
        client = await redis_manager.get_async_client()

        # Ensure job is registered in Redis if invoked directly
        existing_job = await job_registry.get_job(job_id)
        if not existing_job:
            try:
                parsed_peek = FinancialDocumentParser.parse_json_document(raw_payload)
                peek_doc_id = parsed_peek.document_id
                peek_ticker = parsed_peek.ticker
            except Exception:
                peek_doc_id = "unknown"
                peek_ticker = "UNKNOWN"
            await job_registry.register_job(
                job_id=job_id,
                document_id=peek_doc_id,
                ticker=peek_ticker,
                raw_payload=raw_payload
            )

        # Step 1: Parsing
        await job_registry.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            stage="PARSING",
            progress=15,
            worker_id=worker_id,
            message="Parsing financial document structure"
        )
        try:
            doc = FinancialDocumentParser.parse_json_document(raw_payload)
        except Exception as parse_err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            await job_registry.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage="PARSING_FAILED",
                error=f"Document parsing error: {str(parse_err)}",
                processing_time_ms=round(duration_ms, 2)
            )
            metrics_collector.record_job_failed()
            raise IngestionError(f"Parsing failed for job {job_id}: {parse_err}")

        # Step 2: Distributed Idempotency Deduplication Check
        hashes_key = RedisKeys.ingested_hashes_key(prefix=settings.REDIS_KEY_PREFIX)
        cached_hash = await client.hget(hashes_key, doc.document_id)
        if cached_hash:
            cached_hash_str = cached_hash.decode("utf-8") if isinstance(cached_hash, bytes) else str(cached_hash)
            if cached_hash_str == doc.content_hash:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                logger.info(f"Idempotency hit in Redis: Document '{doc.document_id}' unchanged. Skipping re-indexing.")
                job = await job_registry.update_job(
                    job_id,
                    status=JobStatus.SKIPPED_EXISTING,
                    stage="IDEMPOTENT_SKIPPED",
                    progress=100,
                    processing_time_ms=round(duration_ms, 2),
                    message="Document already indexed (idempotency hit; content unchanged)"
                )
                if not job:
                    job = IngestionJob(
                        job_id=job_id,
                        document_id=doc.document_id,
                        ticker=doc.ticker,
                        status=JobStatus.SKIPPED_EXISTING,
                        stage="IDEMPOTENT_SKIPPED",
                        progress=100,
                        processing_time_ms=round(duration_ms, 2),
                        message="Document already indexed (idempotency hit; content unchanged)"
                    )
                self._jobs[job_id] = job
                metrics_collector.record_job_completed()
                return job

        # Step 3: Hierarchical Chunking
        await job_registry.update_job(
            job_id,
            stage="CHUNKING",
            progress=35,
            message=f"Generating hierarchical parent-child chunks for {doc.document_id}"
        )
        try:
            parent_chunks, child_chunks = HierarchicalFinancialChunker.chunk_document(doc)
            all_chunks = parent_chunks + child_chunks
            child_texts = [c.content for c in child_chunks]
        except Exception as chunk_err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            await job_registry.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage="CHUNKING_FAILED",
                error=f"Chunking error: {str(chunk_err)}",
                processing_time_ms=round(duration_ms, 2)
            )
            metrics_collector.record_job_failed()
            raise IngestionError(f"Chunking failed for job {job_id}: {chunk_err}")

        # Step 4: Batched Dense Embedding Generation
        await job_registry.update_job(
            job_id,
            stage="EMBEDDING",
            progress=60,
            message=f"Generating dense embeddings for {len(child_chunks)} child chunks"
        )
        try:
            embeddings = embedding_service.embed_batch(child_texts)
        except Exception as emb_err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            await job_registry.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage="EMBEDDING_FAILED",
                error=f"Embedding error: {str(emb_err)}",
                processing_time_ms=round(duration_ms, 2)
            )
            metrics_collector.record_job_failed()
            raise IngestionError(f"Embedding failed for job {job_id}: {emb_err}")

        # Step 5: Shared Vector & Lexical Index Commits
        await job_registry.update_job(
            job_id,
            stage="INDEXING",
            progress=85,
            message="Committing chunks to vector database and shared BM25 index"
        )
        try:
            # 5a. Store in vector database
            vector_store.add_chunks(all_chunks, embeddings)

            # 5b. Update shared BM25 index on disk and memory
            bm25_retriever.index_chunks(child_chunks)

            # 5c. Record content hash in Redis
            await client.hset(hashes_key, doc.document_id, doc.content_hash)
            self._ingested_doc_hashes[doc.document_id] = doc.content_hash

            # 5d. Increment global index version & publish notification
            version_key = RedisKeys.index_version_key(prefix=settings.REDIS_KEY_PREFIX)
            new_version = await client.incr(version_key)
            channel = RedisKeys.index_updates_channel(prefix=settings.REDIS_KEY_PREFIX)
            await client.publish(channel, str(new_version))

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            job = await job_registry.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                stage="COMPLETED",
                progress=100,
                total_parent_chunks=len(parent_chunks),
                total_child_chunks=len(child_chunks),
                processing_time_ms=round(duration_ms, 2),
                message="Successfully parsed, embedded, and indexed document"
            )
            if not job:
                job = IngestionJob(
                    job_id=job_id,
                    document_id=doc.document_id,
                    ticker=doc.ticker,
                    status=JobStatus.COMPLETED,
                    stage="COMPLETED",
                    progress=100,
                    total_parent_chunks=len(parent_chunks),
                    total_child_chunks=len(child_chunks),
                    processing_time_ms=round(duration_ms, 2),
                    message="Successfully parsed, embedded, and indexed document"
                )
            self._jobs[job_id] = job
            metrics_collector.record_job_completed()
            logger.info(
                f"Successfully ingested {doc.document_id} (job: {job_id}): "
                f"{len(parent_chunks)} parents, {len(child_chunks)} children in {duration_ms:.2f}ms. "
                f"Index version: v{new_version}."
            )
            return job

        except Exception as idx_err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            await job_registry.update_job(
                job_id,
                status=JobStatus.FAILED,
                stage="INDEXING_FAILED",
                error=f"Indexing error: {str(idx_err)}",
                processing_time_ms=round(duration_ms, 2)
            )
            metrics_collector.record_job_failed()
            raise IngestionError(f"Indexing failed for job {job_id}: {idx_err}")

    def ingest_document_payload(self, json_payload: str) -> IngestionJob:
        """Synchronous ingestion helper for tests and offline scripts."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        doc = FinancialDocumentParser.parse_json_document(json_payload)

        # Check in-memory mirror
        if doc.document_id in self._ingested_doc_hashes and self._ingested_doc_hashes[doc.document_id] == doc.content_hash:
            job = IngestionJob(
                job_id=job_id,
                document_id=doc.document_id,
                ticker=doc.ticker,
                status=JobStatus.SKIPPED_EXISTING,
                stage="IDEMPOTENT_SKIPPED",
                progress=100,
                message="Document unchanged. Skipping re-indexing."
            )
            self._jobs[job_id] = job
            return job

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(self.execute_job(job_id, json_payload))).result()
            return loop.run_until_complete(self.execute_job(job_id, json_payload))
        except RuntimeError:
            return asyncio.run(self.execute_job(job_id, json_payload))

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Fetches job status from in-memory mirror or Redis."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(lambda: asyncio.run(job_registry.get_job(job_id))).result()
            return loop.run_until_complete(job_registry.get_job(job_id))
        except RuntimeError:
            return asyncio.run(job_registry.get_job(job_id))

    def ingest_directory(self, dir_path: Path) -> List[IngestionJob]:
        """Ingests all JSON raw filing files from a specified directory."""
        jobs = []
        if not dir_path.exists():
            return jobs
        for json_path in dir_path.glob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    content = f.read()
                job = self.ingest_document_payload(content)
                jobs.append(job)
            except Exception as e:
                logger.error(f"Failed to ingest file {json_path.name}: {e}")
        return jobs


# Global singleton ingestion pipeline instance
ingestion_pipeline = IngestionPipeline()
