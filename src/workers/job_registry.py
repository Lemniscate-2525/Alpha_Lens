"""
AlphaLens - Redis-Backed Distributed Job Status Registry
Tracks ingestion lifecycle, progress, and error details across independent workers and API nodes.
"""

from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from src.config import settings
from src.core.models import IngestionJob, JobStatus
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class DistributedJobRegistry:
    """Manages job state persistence in Redis."""

    def __init__(self, redis_client=None):
        self._custom_client = redis_client
        self.prefix = settings.REDIS_KEY_PREFIX

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    async def register_job(
        self,
        job_id: str,
        document_id: str,
        ticker: str,
        raw_payload: Optional[str] = None
    ) -> IngestionJob:
        """Registers a newly submitted ingestion job in Redis."""
        client = await self._get_client()
        job = IngestionJob(
            job_id=job_id,
            document_id=document_id,
            ticker=ticker,
            status=JobStatus.QUEUED,
            stage="QUEUED",
            progress=0,
            message="Document accepted for asynchronous ingestion"
        )

        job_key = RedisKeys.job_key(job_id, prefix=self.prefix)
        index_key = RedisKeys.jobs_index_key(prefix=self.prefix)
        now_ts = time.time()

        job_data = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "ticker": job.ticker,
            "status": job.status.value,
            "stage": job.stage,
            "progress": str(job.progress),
            "total_parent_chunks": str(job.total_parent_chunks),
            "total_child_chunks": str(job.total_child_chunks),
            "processing_time_ms": str(job.processing_time_ms),
            "retry_count": str(job.retry_count),
            "error": job.error or "",
            "worker_id": job.worker_id or "",
            "message": job.message or "",
            "created_at": job.created_at.isoformat(),
            "started_at": "",
            "completed_at": ""
        }

        # Store job hash
        await client.hset(job_key, mapping=job_data)
        # Store raw payload temporarily if provided (1 day TTL)
        if raw_payload:
            payload_key = f"{job_key}:payload"
            await client.set(payload_key, raw_payload, ex=86400)

        # Add to global jobs sorted set ordered by creation timestamp
        await client.zadd(index_key, {job_id: now_ts})
        metrics_collector.record_job_queued()
        logger.info(f"Registered job '{job_id}' for doc '{document_id}' in Redis.")
        return job

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        total_parent_chunks: Optional[int] = None,
        total_child_chunks: Optional[int] = None,
        processing_time_ms: Optional[float] = None,
        worker_id: Optional[str] = None,
        retry_count: Optional[int] = None,
        message: Optional[str] = None
    ) -> Optional[IngestionJob]:
        """Atomically updates job fields in Redis."""
        client = await self._get_client()
        job_key = RedisKeys.job_key(job_id, prefix=self.prefix)
        exists = await client.exists(job_key)
        if not exists:
            return None

        updates: Dict[str, str] = {}
        if status is not None:
            updates["status"] = status.value
            if status == JobStatus.PROCESSING and not await client.hget(job_key, "started_at"):
                updates["started_at"] = datetime.utcnow().isoformat()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.DEAD_LETTER, JobStatus.SKIPPED_EXISTING):
                updates["completed_at"] = datetime.utcnow().isoformat()

        if stage is not None:
            updates["stage"] = stage
        if progress is not None:
            updates["progress"] = str(progress)
        if error is not None:
            updates["error"] = error
        if total_parent_chunks is not None:
            updates["total_parent_chunks"] = str(total_parent_chunks)
        if total_child_chunks is not None:
            updates["total_child_chunks"] = str(total_child_chunks)
        if processing_time_ms is not None:
            updates["processing_time_ms"] = str(processing_time_ms)
        if worker_id is not None:
            updates["worker_id"] = worker_id
        if retry_count is not None:
            updates["retry_count"] = str(retry_count)
        if message is not None:
            updates["message"] = message

        if updates:
            await client.hset(job_key, mapping=updates)

        return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Fetches job metadata from Redis."""
        client = await self._get_client()
        job_key = RedisKeys.job_key(job_id, prefix=self.prefix)
        raw = await client.hgetall(job_key)
        if not raw:
            return None

        def _val(k):
            v = raw.get(k) or raw.get(k.encode("utf-8"))
            if v is None:
                return None
            return v.decode("utf-8") if isinstance(v, bytes) else str(v)

        def _dt(v_str):
            if not v_str:
                return None
            try:
                return datetime.fromisoformat(v_str)
            except Exception:
                return None

        status_val = _val("status") or "pending"
        try:
            status_enum = JobStatus(status_val)
        except ValueError:
            status_enum = JobStatus.PENDING

        return IngestionJob(
            job_id=_val("job_id") or job_id,
            document_id=_val("document_id") or "",
            ticker=_val("ticker") or "",
            status=status_enum,
            stage=_val("stage") or "PENDING",
            progress=int(_val("progress") or 0),
            total_parent_chunks=int(_val("total_parent_chunks") or 0),
            total_child_chunks=int(_val("total_child_chunks") or 0),
            processing_time_ms=float(_val("processing_time_ms") or 0.0),
            retry_count=int(_val("retry_count") or 0),
            error=_val("error") or None,
            worker_id=_val("worker_id") or None,
            message=_val("message") or None,
            created_at=_dt(_val("created_at")) or datetime.utcnow(),
            started_at=_dt(_val("started_at")),
            completed_at=_dt(_val("completed_at"))
        )

    async def get_job_payload(self, job_id: str) -> Optional[str]:
        """Retrieves raw JSON payload for a job."""
        client = await self._get_client()
        payload_key = f"{RedisKeys.job_key(job_id, prefix=self.prefix)}:payload"
        data = await client.get(payload_key)
        if not data:
            return None
        return data.decode("utf-8") if isinstance(data, bytes) else data

    async def list_jobs(self, offset: int = 0, limit: int = 50) -> Tuple[List[IngestionJob], int]:
        """Lists recent jobs ordered by creation timestamp."""
        client = await self._get_client()
        index_key = RedisKeys.jobs_index_key(prefix=self.prefix)
        total = await client.zcard(index_key)
        # Reverse range (newest first)
        job_ids = await client.zrevrange(index_key, offset, offset + limit - 1)
        jobs = []
        for jid in job_ids:
            j_str = jid.decode("utf-8") if isinstance(jid, bytes) else str(jid)
            job = await self.get_job(j_str)
            if job:
                jobs.append(job)
        return jobs, total


job_registry = DistributedJobRegistry()
