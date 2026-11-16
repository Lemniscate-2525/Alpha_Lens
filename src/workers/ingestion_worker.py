"""
AlphaLens - Independent Ingestion Worker Process
Dedicated background worker that consumes tasks from the Redis stream queue,
executes parsing/chunking/embedding/indexing, and acknowledges committed jobs.
Run as: python -m src.workers.ingestion_worker
"""

import asyncio
import os
import signal
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.core.models import JobStatus
from src.ingestion.pipeline import ingestion_pipeline
from src.workers.job_registry import job_registry
from src.workers.queue_service import ingestion_queue
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class IngestionWorker:
    """Dedicated background consumer for document ingestion tasks."""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.max_retries = getattr(settings, "INGESTION_MAX_RETRIES", 3)
        self._running = False

    async def start(self):
        """Starts worker event loop consuming from Redis queue."""
        self._running = True
        logger.info(f"Ingestion Worker '{self.worker_id}' started. Listening on Redis stream...")

        while self._running:
            try:
                # Dequeue next task
                task = await ingestion_queue.dequeue(worker_id=self.worker_id, block_ms=2000)
                if not task:
                    # Timeout / idle tick
                    continue

                msg_id, fields = task
                job_id = fields.get("job_id")
                document_id = fields.get("document_id")

                if not job_id:
                    logger.warning(f"Malformed message {msg_id} without job_id. Acknowledging and skipping.")
                    await ingestion_queue.acknowledge(msg_id)
                    continue

                logger.info(f"Worker '{self.worker_id}' claimed job '{job_id}' (doc: {document_id}).")
                await self._process_task(job_id, msg_id)

            except asyncio.CancelledError:
                logger.info(f"Worker '{self.worker_id}' received cancellation signal. Stopping.")
                break
            except Exception as e:
                logger.error(f"Worker '{self.worker_id}' encountered unexpected error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

        logger.info(f"Worker '{self.worker_id}' stopped gracefully.")

    async def _process_task(self, job_id: str, msg_id: str):
        """Processes a single ingestion job with retry logic and safe ACK."""
        job = await job_registry.get_job(job_id)
        if not job:
            logger.error(f"Job metadata not found in Redis for job '{job_id}'. Dropping.")
            await ingestion_queue.acknowledge(msg_id)
            return

        payload = await job_registry.get_job_payload(job_id)
        if not payload:
            err_msg = f"Raw document payload missing for job '{job_id}'."
            logger.error(err_msg)
            await job_registry.update_job(job_id, status=JobStatus.FAILED, error=err_msg)
            await ingestion_queue.move_to_dead_letter(job_id, msg_id, err_msg)
            return

        # Attempt execution
        retry_count = job.retry_count
        try:
            # Execute full pipeline
            await ingestion_pipeline.execute_job(
                job_id=job_id,
                raw_payload=payload,
                worker_id=self.worker_id
            )
            # Safe ACK: ONLY acknowledged after successful indexing and commit
            await ingestion_queue.acknowledge(msg_id)
            logger.info(f"Job '{job_id}' successfully completed and acknowledged.")

        except Exception as e:
            retry_count += 1
            logger.warning(f"Job '{job_id}' failed on attempt {retry_count}/{self.max_retries}: {e}")

            if retry_count <= self.max_retries:
                # Mark retrying
                await job_registry.update_job(
                    job_id,
                    status=JobStatus.RETRYING,
                    retry_count=retry_count,
                    error=f"Attempt {retry_count} failed: {str(e)}",
                    stage="RETRYING"
                )
                metrics_collector.record_job_retried()
                # Exponential backoff before re-queueing
                await asyncio.sleep(min(10.0, 1.5 ** retry_count))
                # Re-queue task to end of stream
                await ingestion_queue.enqueue(job_id, job.document_id, job.ticker)
                # Acknowledge the old failed message ID
                await ingestion_queue.acknowledge(msg_id)
            else:
                # Exceeded max retries -> route to Dead Letter Queue
                err_detail = f"Permanently failed after {retry_count} attempts. Last error: {str(e)}"
                await job_registry.update_job(
                    job_id,
                    status=JobStatus.DEAD_LETTER,
                    retry_count=retry_count,
                    error=err_detail,
                    stage="DEAD_LETTER"
                )
                metrics_collector.record_job_dead_letter()
                await ingestion_queue.move_to_dead_letter(job_id, msg_id, err_detail)

    def stop(self):
        """Signals worker to stop."""
        self._running = False


async def run_worker_pool(concurrency: int = 1):
    """Runs a pool of concurrent ingestion workers within this process."""
    workers = [IngestionWorker(worker_id=f"worker-{os.getpid()}-{i+1}") for i in range(concurrency)]

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received termination signal. Shutting down worker pool...")
        for w in workers:
            w.stop()
        stop_event.set()

    # Register OS signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, AttributeError):
            # Windows signal handler limitations
            pass

    tasks = [asyncio.create_task(w.start()) for w in workers]
    logger.info(f"Launched {len(workers)} worker coroutines.")
    await asyncio.gather(*tasks)


def main():
    concurrency = getattr(settings, "INGESTION_WORKER_CONCURRENCY", 2)
    logger.info(f"Starting AlphaLens Ingestion Worker Process (PID: {os.getpid()}, Concurrency: {concurrency})...")
    try:
        asyncio.run(run_worker_pool(concurrency=concurrency))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Ingestion Worker process terminated.")


if __name__ == "__main__":
    main()
