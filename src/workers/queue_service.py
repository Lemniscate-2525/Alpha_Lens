"""
AlphaLens - Redis Streams Distributed Job Queue Service
Provides reliable task enqueueing, consumer group management, acknowledgments,
and dead-letter queue routing for background document ingestion.
"""

import asyncio
from typing import Any, Dict, Optional, Tuple
from src.config import settings
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.observability.logging import logger


class IngestionQueueService:
    """Manages Redis Streams for ingestion task queueing and delivery."""

    def __init__(self, redis_client=None):
        self._custom_client = redis_client
        self.stream_name = getattr(settings, "INGESTION_QUEUE_NAME", "alpha_lens:queue:ingestion")
        self.dead_letter_name = getattr(settings, "INGESTION_DEAD_LETTER_QUEUE", "alpha_lens:queue:ingestion:dead_letter")
        self.consumer_group = getattr(settings, "INGESTION_CONSUMER_GROUP", "alpha_lens_workers")
        self._group_initialized = False

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    async def ensure_consumer_group(self):
        """Ensures the consumer group and stream exist in Redis."""
        if self._group_initialized:
            return
        client = await self._get_client()
        try:
            await client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0",
                mkstream=True
            )
            self._group_initialized = True
            logger.info(f"Created Redis stream consumer group '{self.consumer_group}' on '{self.stream_name}'.")
        except Exception as e:
            # Group already exists (BUSYGROUP)
            self._group_initialized = True
            logger.debug(f"Consumer group '{self.consumer_group}' already exists or ready: {e}")

    async def enqueue(self, job_id: str, document_id: str, ticker: str) -> str:
        """Enqueues an ingestion task onto the Redis stream."""
        client = await self._get_client()
        await self.ensure_consumer_group()
        entry = {
            "job_id": job_id,
            "document_id": document_id,
            "ticker": ticker
        }
        msg_id = await client.xadd(self.stream_name, entry)
        msg_id_str = msg_id.decode("utf-8") if isinstance(msg_id, bytes) else str(msg_id)
        logger.info(f"Enqueued job '{job_id}' (msg_id: {msg_id_str}) into Redis stream '{self.stream_name}'.")
        return msg_id_str

    async def dequeue(
        self,
        worker_id: str,
        block_ms: int = 2000
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Dequeues the next pending task for this consumer group.
        Returns: (message_id, data_dict) or None if timeout.
        """
        client = await self._get_client()
        await self.ensure_consumer_group()

        try:
            entries = await client.xreadgroup(
                groupname=self.consumer_group,
                consumername=worker_id,
                streams={self.stream_name: ">"},
                count=1,
                block=block_ms
            )
            if not entries:
                return None

            # Stream result format: [[stream_name, [(msg_id, fields_dict)]]]
            stream_result = entries[0]
            messages = stream_result[1]
            if not messages:
                return None

            raw_msg_id, raw_fields = messages[0]
            msg_id = raw_msg_id.decode("utf-8") if isinstance(raw_msg_id, bytes) else str(raw_msg_id)

            parsed_fields: Dict[str, str] = {}
            for k, v in raw_fields.items():
                k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                parsed_fields[k_str] = v_str

            return msg_id, parsed_fields
        except Exception as e:
            logger.warning(f"Error reading from Redis queue '{self.stream_name}': {e}")
            return None

    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledges successful processing of a message."""
        client = await self._get_client()
        try:
            res = await client.xack(self.stream_name, self.consumer_group, message_id)
            return bool(res > 0)
        except Exception as e:
            logger.error(f"Error acknowledging message '{message_id}': {e}")
            return False

    async def move_to_dead_letter(self, job_id: str, message_id: str, error_detail: str):
        """Moves permanently failed task to dead-letter queue and ACKs from main stream."""
        client = await self._get_client()
        try:
            entry = {
                "job_id": job_id,
                "original_message_id": message_id,
                "error": error_detail
            }
            await client.xadd(self.dead_letter_name, entry)
            await client.xack(self.stream_name, self.consumer_group, message_id)
            logger.error(f"Moved job '{job_id}' to dead-letter queue '{self.dead_letter_name}'.")
        except Exception as e:
            logger.error(f"Failed to route job '{job_id}' to dead-letter queue: {e}")

    async def get_queue_depth(self) -> int:
        """Returns total messages in the stream."""
        client = await self._get_client()
        try:
            return await client.xlen(self.stream_name)
        except Exception:
            return 0


ingestion_queue = IngestionQueueService()
