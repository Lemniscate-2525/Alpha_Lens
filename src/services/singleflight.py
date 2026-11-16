"""
AlphaLens - Distributed Singleflight Mutex
Coordinates duplicate concurrent queries across multiple independent FastAPI instances.
Ensures that only ONE request acts as Leader and performs retrieval/generation,
while all other concurrent requests wait and share the cached result.
"""

import asyncio
import time
from typing import Awaitable, Callable, Optional, Tuple
from src.config import settings
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.services.distributed_lock import DistributedLock
from src.core.models import QueryResponse
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class DistributedSingleflight:
    """Distributed request coalescing mechanism backed by Redis."""

    def __init__(self, redis_client=None):
        self._custom_client = redis_client
        self.lease_ms = getattr(settings, "SINGLEFLIGHT_LEASE_MS", 30000)
        self.wait_timeout = getattr(settings, "SINGLEFLIGHT_WAIT_TIMEOUT_SECONDS", 35.0)
        self.poll_interval_s = getattr(settings, "SINGLEFLIGHT_POLL_INTERVAL_MS", 50) / 1000.0

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    async def execute_or_wait(
        self,
        exact_key: str,
        leader_fn: Callable[[], Awaitable[QueryResponse]],
        cache_getter: Callable[[], Awaitable[Optional[QueryResponse]]],
    ) -> Tuple[QueryResponse, bool]:
        """
        Executes leader_fn if this instance acquires the lock.
        Otherwise waits for the leader to complete and returns the cached response.
        Returns: (QueryResponse, is_leader: bool)
        """
        lock_key = RedisKeys.singleflight_lock_key(exact_key)
        channel = RedisKeys.singleflight_channel(exact_key)
        client = await self._get_client()

        # Step 1: Pre-check cache
        cached = await cache_getter()
        if cached is not None:
            return cached, False

        lock = DistributedLock(lock_key, lease_ms=self.lease_ms, redis_client=client)
        acquired = await lock.acquire()

        if acquired:
            # === WE ARE THE LEADER ===
            metrics_collector.record_singleflight_leader()
            logger.info(f"Singleflight LEADER elected for key: {exact_key[:12]}")
            lock.start_heartbeat()
            try:
                response = await leader_fn()
                # Publish completion signal to waiters
                try:
                    await client.publish(channel, "COMPLETED")
                except Exception as pub_err:
                    logger.debug(f"PubSub notification failed (non-critical): {pub_err}")
                return response, True
            finally:
                lock.stop_heartbeat()
                await lock.release()

        # === WE ARE A WAITER ===
        metrics_collector.record_singleflight_waiter()
        logger.info(f"Singleflight WAITER queued for key: {exact_key[:12]}, awaiting leader result...")
        start_wait = time.perf_counter()

        # Set up PubSub listener if available
        pubsub = client.pubsub()
        pubsub_subscribed = False
        try:
            await pubsub.subscribe(channel)
            pubsub_subscribed = True
        except Exception as e:
            logger.debug(f"PubSub subscribe fallback to polling: {e}")

        try:
            while True:
                # Check timeout
                elapsed = time.perf_counter() - start_wait
                if elapsed > self.wait_timeout:
                    logger.warning(
                        f"Singleflight waiter timed out after {elapsed:.1f}s for key: {exact_key[:12]}. "
                        "Breaking wait loop."
                    )
                    break

                # 1. Check if response is now available in shared cache
                cached = await cache_getter()
                if cached is not None:
                    wait_ms = (time.perf_counter() - start_wait) * 1000.0
                    metrics_collector.record_duplicate_generation_prevented(wait_duration_ms=wait_ms)
                    logger.info(
                        f"Singleflight waiter received cached response in {wait_ms:.1f}ms for key: {exact_key[:12]}"
                    )
                    return cached, False

                # 2. Check if leader crashed or lock expired
                lock_exists = await client.exists(lock_key)
                if not lock_exists:
                    # Lock is released or expired! Try acquiring to become new leader
                    re_lock = DistributedLock(lock_key, lease_ms=self.lease_ms, redis_client=client)
                    if await re_lock.acquire():
                        logger.info(f"Singleflight waiter promoted to LEADER for key: {exact_key[:12]}")
                        metrics_collector.record_singleflight_leader()
                        re_lock.start_heartbeat()
                        try:
                            # Final cache check
                            cached = await cache_getter()
                            if cached is not None:
                                return cached, False
                            response = await leader_fn()
                            try:
                                await client.publish(channel, "COMPLETED")
                            except Exception:
                                pass
                            return response, True
                        finally:
                            re_lock.stop_heartbeat()
                            await re_lock.release()

                # 3. Wait via pubsub message or bounded sleep
                if pubsub_subscribed:
                    try:
                        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=self.poll_interval_s)
                        if msg and msg.get("data") in (b"COMPLETED", "COMPLETED"):
                            cached = await cache_getter()
                            if cached is not None:
                                wait_ms = (time.perf_counter() - start_wait) * 1000.0
                                metrics_collector.record_duplicate_generation_prevented(wait_duration_ms=wait_ms)
                                return cached, False
                    except Exception:
                        await asyncio.sleep(self.poll_interval_s)
                else:
                    await asyncio.sleep(self.poll_interval_s)

        finally:
            if pubsub_subscribed:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.aclose()
                except Exception:
                    pass

        # If we exited loop without finding cached response, execute as fallback
        logger.warning(f"Singleflight fallback execution for key: {exact_key[:12]}")
        response = await leader_fn()
        return response, True


singleflight = DistributedSingleflight()
