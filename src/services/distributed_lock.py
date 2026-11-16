"""
AlphaLens - Distributed Redis Lock
Provides atomic lock acquisition (SET NX PX), atomic Lua release,
and automatic heartbeat renewal to protect long-running RAG operations.
"""

import asyncio
import uuid
from typing import Optional
from src.config import settings
from src.infrastructure.redis_client import redis_manager
from src.observability.logging import logger

LUA_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

LUA_RENEW_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class DistributedLock:
    """
    Redis-based distributed lock with ownership token, safe atomic release,
    and background heartbeat renewal.
    """

    def __init__(
        self,
        lock_key: str,
        lease_ms: Optional[int] = None,
        redis_client = None
    ):
        self.lock_key = lock_key
        self.lease_ms = lease_ms or getattr(settings, "SINGLEFLIGHT_LEASE_MS", 30000)
        self.owner_token = f"owner-{uuid.uuid4().hex}"
        self._custom_client = redis_client
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._is_locked: bool = False

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    async def acquire(self) -> bool:
        """Attempts to acquire lock with SET NX PX."""
        client = await self._get_client()
        result = await client.set(
            self.lock_key,
            self.owner_token,
            nx=True,
            px=self.lease_ms
        )
        self._is_locked = bool(result)
        return self._is_locked

    async def release(self) -> bool:
        """Atomically releases lock using Lua script verifying owner token."""
        self.stop_heartbeat()
        if not self._is_locked:
            return False

        try:
            client = await self._get_client()
            res = await client.eval(LUA_RELEASE_LOCK, 1, self.lock_key, self.owner_token)
            self._is_locked = False
            return bool(res == 1)
        except Exception as e:
            logger.error(f"Error releasing lock for key {self.lock_key}: {e}")
            self._is_locked = False
            return False

    async def renew(self) -> bool:
        """Renews lock lease by another lease_ms if still owned by this token."""
        try:
            client = await self._get_client()
            res = await client.eval(LUA_RENEW_LOCK, 1, self.lock_key, self.lease_ms)
            return bool(res == 1)
        except Exception as e:
            logger.warning(f"Error renewing lock {self.lock_key}: {e}")
            return False

    def start_heartbeat(self, interval_seconds: Optional[float] = None):
        """Starts background task to periodically renew lock lease."""
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        interval = interval_seconds or max(1.0, (self.lease_ms / 1000.0) / 3.0)

        async def _heartbeat_loop():
            while self._is_locked:
                try:
                    await asyncio.sleep(interval)
                    if not self._is_locked:
                        break
                    renewed = await self.renew()
                    if not renewed:
                        logger.warning(f"Heartbeat failed to renew lock {self.lock_key}; lock lost.")
                        self._is_locked = False
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Heartbeat exception for {self.lock_key}: {e}")
                    break

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())

    def stop_heartbeat(self):
        """Cancels background heartbeat task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def __aenter__(self):
        acquired = await self.acquire()
        if not acquired:
            raise RuntimeError(f"Could not acquire distributed lock for {self.lock_key}")
        self.start_heartbeat()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
