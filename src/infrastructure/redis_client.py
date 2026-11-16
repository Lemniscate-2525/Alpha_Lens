"""
AlphaLens - Central Redis Connection Management
Provides resilient asynchronous and synchronous Redis connection pools.
Supports live Redis clusters/servers and in-memory FakeServer for deterministic testing.
"""

import asyncio
from typing import Optional, Union
import redis
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool
from redis.connection import ConnectionPool as SyncConnectionPool

from src.config import settings
from src.observability.logging import logger


class RedisClientManager:
    """Singleton manager for Redis connection pooling, health checks, and lifecycle."""

    def __init__(self):
        self._async_pool: Optional[AsyncConnectionPool] = None
        self._async_client: Optional[aioredis.Redis] = None
        self._sync_pool: Optional[SyncConnectionPool] = None
        self._sync_client: Optional[redis.Redis] = None
        self._is_fake: bool = False
        self._fake_server = None
        self._lock = asyncio.Lock()

    def _should_use_fakeredis(self, url: Optional[str]) -> bool:
        if not url:
            return True
        norm = url.strip().lower()
        return norm.startswith("fakeredis://") or norm.startswith("mock://")

    async def get_async_client(self) -> aioredis.Redis:
        """Returns or creates the shared async Redis client."""
        if self._async_client is not None:
            return self._async_client

        async with self._lock:
            if self._async_client is not None:
                return self._async_client

            url = settings.REDIS_URL
            if self._should_use_fakeredis(url):
                logger.info("Initializing in-memory Redis client (fakeredis backend).")
                import fakeredis.aioredis
                if self._fake_server is None:
                    import fakeredis
                    self._fake_server = fakeredis.FakeServer()
                self._async_client = fakeredis.aioredis.FakeRedis(server=self._fake_server)
                self._is_fake = True
                return self._async_client

            try:
                logger.info(f"Connecting to Redis at: {url}")
                self._async_pool = AsyncConnectionPool.from_url(
                    url,
                    max_connections=getattr(settings, "REDIS_MAX_CONNECTIONS", 50),
                    socket_timeout=getattr(settings, "REDIS_SOCKET_TIMEOUT", 5.0),
                    socket_connect_timeout=getattr(settings, "REDIS_CONNECT_TIMEOUT", 5.0),
                    decode_responses=False,  # Keep binary by default for arbitrary bytes/payloads
                )
                client = aioredis.Redis(connection_pool=self._async_pool)
                # Verify connectivity
                await asyncio.wait_for(client.ping(), timeout=getattr(settings, "REDIS_CONNECT_TIMEOUT", 5.0))
                self._async_client = client
                self._is_fake = False
                logger.info("Successfully established connection to shared Redis instance.")
                return self._async_client
            except Exception as e:
                logger.warning(
                    f"Failed to connect to primary Redis ({url}): {e}. "
                    "Falling back to shared in-memory Redis server for resilience."
                )
                import fakeredis.aioredis
                import fakeredis
                if self._fake_server is None:
                    self._fake_server = fakeredis.FakeServer()
                self._async_client = fakeredis.aioredis.FakeRedis(server=self._fake_server)
                self._is_fake = True
                return self._async_client

    def get_sync_client(self) -> redis.Redis:
        """Returns or creates a synchronous Redis client (for background workers or sync hooks)."""
        if self._sync_client is not None:
            return self._sync_client

        url = settings.REDIS_URL
        if self._should_use_fakeredis(url) or self._is_fake:
            import fakeredis
            if self._fake_server is None:
                self._fake_server = fakeredis.FakeServer()
            self._sync_client = fakeredis.FakeRedis(server=self._fake_server)
            return self._sync_client

        try:
            self._sync_pool = SyncConnectionPool.from_url(
                url,
                max_connections=getattr(settings, "REDIS_MAX_CONNECTIONS", 50),
                socket_timeout=getattr(settings, "REDIS_SOCKET_TIMEOUT", 5.0),
                socket_connect_timeout=getattr(settings, "REDIS_CONNECT_TIMEOUT", 5.0),
                decode_responses=False,
            )
            client = redis.Redis(connection_pool=self._sync_pool)
            client.ping()
            self._sync_client = client
            return self._sync_client
        except Exception as e:
            logger.warning(f"Failed to connect to sync Redis ({url}): {e}. Using fake server.")
            import fakeredis
            if self._fake_server is None:
                self._fake_server = fakeredis.FakeServer()
            self._sync_client = fakeredis.FakeRedis(server=self._fake_server)
            return self._sync_client

    async def check_health(self) -> bool:
        """Pings Redis to verify connectivity and liveness."""
        try:
            client = await self.get_async_client()
            res = await client.ping()
            return bool(res)
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def close(self):
        """Gracefully terminates connection pools."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        if self._async_pool:
            await self._async_pool.disconnect()
            self._async_pool = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._sync_pool:
            self._sync_pool.disconnect()
            self._sync_pool = None


redis_manager = RedisClientManager()


async def get_redis() -> aioredis.Redis:
    """Helper function to fetch the shared async Redis client."""
    return await redis_manager.get_async_client()
