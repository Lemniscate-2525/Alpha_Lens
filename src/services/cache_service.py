"""
AlphaLens - Redis-Backed Two-Tier Distributed Caching Service
Tier 1: L1 Exact Hash Cache (SHA-256 key, Redis String with TTL)
Tier 2: L2 Semantic Similarity Cache (Embedding similarity >= 0.94, Partitioned Redis Hashes)
Features Redis distributed Singleflight request coalescing to prevent cache stampedes.
"""

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import settings
from src.core.models import QueryResponse
from src.infrastructure.redis_client import redis_manager
from src.infrastructure.redis_keys import RedisKeys
from src.ingestion.embedder import embedding_service
from src.services.singleflight import singleflight
from src.observability.logging import logger
from src.observability.metrics import metrics_collector


class TwoTierCacheService:
    """Distributed exact and semantic cache backed by Redis."""

    def __init__(self, redis_client=None):
        self._custom_client = redis_client
        self.singleflight = singleflight
        self.exact_ttl = getattr(settings, "EXACT_CACHE_TTL_SECONDS", 3600)
        self.semantic_ttl = getattr(settings, "SEMANTIC_CACHE_TTL_SECONDS", 3600)
        self.cosine_threshold = getattr(settings, "SEMANTIC_CACHE_THRESHOLD", 0.94)
        # In-memory dictionary mirror for backward compatibility in mock unit tests
        self.exact_cache: Dict[str, Tuple[QueryResponse, float]] = {}

    async def _get_client(self):
        if self._custom_client is not None:
            return self._custom_client
        return await redis_manager.get_async_client()

    @staticmethod
    def compute_exact_key(query: str, filters: Dict[str, Any], index_version: Optional[int] = None) -> str:
        """Computes deterministic SHA-256 key from normalized query, sorted active filters, and index version."""
        norm_query = query.strip().lower()
        sorted_filters = sorted((k, str(v)) for k, v in filters.items() if v is not None)
        v_str = f"::v{index_version}" if index_version is not None else ""
        combined = f"{norm_query}::{json.dumps(sorted_filters)}{v_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    async def get_exact(self, exact_key: str) -> Optional[QueryResponse]:
        """L1 Exact Cache lookup in Redis."""
        if not (getattr(settings, "ENABLE_EXACT_CACHE", True) and getattr(settings, "EXACT_CACHE_ENABLED", True)):
            return None

        # Check in-memory mirror first (for fast unit testing)
        if exact_key in self.exact_cache:
            resp, expiry = self.exact_cache[exact_key]
            if time.time() < expiry:
                metrics_collector.record_exact_cache_hit()
                return resp
            else:
                del self.exact_cache[exact_key]

        try:
            client = await self._get_client()
            r_key = RedisKeys.exact_cache_key(exact_key, prefix=settings.EXACT_CACHE_PREFIX)
            raw_data = await client.get(r_key)
            if raw_data:
                if isinstance(raw_data, bytes):
                    raw_data = raw_data.decode("utf-8")
                response = QueryResponse.model_validate_json(raw_data)
                metrics_collector.record_exact_cache_hit()
                logger.debug(f"L1 Exact Cache HIT in Redis for key: {exact_key[:12]}")
                return response
            metrics_collector.record_exact_cache_miss()
        except Exception as e:
            logger.warning(f"Error querying exact cache in Redis for {exact_key[:12]}: {e}")
        return None

    async def get_semantic(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> Optional[Tuple[QueryResponse, float]]:
        """
        L2 Semantic Cache lookup in Redis.
        Strict Financial Metadata Isolation:
        Entries are partitioned by (ticker, fiscal_year, fiscal_quarter).
        Only entries within the matching partition are evaluated, preventing cross-period/ticker leakage.
        """
        if not (getattr(settings, "ENABLE_SEMANTIC_CACHE", True) and getattr(settings, "SEMANTIC_CACHE_ENABLED", True)):
            return None

        ticker = filters.get("ticker")
        year = filters.get("fiscal_year")
        quarter = filters.get("fiscal_quarter")

        partition_key = RedisKeys.semantic_partition_key(
            ticker=ticker,
            fiscal_year=year,
            fiscal_quarter=quarter,
            prefix=settings.SEMANTIC_CACHE_PREFIX
        )

        try:
            client = await self._get_client()
            entry_ids = await client.smembers(partition_key)
            if not entry_ids:
                metrics_collector.record_semantic_cache_miss()
                return None

            q_emb = np.array(embedding_service.embed_text(query), dtype=np.float32)
            current_time = time.time()

            best_score = -1.0
            best_resp = None
            stale_ids = []

            for raw_eid in entry_ids:
                eid = raw_eid.decode("utf-8") if isinstance(raw_eid, bytes) else str(raw_eid)
                e_key = RedisKeys.semantic_entry_key(eid, prefix=settings.SEMANTIC_CACHE_PREFIX)
                entry_data = await client.hgetall(e_key)

                if not entry_data:
                    stale_ids.append(eid)
                    continue

                # Parse hash fields
                def _get_field(f_name):
                    val = entry_data.get(f_name) or entry_data.get(f_name.encode("utf-8"))
                    return val.decode("utf-8") if isinstance(val, bytes) else val

                expires_at = float(_get_field("expires_at") or 0.0)
                if current_time >= expires_at:
                    stale_ids.append(eid)
                    continue

                entry_filters = json.loads(_get_field("filters") or "{}")
                # Strict check on filters
                if entry_filters != filters:
                    metrics_collector.record_semantic_metadata_mismatch()
                    continue

                emb_list = json.loads(_get_field("embedding") or "[]")
                cached_emb = np.array(emb_list, dtype=np.float32)
                cos_sim = float(np.dot(q_emb, cached_emb))

                if cos_sim > best_score:
                    best_score = cos_sim
                    raw_resp = _get_field("response")
                    best_resp = QueryResponse.model_validate_json(raw_resp)

            # Clean up stale expired IDs asynchronously
            if stale_ids:
                asyncio.create_task(self._cleanup_stale_entries(client, partition_key, stale_ids))

            if best_score >= self.cosine_threshold and best_resp is not None:
                metrics_collector.record_semantic_cache_hit(score=best_score)
                logger.info(f"L2 Semantic Cache HIT in Redis with cosine similarity: {best_score:.4f}")
                return best_resp, best_score

            metrics_collector.record_semantic_cache_miss()
        except Exception as e:
            logger.warning(f"Error querying semantic cache in Redis: {e}")

        return None

    async def _cleanup_stale_entries(self, client, partition_key: str, entry_ids: List[str]):
        """Helper to remove expired entries from partition set."""
        try:
            for eid in entry_ids:
                await client.srem(partition_key, eid)
                e_key = RedisKeys.semantic_entry_key(eid, prefix=settings.SEMANTIC_CACHE_PREFIX)
                await client.delete(e_key)
        except Exception:
            pass

    async def set(
        self,
        query: str,
        filters: Dict[str, Any],
        response: QueryResponse
    ) -> None:
        """Stores response in both L1 (exact) and L2 (semantic) caches in Redis."""
        exact_key = self.compute_exact_key(query, filters)
        expiry_ts = time.time() + self.exact_ttl
        resp_json = response.model_dump_json()

        # Update in-memory mirror for local sync access
        self.exact_cache[exact_key] = (response, expiry_ts)

        try:
            client = await self._get_client()
            r_exact_key = RedisKeys.exact_cache_key(exact_key, prefix=settings.EXACT_CACHE_PREFIX)
            # Store in L1 Exact Cache
            await client.set(r_exact_key, resp_json, ex=self.exact_ttl)

            # Store in L2 Semantic Cache
            if getattr(settings, "ENABLE_SEMANTIC_CACHE", True) and getattr(settings, "SEMANTIC_CACHE_ENABLED", True):
                entry_id = f"sem-{uuid.uuid4().hex[:12]}"
                e_key = RedisKeys.semantic_entry_key(entry_id, prefix=settings.SEMANTIC_CACHE_PREFIX)

                ticker = filters.get("ticker")
                year = filters.get("fiscal_year")
                quarter = filters.get("fiscal_quarter")
                partition_key = RedisKeys.semantic_partition_key(
                    ticker=ticker,
                    fiscal_year=year,
                    fiscal_quarter=quarter,
                    prefix=settings.SEMANTIC_CACHE_PREFIX
                )

                q_emb = [float(v) for v in embedding_service.embed_text(query)]

                entry_payload = {
                    "entry_id": entry_id,
                    "query": query,
                    "filters": json.dumps(filters),
                    "embedding": json.dumps(q_emb),
                    "response": resp_json,
                    "expires_at": str(time.time() + self.semantic_ttl),
                    "created_at": str(time.time())
                }

                # Save hash and add to partition set
                await client.hset(e_key, mapping=entry_payload)
                await client.expire(e_key, self.semantic_ttl)
                await client.sadd(partition_key, entry_id)
                await client.expire(partition_key, self.semantic_ttl + 300)

                logger.debug(f"Cached response in L1 & L2 for query: '{query[:30]}...'")
        except Exception as e:
            logger.error(f"Failed to write to Redis cache: {e}")

    async def clear(self):
        """Clears all cached entries across L1 and L2."""
        self.exact_cache.clear()
        try:
            client = await self._get_client()
            # Flush keys matching prefix pattern
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=f"{settings.REDIS_KEY_PREFIX}:cache:*", count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Successfully cleared Redis cache namespace.")
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {e}")


cache_service = TwoTierCacheService()
