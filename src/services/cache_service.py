"""
AlphaLens - Two-Tier Caching Service
Tier 1: Exact Hash Cache (SHA-256 key matching, latency < 5ms)
Tier 2: Semantic Similarity Cache (Cosine similarity >= 0.94, latency < 40ms)
Features asynchronous Singleflight mutex locking to prevent cache stampedes under high concurrency.
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.models import QueryResponse
from src.config import settings
from src.ingestion.embedder import embedding_service
from src.observability.logging import logger


class SingleflightGroup:
    """Ensures duplicate concurrent queries for the same key execute only once (stampede prevention)."""
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._master_lock = asyncio.Lock()

    async def get_lock(self, key: str) -> asyncio.Lock:
        async with self._master_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def release_lock(self, key: str):
        async with self._master_lock:
            if key in self._locks and not self._locks[key].locked():
                del self._locks[key]


class TwoTierCacheService:
    """Manages exact and semantic query caching with TTL and cache stampede protection."""

    def __init__(self):
        self.exact_cache: Dict[str, Tuple[QueryResponse, float]] = {}  # key -> (response, expiry_ts)
        self.semantic_queries: List[Dict[str, Any]] = []  # list of {embedding, response, filters, expiry_ts}
        self.singleflight = SingleflightGroup()
        self.cosine_threshold = settings.SEMANTIC_CACHE_COSINE_THRESHOLD
        self.ttl = settings.EXACT_CACHE_TTL_SECONDS

    @staticmethod
    def compute_exact_key(query: str, filters: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 key from normalized query and sorted filters."""
        norm_query = query.strip().lower()
        sorted_filters = sorted((k, str(v)) for k, v in filters.items() if v is not None)
        combined = f"{norm_query}::{json.dumps(sorted_filters)}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get_exact(self, exact_key: str) -> Optional[QueryResponse]:
        """L1 Exact Cache lookup."""
        if not settings.ENABLE_EXACT_CACHE:
            return None

        if exact_key in self.exact_cache:
            resp, expiry = self.exact_cache[exact_key]
            if time.time() < expiry:
                logger.debug(f"L1 Exact Cache HIT for key: {exact_key[:12]}")
                return resp
            else:
                del self.exact_cache[exact_key]
        return None

    def get_semantic(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> Optional[Tuple[QueryResponse, float]]:
        """
        L2 Semantic Cache lookup.
        Computes query embedding and evaluates cosine similarity against cached queries.
        Returns: (cached_response, similarity_score) if score >= threshold, else None.
        """
        if not settings.ENABLE_SEMANTIC_CACHE or not self.semantic_queries:
            return None

        q_emb = np.array(embedding_service.embed_text(query), dtype=np.float32)
        current_time = time.time()

        best_score = -1.0
        best_resp = None

        for item in self.semantic_queries:
            if current_time >= item["expiry_ts"]:
                continue

            # Cached query must match identical metadata filters
            if item["filters"] != filters:
                continue

            cached_emb = item["embedding"]
            cos_sim = float(np.dot(q_emb, cached_emb))

            if cos_sim > best_score:
                best_score = cos_sim
                best_resp = item["response"]

        if best_score >= self.cosine_threshold and best_resp is not None:
            logger.debug(f"L2 Semantic Cache HIT with cosine similarity: {best_score:.4f}")
            return best_resp, best_score

        return None

    def set(
        self,
        query: str,
        filters: Dict[str, Any],
        response: QueryResponse
    ) -> None:
        """Stores response in both L1 (exact) and L2 (semantic) caches."""
        exact_key = self.compute_exact_key(query, filters)
        expiry_ts = time.time() + self.ttl

        # Save to L1 Exact
        self.exact_cache[exact_key] = (response, expiry_ts)

        # Save to L2 Semantic
        if settings.ENABLE_SEMANTIC_CACHE:
            q_emb = np.array(embedding_service.embed_text(query), dtype=np.float32)
            self.semantic_queries.append({
                "query": query,
                "embedding": q_emb,
                "filters": filters,
                "response": response,
                "expiry_ts": expiry_ts
            })
            # Evict oldest if exceeding capacity
            if len(self.semantic_queries) > settings.SEMANTIC_CACHE_MAX_ENTRIES:
                self.semantic_queries = self.semantic_queries[-settings.SEMANTIC_CACHE_MAX_ENTRIES:]

    def clear(self):
        """Clears all cached entries."""
        self.exact_cache.clear()
        self.semantic_queries.clear()


cache_service = TwoTierCacheService()
