"""
AlphaLens - Redis Key Namespace Registry
Centralizes all Redis key patterns, prefixes, and channel names.
Ensures consistency, avoids collisions, and supports Redis Cluster hash tags.
"""

from typing import Optional


class RedisKeys:
    """Namespace manager for all AlphaLens Redis keys."""
    
    DEFAULT_PREFIX = "alpha_lens"

    @classmethod
    def rate_limit_key(cls, client_id: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Key for distributed Token Bucket bucket state."""
        return f"{prefix}:ratelimit:{{{client_id}}}"

    @classmethod
    def singleflight_lock_key(cls, cache_key: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Distributed lock key for request coalescing."""
        return f"{prefix}:lock:singleflight:{{{cache_key}}}"

    @classmethod
    def singleflight_channel(cls, cache_key: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Pub/Sub channel to notify waiters of leader completion."""
        return f"{prefix}:pubsub:singleflight:{{{cache_key}}}"

    @classmethod
    def exact_cache_key(cls, cache_key: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Key for L1 exact hash query cache."""
        return f"{prefix}:cache:exact:{{{cache_key}}}"

    @classmethod
    def semantic_partition_key(
        cls,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        fiscal_quarter: Optional[str] = None,
        prefix: str = DEFAULT_PREFIX
    ) -> str:
        """
        Partition key for financial metadata isolation in semantic cache.
        Groups candidate entries by ticker, year, and quarter to prevent cross-period matching.
        """
        t = (ticker or "ALL").upper()
        y = fiscal_year or "ALL"
        q = (fiscal_quarter or "ALL").upper()
        return f"{prefix}:cache:semantic:partition:{{{t}:{y}:{q}}}"

    @classmethod
    def semantic_entry_key(cls, entry_id: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Hash key for an individual semantic cache entry."""
        return f"{prefix}:cache:semantic:entry:{{{entry_id}}}"

    @classmethod
    def job_key(cls, job_id: str, prefix: str = DEFAULT_PREFIX) -> str:
        """Hash key for ingestion job status metadata."""
        return f"{prefix}:jobs:{{{job_id}}}"

    @classmethod
    def jobs_index_key(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Sorted set of all job IDs ordered by creation timestamp."""
        return f"{prefix}:jobs:all"

    @classmethod
    def ingestion_queue(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Redis Stream / List for asynchronous ingestion tasks."""
        return f"{prefix}:queue:ingestion"

    @classmethod
    def ingestion_dead_letter(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Dead letter queue for permanently failed ingestion tasks."""
        return f"{prefix}:queue:ingestion:dead_letter"

    @classmethod
    def ingested_hashes_key(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Hash mapping document_id -> content_hash for distributed idempotency."""
        return f"{prefix}:metadata:ingested_hashes"

    @classmethod
    def index_version_key(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Atomic integer counter tracking the global corpus/index version."""
        return f"{prefix}:metadata:index_version"

    @classmethod
    def index_updates_channel(cls, prefix: str = DEFAULT_PREFIX) -> str:
        """Pub/Sub channel broadcasting index reload notifications to API instances."""
        return f"{prefix}:channel:index_updates"
