"""
AlphaLens - Metrics and SLI/SLO Monitoring
Maintains thread-safe metrics for distributed rate limiting, singleflight coalescing,
exact/semantic caching, background ingestion queues, and latency percentiles.
"""

import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np


class MetricsCollector:
    """Thread-safe metrics collector for tracking real-time systems performance and latency percentiles."""

    def __init__(self):
        self._lock = threading.Lock()

        # Latency buckets (milliseconds)
        self.e2e_latencies: List[float] = []
        self.retrieval_latencies: List[float] = []
        self.rerank_latencies: List[float] = []
        self.generation_latencies: List[float] = []
        self.singleflight_wait_latencies: List[float] = []

        # Request Counts
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.error_requests: int = 0
        self.rate_limited_requests: int = 0

        # Distributed Singleflight Stats
        self.singleflight_leaders: int = 0
        self.singleflight_waiters: int = 0
        self.duplicate_generations_prevented: int = 0

        # Cache Stats
        self.exact_cache_hits: int = 0
        self.exact_cache_misses: int = 0
        self.semantic_cache_hits: int = 0
        self.semantic_cache_misses: int = 0
        self.semantic_metadata_mismatches: int = 0
        self.semantic_similarity_scores: List[float] = []

        # Background Ingestion Queue Stats
        self.ingestion_jobs_queued: int = 0
        self.ingestion_jobs_completed: int = 0
        self.ingestion_jobs_failed: int = 0
        self.ingestion_jobs_retried: int = 0
        self.ingestion_jobs_dead_letter: int = 0

        # Tokens and Costs
        self.total_tokens_consumed: int = 0
        self.total_estimated_cost_usd: float = 0.0

        # Start time for uptime calculation
        self.start_time: float = time.time()

    def record_query(
        self,
        e2e_latency_ms: float,
        retrieval_latency_ms: float,
        rerank_latency_ms: float,
        generation_latency_ms: float,
        cache_hit_type: Optional[str] = None,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        is_error: bool = False
    ):
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.error_requests += 1
            else:
                self.successful_requests += 1

            self.e2e_latencies.append(e2e_latency_ms)
            self.retrieval_latencies.append(retrieval_latency_ms)
            self.rerank_latencies.append(rerank_latency_ms)
            self.generation_latencies.append(generation_latency_ms)

            if cache_hit_type == "exact":
                self.exact_cache_hits += 1
            elif cache_hit_type == "semantic":
                self.semantic_cache_hits += 1
            elif cache_hit_type is None:
                self.exact_cache_misses += 1

            self.total_tokens_consumed += tokens_used
            self.total_estimated_cost_usd += cost_usd

            # Keep sliding window of last 10,000 requests to avoid unbounded memory
            if len(self.e2e_latencies) > 10000:
                self.e2e_latencies = self.e2e_latencies[-10000:]
                self.retrieval_latencies = self.retrieval_latencies[-10000:]
                self.rerank_latencies = self.rerank_latencies[-10000:]
                self.generation_latencies = self.generation_latencies[-10000:]

    def record_rate_limit_drop(self):
        with self._lock:
            self.total_requests += 1
            self.rate_limited_requests += 1

    def record_singleflight_leader(self):
        with self._lock:
            self.singleflight_leaders += 1

    def record_singleflight_waiter(self):
        with self._lock:
            self.singleflight_waiters += 1

    def record_duplicate_generation_prevented(self, wait_duration_ms: float):
        with self._lock:
            self.duplicate_generations_prevented += 1
            self.singleflight_wait_latencies.append(wait_duration_ms)
            if len(self.singleflight_wait_latencies) > 5000:
                self.singleflight_wait_latencies = self.singleflight_wait_latencies[-5000:]

    def record_exact_cache_hit(self):
        with self._lock:
            self.exact_cache_hits += 1

    def record_exact_cache_miss(self):
        with self._lock:
            self.exact_cache_misses += 1

    def record_semantic_cache_hit(self, score: float):
        with self._lock:
            self.semantic_cache_hits += 1
            self.semantic_similarity_scores.append(score)
            if len(self.semantic_similarity_scores) > 5000:
                self.semantic_similarity_scores = self.semantic_similarity_scores[-5000:]

    def record_semantic_cache_miss(self):
        with self._lock:
            self.semantic_cache_misses += 1

    def record_semantic_metadata_mismatch(self):
        with self._lock:
            self.semantic_metadata_mismatches += 1

    def record_job_queued(self):
        with self._lock:
            self.ingestion_jobs_queued += 1

    def record_job_completed(self):
        with self._lock:
            self.ingestion_jobs_completed += 1

    def record_job_failed(self):
        with self._lock:
            self.ingestion_jobs_failed += 1

    def record_job_retried(self):
        with self._lock:
            self.ingestion_jobs_retried += 1

    def record_job_dead_letter(self):
        with self._lock:
            self.ingestion_jobs_dead_letter += 1

    @staticmethod
    def _compute_percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.array(values)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "mean": round(float(np.mean(arr)), 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            uptime_seconds = time.time() - self.start_time
            rps = self.total_requests / max(uptime_seconds, 1.0)
            cache_hits = self.exact_cache_hits + self.semantic_cache_hits
            cache_misses = self.exact_cache_misses + self.semantic_cache_misses
            total_cache_queries = cache_hits + cache_misses
            hit_rate = (cache_hits / max(total_cache_queries, 1)) * 100.0

            error_rate = (self.error_requests / max(self.total_requests, 1)) * 100.0
            e2e_perc = self._compute_percentiles(self.e2e_latencies)

            return {
                # Flat telemetry aliases expected by UI and tests
                "uptime_seconds": round(uptime_seconds, 1),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "error_requests": self.error_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "current_rps": round(rps, 2),
                "error_rate_pct": round(error_rate, 3),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "p50_latency_ms": e2e_perc["p50"],
                "p90_latency_ms": e2e_perc["p90"],
                "p95_latency_ms": e2e_perc["p95"],
                "p99_latency_ms": e2e_perc["p99"],
                "total_cost_usd": round(self.total_estimated_cost_usd, 5),
                "ingestion_jobs_completed": self.ingestion_jobs_completed,
                "ingestion_jobs_failed": self.ingestion_jobs_failed,

                # Structured namespaces for detailed dashboard & Prometheus exports
                "rate_limiter": {
                    "total_evaluated": self.total_requests,
                    "dropped": self.rate_limited_requests,
                },
                "singleflight": {
                    "leaders": self.singleflight_leaders,
                    "waiters": self.singleflight_waiters,
                    "duplicate_generations_prevented": self.duplicate_generations_prevented,
                    "wait_latency_ms": self._compute_percentiles(self.singleflight_wait_latencies)
                },
                "cache": {
                    "exact_hits": self.exact_cache_hits,
                    "exact_misses": self.exact_cache_misses,
                    "semantic_hits": self.semantic_cache_hits,
                    "semantic_misses": self.semantic_cache_misses,
                    "semantic_metadata_mismatches": self.semantic_metadata_mismatches,
                    "overall_hit_rate_pct": round(hit_rate, 2),
                },
                "ingestion_queue": {
                    "queued": self.ingestion_jobs_queued,
                    "completed": self.ingestion_jobs_completed,
                    "failed": self.ingestion_jobs_failed,
                    "retried": self.ingestion_jobs_retried,
                    "dead_letter": self.ingestion_jobs_dead_letter,
                },
                "tokens": {
                    "total_consumed": self.total_tokens_consumed,
                    "estimated_cost_usd": round(self.total_estimated_cost_usd, 5),
                },
                "latency_e2e_ms": e2e_perc,
                "latency_retrieval_ms": self._compute_percentiles(self.retrieval_latencies),
                "latency_rerank_ms": self._compute_percentiles(self.rerank_latencies),
                "latency_generation_ms": self._compute_percentiles(self.generation_latencies),
            }


metrics_collector = MetricsCollector()
