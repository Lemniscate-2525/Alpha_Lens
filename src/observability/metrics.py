"""
AlphaLens - Metrics and SLI/SLO Monitoring
Maintains thread-safe in-memory latency percentiles, error rates, cache hit ratios, and token costs.
"""
import threading
import time
from typing import Dict, List, Any
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
        self.cache_latencies: List[float] = []
        
        # Counts
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.error_requests: int = 0
        self.rate_limited_requests: int = 0
        
        # Cache stats
        self.exact_cache_hits: int = 0
        self.semantic_cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Tokens and costs
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
        cache_hit_type: str = None,
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
            else:
                self.cache_misses += 1

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
            cache_requests = self.exact_cache_hits + self.semantic_cache_hits + self.cache_misses
            hit_rate = (
                (self.exact_cache_hits + self.semantic_cache_hits) / max(cache_requests, 1)
            ) * 100.0

            error_rate = (self.error_requests / max(self.total_requests, 1)) * 100.0

            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "error_requests": self.error_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "current_rps": round(rps, 2),
                "error_rate_pct": round(error_rate, 3),
                "cache": {
                    "exact_hits": self.exact_cache_hits,
                    "semantic_hits": self.semantic_cache_hits,
                    "misses": self.cache_misses,
                    "overall_hit_rate_pct": round(hit_rate, 2),
                },
                "tokens": {
                    "total_consumed": self.total_tokens_consumed,
                    "estimated_cost_usd": round(self.total_estimated_cost_usd, 4),
                },
                "latency_e2e_ms": self._compute_percentiles(self.e2e_latencies),
                "latency_retrieval_ms": self._compute_percentiles(self.retrieval_latencies),
                "latency_rerank_ms": self._compute_percentiles(self.rerank_latencies),
                "latency_generation_ms": self._compute_percentiles(self.generation_latencies),
            }


# Global singleton metrics collector
metrics_collector = MetricsCollector()
