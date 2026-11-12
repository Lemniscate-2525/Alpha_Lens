"""
AlphaLens - RAG Orchestration Service
Unifies Caching, Query Rewriting, Hybrid Retrieval, Reranking, Generation, and Citations.
"""
import asyncio
import time
from typing import Dict, Optional, Tuple
from src.core.models import QueryRequest, QueryResponse, SearchResult
from src.retrieval.query_rewriter import FinancialQueryRewriter
from src.retrieval.hybrid import hybrid_retriever
from src.retrieval.dense import dense_retriever
from src.retrieval.reranker import neural_reranker
from src.generation.generator import llm_generator
from src.generation.citation_verifier import CitationVerifier
from src.services.cache_service import cache_service
from src.services.circuit_breaker import llm_circuit_breaker
from src.observability.tracer import tracer
from src.observability.metrics import metrics_collector
from src.observability.logging import logger
from src.config import settings


class RAGService:
    """Master orchestration service for AlphaLens query pipeline."""

    async def execute_query(
        self,
        request: QueryRequest,
        trace_id: Optional[str] = None
    ) -> QueryResponse:
        start_time = time.perf_counter()
        trace_id = trace_id or tracer.generate_trace_id()

        # Step 1: Pre-process query & extract implicit metadata filters
        inferred_filters = FinancialQueryRewriter.extract_metadata_filters(request.query)
        # Merge explicit request filters with inferred filters
        effective_filters = {
            "ticker": request.ticker or inferred_filters.get("ticker"),
            "fiscal_quarter": request.fiscal_quarter or inferred_filters.get("fiscal_quarter"),
            "fiscal_year": request.fiscal_year or inferred_filters.get("fiscal_year"),
            "speaker_role": request.speaker_role.value if request.speaker_role else inferred_filters.get("speaker_role"),
        }
        # Clean null values
        active_filters = {k: v for k, v in effective_filters.items() if v is not None}

        # Step 2: Two-Tier Cache Check
        exact_key = cache_service.compute_exact_key(request.query, active_filters)

        if request.use_semantic_cache and settings.ENABLE_EXACT_CACHE:
            cached_resp = cache_service.get_exact(exact_key)
            if cached_resp:
                e2e_ms = (time.perf_counter() - start_time) * 1000.0
                cached_resp.latency_ms = round(e2e_ms, 2)
                cached_resp.cache_hit_type = "exact"
                cached_resp.trace_id = trace_id
                metrics_collector.record_query(
                    e2e_latency_ms=e2e_ms,
                    retrieval_latency_ms=0.0,
                    rerank_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    cache_hit_type="exact",
                    tokens_used=0,
                    cost_usd=0.0
                )
                return cached_resp

        if request.use_semantic_cache and settings.ENABLE_SEMANTIC_CACHE:
            sem_result = cache_service.get_semantic(request.query, active_filters)
            if sem_result:
                cached_resp, cos_sim = sem_result
                e2e_ms = (time.perf_counter() - start_time) * 1000.0
                cached_resp.latency_ms = round(e2e_ms, 2)
                cached_resp.cache_hit_type = "semantic"
                cached_resp.trace_id = trace_id
                metrics_collector.record_query(
                    e2e_latency_ms=e2e_ms,
                    retrieval_latency_ms=0.0,
                    rerank_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    cache_hit_type="semantic",
                    tokens_used=0,
                    cost_usd=0.0
                )
                return cached_resp

        # Step 3: Singleflight Stampede Protection
        lock = await cache_service.singleflight.get_lock(exact_key)
        async with lock:
            # Re-check exact cache in case a concurrent request already populated it while we waited
            if request.use_semantic_cache and settings.ENABLE_EXACT_CACHE:
                cached_resp = cache_service.get_exact(exact_key)
                if cached_resp:
                    e2e_ms = (time.perf_counter() - start_time) * 1000.0
                    cached_resp.latency_ms = round(e2e_ms, 2)
                    cached_resp.cache_hit_type = "exact"
                    cached_resp.trace_id = trace_id
                    return cached_resp

            # Step 4: Hybrid Retrieval
            t_retrieval_start = time.perf_counter()
            expanded_query = FinancialQueryRewriter.expand_query(request.query)

            candidates = hybrid_retriever.retrieve(
                query=expanded_query,
                top_k=settings.FUSED_TOP_K,
                filters=active_filters
            )
            retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

            # Step 5: Cross-Encoder Neural Reranking
            t_rerank_start = time.perf_counter()
            if request.use_reranker and settings.USE_RERANKER and candidates:
                final_context = neural_reranker.rerank(
                    query=request.query,
                    candidates=candidates,
                    top_k=request.top_k
                )
            else:
                final_context = candidates[:request.top_k]
            rerank_ms = (time.perf_counter() - t_rerank_start) * 1000.0

            # Step 6: Circuit Breaker Protected Generation
            t_gen_start = time.perf_counter()
            try:
                answer, tokens, cost, is_unanswerable = llm_circuit_breaker.execute(
                    llm_generator.generate,
                    query=request.query,
                    context_chunks=final_context
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
            gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

            # Step 7: Post-Generation Citation Extraction & Verification
            citations, fidelity = CitationVerifier.verify_and_extract_citations(
                generated_answer=answer,
                retrieved_chunks=final_context
            )

            e2e_ms = (time.perf_counter() - start_time) * 1000.0

            response = QueryResponse(
                query=request.query,
                answer=answer,
                citations=citations,
                search_results=final_context,
                latency_ms=round(e2e_ms, 2),
                retrieval_latency_ms=round(retrieval_ms, 2),
                rerank_latency_ms=round(rerank_ms, 2),
                generation_latency_ms=round(gen_ms, 2),
                cache_hit_type=None,
                is_unanswerable=is_unanswerable,
                tokens_used=tokens,
                estimated_cost_usd=cost,
                trace_id=trace_id
            )

            # Step 8: Update Two-Tier Cache
            if request.use_semantic_cache and not is_unanswerable:
                cache_service.set(request.query, active_filters, response)

            # Record system metrics
            metrics_collector.record_query(
                e2e_latency_ms=e2e_ms,
                retrieval_latency_ms=retrieval_ms,
                rerank_latency_ms=rerank_ms,
                generation_latency_ms=gen_ms,
                cache_hit_type=None,
                tokens_used=tokens,
                cost_usd=cost
            )

            return response


rag_service = RAGService()
