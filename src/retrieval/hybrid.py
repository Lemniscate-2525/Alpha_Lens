"""
AlphaLens - Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
Combines dense semantic vector search with sparse BM25 lexical search.
"""
from typing import Any, Dict, List, Optional
from src.core.models import SearchResult
from src.retrieval.dense import dense_retriever
from src.retrieval.sparse import bm25_retriever
from src.config import settings


class HybridRetriever:
    """Executes dense + sparse search and fuses results using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = None):
        self.rrf_k = rrf_k or settings.RRF_K_PARAMETER
        self.dense = dense_retriever
        self.sparse = bm25_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        top_k = top_k or settings.FUSED_TOP_K

        # Step 1: Query both retrievers
        dense_results = self.dense.retrieve(query=query, top_k=settings.DENSE_TOP_K, filters=filters)
        sparse_results = self.sparse.retrieve(query=query, top_k=settings.SPARSE_TOP_K, filters=filters)

        # Step 2: Compute Reciprocal Rank Fusion (RRF)
        # RRF_Score = 1 / (k + rank)
        rrf_scores: Dict[str, float] = {}
        items_map: Dict[str, SearchResult] = {}

        for rank, item in enumerate(dense_results, start=1):
            items_map[item.chunk_id] = item
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(sparse_results, start=1):
            if item.chunk_id not in items_map:
                items_map[item.chunk_id] = item
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort candidate items by fused RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:top_k]

        fused_results: List[SearchResult] = []
        for cid in top_chunk_ids:
            item = items_map[cid]
            fused_item = SearchResult(
                chunk_id=item.chunk_id,
                parent_chunk_id=item.parent_chunk_id,
                content=item.content,
                parent_content=item.parent_content,
                score=round(rrf_scores[cid], 5),
                retrieval_type="hybrid_rrf",
                metadata=item.metadata
            )
            fused_results.append(fused_item)

        return fused_results


hybrid_retriever = HybridRetriever()
