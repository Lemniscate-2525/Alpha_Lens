"""
AlphaLens - Dense Vector Retrieval
Performs semantic vector similarity search using normalized embeddings and metadata pre-filtering.
"""
from typing import Any, Dict, List, Optional
from src.core.models import SearchResult
from src.database.vector_store import vector_store
from src.ingestion.embedder import embedding_service
from src.config import settings


class DenseRetriever:
    """Executes dense vector similarity search over embedded document chunks."""

    def __init__(self):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        top_k = top_k or settings.DENSE_TOP_K
        query_vector = self.embedding_service.embed_text(query)
        results = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters
        )
        for r in results:
            r.retrieval_type = "dense"
        return results


dense_retriever = DenseRetriever()

