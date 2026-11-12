"""
AlphaLens - Sparse Lexical Retrieval (BM25Okapi)
Implements BM25 search over financial chunks with entity-preserving tokenization.
"""
import re
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from src.core.models import Chunk, SearchResult
from src.database.vector_store import vector_store
from src.config import settings
from src.observability.logging import logger


class SparseBM25Retriever:
    """Performs lexical search using BM25Okapi with financial tokenization."""

    def __init__(self):
        self.corpus_chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text while preserving financial numbers, percentages, currency, and tickers.
        E.g., '$30.04 billion', '75.7%', 'CoWoS-L', 'NVDA'.
        """
        text = text.lower()
        # Match words with hyphens, dollar values, percentages, and alphanumeric symbols
        tokens = re.findall(r"[\$\€\£]?\d+(?:\.\d+)?%?|[a-z0-9]+(?:[\-_][a-z0-9]+)*", text)
        return tokens

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Adds or updates chunks in the BM25 index."""
        # Append only new chunk_ids
        existing_ids = {c.chunk_id for c in self.corpus_chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids and not c.is_parent]

        if not new_chunks:
            return

        self.corpus_chunks.extend(new_chunks)
        new_tokens = [self.tokenize(c.content) for c in new_chunks]
        self.tokenized_corpus.extend(new_tokens)

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Rebuilt BM25 index with total {len(self.corpus_chunks)} chunks.")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not self.bm25 or not self.corpus_chunks:
            return []

        top_k = top_k or settings.SPARSE_TOP_K
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters
        filtered_candidates = []
        for idx, (chunk, score) in enumerate(zip(self.corpus_chunks, scores)):
            if score <= 0.0:
                continue

            if filters:
                match = True
                for k, v in filters.items():
                    if v is not None:
                        val = getattr(chunk, k, None)
                        if val is None and k in chunk.metadata:
                            val = chunk.metadata[k]
                        if hasattr(val, "value"):
                            val = val.value
                        if val != v:
                            match = False
                            break
                if not match:
                    continue

            filtered_candidates.append((chunk, float(score)))

        # Sort by score descending
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = filtered_candidates[:top_k]

        results = []
        for chunk, score in top_candidates:
            parent_chunk = vector_store.get_parent_chunk(chunk.parent_chunk_id) if chunk.parent_chunk_id else None
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                parent_chunk_id=chunk.parent_chunk_id,
                content=chunk.content,
                parent_content=parent_chunk.content if parent_chunk else None,
                score=round(score, 4),
                retrieval_type="sparse",
                metadata={
                    "ticker": chunk.ticker,
                    "fiscal_quarter": chunk.fiscal_quarter,
                    "fiscal_year": chunk.fiscal_year,
                    "speaker_name": chunk.speaker_name or "",
                    "speaker_role": chunk.speaker_role.value if chunk.speaker_role else "",
                    "section": chunk.section.value,
                }
            ))

        return results


bm25_retriever = SparseBM25Retriever()
