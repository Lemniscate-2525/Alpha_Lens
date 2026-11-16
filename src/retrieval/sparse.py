"""
AlphaLens - Sparse Lexical Retrieval (BM25Okapi)
Implements BM25 search over financial chunks with entity-preserving tokenization.
Features shared index synchronization across independent workers and API instances.
"""

import json
import os
from pathlib import Path
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
        self.corpus_file = Path("data/bm25_corpus.json")
        self.last_loaded_mtime: float = 0.0
        self._load_from_disk_if_present()

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

    def _load_from_disk_if_present(self) -> None:
        """Loads chunks from shared disk storage if available."""
        if not self.corpus_file.exists():
            return
        try:
            mtime = self.corpus_file.stat().st_mtime
            if mtime <= self.last_loaded_mtime and self.bm25 is not None:
                return

            with open(self.corpus_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_chunks = [Chunk.model_validate(item) for item in data]
            existing_ids = {c.chunk_id for c in self.corpus_chunks}
            new_chunks = [c for c in loaded_chunks if c.chunk_id not in existing_ids and not c.is_parent]

            if new_chunks:
                self.corpus_chunks.extend(new_chunks)
                new_tokens = [self.tokenize(c.content) for c in new_chunks]
                self.tokenized_corpus.extend(new_tokens)
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Synchronized BM25 index from disk: total {len(self.corpus_chunks)} chunks.")
            self.last_loaded_mtime = mtime
        except Exception as e:
            logger.warning(f"Failed to synchronize BM25 index from disk: {e}")

    def sync_if_needed(self) -> None:
        """Checks shared corpus file on disk and reloads if updated by a background worker."""
        self._load_from_disk_if_present()

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Adds or updates chunks in the BM25 index and persists to shared storage."""
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

        # Persist to shared disk for cross-process visibility
        try:
            self.corpus_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = [c.model_dump(mode="json") for c in self.corpus_chunks]
            # Write atomically using temp file rename
            temp_path = self.corpus_file.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f)
            temp_path.replace(self.corpus_file)
            self.last_loaded_mtime = self.corpus_file.stat().st_mtime
        except Exception as e:
            logger.error(f"Failed to persist shared BM25 corpus to {self.corpus_file}: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        self.sync_if_needed()
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
                    "document_type": chunk.document_type.value,
                    "section": chunk.section.value,
                    "speaker_name": chunk.speaker_name,
                    "speaker_role": chunk.speaker_role.value if chunk.speaker_role else None,
                }
            ))

        return results


bm25_retriever = SparseBM25Retriever()
