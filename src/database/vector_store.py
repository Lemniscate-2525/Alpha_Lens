"""
AlphaLens - Vector Store Abstraction & Implementations
Provides unified interface for vector storage with metadata pre-filtering.
"""
from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.models import Chunk, SearchResult
from src.core.exceptions import VectorStoreError
from src.config import settings, VectorStoreType
from src.observability.logging import logger


class BaseVectorStore(ABC):
    """Abstract interface defining required vector store operations."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        pass

    @abstractmethod
    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[Chunk]:
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


class ChromaVectorStore(BaseVectorStore):
    """Persistent Chroma vector store with metadata filtering."""

    def __init__(self, persist_directory: str = None, collection_name: str = "alphalens_financial"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.persist_dir = persist_directory or str(settings.CHROMA_PERSIST_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=chroma_settings)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._chunks_registry: Dict[str, Chunk] = {}
        self._parent_chunks_registry: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        valid_embeddings = []

        for chunk, emb in zip(chunks, embeddings):
            if chunk.is_parent:
                self._parent_chunks_registry[chunk.chunk_id] = chunk
                continue

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            valid_embeddings.append(emb)
            self._chunks_registry[chunk.chunk_id] = chunk

            # Chroma metadata values must be primitive types (str, int, float, bool)
            meta = {
                "document_id": chunk.document_id,
                "ticker": chunk.ticker,
                "fiscal_quarter": chunk.fiscal_quarter,
                "fiscal_year": chunk.fiscal_year,
                "document_type": chunk.document_type.value,
                "section": chunk.section.value,
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "speaker_name": chunk.speaker_name or "",
                "speaker_role": chunk.speaker_role.value if chunk.speaker_role else "",
            }
            metadatas.append(meta)

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=valid_embeddings,
                metadatas=metadatas
            )
            logger.info(f"Upserted {len(ids)} child chunks into Chroma collection '{self.collection_name}'")

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        chroma_filter = {}
        if filters:
            clean_filters = {k: v for k, v in filters.items() if v is not None}
            if len(clean_filters) == 1:
                k, v = next(iter(clean_filters.items()))
                chroma_filter = {k: {"$eq": v}}
            elif len(clean_filters) > 1:
                chroma_filter = {"$and": [{k: {"$eq": v}} for k, v in clean_filters.items()]}

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=chroma_filter if chroma_filter else None
            )
        except Exception as e:
            raise VectorStoreError(f"Chroma similarity search failed: {str(e)}")

        search_results: List[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
        documents = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)

        for chunk_id, dist, doc_content, meta in zip(ids, distances, documents, metadatas):
            # Chroma returns cosine distance (0 to 2); cosine similarity is 1 - distance
            score = 1.0 - float(dist)
            parent_id = meta.get("parent_chunk_id")
            parent_chunk = self._parent_chunks_registry.get(parent_id)

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                parent_chunk_id=parent_id if parent_id else None,
                content=doc_content,
                parent_content=parent_chunk.content if parent_chunk else None,
                score=round(score, 4),
                retrieval_type="dense",
                metadata=meta
            ))

        return search_results

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks_registry.get(chunk_id)

    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[Chunk]:
        return self._parent_chunks_registry.get(parent_chunk_id)

    def delete_document(self, document_id: str) -> int:
        try:
            self.collection.delete(where={"document_id": document_id})
            # Remove from local registry
            removed = [cid for cid, c in self._chunks_registry.items() if c.document_id == document_id]
            for cid in removed:
                del self._chunks_registry[cid]
            return len(removed)
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {e}")
            return 0

    def count(self) -> int:
        return self.collection.count()


class InMemoryVectorStore(BaseVectorStore):
    """High-speed in-memory vector store for testing and benchmarking."""

    def __init__(self):
        self.chunk_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunks_map: Dict[str, Chunk] = {}
        self.parent_chunks_map: Dict[str, Chunk] = {}

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        new_ids = []
        new_embs = []
        for chunk, emb in zip(chunks, embeddings):
            if chunk.is_parent:
                self.parent_chunks_map[chunk.chunk_id] = chunk
                continue
            new_ids.append(chunk.chunk_id)
            new_embs.append(emb)
            self.chunks_map[chunk.chunk_id] = chunk

        if not new_ids:
            return

        new_emb_arr = np.array(new_embs, dtype=np.float32)
        if self.embeddings is None or len(self.chunk_ids) == 0:
            self.chunk_ids = new_ids
            self.embeddings = new_emb_arr
        else:
            self.chunk_ids.extend(new_ids)
            self.embeddings = np.vstack([self.embeddings, new_emb_arr])

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.embeddings is None or len(self.chunk_ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        # Cosine similarity (vectors are unit normalized)
        scores = np.dot(self.embeddings, q_vec)

        # Apply metadata filters
        valid_indices = []
        for idx, cid in enumerate(self.chunk_ids):
            chunk = self.chunks_map[cid]
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
            valid_indices.append(idx)

        if not valid_indices:
            return []

        filtered_scores = scores[valid_indices]
        top_sub_indices = np.argsort(filtered_scores)[::-1][:top_k]

        results = []
        for sub_idx in top_sub_indices:
            actual_idx = valid_indices[sub_idx]
            cid = self.chunk_ids[actual_idx]
            chunk = self.chunks_map[cid]
            score = float(scores[actual_idx])
            parent_chunk = self.parent_chunks_map.get(chunk.parent_chunk_id) if chunk.parent_chunk_id else None

            results.append(SearchResult(
                chunk_id=cid,
                parent_chunk_id=chunk.parent_chunk_id,
                content=chunk.content,
                parent_content=parent_chunk.content if parent_chunk else None,
                score=round(score, 4),
                retrieval_type="dense",
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

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks_map.get(chunk_id)

    def get_parent_chunk(self, parent_chunk_id: str) -> Optional[Chunk]:
        return self.parent_chunks_map.get(parent_chunk_id)

    def delete_document(self, document_id: str) -> int:
        remove_cids = [cid for cid, c in self.chunks_map.items() if c.document_id == document_id]
        for cid in remove_cids:
            del self.chunks_map[cid]
        # Rebuild array
        keep_indices = [i for i, cid in enumerate(self.chunk_ids) if cid not in remove_cids]
        if keep_indices and self.embeddings is not None:
            self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
        else:
            self.chunk_ids = []
            self.embeddings = None
        return len(remove_cids)

    def count(self) -> int:
        return len(self.chunk_ids)


def get_vector_store() -> BaseVectorStore:
    """Factory creating configured vector store adapter."""
    if settings.VECTOR_STORE_TYPE == VectorStoreType.MEMORY:
        return InMemoryVectorStore()
    return ChromaVectorStore()


# Global singleton vector store instance
vector_store = get_vector_store()
