"""
AlphaLens - Embedding Generation Service
Batched, normalized dense vector embeddings using SentenceTransformers with LRU hashing.
"""
from typing import List, Union
import numpy as np
from src.config import settings
from src.observability.logging import logger


class EmbeddingService:
    """Manages dense vector embedding generation with batching and memory caching."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._cache = {}

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates a normalized embedding for a single string."""
        if text in self._cache:
            return self._cache[text]

        model = self._get_model()
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        vec = embedding.tolist()
        if len(self._cache) < 5000:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Generates normalized embeddings for a batch of texts with batching."""
        if not texts:
            return []

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()


# Global singleton instance
embedding_service = EmbeddingService()
