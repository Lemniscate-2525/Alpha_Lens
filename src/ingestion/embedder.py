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
        self._is_fastembed = False
        self._cache = {}

    def _get_model(self):
        if self._model is not None:
            return self._model

        # Attempt to load ultra-lightweight FastEmbed (ONNX Runtime, ~180MB RAM vs 450MB+ PyTorch)
        if getattr(settings, "USE_FASTEMBED", True):
            try:
                from fastembed import TextEmbedding
                fastembed_name = (
                    "sentence-transformers/all-MiniLM-L6-v2"
                    if "all-MiniLM-L6-v2" in self.model_name
                    else self.model_name
                )
                logger.info(f"Loading FastEmbed (ONNX Runtime) embedding model: {fastembed_name}")
                self._model = TextEmbedding(model_name=fastembed_name)
                self._is_fastembed = True
                return self._model
            except Exception as e:
                logger.warning(f"FastEmbed initialization skipped ({e}); falling back to SentenceTransformers.")

        logger.info(f"Loading SentenceTransformers embedding model: {self.model_name}")
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        self._is_fastembed = False
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates a normalized embedding for a single string."""
        if text in self._cache:
            return self._cache[text]

        model = self._get_model()
        if self._is_fastembed:
            raw_vec = list(model.embed([text]))[0]
            norm = np.linalg.norm(raw_vec)
            vec = (raw_vec / norm).tolist() if norm > 0 else raw_vec.tolist()
        else:
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

        if self._is_fastembed:
            embeddings_list = list(model.embed(texts, batch_size=batch_size))
            result = []
            for raw_vec in embeddings_list:
                norm = np.linalg.norm(raw_vec)
                vec = (raw_vec / norm).tolist() if norm > 0 else raw_vec.tolist()
                result.append(vec)
            return result
        else:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.tolist()


# Global singleton instance
embedding_service = EmbeddingService()
