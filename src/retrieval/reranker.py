"""
AlphaLens - Cross-Encoder Neural Reranking
Applies joint cross-attention over (Query, Document) candidate pairs to maximize Precision@5.
"""
from typing import List, Optional
import numpy as np

from src.core.models import SearchResult
from src.config import settings
from src.observability.logging import logger


class NeuralReranker:
    """Reranks candidate search results using a deep Cross-Encoder model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.top_k = settings.RERANKER_TOP_K
        self.min_score_threshold = settings.RERANKER_MIN_SCORE_THRESHOLD
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading Cross-Encoder reranker: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        if not candidates:
            return []

        top_k = top_k or self.top_k
        pairs = [[query, c.content] for c in candidates]

        try:
            model = self._get_model()
            scores = model.predict(pairs, show_progress_bar=False)
            
            # Sigmoid normalization if scores are logits
            scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
        except Exception as e:
            logger.warning(f"Cross-Encoder inference failed ({e}), falling back to existing ranks.")
            return candidates[:top_k]

        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            score_val = round(float(score), 4)
            reranked_item = SearchResult(
                chunk_id=candidate.chunk_id,
                parent_chunk_id=candidate.parent_chunk_id,
                content=candidate.content,
                parent_content=candidate.parent_content,
                score=score_val,
                retrieval_type="cross_encoder_reranked",
                metadata=candidate.metadata
            )
            scored_candidates.append(reranked_item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]


neural_reranker = NeuralReranker()
