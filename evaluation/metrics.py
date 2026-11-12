"""
AlphaLens - Evaluation Metrics Module
Computes retrieval quality (Recall@K, MRR), context relevance, answer faithfulness, and latency.
"""
import re
from typing import List, Set
from src.core.models import SearchResult, EvaluationMetrics


class RAGEvaluator:
    """Evaluates RAG pipeline outputs against annotated golden benchmark targets."""

    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_results: List[SearchResult],
        ground_truth_doc_ids: List[str],
        is_unanswerable: bool
    ) -> dict:
        """Computes Recall@1, Recall@3, Recall@5, and Reciprocal Rank (RR)."""
        if is_unanswerable:
            return {"recall_1": 1.0, "recall_3": 1.0, "recall_5": 1.0, "rr": 1.0}

        if not retrieved_results:
            return {"recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0, "rr": 0.0}

        retrieved_ids = [r.chunk_id for r in retrieved_results]
        
        # Check matching by document_id or chunk_id prefix
        def is_relevant(cid: str) -> bool:
            return any(gt_id in cid for gt_id in ground_truth_doc_ids)

        hits_1 = int(any(is_relevant(cid) for cid in retrieved_ids[:1]))
        hits_3 = int(any(is_relevant(cid) for cid in retrieved_ids[:3]))
        hits_5 = int(any(is_relevant(cid) for cid in retrieved_ids[:5]))

        # Calculate Reciprocal Rank
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if is_relevant(cid):
                rr = 1.0 / rank
                break

        return {
            "recall_1": float(hits_1),
            "recall_3": float(hits_3),
            "recall_5": float(hits_5),
            "rr": rr,
        }

    @staticmethod
    def calculate_answer_faithfulness(
        answer: str,
        retrieved_chunks: List[SearchResult],
        is_unanswerable: bool
    ) -> float:
        """
        Extracts numbers, currency, and percentages from the answer and verifies
        that they exist in the retrieved context text (grounded factuality).
        """
        if is_unanswerable:
            if "INFORMATION_NOT_DISCLOSED_IN_FILINGS" in answer:
                return 1.0
            return 0.0

        if not retrieved_chunks:
            return 0.0

        combined_context = " ".join([c.content for c in retrieved_chunks]).lower()

        # Extract numeric tokens from answer (e.g. "$30.04", "75.7%", "122%")
        answer_tokens = re.findall(r"[\$\€\£]?\d+(?:\.\d+)?%?", answer.lower())
        if not answer_tokens:
            return 1.0

        grounded_count = sum(1 for token in answer_tokens if token in combined_context)
        return round(grounded_count / len(answer_tokens), 2)
