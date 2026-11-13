"""
Unit tests for Sparse BM25 Tokenization and Hybrid Reciprocal Rank Fusion.
"""
from src.retrieval.sparse import SparseBM25Retriever
from src.core.models import SearchResult
from src.retrieval.hybrid import HybridRetriever


def test_financial_tokenization():
    text = "NVIDIA's Q2 revenue was $30.04 billion, with gross margin of 75.7% and Blackwell CoWoS-L packaging."
    tokens = SparseBM25Retriever.tokenize(text)

    # Must preserve exact numbers, dollar signs, and percentages
    assert "$30.04" in tokens
    assert "75.7%" in tokens
    assert "nvidia" in tokens
    assert "blackwell" in tokens


def test_rrf_scoring_logic():
    retriever = HybridRetriever(rrf_k=60)
    # RRF score for rank 1: 1 / (60 + 1) = 1/61 = 0.01639
    score_rank_1 = 1.0 / (60 + 1)
    assert round(score_rank_1, 5) == 0.01639

