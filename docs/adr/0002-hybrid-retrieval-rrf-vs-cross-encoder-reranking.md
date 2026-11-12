# ADR 0002: Hybrid Retrieval Strategy — Dense + Sparse (BM25) with Reciprocal Rank Fusion & Cross-Encoder Reranking

## Status: ACCEPTED

## 1. Context & Problem Statement
In financial transcripts and regulatory filings, user queries fall into two distinct retrieval categories:
1. **Semantic Conceptual Queries**: e.g., *"How is management addressing GPU supply constraints and foundry capacity bottlenecks?"*
2. **Entity & Metric Specific Queries**: e.g., *"What was GAAP diluted EPS of $0.78 compared to $0.82 guidance in Q2?"* or *"Did CFO Colette Kress comment on gross margin trajectory?"*

Standard dense vector embeddings (e.g., MiniLM, OpenAI text-embedding-3-small) excel at conceptual similarity but perform poorly on exact entities, ticker symbols (`NVDA`), precise dollar amounts (`$30.04 billion`), and rare executive surnames. Sparse lexical search (BM25) excels at exact token matching but fails when queries use paraphrasing or synonyms.

## 2. Alternatives Considered

### Alternative A: Dense Vector Retrieval Only
- **Pros**: Fast, single vector index query, simple architecture.
- **Cons**: Severe failure mode on exact numerical metrics, ticker identifiers, and specific executive names. Retrieval recall drops significantly on metric-heavy queries.

### Alternative B: Sparse BM25 Retrieval Only
- **Pros**: Fast, exact keyword matching, no embedding model latency during search.
- **Cons**: Zero semantic awareness; fails whenever the user's phrasing diverges from the transcript verbatim text (vocabulary mismatch problem).

### Alternative C: Hybrid Retrieval with Linear Weighted Score Combination
- Normalize scores: $Score_{hybrid} = \alpha \cdot Score_{dense} + (1 - \alpha) \cdot Score_{bm25}$.
- **Cons**: Dense cosine similarities ($[0, 1]$) and BM25 scores ($[0, \infty)$) have completely different underlying score distributions. Linear normalization requires fragile manual tuning of $\alpha$ and calibrating min-max bounds that shift across corpora.

### Alternative D: Two-Stage Retrieval: Hybrid Search via Reciprocal Rank Fusion (RRF) + Cross-Encoder Reranker
- **Stage 1 (Candidate Generation)**:
  - Retrieve top-$K_{dense}$ (25 chunks) from Dense Vector index.
  - Retrieve top-$K_{sparse}$ (25 chunks) from BM25 index.
  - Fuse ranks using Reciprocal Rank Fusion (RRF) with smoothing parameter $k=60$:
    $$RRF\_Score(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + rank_m(d)}$$
  - Select top-20 fused candidates.
- **Stage 2 (Precision Neural Reranking)**:
  - Pass the query and top-20 candidate passages to a Cross-Encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or BGE-reranker).
  - The cross-encoder performs joint cross-attention over $(Query, Document)$ tokens, producing a calibrated relevance score.
  - Select top-5 chunks for context injection.

## 3. Decision
We adopt **Alternative D (Two-Stage Hybrid + RRF + Cross-Encoder Reranking)**.
- RRF provides rank-based calibration without fragile score normalization.
- The Cross-Encoder acts as a precision filter, eliminating false positives from the dense/sparse stages and boosting Recall@5 and MRR dramatically.

## 4. Latency & Resource Trade-offs
- Dense search latency: $\approx 25$ ms.
- BM25 search latency: $\approx 10$ ms.
- RRF fusion latency: $\approx 2$ ms.
- Cross-Encoder scoring (20 pairs): $\approx 45$ ms (batched inference).
- Total retrieval pipeline latency: $\approx 82$ ms, well within our P95 retrieval SLO target of $\le 350$ ms.
