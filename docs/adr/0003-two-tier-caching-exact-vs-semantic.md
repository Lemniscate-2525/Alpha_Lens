# ADR 0003: Two-Tier Caching Strategy — Exact Hash vs Semantic Vector Similarity

## Status: ACCEPTED

## 1. Context & Problem Statement
During quarterly earnings calls (e.g. NVIDIA or Apple post-market release), hundreds of financial analysts submit identical or near-identical queries within minutes:
- *"What was datacenter revenue guidance for Q4?"*
- *"What did NVIDIA guide for Q4 datacenter revenue?"*
- *"Datacenter revenue outlook Q4."*

Without caching, every request initiates dense retrieval, reranking, and an LLM generation call, consuming $\approx 2,645$ tokens and incurring $\approx 1.5 - 3.0$ seconds latency. At 80-100 peak QPS, this risks exhausting LLM API provider rate limits and inflating operational expenditure.

## 2. Architecture: Two-Tier Caching

```
User Query
    │
    ▼
[ Tier 1: Exact Cache (L1) ] ──Hit (Latency < 5ms)──> Return Answer
    │ Miss
    ▼
[ Tier 2: Semantic Cache (L2) ] ──Hit (Cosine >= 0.94, Latency < 40ms)──> Return Answer
    │ Miss
    ▼
[ Full Retrieval + Reranking + LLM Generation ]
    │
    ├─> Write to Tier 1 (Key: SHA256(normalized_query + metadata_filters))
    └─> Write to Tier 2 (Vector Index: Query Embedding + Answer)
```

### Tier 1: Exact Hash Cache (L1)
- **Mechanism**: Fast in-memory / Redis key-value lookup.
- **Key Formulation**: `SHA-256(canonical_query_str + sorted_metadata_filters)`.
- **Query Canonicalization**: Strip whitespace, lowercase, remove punctuation.
- **Hit Latency**: $< 5$ ms.
- **TTL**: 1 hour during active earnings calls; 24 hours for historical filings.

### Tier 2: Semantic Similarity Cache (L2)
- **Mechanism**: Dedicated vector index storing recent query embeddings paired with their generated answers.
- **Lookup**: Compute query vector embedding and perform nearest neighbor search over cached queries.
- **Threshold**: Cosine similarity $\ge 0.94$. (Tested empirically to prevent false positive matches on distinct financial quarters or metric types).
- **Metadata Constraint**: Cached query must share the exact same metadata filters (ticker, fiscal year, fiscal quarter).
- **Hit Latency**: $< 40$ ms (embedding generation + top-1 vector match).

### Cache Stampede Mitigation (Mutex / Singleflight Pattern)
- When 50 concurrent requests arrive for an uncached query, they must not all initiate 50 duplicate LLM generation calls.
- We implement an asynchronous **Singleflight / Distributed Lock** mechanism on the query key. The first request obtains the lock and computes the answer; remaining requests await the result and read directly from cache upon completion.

## 3. Decision
Adopt the Two-Tier Caching architecture with singleflight stampede prevention.
- Expected cache hit rate during peak earnings: $30 - 45\%$.
- Directly lowers P95/P99 latency and protects downstream LLM rate limits.

