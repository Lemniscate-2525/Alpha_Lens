# ADR 0004: Asynchronous Ingestion, Hierarchical Chunking & Idempotency

## Status: ACCEPTED

## 1. Context & Problem Statement
SEC 10-Q filings and earnings call transcripts are large unstructured or semi-structured documents:
- Transcripts contain structured speaker turns (Executive Prepared Remarks vs unscripted Analyst Q&A).
- SEC filings contain structured GAAP tabular data embedded within narrative disclosures.
- Naive synchronous ingestion on an API request thread blocks worker processes for 5–20 seconds per document, easily leading to HTTP connection timeouts and thread starvation under load.
- Re-submitting an existing document or re-running an ingestion job must not duplicate chunks in the vector index.

## 2. Chunking Architecture: Parent-Child & Section-Aware Splitting

### Section & Speaker-Aware Pre-Processing
Transcripts are split by conversational turn:
- **Speaker Role Tagging**: `Executive (CEO/CFO)`, `Sell-Side Analyst`, `Operator`.
- **Section Classification**: `Prepared Remarks` vs `Analyst Q&A`.
- This metadata is injected directly into chunk attributes for targeted pre-retrieval filtering.

### Hierarchical (Parent-Child) Chunking
- **The Core Problem**: A small chunk (100–150 tokens) is optimal for fine-grained semantic vector matching, but lacks the broader context necessary for the LLM to synthesize a complete financial answer. A large chunk (1,000 tokens) dilutes vector similarity.
- **The Solution**:
  - **Child Chunk**: 150–250 tokens (used for vector indexing and high-precision embedding search).
  - **Parent Chunk**: 800–1,200 tokens (encompassing the full conversational exchange or complete financial disclosure paragraph).
  - During retrieval, child chunks are matched; before passing to the LLM, the parent chunk is resolved and injected into the prompt context.

### Financial Table Preservation
- Financial tables (Balance Sheet, Income Statement) are extracted and converted into Markdown pipe tables, with column headers explicitly prepended to row chunks to avoid losing unit/date context.

## 3. Asynchronous Execution & Idempotency Guarantee
1. **Background Job Queue**:
   - Ingestion requests return `HTTP 202 Accepted` with a unique `job_id`.
   - Work is processed asynchronously by background worker tasks.
2. **Content-Addressable Hashing (Deduplication)**:
   - Each document is hashed via `SHA-256(raw_content)`.
   - Each chunk receives a deterministic ID: `SHA-256(doc_id + chunk_index + chunk_content)`.
   - If a document is re-ingested with identical content, the worker detects the existing hash and skips re-embedding, guaranteeing strict idempotency.
   - If a document is updated, the previous version's chunks are atomically soft-deprecated or pruned in the vector store before new chunks are committed.

