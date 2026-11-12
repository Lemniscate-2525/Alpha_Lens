# ADR 0001: Vector Storage Architecture — PostgreSQL + pgvector vs Dedicated Vector DB

## Status: ACCEPTED

## 1. Context & Problem Statement
AlphaLens requires storing and querying vector embeddings (e.g. 384-dimensional `all-MiniLM-L6-v2` or 768-dimensional embeddings) alongside rich relational metadata:
- SEC CIK identifier, Ticker (`NVDA`, `MSFT`), Fiscal Year (`2024`), Fiscal Quarter (`Q3`).
- Document type (`10-Q`, `10-K`, `Earnings_Call_Transcript`).
- Transcript metadata (`speaker_role`, `speaker_name`, `section`: `Prepared Remarks` vs `Analyst Q&A`).
- Access control list / tenant permissions and user audit logs.

We must choose between:
1. **Integrated Relational + Vector Storage**: PostgreSQL with `pgvector` extension.
2. **Dedicated Vector Database**: Standalone vector DB (e.g., Qdrant, Chroma, Pinecone, Milvus) paired with a separate PostgreSQL instance for relational metadata.

## 2. Alternatives Considered

### Alternative A: Dual-Database (Dedicated Vector DB + Relational DB)
- **Architecture**: PostgreSQL stores users, document metadata, audit logs. Qdrant/Pinecone stores chunk text and vectors.
- **Pros**:
  - Purpose-built vector indexing algorithms (e.g. HNSW, DiskANN) optimized for billion-scale vector search.
  - Independent scaling of vector compute from transactional database compute.
- **Cons**:
  - **The Dual-Write Problem**: A failure between saving metadata in Postgres and indexing vectors in Qdrant leads to orphaned vectors or dangling metadata references. Requires distributed two-phase commit (2PC) or an event-driven outbox pattern.
  - **Pre-filtering latency overhead**: Filtering by tenant ID, ticker, and quarter in a separate vector DB requires either maintaining duplicate relational indexes inside the vector DB or doing slow client-side joins.
  - **Operational complexity**: Managing two stateful clusters, backups, network boundaries, and monitoring pipelines.

### Alternative B: Integrated PostgreSQL with `pgvector`
- **Architecture**: A single PostgreSQL instance storing documents, chunks, embeddings (`VECTOR(384)` or `VECTOR(768)`), HNSW/IVFFlat indexes, and metadata tables.
- **Pros**:
  - **ACID Transactions**: Inserting a document, its parsed chunks, and their embeddings occurs within a single atomic database transaction. If embedding fails, the transaction rolls back cleanly.
  - **Single Query Pre-Filtering**: Perform relational filters (`WHERE ticker = 'NVDA' AND fiscal_quarter = 'Q3' AND speaker_role = 'CFO'`) combined with cosine vector similarity in one execution plan.
  - **Operational Simplicity**: One backup strategy (WAL archiving), one connection pool (PgBouncer), one security perimeter.
  - Scales easily to 1-5 million vectors with HNSW indexing, far exceeding our corpus need (~165,000 chunks).
- **Cons**:
  - High HNSW index build memory consumption on multi-million vector datasets (mitigated through proper `maintenance_work_mem` configuration).

## 3. Decision
We adopt a **Unified Storage Architecture**:
- In production: **PostgreSQL + pgvector** as the primary datastore for relational metadata, full-text search, and vector embeddings.
- In software design: We introduce a **VectorStoreAdapter** interface pattern. This allows the application to run seamlessly in development and testing environments using an embedded vector store (such as Chroma or in-memory FAISS) while retaining identical query semantics, without requiring developers to maintain a live PostgreSQL instance for unit tests.

## 4. Consequences & Trade-offs
- Eliminates dual-write bugs and distributed transaction overhead.
- Simplifies backup, migration, and local development.
- Bound to PostgreSQL resource limits; if vector corpus ever exceeds 20M vectors, we can partition by ticker or shard via Citus.
