# AlphaLens: Workload Modeling, Capacity Planning & Service Level Objectives (SLOs)

## 1. System Target & Audience Definition

* **Domain**: Wall Street Earnings & Executive Q&A Intelligence (`AlphaLens`)
* **Target Audience**: 10,000 Registered Financial Analysts, Portfolio Managers, and Equity Research Associates.
* **Workload Characteristics**: Bimodal financial market activity. Sustained baseline during open market hours with extreme burst spikes during earnings season release windows (4:00 PM – 5:30 PM EST).

---

## 2. Workload Derivation: From 10,000 Users to Queries Per Second (QPS)

A common mistake in systems design is equating registered accounts with concurrent active traffic. Below is the mathematically grounded workload breakdown.

### 2.1 User Activity Breakdown

| Metric | Normal Market Day | Earnings Season Release Peak | Formula / Basis |
| :--- | :--- | :--- | :--- |
| **Registered Users** | 10,000 | 10,000 | Total addressable user database |
| **Daily Active Users (DAU)** | 2,000 (20%) | 5,000 (50%) | Active users logging in per day |
| **Active Query Window** | 8.0 hours (9 AM – 5 PM EST) | 1.5 hours (4 PM – 5:30 PM EST) | Window where 80% of volume concentrates |
| **Queries per Active User** | 12 queries / day | 25 queries / burst window | Financial research session intensity |
| **Total Daily Queries** | 24,000 queries / day | 125,000 queries / day | $DAU \times QueriesPerUser$ |
| **Average QPS (Off-Peak)** | **0.83 QPS** | **4.34 QPS** | Spread across 8-hour trading day |
| **Peak Concurrent Users** | 120 users | **850 – 1,000 users** | Concurrent active browser/API sessions |
| **Peak Query Throughput (QPS)** | **15 QPS** | **85 – 100 QPS** | Burst factor $F_{peak} \approx 8.5\times$ |

### 2.2 Token Budget & Payload Metrics

* **User Query Size**:
  * Average: 25 tokens (e.g., *"What did CFO Colette Kress say about Blackwell gross margins in Q3?"*)
  * 99th Percentile: 80 tokens (multi-part comparative queries).
* **Retrieved Context Window**:
  * Top-$K$ Chunks retrieved: $K_{dense}=25, K_{sparse}=25 \rightarrow K_{fused}=20 \rightarrow K_{reranked}=5$.
  * Average Chunk Size: 350 tokens.
  * Injected Prompt Context: $5 \times 350 = 1,750$ tokens.
  * System Prompt + Few-Shot Citations: 450 tokens.
  * Total Prompt Tokens: $\approx 2,225$ tokens.
* **Output / Generation Size**:
  * Executive Summary + Structured Bullet Points: 300 tokens.
  * Citing exact quote, speaker, timestamp, and transcript section: 120 tokens.
  * Total Completion Tokens: $\approx 420$ tokens.
* **Total Tokens per End-to-End Query**: $\approx 2,645$ tokens.

### 2.3 Document Ingestion & Corpus Scale

* **Coverage Universe**: S&P 500 Tech & Financial Giants ($\approx 150$ high-priority tickers).
* **Document Volume**:
  * 4 quarterly earnings call transcripts per company per year = 600 transcripts.
  * 4 SEC Form 10-Q/10-K filings per company per year = 600 filings.
  * 8-K material updates = $\approx 300$ filings.
  * Total Annual Ingested Documents: **1,500 documents/year**.
* **Chunk Volume**:
  * Average transcript length: 8,500 words ($\approx 11,500$ tokens) $\rightarrow \approx 35$ parent chunks / 110 child chunks.
  * Total Chunks in Active Vector Space: $\approx 165,000$ chunks.
* **Ingestion Burst Spike**:
  * On "Super Thursday" of earnings week, up to 25 transcripts drop within 90 minutes.
  * Decoupling Requirement: Ingestion must be handled asynchronously via worker queues with idempotency keys (SHA-256 content hashes) to prevent degrading query latency.

---

## 3. Service Level Objectives (SLOs) & Targets

We define measurable, verifiable Service Level Indicators (SLIs) and Service Level Objectives (SLOs).

### 3.1 Latency SLOs

| Stage | Metric | Target (P50) | Target (P95) | Target (P99) |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Cache Hit** | Cache Latency | $\le 10$ ms | $\le 25$ ms | $\le 50$ ms |
| **Semantic Cache Hit** | Vector Similarity + Fetch | $\le 30$ ms | $\le 60$ ms | $\le 100$ ms |
| **Hybrid Retrieval** | Dense + BM25 + RRF | $\le 80$ ms | $\le 180$ ms | $\le 300$ ms |
| **Cross-Encoder Rerank** | Score top-20 to top-5 | $\le 60$ ms | $\le 120$ ms | $\le 200$ ms |
| **LLM Time to First Token (TTFT)** | Streaming first chunk | $\le 450$ ms | $\le 850$ ms | $\le 1,400$ ms |
| **Full End-to-End Query (Cold)** | Uncached query completion | $\le 1,800$ ms | $\le 3,500$ ms | $\le 5,000$ ms |

### 3.2 Reliability & Availability SLOs

* **Service Availability**: $\ge 99.9\%$ (less than 43.8 minutes of downtime per month).
* **Error Rate Budget**:
  * Normal Traffic ($\le 25$ QPS): $< 0.1\%$ 5xx errors.
  * Extreme Peak ($\ge 80$ QPS): System must shed load gracefully with `HTTP 429 Too Many Requests` (with `Retry-After` headers), keeping unhandled `HTTP 500 Internal Server Error` below $0.05\%$.
* **Circuit Breaker Trip Threshold**: 5 consecutive downstream LLM timeouts or $50\%$ failure rate over a 10-second rolling window triggers fallback to cached / degraded retrieval mode.

### 3.3 Quality & Accuracy SLOs

* **Retrieval Recall@5**: $\ge 0.85$ (on the curated financial golden benchmark).
* **Mean Reciprocal Rank (MRR)**: $\ge 0.75$.
* **Faithfulness (Factuality)**: $\ge 0.95$ (zero unbacked hallucinations; every numeric metric verified against cited text).
* **Unanswerable Query Detection**: $\ge 0.90$ precision (the system must explicitly state `"Information not disclosed in official filings"` rather than hallucinate speculative executive commentary).

### 3.4 Financial & Resource Cost Model

* At peak burst ($100$ QPS uncached for 1 hour $= 360,000$ queries):
  * Raw LLM API cost without caching: $360,000 \times 2,645 \text{ tokens} \approx 952\text{M tokens}$.
  * With Two-Tier Caching ($35\%$ exact/semantic cache hit rate during earnings calls):
    * Actual LLM queries: $234,000$ queries.
  * Direct dollar cost reduction: $\approx 35\%$ cost reduction.
* Target cost per served query: $\le \$0.0035$.

