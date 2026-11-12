# ADR 0005: Unanswerable Query Handling, Citation Verification & Hallucination Mitigation

## Status: ACCEPTED

## 1. Context & Problem Statement
In financial intelligence, hallucinations are fatal to credibility. If an analyst queries:
- *"What did NVIDIA's CFO state regarding gross margin guidance for fiscal year 2029?"* (which has never been disclosed), or
- *"Did Apple announce an acquisition of a semiconductor foundry in Q3?"*

A standard LLM with unconstrained RAG will often fabricate plausible-sounding speculative statements or conflate past historical years.
The system must guarantee:
1. Every claim in the generated answer must be attributed to a specific retrieved chunk ID and verbatim quote.
2. If the retrieved context does not contain sufficient factual evidence to answer the query, the system must trigger an explicit structured fallback (`"NOT_DISCLOSED"`) rather than guess.

## 2. Mitigation Strategy

### Step 1: Retrieval Confidence & Relevance Thresholding
- After Cross-Encoder neural reranking, the top passage relevance scores are evaluated.
- If the maximum reranker score falls below the empirical relevance threshold $\tau_{relevance} = 0.25$, retrieval confidence is flagged as insufficient.

### Step 2: Prompt Contract & Grounded Synthesis Guardrails
- The LLM prompt is structured with explicit negative constraints:
  - *"Only use facts directly stated in the context."*
  - *"If the provided context does not explicitly contain the answer, reply with the exact phrase: 'INFORMATION_NOT_DISCLOSED_IN_FILINGS'."*
  - *"For every factual statement, append an inline citation tag: `[Chunk-ID]`."*

### Step 3: Post-Generation Citation Verification
- A verification layer parses generated inline citation tags (`[Chunk-1]`, `[Chunk-2]`).
- For each cited chunk, it validates whether key numerical metrics and entity strings in the sentence actually appear in the referenced chunk text.
- If an ungrounded citation is detected, the statement is flagged for re-generation or marked with an unverified claim alert.

## 3. Decision
Adopt strict three-stage hallucination defense:
1. Pre-generation relevance gating based on reranker scores.
2. Prompt contract with explicit fallback instruction.
3. Post-generation citation parser and verification check.
