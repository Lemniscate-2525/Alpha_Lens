"""
AlphaLens - Domain Models
Pydantic schemas representing domain entities across API, Storage, and Pipeline layers.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    EARNINGS_CALL_TRANSCRIPT = "earnings_call_transcript"
    SEC_10Q = "sec_10q"
    SEC_10K = "sec_10k"
    SEC_8K = "sec_8k"


class SpeakerRole(str, Enum):
    CEO = "CEO"
    CFO = "CFO"
    EXECUTIVE = "Executive"
    ANALYST = "Analyst"
    OPERATOR = "Operator"
    UNKNOWN = "Unknown"


class TranscriptSection(str, Enum):
    PREPARED_REMARKS = "PREPARED_REMARKS"
    ANALYST_QA = "ANALYST_QA"
    DISCLOSURE = "DISCLOSURE"
    GENERAL = "GENERAL"


class Document(BaseModel):
    document_id: str
    ticker: str
    company_name: str
    fiscal_quarter: str  # e.g., "Q2", "Q3"
    fiscal_year: str     # e.g., "2024", "2025"
    report_date: str     # e.g., "2024-08-28"
    document_type: DocumentType
    raw_text: str
    content_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_chunk_id: Optional[str] = None
    chunk_index: int
    content: str
    content_hash: str
    token_count: int
    is_parent: bool = False
    
    # Financial Metadata for pre-retrieval filtering
    ticker: str
    fiscal_quarter: str
    fiscal_year: str
    document_type: DocumentType
    speaker_name: Optional[str] = None
    speaker_role: Optional[SpeakerRole] = None
    section: TranscriptSection = TranscriptSection.GENERAL
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    content: str
    parent_content: Optional[str] = None
    score: float
    retrieval_type: str  # "dense", "sparse", "fused", "reranked"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    ticker: str
    fiscal_quarter: str
    fiscal_year: str
    speaker_name: Optional[str] = None
    speaker_role: Optional[str] = None
    section: Optional[str] = None
    quote_snippet: str
    confidence_score: float = 1.0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Financial research query")
    ticker: Optional[str] = Field(None, description="Filter by ticker symbol (e.g. NVDA, MSFT)")
    fiscal_quarter: Optional[str] = Field(None, description="Filter by quarter (e.g. Q2, Q3)")
    fiscal_year: Optional[str] = Field(None, description="Filter by year (e.g. 2024, 2025)")
    speaker_role: Optional[SpeakerRole] = Field(None, description="Filter by executive role (e.g. CFO, CEO)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to use")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder neural reranking")
    use_semantic_cache: bool = Field(default=True, description="Enable two-tier exact + semantic caching")
    stream: bool = Field(default=False, description="Stream LLM generation response")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)
    latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    cache_hit_type: Optional[str] = None  # None, "exact", "semantic"
    is_unanswerable: bool = False
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    trace_id: Optional[str] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED_EXISTING = "skipped_existing"


class IngestionJob(BaseModel):
    job_id: str
    document_id: str
    ticker: str
    status: JobStatus = JobStatus.PENDING
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class EvaluationQuery(BaseModel):
    query_id: str
    query: str
    ticker: str
    expected_answer: str
    ground_truth_chunk_ids: List[str]
    is_unanswerable: bool = False
    category: str = "factual"  # "factual", "comparative", "unanswerable", "metric_lookup"


class EvaluationMetrics(BaseModel):
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    context_precision: float
    context_recall: float
    answer_faithfulness: float
    unanswerable_detection_accuracy: float
    avg_latency_ms: float
    total_evaluated: int

