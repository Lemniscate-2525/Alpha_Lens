"""
AlphaLens - Configuration Module
Centralized, validated configuration using Pydantic Settings.
"""
import os
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prevent huggingface from making slow network checks only if explicitly testing offline
if os.getenv("ENVIRONMENT") == "testing":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class VectorStoreType(str, Enum):
    MEMORY = "memory"
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class LLMProviderType(str, Enum):
    MOCK = "mock"  # Fast, deterministic for unit tests & offline CI
    OLLAMA = "ollama"  # Local zero-cost inference via Ollama
    TRITON = "triton"  # Low-latency TensorRT-LLM on Triton Inference Server
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Environment
    APP_NAME: str = "AlphaLens Financial RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: EnvironmentType = EnvironmentType.DEVELOPMENT
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"

    # Database & Storage
    VECTOR_STORE_TYPE: VectorStoreType = VectorStoreType.MEMORY
    DATABASE_URL: str = "sqlite+aiosqlite:///data/alphalens.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_SOCKET_TIMEOUT: float = 5.0
    REDIS_CONNECT_TIMEOUT: float = 5.0
    REDIS_KEY_PREFIX: str = "alpha_lens"

    # Embedding Model
    USE_FASTEMBED: bool = True
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Reranker Model
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER: bool = False
    RERANKER_TOP_K: int = 5
    RERANKER_MIN_SCORE_THRESHOLD: float = 0.20

    # Retrieval Parameters
    DENSE_TOP_K: int = 25
    SPARSE_TOP_K: int = 25
    FUSED_TOP_K: int = 20
    RRF_K_PARAMETER: int = 60

    # Chunking Parameters
    PARENT_CHUNK_SIZE: int = 1000
    PARENT_CHUNK_OVERLAP: int = 150
    CHILD_CHUNK_SIZE: int = 250
    CHILD_CHUNK_OVERLAP: int = 40

    # Caching Parameters
    ENABLE_EXACT_CACHE: bool = True
    EXACT_CACHE_ENABLED: bool = True
    EXACT_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    EXACT_CACHE_PREFIX: str = "alpha_lens:cache:exact"
    ENABLE_SEMANTIC_CACHE: bool = True
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_COSINE_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_TTL_SECONDS: int = 3600
    SEMANTIC_CACHE_PREFIX: str = "alpha_lens:cache:semantic"
    SEMANTIC_CACHE_MAX_ENTRIES: int = 1000

    # Singleflight Distributed Mutex Parameters
    SINGLEFLIGHT_LEASE_MS: int = 30000  # 30-second lock lease
    SINGLEFLIGHT_WAIT_TIMEOUT_SECONDS: float = 35.0  # Max waiting period for waiters
    SINGLEFLIGHT_POLL_INTERVAL_MS: int = 50

    # Rate Limiting & Concurrency
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_CAPACITY: int = 50
    RATE_LIMIT_BURST: int = 50
    RATE_LIMIT_REFILL_RATE: float = 25.0
    RATE_LIMIT_RPS: int = 25
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_KEY_PREFIX: str = "alpha_lens:ratelimit"
    RATE_LIMIT_FAIL_CLOSED: bool = True
    MAX_CONCURRENT_LLM_REQUESTS: int = 20

    # Asynchronous Ingestion Queue & Worker Parameters
    INGESTION_QUEUE_NAME: str = "alpha_lens:queue:ingestion"
    INGESTION_DEAD_LETTER_QUEUE: str = "alpha_lens:queue:ingestion:dead_letter"
    INGESTION_CONSUMER_GROUP: str = "alpha_lens_workers"
    INGESTION_MAX_RETRIES: int = 3
    INGESTION_WORKER_CONCURRENCY: int = 2
    INGESTION_JOB_TIMEOUT_SECONDS: int = 1800


    # Circuit Breaker Parameters
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: float = 15.0

    # LLM Settings
    LLM_PROVIDER: LLMProviderType = LLMProviderType.OLLAMA
    LLM_MODEL_NAME: str = "qwen3:1.7b"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 800
    LLM_REQUEST_TIMEOUT_SECONDS: float = 60.0

    # Local Ollama Settings (Zero-Cost Local GPU/CPU)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "qwen3:1.7b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Triton Inference Server / TensorRT-LLM Cluster
    TRITON_SERVER_URL: str = "http://localhost:8000"
    TRITON_LLM_MODEL_NAME: str = "tensorrt_llm"
    TRITON_RERANKER_MODEL_NAME: str = "cross_encoder_trt"

    # API Keys (Optional fallbacks)
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Security & API Auth
    API_KEY_HEADER_NAME: str = "X-API-Key"
    MASTER_API_KEY: str = "alpha-lens-master-key-prod-2025"


# Global singleton settings instance
settings = Settings()
