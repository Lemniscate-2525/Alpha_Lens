"""
AlphaLens - Configuration Module
Centralized, validated configuration using Pydantic Settings.
"""
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Reranker Model
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER: bool = True
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
    ENABLE_SEMANTIC_CACHE: bool = True
    EXACT_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    SEMANTIC_CACHE_COSINE_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_MAX_ENTRIES: int = 1000

    # Rate Limiting & Concurrency
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_RPS: int = 25
    RATE_LIMIT_BURST: int = 50
    MAX_CONCURRENT_LLM_REQUESTS: int = 20

    # Circuit Breaker Parameters
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS: float = 15.0

    # LLM Settings
    LLM_PROVIDER: LLMProviderType = LLMProviderType.MOCK
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 800
    LLM_REQUEST_TIMEOUT_SECONDS: float = 20.0

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Security & API Auth
    API_KEY_HEADER_NAME: str = "X-API-Key"
    MASTER_API_KEY: str = "alpha-lens-master-key-prod-2025"


# Global singleton settings instance
settings = Settings()
