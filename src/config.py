"""
Centralized configuration management for ESGBench.
Validates environment variables and provides type-safe access to settings.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class PathConfig(BaseModel):
    """File and directory paths configuration."""

    # Base directories
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory for embeddings and tables")
    pdf_dir: Path = Field(default=Path("pdfs"), description="Directory containing downloaded PDFs")
    data_dir: Path = Field(default=Path("data"), description="Directory for datasets and outputs")

    # Data files
    docs_seed: Path = Field(default=Path("data/docs_seed.csv"), description="CSV file with document URLs")
    catalog: Path = Field(default=Path("data/esgbench_document_information.jsonl"), description="Document catalog file")
    qa_output: Path = Field(default=Path("data/esgbench_open_source.jsonl"), description="Generated QA pairs output")
    qa_numeric: Path = Field(default=Path("data/esgbench_open_source_num.jsonl"), description="Numeric QA pairs output")
    predictions: Path = Field(default=Path("data/esgbench_preds.jsonl"), description="RAG predictions output")

    # Cache files
    chunks_cache: Path = Field(default=Path("cache/chunks.json"), description="Text chunks cache")
    embeddings_cache: Path = Field(default=Path("cache/chunk_embeddings.npz"), description="Embeddings cache")
    faiss_index: Path = Field(default=Path("cache/faiss.index"), description="FAISS vector index")

    def ensure_directories(self):
        """Create directories if they don't exist."""
        for dir_path in [self.cache_dir, self.pdf_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class LLMConfig(BaseModel):
    """LLM and embedding model configuration."""

    # OpenAI API
    openai_api_key: str = Field(..., description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for QA generation and RAG")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model for vector search")

    # API limits and timeouts
    api_timeout: int = Field(default=40, ge=10, le=120, description="API request timeout in seconds")
    max_retries: int = Field(default=4, ge=1, le=10, description="Maximum API retry attempts")
    rate_limit_delay: float = Field(default=2.0, ge=0.1, le=10.0, description="Base delay for rate limiting")
    max_output_tokens: int = Field(default=200, ge=50, le=1000, description="Maximum output tokens for answers")

    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or not v.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format")
        return v


class ProcessingConfig(BaseModel):
    """Document processing and QA generation configuration."""

    # Text chunking
    chunk_max_chars: int = Field(default=1200, ge=500, le=3000, description="Maximum characters per text chunk")
    chunk_overlap: int = Field(default=200, ge=50, le=500, description="Character overlap between chunks")
    passage_chars: int = Field(default=900, ge=300, le=2000, description="Characters per passage for LLM input")

    # QA generation limits
    max_qas_per_doc: int = Field(default=16, ge=5, le=50, description="Maximum QA pairs per document")
    max_qas_per_passage: int = Field(default=1, ge=1, le=5, description="Maximum QA pairs per passage")
    max_passages_per_doc: int = Field(default=10, ge=5, le=30, description="Maximum passages to try per document")
    max_table_pages_per_doc: int = Field(default=8, ge=3, le=20, description="Maximum table pages per document")

    # Processing limits
    head_docs: int = Field(default=999999, ge=1, description="Maximum documents to process")
    include_tables: bool = Field(default=True, description="Whether to include table parsing")

    # Retrieval settings
    retrieve_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve for RAG")
    embedding_batch_size: int = Field(default=128, ge=32, le=512, description="Batch size for embedding generation")

    # Random seed for reproducibility
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducible results")


class ESGBenchConfig(BaseModel):
    """Master configuration class combining all settings."""

    paths: PathConfig = Field(default_factory=PathConfig)
    llm: LLMConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    @classmethod
    def from_env(cls) -> 'ESGBenchConfig':
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                embedding_model=os.getenv("EMB_MODEL", "text-embedding-3-large"),
                api_timeout=int(os.getenv("API_TIMEOUT", "40")),
                max_retries=int(os.getenv("MAX_RETRIES", "4")),
                rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "2.0")),
                max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "200")),
            ),
            processing=ProcessingConfig(
                chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                passage_chars=int(os.getenv("PASSAGE_CHARS", "900")),
                max_qas_per_doc=int(os.getenv("MAX_QAS_PER_DOC", "16")),
                max_qas_per_passage=int(os.getenv("MAX_QAS_PER_PASSAGE", "1")),
                max_passages_per_doc=int(os.getenv("MAX_PASSAGES_PER_DOC", "10")),
                max_table_pages_per_doc=int(os.getenv("MAX_TABLE_PAGES_PER_DOC", "8")),
                head_docs=int(os.getenv("HEAD_DOCS", "999999")),
                include_tables=os.getenv("INCLUDE_TABLES", "1").lower() not in ("0", "false"),
                retrieve_k=int(os.getenv("RETRIEVE_K", "5")),
                embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "128")),
                random_seed=int(os.getenv("SEED", "42")),
            )
        )

    def ensure_setup(self):
        """Ensure all required directories exist and configuration is valid."""
        self.paths.ensure_directories()

        # Validate OpenAI API key is accessible
        if not self.llm.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )


# Global configuration instance
_config: Optional[ESGBenchConfig] = None


def get_config() -> ESGBenchConfig:
    """Get the global configuration instance, creating it if necessary."""
    global _config
    if _config is None:
        _config = ESGBenchConfig.from_env()
        _config.ensure_setup()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Convenience functions for common config access
def get_paths() -> PathConfig:
    """Get path configuration."""
    return get_config().paths


def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_config().llm


def get_processing_config() -> ProcessingConfig:
    """Get processing configuration."""
    return get_config().processing