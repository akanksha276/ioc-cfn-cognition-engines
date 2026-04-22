# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Environment configuration using Pydantic Settings.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

# Resolve .env from repo root (one level above ingestion/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore env vars for other services (e.g. MOCKED_DB_BASE_URL, NEGOTIATOR_STRATEGY)
    )

    # Service configuration
    service_name: str = Field(default="telemetry-extraction-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8086)
    log_level: str = Field(default="INFO")

    # LLM configuration (litellm provider/model format, e.g. openai/gpt-4o, anthropic/claude-sonnet-4-6)
    llm_model: str = Field(default="openai/gpt-4o")
    llm_api_key: Optional[str] = Field(default=None)
    llm_base_url: Optional[str] = Field(default=None)

    # Knowledge processing configuration
    enable_embeddings: bool = Field(default=True)
    enable_dedup: bool = Field(default=True)
    enable_rag_ingest: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.95)
    # Local path to granite-embedding-30m-english (overrides Hugging Face when set)
    embedding_model_path: Optional[str] = Field(default=None)

    # Remote similarity-search API (concept dedupe via /concepts/similarity-search)
    cfn_url: str | None = os.getenv("MOCKED_DB_BASE_URL") or os.getenv("CFN_URL")
    similarity_top_k: int = Field(default=5)
    similarity_metric: str = Field(default="l2")


# Singleton settings instance
settings = Settings()
