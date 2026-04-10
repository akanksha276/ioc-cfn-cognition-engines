# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import os


class Settings:
    # Use mocked DB (data layer) when set (e.g. http://localhost:8088); otherwise in-process mock repo
    CFN_URL: str | None = os.getenv("MOCKED_DB_BASE_URL") or os.getenv("CFN_URL")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o")
    LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL")
    EG_MAX_DEPTH: int = int(os.getenv("EG_MAX_DEPTH", "4"))
    EG_PATH_LIMIT: int = int(os.getenv("EG_PATH_LIMIT", "20"))
    # RAG vector retrieval when `rag_cache_layer` is injected (parallel to graph evidence); overridable per request via payload.metadata.rag
    EVIDENCE_RAG_TOP_K: int = int(os.getenv("EVIDENCE_RAG_TOP_K", "5"))
    EVIDENCE_RAG_TIMEOUT_SEC: float = float(os.getenv("EVIDENCE_RAG_TIMEOUT_SEC", "60"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    service_name: str = "Evidence Gathering Agent"


settings = Settings()
