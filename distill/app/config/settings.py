# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class Settings:
    DATA_LAYER_BASE_URL: str | None = (
        os.getenv("MOCKED_DB_BASE_URL") or os.getenv("DATA_LAYER_BASE_URL") or os.getenv("CFN_URL") or ""
    ).strip() or None
    # Optional full URL for graph mutation; if unset, URL is derived from DATA_LAYER_BASE_URL + CFN graph/update path.
    CFN_MUTATION_APPLY_URL: str | None = (os.getenv("CFN_MUTATION_APPLY_URL") or "").strip() or None
    # Path segment after ``.../graph/`` for CFN batch update (default: ``update``).
    CODI_GRAPH_UPDATE_SEGMENT: str = (os.getenv("CODI_GRAPH_UPDATE_SEGMENT") or "update").strip().strip("/") or "update"
    CODI_GRAPH_UPDATE_DESCRIPTOR: str = (os.getenv("CODI_GRAPH_UPDATE_DESCRIPTOR") or "Cognition Distillation").strip()
    # §3 — server-side distillation controls (async /distillation/run; clients cannot override).
    CODI_MIN_EDGES: int = int(os.getenv("CODI_MIN_EDGES", "10"))
    # CFN distillation/read filters (see ioc-cfn-svc graph/distillation/read).
    CODI_DISTILL_STATUS_FILTER: str = os.getenv("CODI_DISTILL_STATUS_FILTER", "")
    CODI_RETURN_MISSING_DISTILL_STATUS: bool = _env_bool("CODI_RETURN_MISSING_DISTILL_STATUS", True)
    CODI_MAX_RELATIONS_PER_BATCH: int = int(os.getenv("CODI_MAX_RELATIONS_PER_BATCH", "10"))
    DISTILLATION_MODE: str = (os.getenv("DISTILLATION_MODE", "Summary") or "Summary").strip() or "Summary"
    # Distillation LLM: LiteLLM via CODI_LLM_* (falls back to LLM_* like Evidence).
    CODI_LLM_MODEL: str = (os.getenv("CODI_LLM_MODEL") or os.getenv("LLM_MODEL") or "openai/gpt-4o").strip()
    CODI_LLM_API_KEY: str | None = (os.getenv("CODI_LLM_API_KEY") or os.getenv("LLM_API_KEY") or "").strip() or None
    CODI_LLM_BASE_URL: str | None = (os.getenv("CODI_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip() or None
    CODI_RAG_TOP_K: int = int(os.getenv("CODI_RAG_TOP_K", "5"))
    CODI_RAG_TIMEOUT_SEC: float = float(os.getenv("CODI_RAG_TIMEOUT_SEC", "60"))
    CODI_LLM_MAX_RETRIES: int = int(os.getenv("CODI_LLM_MAX_RETRIES", "5"))
    CODI_LLM_MAX_BACKOFF_SEC: int = int(os.getenv("CODI_LLM_MAX_BACKOFF_SEC", "16"))
    # Graph HTTP path segment for async distillation read (appended under workspace/mas graph prefix).
    CODI_GRAPH_DISTILLATION_READ_SEGMENT: str = (os.getenv("CODI_GRAPH_DISTILLATION_READ_SEGMENT") or "distillation/read").strip().strip("/")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    service_name: str = "Cognition Distillation (CoDi)"


settings = Settings()
