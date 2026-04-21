# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import Request

from .api.schemas import ReasonerCognitionRequest
from .data.mock_repo import MockDataRepository
from .data.http_repo import HttpDataRepository
from .config.settings import settings


def _repository(
    _request: Request,
    workspace_id: Optional[str],
    mas_id: Optional[str],
):
    # Graph (neighbors, paths, concepts/by_ids, etc.) uses HTTP when the data layer URL is set.
    # In-memory cache_layer on app.state is injected into ConceptRepository for similar-concept search.
    if settings.CFN_URL:
        return HttpDataRepository(
            base_url=settings.CFN_URL,
            workspace_id=workspace_id,
            mas_id=mas_id,
        )
    return MockDataRepository()


def get_repository_for_reasoning(request: Request, req: ReasonerCognitionRequest):
    """
    Used by POST /reasoning/evidence only.
    Scopes HttpDataRepository to CFN paths using header.workspace_id and header.mas_id.
    """
    return _repository(request, req.header.workspace_id, req.header.mas_id)


def get_repository(request: Request):
    """
    Used by standalone /graph/* proxy routes (no ReasonerCognitionRequest body).
    HttpDataRepository uses legacy /api/graph/... (no workspace/mas in path).
    """
    return _repository(request, None, None)


def get_cache_layer(request: Request):
    """Return the shared in-memory CachingLayer when running under the unified app (Option A)."""
    return getattr(request.app.state, "cache_layer", None)


def get_rag_cache_layer(request: Request):
    """Optional second cache (vector index) for RAG chunks; unified app may set app.state.rag_cache_layer."""
    return getattr(request.app.state, "rag_cache_layer", None)
