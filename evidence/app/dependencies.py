# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import Request

from .api.schemas import ReasonerCognitionRequest
from .config.settings import settings
from .data.http_repo import HttpDataRepository
from .data.mock_repo import MockDataRepository


def _repository(
    _request: Request,
    workspace_id: Optional[str],
    mas_id: Optional[str],
    agent_id: Optional[str],
):
    # Graph (neighbors, paths, concepts/by_ids, etc.) and similarity search use HTTP
    # when the data layer URL is set.
    if settings.CFN_URL:
        return HttpDataRepository(
            base_url=settings.CFN_URL,
            workspace_id=workspace_id,
            mas_id=mas_id,
            agent_id=agent_id,
        )
    return MockDataRepository()


def get_repository_for_reasoning(request: Request, req: ReasonerCognitionRequest):
    """
    Used by POST /reasoning/evidence only.
    Scopes HttpDataRepository to CFN paths using header.workspace_id and header.mas_id.
    """
    return _repository(
        request,
        req.header.workspace_id,
        req.header.mas_id,
        req.header.agent_id,
    )


def get_repository(request: Request):
    """
    Used by standalone /graph/* proxy routes (no ReasonerCognitionRequest body).
    HttpDataRepository uses legacy /api/graph/... (no workspace/mas in path).
    """
    return _repository(request, None, None, None)


def get_evidence_cognition_engine(request: Request) -> "EvidenceCognitionEngine":
    """
    Per-request factory for :class:`~app.agent.evidence_ce.EvidenceCognitionEngine`.

    Creates the engine with an unscoped repository by default (for graph/* routes).
    The ``/reasoning/evidence`` route overrides ``engine._repo`` with the
    workspace/MAS-scoped repository after injection.
    """
    from common.cognition_engine import ModelConfig

    from .agent.evidence_ce import EvidenceCognitionEngine

    cfg = ModelConfig(
        llm_model=settings.LLM_MODEL,
        llm_api_key=settings.LLM_API_KEY,
        llm_base_url=settings.LLM_BASE_URL,
    )
    repo = _repository(request, None, None, None)
    return EvidenceCognitionEngine(repo_adapter=repo, model_config=cfg)
