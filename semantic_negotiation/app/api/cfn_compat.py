# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""CFN-compatible semantic negotiation API routes.

Exposes:
    POST /api/workspaces/{workspace_id}/multi-agentic-systems/{mas_id}/semantic-negotiation/start
    POST /api/workspaces/{workspace_id}/multi-agentic-systems/{mas_id}/semantic-negotiation/decide

These routes mirror the Go ``ioc-cfn-svc`` binary's API so that evaluation
scripts (e.g. ``test_via_cfn_service.py``) can point to our Python gateway
(port 9004) instead of the Go binary and get correct SAO consensus semantics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel

from ..agent.semantic_neg_ce import NegotiationAction, NegotiationCognitionEngine
from ..agent.semantic_negotiation import (
    SemanticNegotiationInputError,
    SemanticNegotiationSessionNotFoundError,
)
from ..dependencies import get_negotiation_cognition_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cfn-compat"])


# ── Request / response models ─────────────────────────────────────────────────


class _Agent(BaseModel):
    id: str
    name: str


class StartRequest(BaseModel):
    session_id: str
    content_text: str
    agents: List[_Agent]
    n_steps: Optional[int] = 20


class AgentReply(BaseModel):
    agent_id: str
    action: Literal["accept", "reject", "counter_offer"]
    offer: Optional[Dict[str, Any]] = None


class DecideRequest(BaseModel):
    session_id: str
    agent_replies: List[AgentReply]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/workspaces/{workspace_id}/multi-agentic-systems/{mas_id}/semantic-negotiation/start",
    status_code=status.HTTP_200_OK,
)
async def start_negotiation(
    req: StartRequest = Body(...),
    workspace_id: str = Path(...),
    mas_id: str = Path(...),
    engine: NegotiationCognitionEngine = Depends(get_negotiation_cognition_engine),
) -> Dict[str, Any]:
    """Start a semantic negotiation session (CFN-compatible API)."""
    payload = {
        "session_id": req.session_id,
        "content_text": req.content_text,
        "agents": [a.model_dump() for a in req.agents],
        "n_steps": req.n_steps,
        "workspace_id": workspace_id,
        "mas_id": mas_id,
    }
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            return await engine.run(NegotiationAction.INITIATE, payload)
        except SemanticNegotiationInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "cfn-compat start_negotiation attempt %d/3 failed: %s", attempt, exc
            )
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)
    logger.exception("cfn-compat start_negotiation failed after 3 attempts", exc_info=last_exc)
    raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post(
    "/workspaces/{workspace_id}/multi-agentic-systems/{mas_id}/semantic-negotiation/decide",
    status_code=status.HTTP_200_OK,
)
async def decide_negotiation(
    req: DecideRequest = Body(...),
    workspace_id: str = Path(...),
    mas_id: str = Path(...),
    engine: NegotiationCognitionEngine = Depends(get_negotiation_cognition_engine),
) -> Dict[str, Any]:
    """Advance a semantic negotiation session (CFN-compatible API)."""
    try:
        return await engine.run(
            NegotiationAction.DECIDE,
            {
                "session_id": req.session_id,
                "workspace_id": workspace_id,
                "mas_id": mas_id,
                "agent_replies": [
                    {**reply.model_dump(), "participant_id": reply.agent_id}
                    for reply in req.agent_replies
                ],
            },
        )
    except SemanticNegotiationSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except SemanticNegotiationInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unhandled error in cfn-compat decide_negotiation")
        raise HTTPException(status_code=500, detail="Internal Server Error")
