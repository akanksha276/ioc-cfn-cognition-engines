# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
API routes for the Semantic Negotiation Agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is on sys.path so `protocol.sstp` is importable
# regardless of which directory uvicorn is launched from.
_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import (
    Origin,
    PolicyLabels,
    Provenance,
)  # noqa: E402
from protocol.sstp.negotiate import (
    NegotiateSemanticContext,  # noqa: E402
    dump_negotiate_message_json,  # noqa: E402
)

from ..agent.semantic_neg_ce import NegotiationAction, NegotiationCognitionEngine
from ..agent.semantic_negotiation import (
    SemanticNegotiationInputError,
    SemanticNegotiationSessionNotFoundError,
)
from ..config.settings import settings
from ..dependencies import get_negotiation_cognition_engine
from .schemas import (
    InitiateResponse,
    NegotiationError,
    NegotiationHeader,
    NegotiationTrace,
    RoundOffer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["negotiation"])


# ============== Input validation ==============


def _validate_request_scope(workspace_id: str, mas_id: str) -> None:
    """Validate workspace/MAS scope carried in the SSTP origin envelope.

    Raises:
        SemanticNegotiationInputError: If either identifier is missing or blank.
    """
    if not workspace_id or not str(workspace_id).strip():
        raise SemanticNegotiationInputError(
            "origin.tenant_id (workspace_id) is required and must be a non-empty string."
        )
    if not mas_id or not str(mas_id).strip():
        raise SemanticNegotiationInputError(
            "origin.actor_id (mas_id) is required and must be a non-empty string."
        )


def _validate_initiate_payload(payload: Dict[str, Any], session_id: str) -> None:
    """Validate the payload for POST /api/negotiate/initiate.

    Raises:
        SemanticNegotiationInputError: On any constraint violation.
    """
    if not session_id or not session_id.strip():
        raise SemanticNegotiationInputError(
            "session_id is required and must be a non-empty string."
        )

    content_text: str = payload.get("content_text", "")
    if not content_text or not content_text.strip():
        raise SemanticNegotiationInputError(
            "payload.content_text is required and must be a non-empty string."
        )

    agents_raw = payload.get("agents")
    if agents_raw is None:
        raise SemanticNegotiationInputError("payload.agents is required.")
    if not isinstance(agents_raw, list) or len(agents_raw) < 2:
        raise SemanticNegotiationInputError(
            f"payload.agents must be a list of at least 2 agents, got {len(agents_raw) if isinstance(agents_raw, list) else type(agents_raw).__name__!r}."
        )
    seen_ids: set = set()
    for i, agent in enumerate(agents_raw):
        if not isinstance(agent, dict):
            raise SemanticNegotiationInputError(
                f"payload.agents[{i}] must be a dict with 'id' and 'name' keys."
            )
        agent_id = agent.get("id", "")
        agent_name = agent.get("name", "")
        if not agent_id or not str(agent_id).strip():
            raise SemanticNegotiationInputError(
                f"payload.agents[{i}].id is required and must be a non-empty string."
            )
        if not agent_name or not str(agent_name).strip():
            raise SemanticNegotiationInputError(
                f"payload.agents[{i}].name is required and must be a non-empty string."
            )
        if agent_id in seen_ids:
            raise SemanticNegotiationInputError(
                f"payload.agents contains duplicate id {agent_id!r}."
            )
        seen_ids.add(agent_id)

    n_steps = payload.get("n_steps")
    if n_steps is not None:
        if not isinstance(n_steps, int) or n_steps < 1:
            raise SemanticNegotiationInputError(
                f"payload.n_steps must be a positive integer, got {n_steps!r}."
            )


def _validate_decide_payload(payload: Dict[str, Any], session_id: str) -> None:
    """Validate the payload for POST /api/negotiate/decide.

    Raises:
        SemanticNegotiationInputError: On any constraint violation.
    """
    if not session_id or not session_id.strip():
        raise SemanticNegotiationInputError(
            "session_id is required and must be a non-empty string "
            "(set payload.session_id or semantic_context.session_id)."
        )

    agent_replies = payload.get("agent_replies")
    if (
        agent_replies is None
        or not isinstance(agent_replies, list)
        or len(agent_replies) == 0
    ):
        raise SemanticNegotiationInputError(
            "payload.agent_replies is required and must be a non-empty list."
        )


# ============== Helpers ==============


def _wrap_sstp_response(
    session_id: str,
    request_id: str,
    domain_resp: Any,
    *,
    issues: Optional[List[str]] = None,
    options_per_issue: Optional[Dict[str, List[str]]] = None,
) -> SSTPNegotiateMessage:
    """Wrap a domain response in an SSTPNegotiateMessage envelope.

    ``message_id`` is the caller-supplied ``request_id`` — the server never
    generates its own IDs; unique-ID responsibility belongs to the caller.
    ``payload_hash`` is derived from the serialised payload for integrity.

    When the pipeline has run, pass ``issues`` and ``options_per_issue`` so
    ``semantic_context`` carries the negotiation space for agents and tracers.
    """
    if hasattr(domain_resp, "model_dump"):
        payload: Dict[str, Any] = domain_resp.model_dump(mode="json")
    else:
        payload = dict(domain_resp or {})

    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    _issues = list(issues) if issues is not None else []
    _opts = dict(options_per_issue) if options_per_issue is not None else {}
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=request_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="negotiation_server", tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            sao_state=None,
            issues=_issues,
            options_per_issue=_opts,
        ),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=payload,
    )


# ============== Initiate ==============


@router.post(
    "/negotiate/initiate",
    summary="Initiate a semantic negotiation from a mission description",
    description=(
        "Accepts a mission description and a list of agents, then runs:\n\n"
        "1. **Component 1** — `IntentDiscovery` extracts negotiable issues from content_text.\n"
        "2. **Component 2** — `OptionsGeneration` produces candidate options per issue.\n"
        "3. Returns the **first round's** ``List[SSTPNegotiateMessage]`` batch inside the SSTP envelope.\n\n"
        "The caller must dispatch those messages to the agents, collect their replies,\n"
        "and POST them to ``POST /api/negotiate/decide`` to advance the negotiation.\n\n"
        "Request body is a full SSTP **`SSTPNegotiateMessage`** envelope.\n"
        "- `semantic_context.session_id` — caller-supplied session ID (required).\n"
        "- `payload.content_text` — natural-language mission or negotiation goal.\n"
        "- `payload.agents` — list of `{id, name}` for each agent (min 2).\n"
        "- `payload.n_steps` — optional SAO round budget."
    ),
)
async def negotiate_initiate(
    body: SSTPNegotiateMessage,
    engine: NegotiationCognitionEngine = Depends(get_negotiation_cognition_engine),
) -> SSTPNegotiateMessage:
    """Run Components 1+2, seed round 1, return first-round messages."""
    session_id = body.semantic_context.session_id
    request_id = body.message_id
    workspace_id = body.origin.tenant_id
    mas_id = body.origin.actor_id
    header = NegotiationHeader(
        workspace_id=workspace_id,
        mas_id=mas_id,
    )
    payload = body.payload
    content_text: str = payload.get("content_text", "")
    agents_raw: List[Dict[str, Any]] = payload.get("agents", [])
    n_steps: Optional[int] = payload.get("n_steps")

    try:
        _validate_request_scope(workspace_id, mas_id)
        _validate_initiate_payload(payload, session_id)
        logger.info(
            "initiate validation passed session_id=%s agents=%d content_len=%d n_steps=%s",
            session_id,
            len(agents_raw),
            len(content_text),
            n_steps,
        )
        agent_names = [a["name"] for a in agents_raw if isinstance(a, dict) and a.get("name")]
        initiate_payload = {
            "session_id": session_id,
            "n_steps": n_steps,
            "content_text": content_text,
            "agents": agents_raw,
            "initiate_message": dump_negotiate_message_json(body),
            "workspace_id": workspace_id,
            "mas_id": mas_id,
            "fabric_node_base_url": settings.cfn_url,
            "agent_names": agent_names,
        }
        result = await engine.run(NegotiationAction.INITIATE, initiate_payload)
    except (SemanticNegotiationInputError, ValueError) as exc:
        logger.warning(
            "initiate validation failed session_id=%s reason=%s",
            session_id,
            exc,
        )
        trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
        error_resp = InitiateResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            current_round=RoundOffer(round=0, proposer_id="", offer={}),
            total_rounds=0,
            trace=trace,
            error=NegotiationError(message="BAD_REQUEST", detail={"reason": str(exc)}),
        )
        return JSONResponse(
            status_code=400,
            content=dump_negotiate_message_json(
                _wrap_sstp_response(session_id, request_id, error_resp)
            ),
        )
    except Exception:
        logger.exception("Unexpected error in /api/negotiate/initiate [%s]", request_id)
        trace = NegotiationTrace(rounds=[], timedout=False, broken=True)
        error_resp = InitiateResponse(
            header=header,
            session_id=session_id,
            response_id=request_id,
            status="broken",
            current_round=RoundOffer(round=0, proposer_id="", offer={}),
            total_rounds=0,
            trace=trace,
            error=NegotiationError(
                message="INTERNAL_ERROR", detail={"traceback": traceback.format_exc()}
            ),
        )
        return JSONResponse(
            status_code=500,
            content=dump_negotiate_message_json(
                _wrap_sstp_response(session_id, request_id, error_resp)
            ),
        )

    # Extract token metadata and include in response meta
    token_metadata = result.pop("token_metadata", None)
    logger.info(f"Token metadata extracted: {token_metadata is not None}")
    if token_metadata:
        from .schemas import TokenUsage, TokenUsageMeta
        from gateway.app.registration import CE_SEMANTIC_NEG_NAME, get_ce_id
        # Construct meta field
        result_meta = TokenUsageMeta(
            tokens=TokenUsage(
                prompt=token_metadata.prompt_tokens,
                completion=token_metadata.completion_tokens,
                total=token_metadata.total_tokens,
                model=token_metadata.model,
            ),
            latency_ms=token_metadata.latency_ms,
            cost_usd=token_metadata.cost_usd,
            timestamp=token_metadata.timestamp,
            ce_id=get_ce_id(CE_SEMANTIC_NEG_NAME),
        )
        logger.info(f"Token meta constructed: prompt={result_meta.tokens.prompt} completion={result_meta.tokens.completion}")
        # Add to result payload
        if "payload" not in result:
            result["payload"] = {}
        result["payload"]["meta"] = result_meta.model_dump()
        logger.info("Token meta added to result.payload")

    envelope = _wrap_sstp_response(session_id, request_id, result)
    return JSONResponse(content=dump_negotiate_message_json(envelope))


# ============== Decide (turn-by-turn) ==============


@router.post(
    "/negotiate/decide",
    summary="Advance the negotiation by one batch of agent decisions",
    description=(
        "Accepts the agents' replies to the last dispatched message batch and returns\n"
        "either the next round's messages (``status='ongoing'``) or the final result\n"
        "(``status='agreed'|'broken'|'timeout'``).\n\n"
        "Request body: ``SSTPNegotiateMessage`` whose ``payload.session_id`` identifies\n"
        "the active session, and ``payload.agent_replies`` is the list of\n"
        "``SSTPNegotiateMessage`` dicts returned by the agents."
    ),
)
async def negotiate_decide(
    body: SSTPNegotiateMessage,
    engine: NegotiationCognitionEngine = Depends(get_negotiation_cognition_engine),
) -> JSONResponse:
    """Apply agent replies and advance the SAO by one step."""
    payload = body.payload
    workspace_id = body.origin.tenant_id
    mas_id = body.origin.actor_id
    session_id: str = payload.get("session_id") or body.semantic_context.session_id
    request_id = body.message_id
    agent_replies = payload.get("agent_replies", [])

    decide_payload = {
        "session_id": session_id,
        "agent_replies": agent_replies,
        "commit_message_id": request_id,
        "workspace_id": workspace_id,
        "mas_id": mas_id,
    }

    try:
        _validate_request_scope(workspace_id, mas_id)
        _validate_decide_payload(payload, session_id)
        logger.info(
            "decide validation passed session_id=%s replies=%d",
            session_id,
            len(agent_replies),
        )
        exec_result = await engine.run(NegotiationAction.DECIDE, decide_payload)
    except SemanticNegotiationInputError as exc:
        logger.warning(
            "decide validation failed session_id=%s reason=%s",
            session_id,
            exc,
        )
        return JSONResponse(
            status_code=400,
            content={"error": "BAD_REQUEST", "detail": str(exc)},
        )
    except SemanticNegotiationSessionNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"No active session for session_id={session_id!r}"},
        )
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": f"No active session for session_id={session_id!r}"},
        )

    if exec_result["status"] == "ongoing":
        return JSONResponse(
            content={
                "session_id": session_id,
                "status": "ongoing",
                "round": exec_result["round"],
                "messages": exec_result["messages"],
            }
        )

    # Terminal — commit envelope was built inside pipeline.execute().
    _status_str = exec_result["status"]
    total_rounds = exec_result["round"]
    final_envelope = exec_result["final_result"]
    return JSONResponse(
        content={
            "session_id": session_id,
            "status": _status_str,
            "round": total_rounds,
            "final_result": final_envelope.model_dump(mode="json"),
        }
    )
