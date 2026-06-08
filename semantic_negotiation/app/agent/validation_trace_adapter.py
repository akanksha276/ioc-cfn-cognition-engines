# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt in-memory negotiation traces for :class:`SemanticAlignmentValidationPipeline`.

The CFN ``/semantic-negotiation/start`` and ``/decide`` API uses compact JSON (no full
``SSTPNegotiateMessage`` shells).  ``BatchCallbackRunner`` therefore records bare agent
dicts and may omit a gateway-style **initiate** row with ``payload.content_text``.

Step 4's ``validate_input`` expects the same shape as ``dump_negotiate_message_json``
produces.  This module builds a **throwaway copy** of the trace for validation only —
``NegotiationSession.sstp_message_trace`` is never mutated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from protocol.sstp import SSTPNegotiateMessage
from protocol.sstp._base import Origin, PolicyLabels, Provenance
from protocol.sstp.negmas_sao import ResponseType, SAOResponse
from protocol.sstp.negotiate import NegotiateSemanticContext, dump_negotiate_message_json

logger = logging.getLogger(__name__)

# Re-used on every synthetic SSTP row (matches ``build_callback_message`` / agents).
_POLICY = PolicyLabels(
    sensitivity="internal",
    propagation="restricted",
    retention_policy="default",
)
_PROVENANCE = Provenance(sources=[], transforms=[])


def _slug(name: str) -> str:
    """Lowercase slug for ``Origin.actor_id`` (same rule as ``base_agent.build_sstp_reply``)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _mission_text(msg: Any) -> str:
    """Non-empty mission string from ``payload.content_text``, or ``\"\"``."""
    if not isinstance(msg, dict):
        return ""
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return ""
    ct = payload.get("content_text")
    return ct.strip() if isinstance(ct, str) else ""


def _is_wire_negotiate_message(msg: Any) -> bool:
    """``True`` if *msg* looks like ``dump_negotiate_message_json(SSTPNegotiateMessage)``."""
    return (
        isinstance(msg, dict)
        and msg.get("kind") == "negotiate"
        and isinstance(msg.get("message_id"), str)
        and bool(msg.get("message_id"))
        and isinstance(msg.get("semantic_context"), dict)
    )


def _is_cfn_minimal_envelope(msg: Any) -> bool:
    """True for cfn-svc minimal SSTP envelopes: wire format but sao_response absent,
    and action lives in payload rather than at the top level.
    buildAgentReplyEnvelopes in cfn-svc produces these — crucially, it omits the
    ``origin`` field entirely, which distinguishes them from server respond messages."""
    if not _is_wire_negotiate_message(msg):
        return False
    # Server messages (respond, initiate) always carry a populated ``origin`` dict.
    # cfn-svc buildAgentReplyEnvelopes never sets ``origin``.
    if msg.get("origin"):
        return False
    sc = msg.get("semantic_context") or {}
    if "sao_response" in sc:
        return False
    payload = msg.get("payload") or {}
    return "action" in payload


def _is_bare_cfn_agent_reply(msg: Any) -> bool:
    """Flat agent dict from CFN ``/decide`` (action at top level) OR cfn-svc minimal
    SSTP envelope (kind=negotiate but no sao_response; action inside payload)."""
    if isinstance(msg, dict) and "action" in msg and not _is_wire_negotiate_message(msg):
        return True
    return _is_cfn_minimal_envelope(msg)


def _normalize_cfn_reply(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten cfn-svc minimal envelopes so _wrap_bare_agent_reply sees a flat dict."""
    if _is_cfn_minimal_envelope(msg):
        return dict(msg.get("payload") or {})
    return msg


def _participant_name(participants: List[Any], participant_id: str) -> str:
    return next(
        (
            str(getattr(p, "name", participant_id) or participant_id)
            for p in participants
            if getattr(p, "id", None) == participant_id
        ),
        participant_id,
    )


def _negotiate_wire_dict(
    *,
    uuid_key_prefix: str,
    origin_actor_id: str,
    tenant_id: str,
    semantic_context: NegotiateSemanticContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialised ``SSTPNegotiateMessage`` dict with deterministic ``message_id``."""
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"{uuid_key_prefix}:{payload_hash}")
    )
    return dump_negotiate_message_json(
        SSTPNegotiateMessage(
            kind="negotiate",
            message_id=message_id,
            dt_created=datetime.now(timezone.utc).isoformat(),
            origin=Origin(actor_id=origin_actor_id, tenant_id=tenant_id),
            semantic_context=semantic_context,
            payload_hash=payload_hash,
            policy_labels=_POLICY,
            provenance=_PROVENANCE,
            payload=payload,
        )
    )


def _synthetic_initiate_message(
    *,
    session_id: str,
    content_text: str,
    issues: List[str],
    options_per_issue: Dict[str, List[str]],
    participants: List[Any],
    n_steps: Optional[int],
) -> Dict[str, Any]:
    agents = [
        {"id": getattr(p, "id", ""), "name": getattr(p, "name", "")}
        for p in participants
        if getattr(p, "id", None)
    ]
    payload: Dict[str, Any] = {
        "content_text": content_text,
        "agents": agents,
        "session_id": session_id,
    }
    if n_steps is not None:
        payload["n_steps"] = n_steps
    return _negotiate_wire_dict(
        uuid_key_prefix=f"{session_id}:initiate",
        origin_actor_id="semantic-negotiation-client",
        tenant_id=session_id,
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            issues=list(issues),
            options_per_issue=dict(options_per_issue),
        ),
        payload=payload,
    )


def _wrap_bare_agent_reply(
    bare: Dict[str, Any],
    *,
    session_id: str,
    participants: List[Any],
) -> Dict[str, Any]:
    """Wrap a CFN bare ``AgentReply`` dict in a full SSTP envelope for validation.

    Preserves ``reason`` (and ``offer``) from the bare dict in ``payload`` so
    alignment judges see the same fields as on the live wire.
    """
    participant_id = str(bare.get("participant_id") or "")
    agent_name = _participant_name(participants, participant_id)
    action = str(bare.get("action") or "").lower()
    # Copy action, offer, reason, round, etc. — only participant_id is re-slotted.
    inner = {k: v for k, v in bare.items() if k != "participant_id"}
    inner["participant_id"] = participant_id
    payload_str = json.dumps(inner, sort_keys=True)
    message_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{participant_id}:{payload_str}"))
    return {
        "kind": "negotiate",
        "message_id": message_id,
        "origin": {"actor_id": _slug(agent_name), "tenant_id": session_id},
        "semantic_context": {"session_id": session_id},
        "payload": inner,
    }


def adapt_sstp_trace_for_alignment_validation(
    raw_trace: List[Dict[str, Any]],
    *,
    session_id: str,
    content_text: str,
    issues: List[str],
    options_per_issue: Dict[str, List[str]],
    participants: List[Any],
    n_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a trace copy that satisfies ``SemanticAlignmentValidationPipeline.validate_input``.

    Does not modify *raw_trace*.  Inserts a synthetic initiate when the first row has
    no mission text; wraps CFN-style bare agent dicts in full SSTP envelopes.
    """
    if not raw_trace:
        return list(raw_trace)

    out: List[Dict[str, Any]] = [dict(m) if isinstance(m, dict) else m for m in raw_trace]  # type: ignore[misc]
    adapted = False

    if not _mission_text(out[0]):
        mission = (content_text or "").strip()
        if mission:
            try:
                out.insert(
                    0,
                    _synthetic_initiate_message(
                        session_id=session_id,
                        content_text=mission,
                        issues=list(issues or []),
                        options_per_issue=dict(options_per_issue or {}),
                        participants=participants,
                        n_steps=n_steps,
                    ),
                )
                adapted = True
            except Exception as exc:
                logger.warning("validation_trace_adapter: synthetic initiate failed: %s", exc)
        else:
            logger.debug(
                "validation_trace_adapter: skip synthetic initiate (empty content_text)"
            )

    for i, m in enumerate(out):
        if not _is_bare_cfn_agent_reply(m):
            continue
        try:
            out[i] = _wrap_bare_agent_reply(
                _normalize_cfn_reply(m),
                session_id=session_id,
                participants=participants,
            )
            adapted = True
        except Exception as exc:
            logger.warning(
                "validation_trace_adapter: wrap bare reply at index %d failed: %s",
                i,
                exc,
            )

    if adapted:
        logger.debug(
            "validation_trace_adapter: adapted trace len=%d (raw_len=%d)",
            len(out),
            len(raw_trace),
        )
    return out
