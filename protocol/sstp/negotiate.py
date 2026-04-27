# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""sstp/negotiate.py — SSTPNegotiateMessage kind with NegMAS SAO semantic context."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from ._base import EncodingType, _STBaseMessage
from .negmas_sao import SAONMI, SAOResponse, SAOState


class NegotiateSemanticContext(BaseModel):
    """
    SAO-specific semantic context for ``kind='negotiate'`` messages.

    Carries a full NegMAS SAO negotiation snapshot so that receivers have
    complete mechanism state, the latest response, and optionally the NMI
    configuration — all in one well-typed envelope.
    """

    schema_id: str = "urn:ioc:schema:negotiate:negmas-sao:v1"
    """Canonical schema URN identifying this as a NegMAS SAO negotiate context."""

    schema_version: str = "1.0"
    """Schema version."""

    encoding: EncodingType = "json"
    """Payload encoding (always json for NegMAS snapshots)."""

    session_id: str
    """Negotiation session / mechanism ID that this message belongs to."""

    issues: list[str] = Field(default_factory=list)
    """Ordered list of negotiable issue identifiers for this session."""

    options_per_issue: dict[str, list[str]] = Field(default_factory=dict)
    """Candidate options for each issue.  Shape: ``{issue_id: [option, ...]}``.

    Populated by the negotiation server on every round message so that
    receiving agents have the full negotiation space without parsing the
    payload.
    """

    sao_state: SAOState | None = None
    """Full SAO mechanism state snapshot at the time this message was created.

    ``None`` on initiate requests (no prior state exists yet).
    """

    sao_response: SAOResponse | None = None
    """The response produced by the sending negotiator in this round (if any)."""

    nmi: SAONMI | None = None
    """SAO NegotiatorMechanismInterface config snapshot (optional; omit if large)."""

    offer_validation_failure: dict[str, Any] | None = None
    """Populated when the previous counter_offer from this session was rejected due
    to unrecognised issue keys or option values (i.e. validation failure, not a
    strategic reject).

    Shape::

        {
            "rejected_agent_id": str,   # agent whose offer failed
            "round": int,               # round number of the bad offer
            "problems": list[str],      # specific snap failures
            "hint": str,                # human-readable correction guidance
        }

    ``None`` when the previous offer was valid or no counter_offer was made.
    Agents should check this field and correct their issue keys / option values
    before submitting the next counter_offer.
    """


class SSTPNegotiateMessage(_STBaseMessage):
    """
    A negotiation-round message backed by NegMAS SAO semantics.

    Unlike other kinds, ``semantic_context`` is typed as
    :class:`NegotiateSemanticContext` instead of the generic
    :class:`SemanticContext`, carrying a full SAO state snapshot,
    the sender's response, and an optional NMI configuration.
    """

    kind: Literal["negotiate"]

    # Override: narrow semantic_context to the SAO-specific subtype
    semantic_context: NegotiateSemanticContext  # type: ignore[override]


# Optional SSTP envelope fields omitted from negotiate wire JSON.
_NEGOTIATE_SLIM_ENVELOPE_KEYS: Final[tuple[str, ...]] = (
    "state_object_id",
    "parent_ids",
    "logical_clock",
    "payload_refs",
    "confidence_score",
    "ttl_seconds",
    "merge_strategy",
    "risk_score",
)

# Shared with FastAPI ``response_model_exclude`` for negotiate endpoints.
NEGOTIATE_MESSAGE_JSON_EXCLUDE: Final[dict[str, Any]] = {
    "provenance": True,
    "policy_labels": True,
    **{k: True for k in _NEGOTIATE_SLIM_ENVELOPE_KEYS},
    "semantic_context": {
        "sao_state": True,
        "sao_response": True,
        "nmi": True,
    },
}


def dump_negotiate_message_json(msg: SSTPNegotiateMessage) -> dict[str, Any]:
    """JSON-ready dict for API/trace/callback; omits verbose envelope/context fields."""
    data = msg.model_dump(mode="json", exclude=NEGOTIATE_MESSAGE_JSON_EXCLUDE)
    # Belt-and-suspenders: ensure omitted keys never appear (e.g. as null) on any Pydantic version.
    sc = data.get("semantic_context")
    if isinstance(sc, dict):
        for key in ("sao_state", "sao_response", "nmi"):
            sc.pop(key, None)
    for key in (
        "policy_labels",
        "provenance",
        *_NEGOTIATE_SLIM_ENVELOPE_KEYS,
    ):
        data.pop(key, None)
    pl = data.get("payload")
    if isinstance(pl, dict):
        pl.pop("is_shadow_call", None)
    return data
