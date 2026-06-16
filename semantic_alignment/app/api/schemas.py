# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic request/response models for the Semantic Alignment Agent API.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ============== Token tracking ==============


class TokenUsage(BaseModel):
    """LLM token consumption metrics."""
    prompt: int = Field(..., description="Input tokens consumed")
    completion: int = Field(..., description="Output tokens generated")
    total: int = Field(..., description="Total tokens (prompt + completion)")
    model: str = Field(..., description="Model identifier")


class TokenUsageMeta(BaseModel):
    """LLM call metadata including token usage and CE attribution."""
    tokens: TokenUsage
    latency_ms: float = Field(..., description="Time taken for LLM call in milliseconds")
    cost_usd: Optional[float] = Field(None, description="Calculated cost in USD")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    ce_id: Optional[str] = Field(None, description="CE that performed the operation (for CFN metrics attribution)")


# ============== Shared header ==============


class NegotiationHeader(BaseModel):
    """Header common to all negotiation requests."""

    workspace_id: str = Field(..., description="Workspace identifier")
    mas_id: str = Field(..., description="MAS identifier")
    agent_id: Optional[str] = Field(
        None, description="Calling agent identifier (optional)"
    )


# ============== Negotiate request ==============


class ParticipantRequest(BaseModel):
    """One negotiating party supplied by the caller."""

    id: str = Field(..., description="Unique identifier for this participant")
    name: str = Field(..., description="Human-readable display name")
    preferences: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-issue option utilities. "
            "Shape: {issue_id: {option_label: utility_value}}. "
            "Values should be in [0.0, 1.0]."
        ),
    )
    issue_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-issue importance weights. Defaults to uniform weights when omitted.",
    )


class NegotiateRequest(BaseModel):
    """Top-level request body for POST /negotiate."""

    header: NegotiationHeader
    request_id: str = Field(
        ...,
        description="Client-supplied request ID; echoed back as response_id",
    )
    issues: List[str] = Field(
        ...,
        description="Ordered list of issue identifiers to negotiate over",
    )
    options_per_issue: Dict[str, List[str]] = Field(
        ...,
        description="Candidate options for each issue. Shape: {issue_id: [option, ...]}",
    )
    participants: List[ParticipantRequest] = Field(
        ...,
        min_length=2,
        description="Negotiating parties (minimum 2)",
    )
    n_steps: Optional[int] = Field(
        None,
        description="Maximum SAO rounds. Falls back to the service default when omitted.",
    )


# ============== Negotiate response ==============


class NegotiationOutcomeResponse(BaseModel):
    """Agreed value for a single issue."""

    issue_id: str
    chosen_option: str


class NegotiationError(BaseModel):
    """Error block returned when negotiation fails."""

    message: str = Field(..., description="User-meaningful error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Debugging context")


class NegotiateResponse(BaseModel):
    """
    Response for POST /negotiate.

    Either ``error`` is set **or** the outcome fields are set — never both.
    """

    header: NegotiationHeader
    response_id: str = Field(..., description="Echoed from request_id")
    agreement: Optional[List[NegotiationOutcomeResponse]] = Field(
        None,
        description="Agreed option per issue, or null when no agreement was reached",
    )
    timedout: Optional[bool] = None
    broken: Optional[bool] = None
    steps: Optional[int] = None
    history: Optional[List[Any]] = Field(
        None,
        description="SAO trace as (step, negotiator_id, offer) tuples",
    )
    error: Optional[NegotiationError] = None
    meta: Optional[TokenUsageMeta] = Field(None, description="LLM token usage metadata")


# ============== Initiate ==============
# Note: the request body for /api/negotiate/initiate and /api/negotiate/offer-response
# is SSTPNegotiateMessage (protocol.sstp.SSTPNegotiateMessage).
# session_id is carried in semantic_context.session_id and is REQUIRED —
# the caller must always supply it; the server never generates one.


class AgentDecision(BaseModel):
    """One participant's decision within a SAO round.

    Mirrors the per-agent entry in ``BatchCallbackRunner.round_decisions`` and
    the agent reply ``payload`` on the SSTP wire (including optional ``reason``).
    """

    participant_id: str = Field(
        ..., description="ID of the participant who made this decision"
    )
    action: str = Field(..., description="'accept', 'reject', or 'counter_offer'")
    offer: Optional[Dict[str, str]] = Field(
        None,
        description="Proposed offer when action='counter_offer'. Shape: {issue_id: option}",
    )
    reason: Optional[str] = Field(
        None,
        description=(
            "Agent's explanation for this action (accept, reject, or counter_offer). "
            "Pass-through metadata; not used by SAO outcome logic."
        ),
    )


class RoundOffer(BaseModel):
    """The offer produced in a single SAO round."""

    round: int = Field(..., description="1-based round number within the SAO trace")
    proposer_id: str = Field(
        ..., description="ID of the participant who made this proposal"
    )
    next_proposer_id: Optional[str] = Field(
        None,
        description="ID of the participant who will propose in the next round. None for the final round.",
    )
    offer: Dict[str, str] = Field(
        ..., description="Proposed option per issue. Shape: {issue_id: option}"
    )
    decisions: List[AgentDecision] = Field(
        default_factory=list,
        description="Each participant's decision in response to this round's offer (accept / reject / counter_offer)",
    )


class NegotiationTrace(BaseModel):
    """Immutable pre-computed SAO trace returned by /api/negotiate/initiate.

    Pass this back verbatim in every /api/negotiate/offer-response request.
    The server holds **no state** — all session data lives in this object.
    """

    rounds: List[RoundOffer] = Field(
        ..., description="All SAO rounds in order (1-based)"
    )
    final_agreement: Optional[List[NegotiationOutcomeResponse]] = Field(
        None,
        description="Agreement reached by NegMAS at the end of its own run, if any",
    )
    timedout: bool = Field(
        False, description="Whether NegMAS exhausted its step budget"
    )
    broken: bool = Field(
        False, description="Whether a participant explicitly broke off"
    )
    sstp_message_trace: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "All SSTPNegotiateMessage envelopes exchanged during this negotiation "
            "(initiate request first, then interleaved server→agent and agent→server "
            "messages for every round), in chronological order."
        ),
    )


class AcceptedResponse(BaseModel):
    """Immediate 202 response when result_callback_url is provided.

    The negotiation runs in the background; the full result is POSTed to
    ``result_callback_url`` when it completes.
    """

    header: NegotiationHeader
    session_id: str = Field(..., description="Correlation ID for this negotiation run")
    response_id: str = Field(..., description="Echoed from message_id")
    status: Literal["accepted"] = "accepted"
    result_callback_url: str = Field(
        ..., description="The URL the final result will be POSTed to"
    )


class InitiateResponse(BaseModel):
    """Response for POST /api/negotiate/initiate (synchronous mode)."""

    header: NegotiationHeader
    session_id: str = Field(
        ...,
        description="Opaque correlation identifier — echo back in every /api/negotiate/offer-response call",
    )
    response_id: str = Field(..., description="Echoed from request_id")
    status: Literal["ongoing", "agreed", "broken", "timeout"] = Field(
        ...,
        description=(
            "'ongoing' when more rounds are available; "
            "'agreed'/'broken'/'timeout' when the negotiation already resolved in round 1"
        ),
    )
    current_round: RoundOffer = Field(..., description="The first offer in the trace")
    total_rounds: int = Field(
        ..., description="Total SAO rounds in the pre-computed trace"
    )
    trace: NegotiationTrace = Field(
        ...,
        description=(
            "Complete pre-computed SAO trace. "
            "Store this client-side and pass it back with every /api/negotiate/offer-response request."
        ),
    )
    error: Optional[NegotiationError] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# ============== Turn-by-turn /decide ==============


class DecideRoundResponse(BaseModel):
    """Response for POST /api/negotiate/decide — returned each turn.

    When ``status='ongoing'`` the caller should dispatch ``next_messages`` to
    all agents, collect their replies, and POST them back to /decide again.
    When ``status`` is ``'agreed'``, ``'broken'``, or ``'timeout'`` the
    negotiation is finished and ``final_result`` carries the full outcome.
    """

    session_id: str
    round: int = Field(..., description="Round number that was just evaluated")
    status: Literal["ongoing", "agreed", "broken", "timeout"]
    # Present when status='ongoing': the next batch of SSTP messages to forward to agents
    next_messages: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Next round's SSTPNegotiateMessage list ready to dispatch to agents",
    )
    # Present when status != 'ongoing': full negotiation result
    final_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Final SSTPNegotiateMessage/SSTPCommitMessage envelope when done",
    )
