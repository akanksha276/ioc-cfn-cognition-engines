# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .negotiation_model import NegotiationResult


# ── Failure mode constants (based on MAST taxonomy, arxiv:2503.13657) ────────

class FailureMode:
    """
    Catalogue of observable failure modes in multi-agent negotiation systems.

    Derived from the MAST (Multi-Agent System Failure Taxonomy) framework
    (arxiv:2503.13657), adapted for semantic negotiation contexts.
    """

    # Communication breakdowns
    COMMUNICATION_BREAKDOWN = "communication_breakdown"
    """Agents failed to convey information accurately across rounds, leading to
    misaligned shared understanding of the negotiation state or outstanding offers."""

    CONVERSATION_HISTORY_LOSS = "conversation_history_loss"
    """An agent lost access to prior round context, causing it to repeat rejected
    offers or ignore previously accepted concessions."""

    # Role and instruction failures
    ROLE_DISOBEDIENCE = "role_disobedience"
    """An agent acted contrary to its designated role (e.g., a responder issued
    unsolicited proposals, or an agent ignored its utility function)."""

    ROLE_SPECIFICATION_AMBIGUITY = "role_specification_ambiguity"
    """The agent's responsibilities were underspecified, causing it to operate
    outside its intended scope during propose or respond phases."""

    # Termination failures
    PREMATURE_TERMINATION = "premature_termination"
    """The negotiation halted before a genuine agreement or timeout, typically
    because an agent incorrectly signalled acceptance of an unresolved offer."""

    UNAWARE_OF_STOPPING_CONDITIONS = "unaware_of_stopping_conditions"
    """An agent continued proposing beyond the step budget or after a final
    agreement had already been reached, indicating missing termination awareness."""

    STEP_REPETITION = "step_repetition"
    """An agent repeatedly submitted identical offers across consecutive rounds
    without concession, indicating a reasoning loop or frozen aspiration curve."""

    # Awareness and state failures
    AWARENESS_GAP = "awareness_gap"
    """An agent operated with incomplete knowledge of the negotiation state —
    missing current offer, issue list, or options — leading to invalid decisions."""

    # Alignment-specific failures
    UTILITY_MISALIGNMENT = "utility_misalignment"
    """The agreed outcome yields a utility distribution that is asymmetrically
    unfair: one party's utility is near reservation value while the other's is
    near maximum, suggesting coercion or defective preference expression."""

    SEMANTIC_DRIFT = "semantic_drift"
    """The final agreement diverges semantically from the original negotiation
    intent expressed in the mission text — e.g., agreeing on options that do
    not correspond to the discovered issues."""

    INVALID_OFFER = "invalid_offer"
    """An agent submitted an offer containing issue keys or option values not
    present in the options_per_issue space defined for this session."""

    LOW_PARETO_EFFICIENCY = "low_pareto_efficiency"
    """The agreement is dominated by an alternative outcome that would improve
    at least one party's utility without harming the other, indicating the
    negotiation settled for a suboptimal joint outcome."""

    IGNORED_OTHER_AGENT_INPUT = "ignored_other_agent_input"
    """An agent disregarded the other party's most recent offer or concession,
    continuing to propose as if no input had been received — indicating a failure
    to model or incorporate the counterpart's negotiation moves."""

    NO_CONSENSUS_REACHED = "no_consensus_reached"
    """The negotiation exhausted its full step budget without any party accepting
    an offer — distinct from premature termination in that all rounds were used.
    Typically caused by incompatible preferences, overly rigid strategies (e.g.
    Boulware deadlock), or insufficient step budget relative to the issue space."""


# ── ValidationResult ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Output of the SemanticAlignmentValidationPipeline.

    Summarises whether a completed negotiation agreement is semantically sound,
    fair, and free of the failure modes catalogued in the MAST taxonomy
    (arxiv:2503.13657).
    """

    needs_intervention: bool
    """True when the validation pipeline determines that a human or system-level
    intervention is required before the agreement can be safely committed.
    Typically set when severity is 'high' or a critical failure mode is present."""

    severity: str
    """Estimated severity of any detected issues.  One of:
      - ``"low"``    — minor anomaly, agreement is likely acceptable as-is.
      - ``"medium"`` — notable concern; justification or review is advisable.
      - ``"high"``   — critical problem; the agreement should not be committed
                       without escalation or re-negotiation.
    Set to ``"low"`` with an empty ``failure_modes`` list when no issues are found."""

    alignment_score: float
    """A scalar in ``[0.0, 1.0]`` reflecting how well the final agreement aligns
    with the original negotiation intent and the participants' stated preferences.
    Computed as a weighted combination of utility balance, Pareto efficiency, and
    semantic fidelity to the mission text.  Values below 0.4 typically trigger
    ``needs_intervention = True``."""

    failure_modes: List[str]
    """Zero or more failure mode identifiers detected in this negotiation session.
    Use constants from :class:`FailureMode` (e.g. ``FailureMode.STEP_REPETITION``).
    An empty list indicates a clean negotiation with no observed anomalies."""

    reasoning: str
    """Human-readable explanation of the validation outcome.  Describes which
    failure modes were detected, why the alignment score was assigned, and what
    evidence from the negotiation trace led to the recommendation."""

    recommendation: str
    """Suggested next action for the orchestrating system.  One of:
      - ``"accept"``                — agreement is sound; proceed to commit.
      - ``"request_justification"`` — ask one or both agents to explain their
                                      final offer before committing.
      - ``"restart_negotiation"``   — discard the current result and re-run the
                                      full negotiation pipeline.
      - ``"escalate"``              — surface the issue to a human operator or
                                      a higher-authority decision system."""

    statistical_events: Optional[int] = field(default=None)
    """Reserved for future implementation."""


# ── Exceptions ────────────────────────────────────────────────────────────────

class ValidationInputError(ValueError):
    """Raised when the trace passed to the pipeline is malformed."""


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SemanticAlignmentValidationPipeline:
    def __init__(self):
        pass

    # ── Input validation ──────────────────────────────────────────────────────

    def validate_input(self, trace: Any) -> None:
        """Validate the structure of an SSTP message trace before processing.

        The expected input is a **list of SSTPNegotiateMessage dicts** as written
        by the negotiation server to ``sstp_message_trace.json`` — the same format
        produced by ``BatchCallbackRunner`` and persisted by
        ``SemanticNegotiationPipeline.build_commit_envelope``.

        Concretely, each message in the list is a dict with:

        * ``message_id`` — non-empty string
        * ``payload`` (first message only) — dict with ``content_text`` (non-empty
          mission description string)
        * ``semantic_context`` — dict with:
            - ``session_id`` — non-empty string (same across all messages)
            - ``issues`` — list of issue name strings (present and non-empty on
              at least one message after the initiate)
            - ``options_per_issue`` — dict mapping each issue to a list of options
              (non-empty on at least one message)
            - ``sao_state`` (optional on wire) — when present, dict with
              ``current_offer`` (must be non-None)
            - ``sao_response`` (optional on wire; typical on agent-reply messages)
              — when present, dict with ``response`` and ``outcome`` as above

        Additionally:

        * All messages must share the same ``session_id``.
        * The first message's ``payload.content_text`` must be a non-empty string.
        * At least one message must carry non-empty ``issues`` and
          ``options_per_issue`` in its ``semantic_context``.
        * Every message that carries a ``sao_state`` must have a non-None
          ``current_offer`` inside it.
        * Every negotiation step must have at least one agent decision (a message
          with a non-None ``sao_response`` at that step).
        * When the last message's ``semantic_context.sao_response.response`` is
          ``"ACCEPT_OFFER"``, ``sao_response.outcome`` must be a non-empty dict.

        Raises:
            ValidationInputError: with a human-readable description of every
                field that failed validation.
        """
        errors: List[str] = []

        if trace is None:
            raise ValidationInputError("trace must not be None")

        if not isinstance(trace, list):
            raise ValidationInputError(
                f"trace must be a list of SSTPNegotiateMessage dicts, "
                f"got {type(trace).__name__}"
            )

        if len(trace) == 0:
            raise ValidationInputError("trace must not be empty")

        # ── First message: mission context ────────────────────────────────────
        first = trace[0] if isinstance(trace[0], dict) else {}
        payload = first.get("payload")
        if not isinstance(payload, dict) or not payload.get("content_text"):
            errors.append(
                "trace[0].payload.content_text must be a non-empty mission description string"
            )

        # ── Per-message checks ────────────────────────────────────────────────
        session_ids: set = set()
        has_issues = False
        has_options = False
        responses_by_step: dict = {}  # step -> list of sao_response values

        for i, msg in enumerate(trace):
            if not isinstance(msg, dict):
                errors.append(f"trace[{i}] must be a dict, got {type(msg).__name__}")
                continue

            # message_id
            message_id = msg.get("message_id")
            if not message_id or not isinstance(message_id, str):
                errors.append(f"trace[{i}].message_id must be a non-empty string")

            # semantic_context
            sc = msg.get("semantic_context")
            if not isinstance(sc, dict):
                errors.append(f"trace[{i}].semantic_context must be a dict")
                continue

            session_id = sc.get("session_id")
            if not session_id or not isinstance(session_id, str):
                errors.append(
                    f"trace[{i}].semantic_context.session_id must be a non-empty string"
                )
            else:
                session_ids.add(session_id)

            issues = sc.get("issues")
            if isinstance(issues, list) and issues:
                has_issues = True

            options = sc.get("options_per_issue")
            if isinstance(options, dict) and options:
                has_options = True

            # current_offer must be non-None on every message that has a sao_state;
            # also collect sao_response per step to verify agent decisions
            sao_state = sc.get("sao_state")
            if isinstance(sao_state, dict):
                if sao_state.get("current_offer") is None:
                    errors.append(
                        f"trace[{i}].semantic_context.sao_state.current_offer must not be None"
                    )
                step = sao_state.get("step")
                if step is not None:
                    responses_by_step.setdefault(step, []).append(sc.get("sao_response"))

        # ── Cross-message checks ──────────────────────────────────────────────
        if len(session_ids) > 1:
            errors.append(
                f"all messages must share the same session_id, "
                f"found multiple: {sorted(session_ids)}"
            )

        if not has_issues:
            errors.append(
                "no message in the trace carries a non-empty 'issues' list in "
                "semantic_context — at least one server message must include the "
                "discovered issues"
            )

        if not has_options:
            errors.append(
                "no message in the trace carries a non-empty 'options_per_issue' "
                "dict in semantic_context"
            )

        # ── Per-step agent decision checks ────────────────────────────────────
        for step, resps in responses_by_step.items():
            if not any(r is not None for r in resps):
                errors.append(
                    f"step {step} has no agent decision — at least one message "
                    f"in each step must carry a non-None sao_response"
                )

        # ── Last message / outcome checks ─────────────────────────────────────
        last = trace[-1] if isinstance(trace[-1], dict) else {}
        last_sc = last.get("semantic_context") if isinstance(last, dict) else {}
        sao_response = last_sc.get("sao_response") if isinstance(last_sc, dict) else None
        if isinstance(sao_response, dict):
            response = sao_response.get("response")
            if response == "ACCEPT_OFFER":
                outcome = sao_response.get("outcome")
                if not isinstance(outcome, dict) or not outcome:
                    errors.append(
                        "last message semantic_context.sao_response.outcome must be a "
                        "non-empty dict when response is 'ACCEPT_OFFER'"
                    )

        if errors:
            raise ValidationInputError(
                "Invalid trace passed to SemanticAlignmentValidationPipeline:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def run(self, trace: Any) -> ValidationResult:
        """Validate a completed SSTP negotiation message trace.

        Args:
            trace: A list of SSTPNegotiateMessage dicts as written by the
                negotiation server to ``sstp_message_trace.json``.  Each entry
                is a ``kind='negotiate'`` message carrying ``semantic_context``
                (with ``session_id``, ``issues``, ``options_per_issue``,
                ``sao_state``) and, for agent-reply messages, ``sao_response``.

        Returns:
            A :class:`ValidationResult` describing any detected failure modes
            and the recommended next action.

        Raises:
            ValidationInputError: If *trace* is malformed or missing required fields.
        """
        self.validate_input(trace)

        # TODO - implement the actual validation logic here
        return ValidationResult(
            needs_intervention=False,
            severity="low",
            alignment_score=1.0,
            failure_modes=[],
            reasoning="Validation not yet implemented.",
            recommendation="accept",
        )
