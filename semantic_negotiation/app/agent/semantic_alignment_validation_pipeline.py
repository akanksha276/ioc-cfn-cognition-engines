# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .acse import (
    GoalSpecExtractor,
    InteractionSignalExtractor,
    IssueEvaluation,
    NegotiationTrace,
    RoundRecord,
    SemanticAlignmentEvaluator,
    TraceStateBuilder,
)
from ..config.utils import get_llm_provider

logger = logging.getLogger(__name__)


# ── ValidationResult ─────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Output of SemanticAlignmentValidationPipeline.run().

    Combines the full ACSE semantic evaluation with protocol-level outcomes
    and an orchestration recommendation.  All fields are always populated.
    """

    # ── Semantic evaluation (from ACSE) ──────────────────────────────────────
    needs_intervention: bool
    """True when the agreement requires human or system review before committing."""

    severity: str
    """One of "low", "medium", or "high"."""

    alignment_score: float
    """Scalar in [0.0, 1.0] — weighted average of per-issue ACSE scores."""

    cognitive_alignment: float
    """Overall confidence that agents reached genuine shared understanding."""

    failure_modes: List[str]
    """SM-X failure strings from the ACSE evaluator (semantic layer only).
    Protocol-level outcomes are expressed as dedicated fields (e.g. timed_out)."""

    per_issue_evaluations: List[IssueEvaluation]
    """Per-issue breakdown: resolution quality, constraint fit, consistency, focus."""

    reasoning: str
    """Human-readable explanation of the evaluation outcome."""

    agreement_coherence: float
    """Cross-issue coherence: do the final choices across issues contradict each other?"""

    cross_issue_conflicts: List[str]
    """Pairs of issues whose final choices are mutually incompatible."""

    # ── Protocol-level ────────────────────────────────────────────────────────
    timed_out: bool
    """True when the negotiation exhausted its step budget without agreement."""

    # ── Orchestration ─────────────────────────────────────────────────────────
    recommendation: str
    """Suggested next action: "accept", "request_justification", or "escalate"."""

    # ── Debug ─────────────────────────────────────────────────────────────────
    raw_llm: Optional[Dict[str, Any]] = None
    """Raw LLM response, present only when the LLM evaluation path was used."""


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

            # current_offer must be non-None on every message that has a sao_state
            sao_state = sc.get("sao_state")
            if isinstance(sao_state, dict):
                if sao_state.get("current_offer") is None:
                    errors.append(
                        f"trace[{i}].semantic_context.sao_state.current_offer must not be None"
                    )

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

        # ── options_per_issue / issues alignment ──────────────────────────────
        # Check using the first non-empty issues/options found in the trace
        found_issues: List[str] = []
        found_options: Dict[str, List] = {}
        for msg in trace:
            if not isinstance(msg, dict):
                continue
            sc = msg.get("semantic_context") or {}
            if not found_issues and isinstance(sc.get("issues"), list) and sc["issues"]:
                found_issues = sc["issues"]
            if not found_options and isinstance(sc.get("options_per_issue"), dict) and sc["options_per_issue"]:
                found_options = sc["options_per_issue"]
            if found_issues and found_options:
                break

        if found_issues and found_options:
            for issue in found_issues:
                if issue not in found_options:
                    errors.append(
                        f"options_per_issue is missing key for issue '{issue}'"
                    )
                elif not found_options[issue]:
                    errors.append(
                        f"options_per_issue['{issue}'] must have at least one option"
                    )

        # ── server respond messages must have non-empty current_offer and proposer_id ──
        for i, msg in enumerate(trace):
            if not isinstance(msg, dict):
                continue
            origin = msg.get("origin") or {}
            if origin.get("actor_id") != "negotiation-server":
                continue
            payload = msg.get("payload") or {}
            if payload.get("action") == "respond":
                if not payload.get("current_offer"):
                    errors.append(
                        f"trace[{i}]: server 'respond' message has empty or missing current_offer"
                    )
                if not payload.get("proposer_id"):
                    errors.append(
                        f"trace[{i}]: server 'respond' message has empty or missing proposer_id"
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
                negotiation server to ``sstp_message_trace.json``.

        Returns:
            A :class:`ValidationResult` with the full ACSE semantic
            evaluation, protocol-level outcomes, and orchestration recommendation.

        Raises:
            ValidationInputError: If *trace* is malformed or missing required fields.
        """
        self.validate_input(trace)

        mission_goal, issues, options_per_issue, neg_trace = self._extract_from_sstp_trace(trace)

        logger.info(
            "SemanticAlignmentValidationPipeline.run: session=%s issues=%d rounds=%d status=%s",
            (trace[0].get("semantic_context") or {}).get("session_id", "?"),
            len(issues),
            neg_trace.total_rounds,
            "timeout" if neg_trace.timedout else "agreed",
        )

        llm_provider = get_llm_provider()
        goal_spec = GoalSpecExtractor(llm_provider=llm_provider).extract(
            mission_goal=mission_goal,
            issues=issues,
            options_per_issue=options_per_issue,
        )
        trace_state = TraceStateBuilder().build(neg_trace, issues)
        interaction_signals = InteractionSignalExtractor().extract(trace_state, options_per_issue)
        ae = SemanticAlignmentEvaluator(llm_provider=llm_provider).evaluate(
            goal_spec=goal_spec,
            trace_state=trace_state,
            interaction_signals=interaction_signals,
        )

        recommendation = self._derive_recommendation(ae.needs_intervention, ae.severity.value)

        return ValidationResult(
            needs_intervention=ae.needs_intervention,
            severity=ae.severity.value,
            alignment_score=ae.alignment_score,
            cognitive_alignment=ae.cognitive_alignment,
            failure_modes=list(ae.failure_modes),
            per_issue_evaluations=list(ae.issue_scores.values()),
            reasoning=ae.reasoning,
            agreement_coherence=ae.agreement_coherence or 1.0,
            cross_issue_conflicts=list(ae.cross_issue_conflicts),
            timed_out=neg_trace.timedout,
            recommendation=recommendation,
            raw_llm=ae.raw_llm,
        )

    # ── SSTP trace extraction ─────────────────────────────────────────────────

    def _extract_from_sstp_trace(
        self, trace: List[Any]
    ) -> Tuple[str, List[str], Dict[str, List[str]], NegotiationTrace]:
        """Extract mission_goal, issues, options_per_issue and a NegotiationTrace
        from the SSTP message list accepted by ``run()``.

        Offers are carried in ``payload.action`` / ``payload.round`` /
        ``payload.current_offer`` (server "respond" messages) or
        ``payload.offer`` (agent "counter_offer" messages).  Both the server
        and the responding agent share the same ``round`` value, so a
        sequential list is used to preserve both entries without collision.
        """
        # Mission goal from the first message's payload
        mission_goal: str = (
            (trace[0].get("payload") or {}).get("content_text", "")
            if trace
            else ""
        )

        # Issues and options_per_issue: take first non-empty occurrence
        issues: List[str] = []
        options_per_issue: Dict[str, List[str]] = {}
        for msg in trace:
            sc = msg.get("semantic_context") or {}
            if not issues and isinstance(sc.get("issues"), list) and sc["issues"]:
                issues = sc["issues"]
            if not options_per_issue and isinstance(sc.get("options_per_issue"), dict) and sc["options_per_issue"]:
                options_per_issue = sc["options_per_issue"]
            if issues and options_per_issue:
                break

        # Rounds: server "respond" and agent "counter_offer" messages both carry
        # the same round number, so a sequential list preserves both entries.
        records: List[Tuple[str, Dict[str, str]]] = []  # (proposer, offer)

        for msg in trace:
            origin = msg.get("origin") or {}
            actor = origin.get("actor_id", "")
            payload = msg.get("payload") or {}
            action = payload.get("action", "")
            if action == "respond" and actor == "negotiation-server":
                current_offer = payload.get("current_offer") or {}
                if current_offer:
                    records.append((payload.get("proposer_id") or "server", current_offer))

        rounds: List[RoundRecord] = [
            RoundRecord(round_index=i, proposer_id=proposer, offer=offer)
            for i, (proposer, offer) in enumerate(records)
        ]
        total_rounds = len(records)

        # Final agreement: last ACCEPT_OFFER with a non-empty outcome
        final_agreement: Dict[str, str] = {}
        for msg in reversed(trace):
            sc = msg.get("semantic_context") or {}
            resp = sc.get("sao_response") or {}
            if resp.get("response") == "ACCEPT_OFFER" and isinstance(resp.get("outcome"), dict) and resp["outcome"]:
                final_agreement = {str(k): str(v) for k, v in resp["outcome"].items()}
                break

        timedout = bool(records) and not final_agreement

        neg_trace = NegotiationTrace(
            rounds=rounds,
            final_agreement=final_agreement,
            timedout=timedout,
            broken=False,
            total_rounds=total_rounds,
        )
        return mission_goal, issues, options_per_issue, neg_trace

    # ── Recommendation logic ──────────────────────────────────────────────────

    @staticmethod
    def _derive_recommendation(needs_intervention: bool, severity: str) -> str:
        if not needs_intervention:
            return "accept"
        if severity == "medium":
            return "request_justification"
        return "escalate"
