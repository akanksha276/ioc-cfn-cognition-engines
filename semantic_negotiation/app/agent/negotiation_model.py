# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Negotiation model — component 3 of the semantic negotiation pipeline.

This module implements the negotiation runner that takes issues (from component 1 –
intent discovery) and options per issue (from component 2 – options generation) and
runs a multi-issue bilateral or multilateral negotiation using the NegMAS SAO
(Stacked Alternating Offers) mechanism.

Components 1 and 2 are *not* implemented here; they are expected to produce the
inputs consumed by this module.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import negmas.sao.negotiators as _sao_negotiators
from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as UFun

_STRATEGY_MODULES = [
    "negmas.sao.negotiators",
    "negmas.sao.negotiators.controlled",
    "negmas.sao.negotiators.limited",
    "negmas.sao.negotiators.timebased",
]

# ---------------------------------------------------------------------------
# Per-session concurrency guard
# ---------------------------------------------------------------------------
# Maps (workspace_id, mas_id, session_id) → threading.current_thread().name for
# the running thread. Prevents two concurrent requests with the same scoped
# session from both entering negotiation logic, which would cause ambiguous
# results in SAOMechanism paths.

ActiveSessionKey = Tuple[Optional[str], Optional[str], str]

_ACTIVE_SESSIONS: dict[ActiveSessionKey, str] = {}
_ACTIVE_SESSIONS_LOCK = threading.Lock()


def _active_session_key(
    session_id: str,
    workspace_id: Optional[str] = None,
    mas_id: Optional[str] = None,
) -> ActiveSessionKey:
    """Return the scoped key used for active-session concurrency checks."""
    return (workspace_id, mas_id, session_id)


def _resolve_strategy(name: str) -> type:
    """Resolve a NegMAS negotiator class by name.

    Searches ``negmas.sao.negotiators`` and its common sub-modules.
    Raises ``ValueError`` with available names when not found.
    """
    # Fast path — class is re-exported at the top-level package
    cls = getattr(_sao_negotiators, name, None)
    if cls is not None:
        return cls
    # Slow path — try sub-modules
    for mod_path in _STRATEGY_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    available = sorted(
        n for n in dir(_sao_negotiators) if "Negotiator" in n or "Agent" in n
    )
    raise ValueError(
        f"Unknown negotiator strategy '{name}'. " f"Available: {available}"
    )


@dataclass
class NegotiationParticipant:
    """Represents one participant (agent) in a negotiation session.

    Attributes:
        id: Unique identifier for the participant.
        name: Human-readable name.
        preferences: Per-issue option utilities.
            Shape: ``{issue_id: {option_label: utility_value}}``.
            Utilities should be in ``[0.0, 1.0]`` for best results.
        issue_weights: Optional per-issue importance weights for the linear additive
            utility function. Defaults to equal weights when omitted.
    """

    id: str
    name: str
    preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    issue_weights: Optional[Dict[str, float]] = None


@dataclass
class NegotiationOutcome:
    """Agreed value for a single issue."""

    issue_id: str
    chosen_option: str


@dataclass
class NegotiationResult:
    """Full result returned by :class:`NegotiationModel.run`.

    Attributes:
        agreement: Agreed outcomes per issue, or ``None`` if no agreement was reached.
        timedout: Whether the negotiation exhausted the step budget without agreement.
        broken: Whether a participant explicitly ended the negotiation.
        steps: Number of SAO rounds executed.
        history: Raw NegMAS extended trace as ``(step, negotiator_id, offer)`` tuples.
        round_decisions: Per-round agent decisions keyed by 1-based round number.
            Each value is a list of ``{participant_id, action, offer?}`` dicts.
        round_next_proposer: Participant id of the proposer for the *next* round,
            keyed by 1-based round number.  ``None`` for the final round.
        raw_state: The final ``SAOState`` object for advanced inspection.
    """

    agreement: Optional[List[NegotiationOutcome]]
    timedout: bool
    broken: bool
    steps: int
    history: List[Tuple[int, str, Any]] = field(default_factory=list)
    round_decisions: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    round_next_proposer: Dict[int, Optional[str]] = field(default_factory=dict)
    raw_state: Any = field(default=None, repr=False)
    # All SSTPNegotiateMessage envelopes exchanged during this negotiation, in
    # chronological order (initiate → server outgoing → agent replies, per round).
    sstp_message_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CounterOfferResult:
    """Result of :func:`counter_offer`.

    The route layer maps this to an HTTP response without containing any
    negotiation logic itself.

    Attributes:
        counter_offer_valid: ``True`` when the requesting agent owned the next
            proposer slot and the offer was valid.  ``False`` otherwise.
        rejection_reason: Set when the negotiation ends without a new offer.
            One of ``'invalid_round'``, ``'wrong_turn'``, ``'explicit_reject'``,
            ``'missing_offer'``, or ``'incomplete_offer'``.
        expected_proposer_id: The participant whose turn it actually was
            (populated only on ``rejection_reason='wrong_turn'``).
        accepted_offer: Present when ``action='accept'``.
            Shape: ``{issue_id: chosen_option}``.
        new_offer: Present when a valid counter-offer is accepted.
            Shape: ``{issue_id: option}``.  The route appends this as the
            next round in the trace and returns it for the other agent to
            accept, reject, or counter-offer.
        error_detail: Arbitrary dict forwarded verbatim to the HTTP error body.
    """

    counter_offer_valid: bool
    rejection_reason: Optional[str] = None
    expected_proposer_id: Optional[str] = None
    accepted_offer: Optional[Dict[str, str]] = None
    new_offer: Optional[Dict[str, str]] = None
    error_detail: Optional[Dict[str, Any]] = None


def counter_offer(
    *,
    action: str,
    round_num: int,
    agent_id: str,
    offer_dict: Optional[Dict[str, str]],
    trace_rounds: List[Dict[str, Any]],
    issues: List[str],
    participant_ids: List[str],
    session_id: str = "unknown",
) -> CounterOfferResult:
    """Evaluate an agent's accept / reject / counter-offer decision.

    Pure domain logic — no NegMAS run is triggered.  When the agent supplies a
    valid counter-offer the offer is returned as :attr:`CounterOfferResult.new_offer`
    for the route to append to the running trace and present to the other agent.

    The turn-ownership rule is: each round rotates the proposer slot round-robin
    among the participants.  If an agent submits a counter-offer when it is not
    the expected next proposer the offer is silently discarded
    (``counter_offer_valid=False``, ``rejection_reason='wrong_turn'``).

    Args:
        action: ``'accept'`` | ``'reject'`` | ``'counter_offer'``.
        round_num: 1-based round number the agent is responding to.
        agent_id: Participant id making this decision.
        offer_dict: ``{issue_id: option}`` map.  Required when
            ``action == 'counter_offer'``.
        trace_rounds: Serialised trace rounds so far, each a dict with keys
            ``round``, ``proposer_id``, and ``offer``.
        issues: Ordered list of issue identifiers (used to validate completeness).
        participant_ids: Ordered list of all participant ids (used for
            round-robin turn determination).
        session_id: Used only for log correlation.

    Returns:
        :class:`CounterOfferResult` — the route maps this to the API response.
    """
    total_rounds = len(trace_rounds)

    # ── validate round ────────────────────────────────────────────────
    if round_num < 1 or round_num > total_rounds:
        return CounterOfferResult(
            counter_offer_valid=False,
            rejection_reason="invalid_round",
            error_detail={"round": round_num, "total_rounds": total_rounds},
        )

    # ── accept ────────────────────────────────────────────────────────
    if action == "accept":
        accepted = dict(trace_rounds[round_num - 1]["offer"])
        return CounterOfferResult(
            counter_offer_valid=True,
            accepted_offer=accepted,
        )

    # ── reject ────────────────────────────────────────────────────────
    if action == "reject":
        return CounterOfferResult(
            counter_offer_valid=True,
            rejection_reason="explicit_reject",
        )

    # ── counter_offer — enforce turn ownership ────────────────────────
    current_proposer_id: str = trace_rounds[round_num - 1]["proposer_id"]

    # Round-robin: next proposer is the participant after the current one.
    # Falls back to anyone other than the current proposer for 2-agent sessions.
    if round_num < total_rounds:
        # Trace already has the next round pre-computed (from /initiate).
        expected_next_proposer_id: str = trace_rounds[round_num]["proposer_id"]
    else:
        candidates = [pid for pid in participant_ids if pid != current_proposer_id]
        expected_next_proposer_id = candidates[0] if candidates else current_proposer_id

    logger.info(
        "[%s] counter-offer — agent=%s expected=%s round=%d",
        session_id,
        agent_id,
        expected_next_proposer_id,
        round_num,
    )

    if agent_id != expected_next_proposer_id:
        logger.warning(
            "[%s] counter-offer REJECTED — '%s' offered out-of-turn (expected '%s').",
            session_id,
            agent_id,
            expected_next_proposer_id,
        )
        return CounterOfferResult(
            counter_offer_valid=False,
            rejection_reason="wrong_turn",
            expected_proposer_id=expected_next_proposer_id,
        )

    # ── validate offer completeness ───────────────────────────────────
    if not offer_dict or not isinstance(offer_dict, dict):
        return CounterOfferResult(
            counter_offer_valid=False,
            rejection_reason="missing_offer",
            error_detail={"detail": "action='counter_offer' requires 'offer' key"},
        )

    missing_issues = [i for i in issues if i not in offer_dict]
    if missing_issues:
        return CounterOfferResult(
            counter_offer_valid=False,
            rejection_reason="incomplete_offer",
            error_detail={"missing_issues": missing_issues},
        )

    # ── valid counter-offer — return the new offer for the route to surface ───
    logger.info(
        "[%s] Counter-offer VALID — agent '%s' proposes %s",
        session_id,
        agent_id,
        offer_dict,
    )
    return CounterOfferResult(
        counter_offer_valid=True,
        new_offer=dict(offer_dict),
    )


class NegotiationModel:
    """Runs a multi-issue SAO negotiation via NegMAS.

    This is component 3 of the semantic negotiation pipeline. It consumes:

    * A list of issue identifiers produced by component 1 (intent discovery).
    * A mapping of options per issue produced by component 2 (options generation).
    * A list of :class:`NegotiationParticipant` objects describing each party.

    Internally it builds NegMAS ``Issue`` objects, constructs a
    :class:`~negmas.SAOMechanism`, attaches one negotiator per participant
    (class controlled by *strategy*) with a normalised
    ``LinearAdditiveUtilityFunction`` derived from that participant's stated
    preferences, then runs the mechanism and packages the outcome into a
    :class:`NegotiationResult`.

    Args:
        n_steps: Maximum number of SAO rounds before the negotiation times out.
        strategy: NegMAS negotiator class name (e.g. ``"ConcederTBNegotiator"``)
            or a class object directly.  Defaults to ``"BoulwareTBNegotiator"``
            unless overridden by the ``NEGOTIATOR_STRATEGY`` env variable.
    """

    def __init__(
        self, n_steps: int = 100, strategy: "str | type | None" = None
    ) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps

        if strategy is None:
            from ..config.settings import settings

            strategy = settings.negotiator_strategy

        if isinstance(strategy, str):
            self._negotiator_cls = _resolve_strategy(strategy)
        else:
            self._negotiator_cls = strategy

        logger.info(
            "NegotiationModel — strategy: %s", self._negotiator_cls.__name__
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
        session_id: str = "unknown",
        *,
        workspace_id: Optional[str] = None,
        mas_id: Optional[str] = None,
    ) -> NegotiationResult:
        """Run a full SAO negotiation session and return the result.

        Args:
            issues: Ordered list of issue identifiers.
            options_per_issue: Mapping from each issue id to its candidate options.
            participants: The negotiating parties.  At least two are required.
            session_id: Negotiation session identifier threaded through to any
                :class:`~app.agent.callback_negotiator.SSTPCallbackNegotiator`
                instances so outgoing SSTP messages carry the correct session.
            workspace_id: Optional workspace identifier used to scope the
                in-process concurrency guard.
            mas_id: Optional MAS identifier used to scope the in-process
                concurrency guard.

        Returns:
            A :class:`NegotiationResult` describing what was (or was not) agreed.

        Raises:
            ValueError: If fewer than two participants are supplied, if any issue
                is missing from ``options_per_issue``, or if an options list is empty.
        """
        self._validate_inputs(issues, options_per_issue, participants)

        logger.info(
            "[%s] NegotiationModel.run starting issues=%d participants=%d n_steps=%d",
            session_id,
            len(issues),
            len(participants),
            self.n_steps,
        )
        active_session_key = _active_session_key(
            session_id,
            workspace_id=workspace_id,
            mas_id=mas_id,
        )

        # ── session concurrency guard ──────────────────────────────────────
        # Reject a second concurrent request carrying the same scoped session.
        # Two simultaneous runs would produce ambiguous results.
        with _ACTIVE_SESSIONS_LOCK:
            if active_session_key in _ACTIVE_SESSIONS:
                owner = _ACTIVE_SESSIONS[active_session_key]
                raise ValueError(
                    "Session "
                    f"(workspace_id={workspace_id!r}, mas_id={mas_id!r}, session_id={session_id!r}) "
                    f"is already running in thread '{owner}'. "
                    "Concurrent negotiations must use unique scoped session identifiers."
                )
            _ACTIVE_SESSIONS[active_session_key] = threading.current_thread().name
        logger.debug(
            "[%s] session acquired (thread=%s)",
            session_id,
            threading.current_thread().name,
        )

        try:
            return self._run_negotiation(
                issues, options_per_issue, participants, session_id
            )
        finally:
            with _ACTIVE_SESSIONS_LOCK:
                _ACTIVE_SESSIONS.pop(active_session_key, None)
            logger.debug("[%s] session released", session_id)
            # Purge any _DECISIONS entries that were stored for this session
            # but never consumed (e.g. the runner broke before reading them).
            try:
                from ..agent.batch_callback_runner import _purge_session_decisions

                _purge_session_decisions(session_id)
            except ImportError:
                pass

    def _run_negotiation(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
        session_id: str,
    ) -> NegotiationResult:
        """Internal: execute the negotiation after the session lock is held."""
        negmas_issues = self._build_issues(issues, options_per_issue)
        mechanism = SAOMechanism(issues=negmas_issues, n_steps=self.n_steps)

        for participant in participants:
            ufun = self._build_ufun(participant, issues, options_per_issue, mechanism)
            mechanism.add(self._negotiator_cls(name=participant.name), ufun=ufun)

        state = mechanism.run()
        logger.info(
            "[%s] NegMAS mechanism.run finished agreement=%s timedout=%s broken=%s step=%s",
            session_id,
            state.agreement is not None,
            getattr(state, "timedout", False),
            getattr(state, "broken", False),
            getattr(state, "step", "?"),
        )
        return self._build_result(state, issues, mechanism)

    def counter_offer(
        self,
        *,
        action: str,
        round_num: int,
        agent_id: str,
        offer_dict: Optional[Dict[str, str]],
        trace_rounds: List[Dict[str, Any]],
        issues: List[str],
        participant_ids: List[str],
        session_id: str = "unknown",
    ) -> "CounterOfferResult":
        """Delegate to the module-level :func:`counter_offer` function.

        Kept as an instance method for API consistency; the implementation
        is stateless and does not use any NegMAS machinery.
        """
        return counter_offer(
            action=action,
            round_num=round_num,
            agent_id=agent_id,
            offer_dict=offer_dict,
            trace_rounds=trace_rounds,
            issues=issues,
            participant_ids=participant_ids,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
    ) -> None:
        if len(participants) < 2:
            raise ValueError("At least two participants are required for a negotiation")
        if not issues:
            raise ValueError(
                "At least one negotiable issue is required. "
                "If intent discovery returned none, check the LLM response (JSON with "
                "`negotiable_entities` / `term` fields) and API credentials."
            )
        for issue_id in issues:
            if issue_id not in options_per_issue:
                raise ValueError(
                    f"Issue '{issue_id}' has no entry in options_per_issue"
                )
            if not options_per_issue[issue_id]:
                raise ValueError(f"Issue '{issue_id}' has an empty options list")

    def _build_issues(self, issues: List[str], options_per_issue: Dict[str, List[str]]):
        """Convert issue ids and option lists into NegMAS Issue objects."""
        return [
            make_issue(values=options_per_issue[issue_id], name=issue_id)
            for issue_id in issues
        ]

    def _build_ufun(
        self,
        participant: NegotiationParticipant,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        mechanism: SAOMechanism,
    ) -> UFun:
        """Build a normalised LinearAdditiveUtilityFunction for one participant.

        Options not listed in the participant's preferences receive a utility of
        ``0.0``. Issue weights default to uniform when ``issue_weights`` is None.
        """
        values: Dict[str, Dict[str, float]] = {}
        for issue_id in issues:
            issue_prefs = participant.preferences.get(issue_id, {})
            values[issue_id] = {
                opt: float(issue_prefs.get(opt, 0.0))
                for opt in options_per_issue[issue_id]
            }

        n_issues = len(issues)
        weights: Dict[str, float] = {
            issue_id: (
                float(participant.issue_weights[issue_id])
                if participant.issue_weights and issue_id in participant.issue_weights
                else 1.0 / n_issues
            )
            for issue_id in issues
        }

        return UFun(
            values=values, weights=weights, outcome_space=mechanism.outcome_space
        ).normalize()

    def _build_result(
        self,
        state: Any,
        issues: List[str],
        mechanism: SAOMechanism,
    ) -> NegotiationResult:
        """Translate a raw SAOState into a NegotiationResult."""
        agreement: Optional[List[NegotiationOutcome]] = None
        if state.agreement is not None:
            agreement = [
                NegotiationOutcome(
                    issue_id=issue_id, chosen_option=str(state.agreement[idx])
                )
                for idx, issue_id in enumerate(issues)
            ]

        return NegotiationResult(
            agreement=agreement,
            timedout=state.timedout,
            broken=state.broken,
            steps=state.step,
            history=list(mechanism.extended_trace),
            raw_state=state,
        )
