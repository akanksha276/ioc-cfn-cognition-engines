# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Batch callback runner — drives SAO-style rounds via batched agent messages.
------------
This runner collects **all** per-participant decision requests for a round into
a **single** batch:

    Forward:   List[SSTPNegotiateMessage]  (one message per participant)
    Return:    List[SSTPNegotiateMessage]  (same length)

Each message carries ``payload.participant_id`` so the receiving endpoint knows
which participant it is addressing.

Round semantics
---------------
**Round 1 (server-seeded):**
  The negotiation server picks a *random* initial offer (one option per issue
  chosen uniformly at random) and forwards it to **all** participants as
  ``action=respond``.  There are no shadow calls.  A participant that accepts
  immediately ends the negotiation.  The server also randomly selects which
  participant will be the proposer in round 2.

**Rounds 2 … N (alternating SAO):**
  - **Proposer** (rotates, starting from the server's random pick):
    receives ``action=propose``.  Returns a new offer via
    ``{ "action": "counter_offer", "offer": {...}, "reason": "..." }``.
  - **Responders** (all others): receive ``action=respond`` with
    ``current_offer`` = the standing offer from the previous round.
    Return ``{ "action": "accept" | "reject", "reason": "..." }``.
  Every message represents a real decision (no shadow / seeding flag on wire).

Evaluation after each call
--------------------------
1. Round 1: if any participant accepts the random offer → agreement.
2. Rounds 2+: if any responder accepts the standing offer → agreement.
3. If the proposer returns no valid offer → ``broken``.
4. If all rounds exhaust without agreement → ``timedout``.

Agent reply ``reason`` field
----------------------------
Every agent reply may include ``payload.reason`` (accept, reject, or
counter_offer).  The server copies it into ``round_decisions`` and the
``sstp_message_trace`` via :mod:`reply_payload_utils`; it does not affect
offer validation or agreement detection.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

_workspace_root = str(Path(__file__).resolve().parents[3])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negmas_sao import SAOState  # noqa: E402
from protocol.sstp.negotiate import (  # noqa: E402
    NegotiateSemanticContext,
    dump_negotiate_message_json,
)

from .negotiation_model import (  # noqa: E402  (same package)
    NegotiationOutcome,
    NegotiationParticipant,
    NegotiationResult,
)
from .offer_validation import validate_and_snap_offer  # noqa: E402

# Persist action/offer/reason on each participant decision (reason is metadata only).
from .reply_payload_utils import round_decision_from_reply  # noqa: E402

logger = logging.getLogger(__name__)


def store_decisions(
    key: str, replies: list
) -> None:  # noqa: U100 — kept for backward compat
    """No-op stub retained so existing imports don't break."""


def _purge_session_decisions(session_id: str) -> None:  # noqa: U100
    """No-op stub retained so negotiation_model.py's finally-block call doesn't break."""


# ---------------------------------------------------------------------------
# Session state — holds everything needed to continue a paused negotiation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NegotiationSession:
    """Mutable state snapshot for one in-progress turn-by-turn negotiation.

    Created by :meth:`BatchCallbackRunner.start()` and updated in place by
    :meth:`BatchCallbackRunner.step()`.

    ``phase`` tracks exactly what kind of reply the server is waiting for next:
    - ``"round1_respond"``  — all agents must accept/reject the server's seed offer.
    - ``"propose"``         — the designated proposer must return a counter-offer.
    - ``"respond"``         — all non-proposers must accept/reject the proposer's offer.
    """

    session_id: str
    issues: list[str]
    options_per_issue: dict[str, list[str]]
    participants: list[Any]  # list[NegotiationParticipant]
    n_steps: int
    # current SAO step index (1-based, advances after each full propose+respond pair)
    sao_step: int = 1
    standing_offer: dict[str, str] = dataclasses.field(default_factory=dict)
    standing_offer_proposer_id: str = "server"
    first_proposer_idx: int = 0
    negmas_history: list[tuple] = dataclasses.field(default_factory=list)
    round_decisions: dict[int, list[dict[str, Any]]] = dataclasses.field(
        default_factory=dict
    )
    round_next_proposer: dict[int, Optional[str]] = dataclasses.field(
        default_factory=dict
    )
    sstp_message_trace: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # Mission metadata — stored at initiation so it can be embedded in the commit.
    content_text: str = ""
    agents_negotiating: list[str] = dataclasses.field(default_factory=list)
    # Phase drives what step() does with the incoming replies.
    phase: str = "respond"  # always "respond" — every round is a single broadcast
    # Index into participants of the agent authorised to counter_offer this round.
    next_proposer_idx: int = 0
    # Kept for backward compat, no longer consumed by step().
    pending_new_offer: dict[str, str] = dataclasses.field(default_factory=dict)
    pending_proposer_id: str = ""
    pending_round_decs: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # Set when a counter_offer was downgraded to reject due to offer validation failure.
    # Carried into the next broadcast so the offending agent can self-correct.
    prev_offer_validation_failure: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# SSTP message helpers
# ---------------------------------------------------------------------------


def build_callback_message(
    payload: dict[str, Any],
    participant_id: str,
    session_id: str,
    sao_state: SAOState | None = None,
    issues: list[str] | None = None,
    options_per_issue: dict[str, list[str]] | None = None,
    offer_validation_failure: dict[str, Any] | None = None,
) -> SSTPNegotiateMessage:
    """Build a validated ``SSTPNegotiateMessage`` for one participant decision request.

    ``message_id`` is a deterministic UUID-5 derived from session, participant,
    and payload content — stable across retries of the same round state.
    ``sao_state`` carries the full SAO mechanism snapshot so the receiver has
    complete context about the current negotiation state.
    ``issues`` and ``options_per_issue`` are placed in ``semantic_context`` (not
    in ``payload``) so that the full negotiation space travels in the structured
    envelope rather than the free-form payload dict.
    ``offer_validation_failure`` is placed in ``semantic_context`` so it is a
    typed, schema-visible field — not buried in the free-form payload dict.
    """
    payload_str = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{session_id}:{participant_id}:{payload_hash}",
        )
    )
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="negotiation-server", tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            sao_state=sao_state,
            issues=issues or [],
            options_per_issue=options_per_issue or {},
            offer_validation_failure=offer_validation_failure,
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


def unwrap_reply(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the inner payload from an SSTP-wrapped reply dict.

    If the reply is a full ``SSTPNegotiateMessage`` envelope (has a ``"payload"``
    key whose value is a dict), return that inner payload.  Otherwise return
    *data* unchanged (plain-JSON reply compatibility).
    """
    inner = data.get("payload")
    if isinstance(inner, dict):
        return inner
    return data


# ---------------------------------------------------------------------------
# Step budget heuristic
# ---------------------------------------------------------------------------


def compute_n_steps(
    n_agents: int,
    n_issues: int,
    options_per_issue: dict[str, list[str]],
    *,
    min_steps: int = 50,
    proposer_rounds: int = 3,
    concession_fraction: float = 0.3,
    safety_buffer: float = 1.5,
) -> int:
    """Estimate a sensible SAO step budget from first principles of the SAO mechanism.

    Derivation
    ----------
    In SAO, each step gives one agent the proposer role while all others
    respond.  For convergence under BoulwareTBNegotiator:

    1. **Agent rotation budget** — every agent needs at least *proposer_rounds*
       turns as proposer to explore offers: ``base = proposer_rounds * n_agents``.

    2. **Boulware concession window** — BoulwareTBNegotiator uses
       ``relative_time = step / n_steps`` and only enters a meaningful
       concession zone when ``relative_time → 1`` (empirically the last
       ~30 % of steps are productive).  All n agents must be in this window
       simultaneously, so the productive window must contain the full rotation
       budget::

           productive_steps ≥ proposer_rounds * n_agents
           concession_fraction * n_steps ≥ proposer_rounds * n_agents
           n_steps ≥ (proposer_rounds * n_agents) / concession_fraction

    3. **Issue dimensionality** — each issue requires independent convergence;
       the budget scales linearly with the number of issues.

    4. **Option space** — more options give finer concession resolution, but
       agents follow a concession path rather than enumerate all options.
       The relevant factor is the number of "concession levels", which grows
       logarithmically: ``ln(avg_options)``.

    Combined formula::

        n_steps = ceil(
            (proposer_rounds * n_agents / concession_fraction)
            * n_issues
            * ln(avg_options)
            * safety_buffer
        )

    Args:
        n_agents:            Number of negotiating participants.
        n_issues:            Number of negotiable issues in the space.
        options_per_issue:   Mapping of issue → list of options (used to
                             compute average option count per issue).
        min_steps:           Hard floor — never return fewer than this.
        proposer_rounds:     Minimum turns each agent needs as proposer
                             (default 3).
        concession_fraction: Fraction of steps that fall in BoulwareTBNegotiator's
                             productive concession window (default 0.3).
        safety_buffer:       Multiplicative headroom on top of the theoretical
                             estimate (default 1.5 → 50 % headroom).

    Returns:
        Recommended ``n_steps`` value to pass to :class:`BatchCallbackRunner`.
    """
    import math

    avg_options: float = (
        sum(len(v) for v in options_per_issue.values()) / len(options_per_issue)
        if options_per_issue
        else 3.0
    )
    # Clamp to ≥ 2 to keep log positive and meaningful
    avg_options = max(avg_options, 2.0)

    rotation_budget = proposer_rounds * n_agents / concession_fraction
    estimate = math.ceil(
        rotation_budget * n_issues * math.log(avg_options) * safety_buffer
    )
    out = max(min_steps, estimate)
    logger.info(
        "compute_n_steps n_agents=%d n_issues=%d -> n_steps=%d",
        n_agents,
        n_issues,
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Batch round runner
# ---------------------------------------------------------------------------


class BatchCallbackRunner:
    """SAO-style negotiation runner that drives rounds via batch HTTP calls.

    Each round dispatches a ``List[SSTPNegotiateMessage]`` (one per participant)
    to a shared agent endpoint and waits for the replies.

    Args:
        n_steps: Maximum rounds before the negotiation times out.
        timeout: HTTP request timeout in seconds (per batch call).
    """

    def __init__(self, n_steps: int = 100, timeout: float = 30.0) -> None:
        self.n_steps = n_steps
        self._timeout = timeout
        self._http = httpx.Client(timeout=timeout)
        logger.info(
            "BatchCallbackRunner initialized n_steps=%d timeout=%s",
            n_steps,
            timeout,
        )
        # Maps "session_id:round_num:participant_id" → SHA-256 hex of outbound
        # sao_state so we can detect agent tampering on the echo.
        self._sao_state_checksums: dict[str, str] = {}

    def run(
        self,
        issues: list[str],
        options_per_issue: dict[str, list[str]],
        participants: list[NegotiationParticipant],
        session_id: str = "unknown",
        agent_url: str = "",
    ) -> NegotiationResult:
        """Execute the negotiation and return the result.

        Args:
            issues: Ordered list of issue identifiers.
            options_per_issue: ``{issue_id: [option, ...]}`` for every issue.
            participants: All negotiating parties.
            session_id: Correlation identifier threaded through every message.
            agent_url: Shared HTTP endpoint to POST batched messages to.

        Returns:
            :class:`~app.agent.negotiation_model.NegotiationResult`.
        """
        callback_url: str = agent_url

        # ── Round 1: server picks a random initial offer ──────────────
        standing_offer: dict[str, str] = {
            issue: random.choice(options_per_issue[issue]) for issue in issues
        }
        # The proposer id of the current standing offer (used to prevent an
        # agent from triggering agreement by accepting their own proposal).
        standing_offer_proposer_id: str = "server"
        # Randomly decide which participant proposes in round 2.
        first_proposer_idx: int = random.randrange(len(participants))
        logger.info(
            "[%s] server initial offer: %s  |  first proposer: %s",
            session_id,
            standing_offer,
            participants[first_proposer_idx].id,
        )

        # NegMAS-compatible history: (step, proposer_name, offer_tuple)
        negmas_history: list[tuple] = [
            (
                -1,
                "server",
                tuple(standing_offer[issue] for issue in issues),
            )
        ]
        # Per-round agent decisions: {round_num: [{participant_id, action, offer?}]}
        round_decisions: dict[int, list[dict[str, Any]]] = {}
        # Per-round next proposer: {round_num: id of proposer in next round, None for final}
        round_next_proposer: dict[int, Optional[str]] = {}
        # Chronological log of every SSTPNegotiateMessage sent to or received from agents.
        sstp_message_trace: list[dict[str, Any]] = []

        # next_proposer_idx cycles through participants each round.
        next_proposer_idx = first_proposer_idx
        # Key 0 = next proposer after the server's initial offer (first SAO proposer).
        round_next_proposer[0] = participants[next_proposer_idx].id
        # Tracks the last offer validation failure so it can be fed back in the
        # next broadcast, giving the LLM agent context to self-correct.
        offer_validation_failure: dict[str, Any] | None = None

        for step in range(self.n_steps):
            round_num = step + 1
            next_proposer_id = participants[next_proposer_idx].id

            # ── Build and dispatch a respond broadcast to ALL agents ──
            respond_state = SAOState(
                step=step,
                relative_time=step / self.n_steps,
                running=True,
                started=True,
                n_negotiators=len(participants),
                current_offer=standing_offer,
                current_proposer=standing_offer_proposer_id,
            )
            payload: dict[str, Any] = {
                "action": "respond",
                "participant_id": "server",
                "next_proposer_id": next_proposer_id,
                "round": round_num,
                "n_steps": self.n_steps,
                "allowed_actions": ["accept", "reject", "counter_offer"],
                "current_offer": standing_offer,
                "proposer_id": standing_offer_proposer_id,
            }
            broadcast_msg = build_callback_message(
                payload,
                "broadcast",
                session_id,
                sao_state=respond_state,
                issues=issues,
                options_per_issue=options_per_issue,
                offer_validation_failure=offer_validation_failure,
            )
            offer_validation_failure = None  # consumed — reset for next round
            messages: list[SSTPNegotiateMessage] = [broadcast_msg]

            sstp_message_trace.extend(dump_negotiate_message_json(m) for m in messages)
            self._store_sao_checksums(messages, session_id, round_num)
            replies_raw = self._post_batch(
                callback_url,
                messages,
                session_id,
                round_num,
                n_expected_replies=len(participants),
            )
            if replies_raw is None:
                return NegotiationResult(
                    agreement=None,
                    timedout=False,
                    broken=True,
                    steps=round_num,
                    history=negmas_history,
                    sstp_message_trace=sstp_message_trace,
                )
            self._verify_sao_checksums(messages, replies_raw, session_id, round_num)
            sstp_message_trace.extend(replies_raw)

            replies_by_pid = {
                unwrap_reply(r).get("participant_id", ""): unwrap_reply(r)
                for r in replies_raw
            }

            # ── Downgrade unauthorized counter_offers → reject ────────
            for pid, reply in replies_by_pid.items():
                if reply.get("action") == "counter_offer" and pid != next_proposer_id:
                    logger.warning(
                        "[%s] round %d — agent '%s' submitted counter_offer but is not"
                        " next_proposer ('%s'); downgrading to reject",
                        session_id,
                        round_num,
                        pid,
                        next_proposer_id,
                    )
                    reply["action"] = "reject"
                    reply.pop("offer", None)

            next_proposer_reply = replies_by_pid.get(
                next_proposer_id, {"action": "reject"}
            )
            counter_offered = next_proposer_reply.get("action") == "counter_offer"

            if counter_offered:
                offer_raw = next_proposer_reply.get("offer")
                if isinstance(offer_raw, dict):
                    new_offer, problems = validate_and_snap_offer(
                        offer_raw,
                        issues,
                        options_per_issue,
                        session_id=session_id,
                        agent_id=next_proposer_id,
                    )
                else:
                    new_offer, problems = {}, ["offer is not a dict"]

                if len(new_offer) == len(issues):
                    next_proposer = next(
                        p for p in participants if p.id == next_proposer_id
                    )
                    if problems:
                        logger.info(
                            "[%s] round %d — next_proposer '%s' offer snapped: %s",
                            session_id,
                            round_num,
                            next_proposer_id,
                            "; ".join(problems),
                        )
                    logger.info(
                        "[%s] round %d — next_proposer '%s' counter_offers %s",
                        session_id,
                        round_num,
                        next_proposer_id,
                        new_offer,
                    )
                    negmas_history.append(
                        (
                            step,
                            next_proposer.name,
                            tuple(new_offer[issue] for issue in issues),
                        )
                    )
                    standing_offer = new_offer
                    standing_offer_proposer_id = next_proposer_id
                else:
                    logger.warning(
                        "[%s] round %d — next_proposer '%s' returned invalid offer: %s;"
                        " treating as reject (problems: %s)",
                        session_id,
                        round_num,
                        next_proposer_id,
                        next_proposer_reply,
                        problems,
                    )
                    offer_validation_failure = {
                        "rejected_agent_id": next_proposer_id,
                        "round": round_num,
                        "problems": problems,
                        "hint": (
                            "Your counter_offer was rejected because it contained "
                            "unrecognised issue keys or option values. "
                            "Use only the issue identifiers and option values listed "
                            "in semantic_context.issues and semantic_context.options_per_issue."
                        ),
                    }
                    next_proposer_reply["action"] = "reject"
                    next_proposer_reply.pop("offer", None)
                    counter_offered = False
            else:
                # next_proposer did not counter-offer (accepted or rejected).
                # Only append a history entry if it is NOT the final agreement
                # round — when every agent accepted, the outcome is captured in
                # final_agreement; adding an entry here would produce a spurious
                # extra round in the trace.
                if not all(
                    replies_by_pid.get(p.id, {"action": "reject"}).get("action")
                    == "accept"
                    for p in participants
                ):
                    next_proposer = next(
                        p for p in participants if p.id == next_proposer_id
                    )
                    logger.info(
                        "[%s] round %d — next_proposer '%s' %s; re-tabling standing offer in history",
                        session_id,
                        round_num,
                        next_proposer_id,
                        next_proposer_reply.get("action", "reject"),
                    )
                    negmas_history.append(
                        (
                            step,
                            next_proposer.name,
                            tuple(standing_offer[issue] for issue in issues),
                        )
                    )

            # ── Record all N decisions ────────────────────────────────
            all_replies = [
                replies_by_pid.get(p.id, {"action": "reject"}) for p in participants
            ]
            round_decs = [
                round_decision_from_reply(p.id, r) for p, r in zip(participants, all_replies)
            ]
            round_decisions[round_num] = round_decs
            round_next_proposer[round_num] = (
                None  # updated below if negotiation continues
            )

            # Agreement: all N accepted AND next_proposer did not counter_offer.
            if not counter_offered and all(
                r.get("action") == "accept" for r in all_replies
            ):
                logger.info(
                    "[%s] round %d — agreement on offer: %s",
                    session_id,
                    round_num,
                    standing_offer,
                )
                agreement = [
                    NegotiationOutcome(
                        issue_id=issue, chosen_option=standing_offer[issue]
                    )
                    for issue in issues
                ]
                return NegotiationResult(
                    agreement=agreement,
                    timedout=False,
                    broken=False,
                    steps=round_num,
                    history=negmas_history,
                    round_decisions=round_decisions,
                    round_next_proposer=round_next_proposer,
                    sstp_message_trace=sstp_message_trace,
                )

            # Cycle next_proposer for the next round.
            next_proposer_idx = (next_proposer_idx + 1) % len(participants)
            round_next_proposer[round_num] = participants[next_proposer_idx].id

        # exhausted step budget
        return NegotiationResult(
            agreement=None,
            timedout=True,
            broken=False,
            steps=self.n_steps,
            history=negmas_history,
            round_decisions=round_decisions,
            round_next_proposer=round_next_proposer,
            sstp_message_trace=sstp_message_trace,
        )

    # ------------------------------------------------------------------
    # Turn-by-turn API (stateful stepper)
    # ------------------------------------------------------------------

    def start(
        self,
        issues: list[str],
        options_per_issue: dict[str, list[str]],
        participants: list[NegotiationParticipant],
        session_id: str = "unknown",
    ) -> tuple[NegotiationSession, list[dict[str, Any]]]:
        """Initialise a new negotiation session and return the first round's messages.

        Creates a :class:`NegotiationSession`, seeds the server's random initial
        offer, builds the round-1 ``respond`` batch, and returns:

        ``(session, first_round_messages)``

        where ``first_round_messages`` is a list of serialised
        ``SSTPNegotiateMessage`` dicts ready to be forwarded to the agents.
        The caller must persist *session* and pass it back to :meth:`step`.
        """
        standing_offer: dict[str, str] = {
            issue: random.choice(options_per_issue[issue]) for issue in issues
        }
        first_proposer_idx = random.randrange(len(participants))
        history: list[tuple] = [
            (-1, "server", tuple(standing_offer[issue] for issue in issues))
        ]
        logger.info(
            "[%s] start — server initial offer: %s  |  first proposer: %s",
            session_id,
            standing_offer,
            participants[first_proposer_idx].id,
        )

        sess = NegotiationSession(
            session_id=session_id,
            issues=issues,
            options_per_issue=options_per_issue,
            participants=participants,
            n_steps=self.n_steps,
            sao_step=1,
            standing_offer=standing_offer,
            standing_offer_proposer_id="server",
            first_proposer_idx=first_proposer_idx,
            negmas_history=history,
            phase="respond",
            next_proposer_idx=first_proposer_idx,
        )

        # Build round-1 messages (all agents respond to server's random offer).
        round1_state = SAOState(
            step=0,
            relative_time=0.0,
            running=True,
            started=True,
            n_negotiators=len(participants),
            current_offer=standing_offer,
            current_proposer="server",
        )
        # next_proposer_r1 = who will propose in round 1 (first SAO proposer).
        next_proposer_r1 = participants[first_proposer_idx].id
        # Seed round 0 = next proposer for the server's initial-offer row in history.
        round_next_proposer: dict[int, Optional[str]] = {0: next_proposer_r1}
        broadcast_msg = build_callback_message(
            {
                "action": "respond",
                "participant_id": "server",
                "next_proposer_id": next_proposer_r1,
                "round": 1,
                "n_steps": self.n_steps,
                "allowed_actions": ["accept", "reject", "counter_offer"],
                "current_offer": standing_offer,
                "proposer_id": "server",
            },
            "broadcast",
            session_id,
            sao_state=round1_state,
            issues=issues,
            options_per_issue=options_per_issue,
        )

        serialised = [dump_negotiate_message_json(broadcast_msg)]
        sess.sstp_message_trace.extend(serialised)
        sess.round_next_proposer = round_next_proposer
        return sess, serialised

    def step(
        self,
        sess: NegotiationSession,
        agent_replies: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]] | None, NegotiationResult | None]:
        """Advance the negotiation by one batch of agent replies.

        Args:
            sess: The mutable session returned by :meth:`start` (or updated by
                  a previous :meth:`step` call).
            agent_replies: The list of raw reply dicts returned by the agents for
                           the messages dispatched in the previous call.

        Returns:
            A ``(status, next_messages, result)`` triple:

            - ``("ongoing", next_messages, None)`` — more rounds to go.
            - ``("agreed"|"broken"|"timeout", None, result)`` — negotiation done.
        """
        issues = sess.issues
        options_per_issue = sess.options_per_issue
        participants = sess.participants
        session_id = sess.session_id
        logger.info(
            "[%s] step round=%d phase=%s replies=%d",
            session_id,
            sess.sao_step,
            sess.phase,
            len(agent_replies),
        )

        # Record agent replies in the trace.
        sess.sstp_message_trace.extend(agent_replies)
        # Map replies by participant_id for position-independent matching.
        replies_by_pid = {
            unwrap_reply(r).get("participant_id", ""): unwrap_reply(r)
            for r in agent_replies
        }

        # ── Phase: respond ────────────────────────────────────────────
        # Single unified handler — every round is a respond broadcast.
        # The agent at next_proposer_idx is authorised to counter_offer;
        # all others may only accept or reject.
        if sess.phase == "respond":
            round_num = sess.sao_step
            next_proposer_id = participants[sess.next_proposer_idx].id

            # Downgrade any unauthorized counter_offer → reject.
            for pid, reply in replies_by_pid.items():
                if reply.get("action") == "counter_offer" and pid != next_proposer_id:
                    logger.warning(
                        "[%s] round %d — agent '%s' submitted counter_offer but is not"
                        " next_proposer ('%s'); downgrading to reject",
                        session_id,
                        round_num,
                        pid,
                        next_proposer_id,
                    )
                    reply["action"] = "reject"
                    reply.pop("offer", None)

            next_proposer_reply = replies_by_pid.get(
                next_proposer_id, {"action": "reject"}
            )
            counter_offered = next_proposer_reply.get("action") == "counter_offer"

            if counter_offered:
                offer_raw = next_proposer_reply.get("offer")
                if isinstance(offer_raw, dict):
                    new_offer, problems = validate_and_snap_offer(
                        offer_raw,
                        issues,
                        options_per_issue,
                        session_id=session_id,
                        agent_id=next_proposer_id,
                    )
                else:
                    new_offer, problems = {}, ["offer is not a dict"]

                if len(new_offer) == len(issues):
                    next_proposer = next(
                        p for p in participants if p.id == next_proposer_id
                    )
                    if problems:
                        logger.info(
                            "[%s] round %d — next_proposer '%s' offer snapped: %s",
                            session_id,
                            round_num,
                            next_proposer_id,
                            "; ".join(problems),
                        )
                    sess.negmas_history.append(
                        (
                            round_num - 1,  # 0-indexed step, consistent with run() path
                            next_proposer.name,
                            tuple(new_offer[issue] for issue in issues),
                        )
                    )
                    sess.standing_offer = new_offer
                    sess.standing_offer_proposer_id = next_proposer_id
                else:
                    logger.warning(
                        "[%s] round %d — next_proposer '%s' returned invalid offer: %s;"
                        " treating as reject (problems: %s)",
                        session_id,
                        round_num,
                        next_proposer_id,
                        next_proposer_reply,
                        problems,
                    )
                    sess.prev_offer_validation_failure = {
                        "rejected_agent_id": next_proposer_id,
                        "round": round_num,
                        "problems": problems,
                        "hint": (
                            "Your counter_offer was rejected because it contained "
                            "unrecognised issue keys or option values. "
                            "Use only the issue identifiers and option values listed "
                            "in semantic_context.issues and semantic_context.options_per_issue."
                        ),
                    }
                    next_proposer_reply["action"] = "reject"
                    next_proposer_reply.pop("offer", None)
                    counter_offered = False
            else:
                # next_proposer did not counter-offer (accepted or rejected).
                # Only append a history entry if it is NOT the final agreement
                # round — when every agent accepted, the outcome is captured in
                # final_agreement; adding an entry here would produce a spurious
                # extra round in the trace.
                if not all(
                    replies_by_pid.get(p.id, {"action": "reject"}).get("action")
                    == "accept"
                    for p in participants
                ):
                    next_proposer = next(
                        p for p in participants if p.id == next_proposer_id
                    )
                    logger.info(
                        "[%s] round %d — next_proposer '%s' %s; re-tabling standing offer in history",
                        session_id,
                        round_num,
                        next_proposer_id,
                        next_proposer_reply.get("action", "reject"),
                    )
                    sess.negmas_history.append(
                        (
                            round_num - 1,  # 0-indexed step, consistent with run() path
                            next_proposer.name,
                            tuple(sess.standing_offer[issue] for issue in issues),
                        )
                    )

            # Record all N decisions for this round.
            all_replies = [
                replies_by_pid.get(p.id, {"action": "reject"}) for p in participants
            ]
            decs = [
                round_decision_from_reply(p.id, r) for p, r in zip(participants, all_replies)
            ]
            sess.round_decisions[round_num] = decs
            # Mark this as the final round (no next proposer) until proven otherwise.
            sess.round_next_proposer[round_num] = None

            # Agreement: all N agents accepted the current offer AND
            # next_proposer did not counter_offer (which replaces it).
            if not counter_offered and all(
                r.get("action") == "accept" for r in all_replies
            ):
                logger.info(
                    "[%s] step agreement round=%d offer=%s",
                    session_id,
                    round_num,
                    sess.standing_offer,
                )
                return (
                    "agreed",
                    None,
                    NegotiationResult(
                        agreement=[
                            NegotiationOutcome(
                                issue_id=issue,
                                chosen_option=sess.standing_offer[issue],
                            )
                            for issue in issues
                        ],
                        timedout=False,
                        broken=False,
                        steps=round_num,
                        history=sess.negmas_history,
                        round_decisions=sess.round_decisions,
                        round_next_proposer=sess.round_next_proposer,
                        sstp_message_trace=sess.sstp_message_trace,
                    ),
                )

            # Not agreed — advance step, cycle next_proposer.
            sess.sao_step += 1
            sess.next_proposer_idx = (sess.next_proposer_idx + 1) % len(participants)
            # Record next proposer for the just-completed round.
            sess.round_next_proposer[round_num] = participants[
                sess.next_proposer_idx
            ].id
            if sess.sao_step > sess.n_steps:
                logger.info(
                    "[%s] step timeout after sao_step=%d n_steps=%d",
                    session_id,
                    sess.sao_step,
                    sess.n_steps,
                )
                return (
                    "timeout",
                    None,
                    NegotiationResult(
                        agreement=None,
                        timedout=True,
                        broken=False,
                        steps=sess.n_steps,
                        history=sess.negmas_history,
                        round_decisions=sess.round_decisions,
                        round_next_proposer=sess.round_next_proposer,
                        sstp_message_trace=sess.sstp_message_trace,
                    ),
                )
            return self._dispatch_respond(sess)

        # Should never reach here.
        raise RuntimeError(f"[{session_id}] Unknown phase: {sess.phase!r}")

    def _dispatch_respond(
        self, sess: NegotiationSession
    ) -> tuple[str, list[dict[str, Any]], None]:
        """Build a respond broadcast for ALL participants and return an 'ongoing' triple.

        ``next_proposer_idx`` must already be set to the current round's value
        before calling.  The agent at that index is authorised to counter_offer;
        all others may only accept or reject.  The server enforces this in step().
        """
        issues = sess.issues
        options_per_issue = sess.options_per_issue
        participants = sess.participants
        session_id = sess.session_id
        round_num = sess.sao_step  # sao_step has already been incremented to this round
        next_proposer_id = participants[sess.next_proposer_idx].id

        respond_state = SAOState(
            step=round_num - 1,  # SAOState uses 0-based step
            relative_time=(round_num - 1) / sess.n_steps,
            running=True,
            started=True,
            n_negotiators=len(participants),
            current_offer=sess.standing_offer,
            current_proposer=sess.standing_offer_proposer_id,
        )
        respond_broadcast = build_callback_message(
            {
                "action": "respond",
                "participant_id": "server",
                "next_proposer_id": next_proposer_id,
                "round": round_num,
                "n_steps": sess.n_steps,
                "allowed_actions": ["accept", "reject", "counter_offer"],
                "current_offer": sess.standing_offer,
                "proposer_id": sess.standing_offer_proposer_id,
            },
            "broadcast",
            session_id,
            sao_state=respond_state,
            issues=issues,
            options_per_issue=options_per_issue,
            offer_validation_failure=sess.prev_offer_validation_failure,
        )
        serialised = [dump_negotiate_message_json(respond_broadcast)]
        sess.sstp_message_trace.extend(serialised)
        sess.prev_offer_validation_failure = None  # consumed — reset for next round
        sess.phase = "respond"
        logger.info(
            "[%s] dispatch respond round %d agents=%d next_proposer=%s",
            session_id,
            round_num,
            len(participants),
            next_proposer_id,
        )
        return "ongoing", serialised, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_batch(
        self,
        callback_url: str,
        messages: list[SSTPNegotiateMessage],
        session_id: str,
        round_num: int,
        n_expected_replies: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """POST ``messages`` to the agent's /decide endpoint and return the
        replies directly from the HTTP response body.

        The agent's /decide endpoint is a **blocking** call: it processes all
        messages synchronously and returns ``List[SSTPNegotiateMessage]`` as
        the response body.  No separate callback to /agents-decisions is needed.

        Returns the list of reply dicts (one per message) or None on error.
        """
        try:
            resp = self._http.post(
                callback_url,
                json=[dump_negotiate_message_json(m) for m in messages],
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as exc:
            logger.error(
                "[%s] round %d — /decide HTTP error: %s", session_id, round_num, exc
            )
            return None
        except Exception as exc:
            logger.error(
                "[%s] round %d — /decide unexpected error: %s",
                session_id,
                round_num,
                exc,
            )
            return None

        if not isinstance(result, list):
            logger.error(
                "[%s] round %d — /decide returned non-list body: %s",
                session_id,
                round_num,
                type(result),
            )
            return None

        expected = (
            n_expected_replies if n_expected_replies is not None else len(messages)
        )
        if len(result) != expected:
            logger.error(
                "[%s] round %d — /decide returned %d replies (expected %d)",
                session_id,
                round_num,
                len(result),
                expected,
            )
            return None
        logger.info(
            "[%s] round %d — /decide OK replies=%d",
            session_id,
            round_num,
            len(result),
        )
        return result

    def _store_sao_checksums(
        self,
        messages: list[SSTPNegotiateMessage],
        session_id: str,
        round_num: int,
    ) -> None:
        """Compute and store SHA-256 checksums for every outbound ``sao_state``.

        Keys follow the pattern ``"{session_id}:{round_num}:{participant_id}"``
        so they can be looked up when the corresponding reply arrives.
        """
        for msg in messages:
            participant_id = (msg.payload or {}).get("participant_id", "unknown")
            ck_key = f"{session_id}:{round_num}:{participant_id}"
            sc = msg.semantic_context
            sao_state_dict = (
                sc.sao_state.model_dump(mode="json") if (sc and sc.sao_state) else None
            )
            if sao_state_dict is not None:
                self._sao_state_checksums[ck_key] = hashlib.sha256(
                    json.dumps(sao_state_dict, sort_keys=True).encode()
                ).hexdigest()

    def _verify_sao_checksums(
        self,
        messages: list[SSTPNegotiateMessage],
        replies_raw: list[dict[str, Any]],
        session_id: str,
        round_num: int,
    ) -> None:
        """Verify that each reply's echoed ``sao_state`` matches the stored checksum.

        Outbound negotiate JSON omits ``semantic_context.sao_state``, so agents
        often omit it in replies — those cases are skipped without a warning.
        Logs WARNING on checksum mismatch; DEBUG on successful verification.
        """
        for msg, reply in zip(messages, replies_raw):
            participant_id = (msg.payload or {}).get("participant_id", "unknown")
            ck_key = f"{session_id}:{round_num}:{participant_id}"
            stored_ck = self._sao_state_checksums.get(ck_key)
            if stored_ck is None:
                continue  # nothing was stored (no sao_state was sent)

            sc_dict = reply.get("semantic_context") or {}
            echoed_sao_dict = (
                sc_dict.get("sao_state") if isinstance(sc_dict, dict) else None
            )
            if echoed_sao_dict is None:
                continue

            echoed_ck = hashlib.sha256(
                json.dumps(echoed_sao_dict, sort_keys=True).encode()
            ).hexdigest()
            if echoed_ck != stored_ck:
                logger.warning(
                    "[%s] round %d — TAMPERED sao_state from participant '%s' "
                    "(expected=…%s  received=…%s)",
                    session_id,
                    round_num,
                    participant_id,
                    stored_ck[-8:],
                    echoed_ck[-8:],
                )
            else:
                logger.debug(
                    "[%s] round %d — sao_state integrity OK for participant '%s' (…%s)",
                    session_id,
                    round_num,
                    participant_id,
                    stored_ck[-8:],
                )
