# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all negotiation agents.

Every agent must implement two methods that mirror the SAO decision points:

* :meth:`BaseAgent.decide_propose` — called when the agent is the proposer
  for this round.  Must return a complete offer dict.
* :meth:`BaseAgent.decide_respond` — called when the agent is a responder.
  Must return ``"accept"`` or ``"reject"``.

Both methods receive the negotiation space (``issues`` / ``options_per_issue``)
from the incoming ``SSTPNegotiateMessage.semantic_context`` so agents have
**no prior knowledge** of the space — they discover it live, exactly as in
``test_callback_agents.py``.

The optional :meth:`BaseAgent.handle_message` entry-point is the high-level
hook used by :func:`~evaluation.framework.agents.agent_server.make_decide_app`.
The default implementation delegates to ``decide_propose`` / ``decide_respond``
and wraps the result in an SSTP reply envelope.  Subclasses (e.g.
:class:`~evaluation.framework.agents.llm_agent.LLMAgent`) can override it to
read the raw ``SSTPNegotiateMessage`` body directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# ── sys.path: ensure repo root is importable ─────────────────────────────────
_repo_root = str(Path(__file__).resolve().parents[4])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# SSTP reply helper (shared by all agents)
# ─────────────────────────────────────────────────────────────────────────────


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_sstp_reply(
    session_id: str,
    agent_name: str,
    reply_payload: Dict[str, Any],
    sao_response: Optional[SAOResponse] = None,
    sao_state: Optional[SAOState] = None,
) -> Dict[str, Any]:
    """Wrap a decision payload in a full ``SSTPNegotiateMessage`` envelope.

    This is the canonical reply format expected by both
    ``BatchCallbackRunner._post_batch`` (direct env) and the external
    negotiation server's ``/decide`` handler (callback env).

    Args:
        session_id: Current negotiation session identifier.
        agent_name: Agent's display name — slugified to form ``origin.actor_id``.
        reply_payload: Inner payload dict (``{"action": ..., "offer": ...}``).
        sao_response: Structured SAO decision for the server to parse.
        sao_state: Reflected ``SAOState`` echoed back from the inbound message.

    Returns:
        ``SSTPNegotiateMessage`` serialised as a plain dict (``mode="json"``).
    """
    payload_str = json.dumps(reply_payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{session_id}:{_slug(agent_name)}:{payload_hash}",
        )
    )
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id=_slug(agent_name), tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id,
            sao_state=sao_state,
            sao_response=sao_response,
        ),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=reply_payload,
    ).model_dump(mode="json")


# ─────────────────────────────────────────────────────────────────────────────
# BaseAgent
# ─────────────────────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base for all negotiation agents.

    Subclasses must implement :meth:`decide_propose` and
    :meth:`decide_respond`.  The default :meth:`handle_message` dispatches to
    those methods and returns a fully-wrapped SSTP reply dict.

    Attributes:
        agent_id: Unique participant identifier (matches SSTP ``participant_id``).
        prefer_low: When ``True`` index-0 options are considered best;
            when ``False`` the last index is best.  Used by the default
            preference-building helpers.
    """

    def __init__(self, agent_id: str, prefer_low: bool = True) -> None:
        self.agent_id = agent_id
        self.prefer_low = prefer_low

    # ------------------------------------------------------------------
    # Abstract decision interface
    # ------------------------------------------------------------------

    @abstractmethod
    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> Tuple[Dict[str, str], Optional[float]]:
        """Return ``(offer, aspiration_or_None)`` for a propose turn.

        Args:
            round_num: 1-based current SAO round.
            n_steps: Maximum SAO rounds for this session.
            options_per_issue: Complete negotiation space ``{issue: [option, …]}``.

        Returns:
            A tuple ``(offer, aspiration)`` where ``offer`` maps every issue to
            the chosen option string.  ``aspiration`` may be ``None`` when the
            agent does not use an aspiration curve.
        """

    @abstractmethod
    def decide_respond(
        self,
        offer: Dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> str:
        """Return ``"accept"`` or ``"reject"`` for a respond turn.

        Args:
            offer: The standing offer from the current proposer.
            round_num: 1-based current SAO round.
            n_steps: Maximum SAO rounds for this session.
            options_per_issue: Complete negotiation space.
        """

    # ------------------------------------------------------------------
    # High-level entry point — override in LLMAgent for raw SSTP access
    # ------------------------------------------------------------------

    def handle_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process one inbound ``SSTPNegotiateMessage`` and return a reply dict.

        Called by :func:`~evaluation.framework.agents.agent_server.make_decide_app`
        for every message in the incoming batch.  The default implementation
        reads ``payload.action`` / ``semantic_context`` and delegates to
        :meth:`decide_propose` or :meth:`decide_respond`, then wraps the
        result with :func:`build_sstp_reply`.

        Subclasses that need the full raw envelope can override this method
        directly (see :class:`~evaluation.framework.agents.llm_agent.LLMAgent`).

        Args:
            body: Deserialised ``SSTPNegotiateMessage`` dict.

        Returns:
            Fully-wrapped ``SSTPNegotiateMessage`` reply dict.
        """
        payload: Dict[str, Any] = body.get("payload") or {}
        sc: Dict[str, Any] = body.get("semantic_context") or {}

        action: str = payload.get("action", "respond")
        round_num: int = payload.get("round", 1)
        n_steps: int = payload.get("n_steps") or 200
        issues: List[str] = sc.get("issues") or []
        options_per_issue: Dict[str, List[str]] = sc.get("options_per_issue") or {}
        session_id: str = sc.get("session_id") or "unknown"
        sao_state_raw = sc.get("sao_state")
        incoming_sao_state: Optional[SAOState] = (
            SAOState(**sao_state_raw) if sao_state_raw else None
        )

        if action == "propose":
            offer, aspiration = self.decide_propose(
                round_num, n_steps, options_per_issue
            )
            asp_str = f"  asp={aspiration:.3f}" if aspiration is not None else ""
            print(
                f"  [{self.agent_id}] propose  round={round_num}{asp_str}  offer={offer}",
                flush=True,
            )
            reply_payload: Dict[str, Any] = {
                "action": "counter_offer",
                "round": round_num,
                "issues": issues,
                "options_per_issue": options_per_issue,
                "offer": offer,
            }
            sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER, outcome=offer)

        else:  # respond (or unknown action)
            current_offer: Dict[str, str] = payload.get("current_offer") or {}
            decision = self.decide_respond(
                current_offer, round_num, n_steps, options_per_issue
            )
            print(
                f"  [{self.agent_id}] respond  round={round_num}  → {decision}",
                flush=True,
            )
            reply_payload = {
                "action": decision,
                "round": round_num,
                "issues": issues,
                "options_per_issue": options_per_issue,
            }
            sao_resp = SAOResponse(
                response=(
                    ResponseType.ACCEPT_OFFER
                    if decision == "accept"
                    else ResponseType.REJECT_OFFER
                ),
                outcome=current_offer if decision == "accept" else None,
            )

        return build_sstp_reply(
            session_id,
            self.agent_id,
            reply_payload,
            sao_response=sao_resp,
            sao_state=incoming_sao_state,
        )

    # ------------------------------------------------------------------
    # Utility helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _build_prefs(
        self, options_per_issue: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Build a ``{issue: {option: utility}}`` map from the option order.

        Utility is linearly interpolated in ``[0, 1]``:
        * ``prefer_low=True``  → index 0 gets 1.0, last index gets 0.0
        * ``prefer_low=False`` → index 0 gets 0.0, last index gets 1.0
        """
        prefs: Dict[str, Dict[str, float]] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            d = max(n - 1, 1)
            if self.prefer_low:
                prefs[issue] = {o: round(1.0 - i / d, 3) for i, o in enumerate(opts)}
            else:
                prefs[issue] = {o: round(i / d, 3) for i, o in enumerate(opts)}
        return prefs

    def utility(
        self,
        offer: Dict[str, str],
        options_per_issue: Dict[str, List[str]],
        prefs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """Equal-weighted mean utility of *offer* across all issues.

        Args:
            offer: ``{issue: chosen_option}`` dict.
            options_per_issue: Negotiation space (used to build prefs if None).
            prefs: Pre-built preference map; built on-the-fly when omitted.
        """
        p = prefs if prefs is not None else self._build_prefs(options_per_issue)
        known = [iss for iss in offer if iss in p]
        if not known:
            return 0.0
        return sum(p[iss].get(offer[iss], 0.0) for iss in known) / len(known)
