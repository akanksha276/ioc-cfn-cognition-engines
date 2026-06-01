# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""CaSiNo callback agent — mock SSTP decision server for Phase 3 callback eval.

This module provides a lightweight FastAPI agent server that mirrors the
architecture of ``test_callback_agents.py`` but initialises agent preferences
directly from CaSiNo dataset priority data (``value2issue``) rather than the
generic ``prefer_low`` flag.

It is used exclusively by :func:`~evaluation.casino.eval_negotiation.evaluate_negotiation_callback`
to exercise the production ``BatchCallbackRunner`` code path without needing
a full negotiation server.

Architecture (in-process, no external server required)
-------------------------------------------------------
``BatchCallbackRunner._post_batch`` sends ``List[SSTPNegotiateMessage]`` to
the agent's ``/decide`` endpoint, then waits for decisions to appear in the
``batch_callback_runner._DECISIONS`` dict (populated by
``batch_callback_runner.store_decisions``).

The agent server in this module calls ``store_decisions`` directly (in-process)
instead of POSTing to ``/api/negotiate/agents-decisions``, so the eval loop
never needs a running negotiation server — it's entirely self-contained.

Round semantics (inherited from BatchCallbackRunner)
----------------------------------------------------
* **Round 1**: server picks a random offer; all agents receive
  ``action=respond``.
* **Rounds 2+**: alternating SAO — proposer gets ``action=propose``;
  all other participants get ``action=respond`` with the standing offer.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── sys.path: ensure repo root and agent root are importable ──────────────────
_repo_root = str(Path(__file__).resolve().parents[3])   # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_agent_root = str(Path(__file__).resolve().parents[2])  # semantic_negotiation/
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402

_sn_root = str(Path(__file__).resolve().parents[2])
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)
from app.agent.reply_payload_utils import attach_reason  # noqa: E402

from ...app.agent.batch_callback_runner import store_decisions  # noqa: E402
from ..casino.loader import (  # noqa: E402
    AgentData,
    ISSUES,
    PRIORITY_WEIGHTS,
    TOTAL_PACKAGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _build_sstp_reply(
    session_id: str,
    agent_name: str,
    reply_payload: dict[str, Any],
    sao_response: SAOResponse | None = None,
    sao_state: SAOState | None = None,
) -> dict[str, Any]:
    """Wrap an agent decision in a full SSTPNegotiateMessage envelope."""
    payload_str = json.dumps(reply_payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{session_id}:{_slug(agent_name)}:{payload_hash}",
        )
    )
    msg = SSTPNegotiateMessage(
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
    )
    return msg.model_dump(mode="json")


# ---------------------------------------------------------------------------
# CasinoCallbackAgent
# ---------------------------------------------------------------------------

class CasinoCallbackAgent:
    """NegMAS Boulware concession agent with preferences derived from CaSiNo data.

    Preferences are kept private — they are never sent to the negotiation
    server — exactly as in production.  The agent receives the negotiation
    space (issues + options) from the ``semantic_context`` field of each
    incoming ``SSTPNegotiateMessage``.

    Utility model
    -------------
    The outcome space is expressed from **Agent A's** perspective: each option
    label is the number of packages (``"0"``–``"3"``) that Agent A receives.

    * **Agent A**  utility for option ``q`` on issue ``i``:
      ``weight(i) × q / 3``
    * **Agent B**  utility for option ``q`` on issue ``i``:
      ``weight(i) × (3 - q) / 3``

    Issue weights are drawn from CaSiNo priority levels:
    ``High=5/12``, ``Medium=4/12``, ``Low=3/12``.

    Concession curve (Boulware)
    ---------------------------
    At relative time ``t = round / n_steps``::

        aspiration(t) = max(min_reservation, 1 - t ** exponent)

    * **Propose**: enumerate all outcomes, return the one with minimum utility
      that still satisfies ``utility ≥ aspiration(t)``.  Falls back to ideal.
    * **Respond**: accept if ``utility(offer) ≥ aspiration(t)``.

    Args:
        agent_data: Parsed CaSiNo agent data with ``value2issue`` priorities.
        is_agent_a: True if this agent is on the "more packages" side.
        exponent: Boulware concession exponent (default 2.0).
        min_reservation: Hard utility floor — never accept below this.
    """

    def __init__(
        self,
        agent_data: AgentData,
        is_agent_a: bool,
        exponent: float = 2.0,
        min_reservation: float = 0.0,
    ) -> None:
        self.name = agent_data.agent_id
        self.is_agent_a = is_agent_a
        self.exponent = exponent
        self.min_reservation = min_reservation

        # Build issue → weight from priority data
        issue2priority: Dict[str, str] = {
            v.lower(): k for k, v in agent_data.value2issue.items()
        }
        self._issue_weights: Dict[str, float] = {
            issue: PRIORITY_WEIGHTS.get(issue2priority.get(issue, "Low"), PRIORITY_WEIGHTS["Low"])
            for issue in ISSUES
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _option_utility(self, issue: str, option: str) -> float:
        """Return the raw (unweighted) utility for a single option on one issue."""
        try:
            q = int(option)
        except (ValueError, TypeError):
            return 0.0
        if self.is_agent_a:
            return q / TOTAL_PACKAGES
        else:
            return (TOTAL_PACKAGES - q) / TOTAL_PACKAGES

    def utility(self, offer: dict[str, str], options_per_issue: dict[str, list[str]]) -> float:
        """Weighted utility of a complete offer."""
        total = 0.0
        for issue in offer:
            w = self._issue_weights.get(issue, 1.0 / len(self._issue_weights))
            total += w * self._option_utility(issue, offer[issue])
        return total

    # ------------------------------------------------------------------
    # Aspiration curve
    # ------------------------------------------------------------------

    def _aspiration(self, t: float) -> float:
        return max(self.min_reservation, 1.0 - (t ** self.exponent))

    # ------------------------------------------------------------------
    # All outcomes
    # ------------------------------------------------------------------

    def _all_outcomes_sorted(
        self, options_per_issue: dict[str, list[str]]
    ) -> list[tuple[dict, float]]:
        issues = list(options_per_issue.keys())
        results: list[tuple[dict, float]] = []
        for combo in itertools.product(*[options_per_issue[i] for i in issues]):
            offer = dict(zip(issues, combo))
            u = self.utility(offer, options_per_issue)
            results.append((offer, round(u, 6)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> tuple[dict[str, str], float]:
        """Return ``(offer, aspiration)`` using Boulware concession."""
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        outcomes = self._all_outcomes_sorted(options_per_issue)
        best_qualifying = outcomes[0][0]  # fallback = ideal offer
        for offer, u in outcomes:
            if u >= asp:
                best_qualifying = offer
            else:
                break
        return best_qualifying, asp

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Return ``'accept'`` or ``'reject'``."""
        u = self.utility(offer, options_per_issue)
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        return "accept" if u >= asp else "reject"


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def make_casino_agent_app(agents_registry: dict[str, CasinoCallbackAgent]) -> FastAPI:
    """Return a FastAPI app whose ``POST /decide`` handles all agents.

    The app is intentionally stateless w.r.t. the registry — the eval loop
    mutates ``agents_registry`` in-place between dialogues so the same running
    server handles different agent pairs without restart.

    Instead of POSTing decisions back to a negotiation server, this endpoint
    calls :func:`~app.agent.batch_callback_runner.store_decisions` directly
    (in-process), making the whole eval loop self-contained.

    Args:
        agents_registry: Mutable dict mapping participant_id → agent.
            Updated by the caller between dialogues.
    """
    app = FastAPI(title="CaSiNo Eval Agent Server")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: list[dict[str, Any]] = await request.json()

        def _decide_for_agent(
            body: dict[str, Any], agent: "CasinoCallbackAgent"
        ) -> dict[str, Any]:
            """Run one agent's decision for the given message body."""
            payload: dict[str, Any] = body.get("payload", {})
            action: str = payload.get("action", "respond")
            round_num: int = payload.get("round", 1)
            n_steps: int = payload.get("n_steps") or 100

            # Issues and options_per_issue live in semantic_context (PR #38)
            _sc: dict[str, Any] = body.get("semantic_context") or {}
            issues: list[str] = _sc.get("issues") or []
            options_per_issue: dict[str, list[str]] = _sc.get("options_per_issue") or {}
            session_id: str = _sc.get("session_id") or "unknown"

            _sao_state_dict = _sc.get("sao_state")
            incoming_sao_state: SAOState | None = (
                SAOState(**_sao_state_dict) if _sao_state_dict else None
            )

            if action == "propose":
                offer, aspiration = agent.decide_propose(round_num, n_steps, options_per_issue)
                reply_payload = attach_reason(
                    {
                        "action": "counter_offer",
                        "participant_id": agent.name,
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "offer": offer,
                    },
                    f"{agent.name}: counter-offer.",
                )
                sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER, outcome=offer)
                return _build_sstp_reply(
                    session_id, agent.name, reply_payload,
                    sao_response=sao_resp, sao_state=incoming_sao_state,
                )

            elif action == "respond":
                current_offer: dict[str, str] = payload.get("current_offer") or {}
                decision = agent.decide_respond(
                    current_offer, round_num, n_steps, options_per_issue
                )
                reply_payload = attach_reason(
                    {
                        "action": decision,
                        "participant_id": agent.name,
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                    },
                    (
                        f"{agent.name}: accepting the offer."
                        if decision == "accept"
                        else f"{agent.name}: rejecting the offer."
                    ),
                )
                sao_resp = SAOResponse(
                    response=(
                        ResponseType.ACCEPT_OFFER if decision == "accept"
                        else ResponseType.REJECT_OFFER
                    ),
                    outcome=current_offer if decision == "accept" else None,
                )
                return _build_sstp_reply(
                    session_id, agent.name, reply_payload,
                    sao_response=sao_resp, sao_state=incoming_sao_state,
                )

            else:
                # Unknown action — reject
                reply_payload = attach_reason(
                    {
                        "action": "reject",
                        "participant_id": agent.name,
                        "round": round_num,
                    },
                    f"{agent.name}: unknown action; rejecting.",
                )
                sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER)
                return _build_sstp_reply(
                    session_id, agent.name, reply_payload,
                    sao_response=sao_resp, sao_state=incoming_sao_state,
                )

        def _process_one(body: dict[str, Any]) -> list[dict[str, Any]]:
            """Route one inbound message to the correct agent(s).

            * **Targeted** (``participant_id`` is a non-null string): route to
              a single matching agent; return a one-item list.
            * **Broadcast** (``participant_id`` is ``null`` / absent): dispatch
              to every registered agent and return one reply per agent.
            """
            payload: dict[str, Any] = body.get("payload", {})
            participant_id = payload.get("participant_id")

            if participant_id is None:
                # Broadcast \u2014 all agents reply
                return [
                    _decide_for_agent(body, agent)
                    for agent in agents_registry.values()
                ]

            # Targeted \u2014 exact match first, then fuzzy, then first agent
            agent = agents_registry.get(participant_id)
            if agent is None:
                for pid, a in agents_registry.items():
                    if participant_id in pid or pid in participant_id:
                        agent = a
                        break
            if agent is None:
                agent = next(iter(agents_registry.values()))

            return [_decide_for_agent(body, agent)]

        # Flatten: each message may produce 1 or N replies (broadcast)
        replies = [reply for msg in messages for reply in _process_one(msg)]

        # Store decisions in-process so BatchCallbackRunner can pick them up
        # without needing a running negotiation server.
        if messages:
            _sc0: dict[str, Any] = messages[0].get("semantic_context") or {}
            session_id_0: str = _sc0.get("session_id") or "unknown"
            round_num_0: int = (messages[0].get("payload") or {}).get("round", 1)
            key = f"{session_id_0}:{round_num_0}"
            store_decisions(key, replies)

        return JSONResponse(replies)

    return app


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_casino_agent_server(
    agents_registry: dict[str, CasinoCallbackAgent],
    port: int = 8093,
) -> tuple[uvicorn.Server, threading.Thread]:
    """Start the shared agent FastAPI server in a daemon thread.

    Args:
        agents_registry: Mutable dict keyed by participant_id.  The eval loop
            replaces values in-place between dialogues — no server restart needed.
        port: TCP port to listen on (default 8093, avoids conflicts with
              the negotiation server on 8089 and test agents on 8091).

    Returns:
        ``(server, thread)`` — the caller can call ``server.should_exit = True``
        to stop gracefully after all dialogues complete.
    """
    app = make_casino_agent_app(agents_registry)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server, t


def wait_for_server(port: int, retries: int = 30, delay: float = 0.2) -> None:
    """Block until the agent server is accepting connections."""
    for _ in range(retries):
        try:
            httpx.get(f"http://localhost:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            import time
            time.sleep(delay)
    raise RuntimeError(f"CaSiNo agent server on port {port} did not start in time")
