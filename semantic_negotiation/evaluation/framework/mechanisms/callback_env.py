# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Sample environment — Callback mode (mirrors test_via_semantic_neg_agents.py).

This sample drives negotiations through the **external negotiation server** at
port 8089, exactly like ``test_via_semantic_neg_agents.py``.  The agent logic runs
locally in a FastAPI server; the negotiation server calls back on every SAO
round and drives the turn-by-turn loop via ``POST /api/negotiate/initiate``
→ ``POST /api/negotiate/decide`` until the session resolves.

Comparison with direct_env.py
------------------------------
``direct_env.py`` uses :class:`~evaluation.framework.runner.EvaluationRunner`
(BatchCallbackRunner — self-contained, no external server required).  Issues
and options must be pre-declared in the missions.

**This sample** requires a running negotiation server.  Issues and options are
**discovered by the server's LLM pipeline** from the ``content_text`` alone —
the missions YAML does *not* need ``issues`` / ``options_per_issue``.

Architecture::

    ┌────────────────────────────────────────────────────────────┐
    │  callback_env.py                                            │
    │                                                             │
    │  Agents loaded from callback_env.yaml                       │
    │           │  FastAPI /decide  (port from config)            │
    │           │                        ▲                        │
    │           │  POST /initiate        │ callback per round     │
    │           ▼                        │                        │
    │  Negotiation server :8089  ────────┘                        │
    │  (SAO + LLM intent/options discovery)                       │
    │           │                                                  │
    │           │  POST /decide (turn-by-turn loop)               │
    │           ▼                                                  │
    │  CallbackEnvRunner.run() → EvaluationResult                 │
    └────────────────────────────────────────────────────────────┘

Usage::

    # Terminal 1: start the negotiation server
    cd semantic_negotiation
    poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089

    # Terminal 2: run this sample (uses callback_env.yaml by default)
    cd ioc-cfn-cognitive-agents
    .venv/bin/python -m semantic_negotiation.evaluation.framework.mechanisms.callback_env

    # Custom config file:
    .venv/bin/python -m semantic_negotiation.evaluation.framework.mechanisms.callback_env \\
        --config path/to/my_config.yaml \\
        --neg-server http://localhost:8089
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── sys.path ──────────────────────────────────────────────────────────────────
_repo_root = str(Path(__file__).resolve().parents[4])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_sn_root = str(Path(__file__).resolve().parents[3])  # semantic_negotiation/
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402

_sn_root = str(Path(__file__).resolve().parents[3])
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)
from app.agent.reply_payload_utils import attach_reason  # noqa: E402

from semantic_negotiation.evaluation.framework.config import (  # noqa: E402
    AgentConfig,
    EvaluationConfig,
    MissionEntry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent decision engine (Boulware / linear concession — issues discovered live)
# ─────────────────────────────────────────────────────────────────────────────


class _CallbackAgent:
    """Stateless decision engine for one callback participant.

    Preferences are private — never sent to the negotiation server.
    Issues and options are read from ``semantic_context`` in every incoming
    SSTPNegotiateMessage so the agent works with any negotiation space.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.name = cfg.agent_id
        self.prefer_low = cfg.prefer_low
        self.exponent = float(cfg.params.get("exponent", 2.0))
        self.min_reservation = float(cfg.params.get("min_reservation", 0.0))
        self._prefs: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Preferences (built lazily; rebuilt when issue set changes)
    # ------------------------------------------------------------------

    def _ensure_prefs(
        self, options_per_issue: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        if set(options_per_issue.keys()) != set(self._prefs.keys()):
            self._prefs = {}
            for issue, opts in options_per_issue.items():
                n = len(opts)
                d = max(n - 1, 1)
                if self.prefer_low:
                    self._prefs[issue] = {
                        o: round(1.0 - i / d, 3) for i, o in enumerate(opts)
                    }
                else:
                    self._prefs[issue] = {
                        o: round(i / d, 3) for i, o in enumerate(opts)
                    }
        return self._prefs

    def utility(
        self, offer: Dict[str, str], options_per_issue: Dict[str, List[str]]
    ) -> float:
        prefs = self._ensure_prefs(options_per_issue)
        known = [iss for iss in offer if iss in prefs]
        if not known:
            return 0.0
        return sum(prefs[iss].get(offer[iss], 0.0) for iss in known) / len(known)

    # ------------------------------------------------------------------
    # Aspiration (Boulware)
    # ------------------------------------------------------------------

    def _aspiration(self, t: float) -> float:
        return max(self.min_reservation, 1.0 - (t**self.exponent))

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def decide_propose(
        self, round_num: int, n_steps: int, options_per_issue: Dict[str, List[str]]
    ) -> tuple:
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        prefs = self._ensure_prefs(options_per_issue)
        issues = list(options_per_issue.keys())
        # Enumerate all outcomes, pick minimum-utility offer that still >= asp
        outcomes: List[tuple] = []
        for combo in itertools.product(*[options_per_issue[i] for i in issues]):
            offer = dict(zip(issues, combo))
            u = sum(prefs[iss].get(offer[iss], 0.0) for iss in issues) / len(issues)
            outcomes.append((offer, round(u, 4)))
        outcomes.sort(key=lambda x: x[1], reverse=True)
        best = outcomes[0][0]
        for offer, u in outcomes:
            if u >= asp:
                best = offer
            else:
                break
        return best, asp

    def decide_respond(
        self,
        offer: Dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> str:
        u = self.utility(offer, options_per_issue)
        t = round_num / max(n_steps, 1)
        return "accept" if u >= self._aspiration(t) else "reject"


# ─────────────────────────────────────────────────────────────────────────────
# SSTP message helpers (identical pattern to test_via_semantic_neg_agents.py)
# ─────────────────────────────────────────────────────────────────────────────


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _build_sstp_reply(
    session_id: str,
    agent_name: str,
    reply_payload: Dict[str, Any],
    sao_response: Optional[SAOResponse] = None,
    sao_state: Optional[SAOState] = None,
) -> Dict[str, Any]:
    payload_str = json.dumps(reply_payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"{session_id}:{_slug(agent_name)}:{payload_hash}"
        )
    )
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=message_id,
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id=_slug(agent_name), tenant_id=session_id),
        semantic_context=NegotiateSemanticContext(
            session_id=session_id, sao_state=sao_state, sao_response=sao_response
        ),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal", propagation="restricted", retention_policy="default"
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=reply_payload,
    ).model_dump(mode="json")


def _build_decide_payload(
    session_id: str, agent_replies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Wrap agent replies in an SSTPNegotiateMessage for the /decide endpoint."""
    inner = {"session_id": session_id, "agent_replies": agent_replies}
    payload_str = json.dumps(inner, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=str(uuid.uuid4()),
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="callback-env-runner", tenant_id="demo"),
        semantic_context=NegotiateSemanticContext(session_id=session_id),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal", propagation="restricted", retention_policy="default"
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=inner,
    ).model_dump(mode="json")


# ─────────────────────────────────────────────────────────────────────────────
# Agent FastAPI server  (same /decide pattern as test_via_semantic_neg_agents.py)
# ─────────────────────────────────────────────────────────────────────────────


def _make_agent_app(registry: Dict[str, _CallbackAgent]) -> FastAPI:
    app = FastAPI(title="Callback Env Agent Server")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: List[Dict[str, Any]] = await request.json()

        def _one(body: Dict[str, Any]) -> Dict[str, Any]:
            payload = body.get("payload") or {}
            pid: str = payload.get("participant_id") or ""
            agent = registry.get(pid)
            if agent is None:
                for k, a in registry.items():
                    if pid in k or k in pid:
                        agent = a
                        break
            if agent is None:
                agent = next(iter(registry.values()))

            sc = body.get("semantic_context") or {}
            issues: List[str] = sc.get("issues") or []
            options_per_issue: Dict[str, List[str]] = sc.get("options_per_issue") or {}
            session_id: str = sc.get("session_id") or "unknown"
            sao_state_raw = sc.get("sao_state")
            incoming_sao_state: Optional[SAOState] = (
                SAOState(**sao_state_raw) if sao_state_raw else None
            )
            action: str = payload.get("action", "respond")
            round_num: int = payload.get("round", 1)
            n_steps: int = payload.get("n_steps") or 200

            if action == "propose":
                offer, asp = agent.decide_propose(round_num, n_steps, options_per_issue)
                print(
                    f"  [{agent.name}] propose  round={round_num}"
                    f"  asp={asp:.3f}  offer={offer}",
                    flush=True,
                )
                reply_payload = attach_reason(
                    {
                        "action": "counter_offer",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "offer": offer,
                    },
                    f"{agent.name}: counter-offer from utility curve.",
                )
                sao_resp = SAOResponse(
                    response=ResponseType.REJECT_OFFER, outcome=offer
                )
            else:  # respond / unknown
                current_offer: Dict[str, str] = payload.get("current_offer") or {}
                decision = agent.decide_respond(
                    current_offer, round_num, n_steps, options_per_issue
                )
                u = agent.utility(current_offer, options_per_issue)
                print(
                    f"  [{agent.name}] respond  round={round_num}"
                    f"  utility={u:.3f}  → {decision}",
                    flush=True,
                )
                reply_payload = attach_reason(
                    {
                        "action": decision,
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
                        ResponseType.ACCEPT_OFFER
                        if decision == "accept"
                        else ResponseType.REJECT_OFFER
                    ),
                    outcome=current_offer if decision == "accept" else None,
                )

            # participant_id MUST be included in the payload so BatchCallbackRunner.step()
            # can build replies_by_pid correctly (mirrors test_via_semantic_neg_agents.py).
            return _build_sstp_reply(
                session_id,
                agent.name,
                {**reply_payload, "participant_id": pid},
                sao_response=sao_resp,
                sao_state=incoming_sao_state,
            )

        # Run all decisions in parallel threads (matches test_via_semantic_neg_agents.py)
        # Expand broadcast messages (participant_id=None) into per-agent copies so
        # each agent replies individually — the negotiate server's step() expects
        # exactly N replies (one per registered participant).
        expanded: List[Dict[str, Any]] = []
        for msg in messages:
            pid = (msg.get("payload") or {}).get("participant_id")
            if pid is None:
                for agent_pid in registry:
                    msg_copy = copy.deepcopy(msg)
                    msg_copy["payload"]["participant_id"] = agent_pid
                    expanded.append(msg_copy)
            else:
                expanded.append(msg)

        replies = await asyncio.gather(
            *[asyncio.to_thread(_one, msg) for msg in expanded]
        )
        return JSONResponse(list(replies))

    return app


def _start_agent_server(registry: Dict[str, _CallbackAgent], port: int) -> None:
    """Start the shared agent FastAPI server in a daemon thread."""
    config = uvicorn.Config(
        _make_agent_app(registry), host="0.0.0.0", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()


def _wait_for_server(port: int, retries: int = 20, delay: float = 0.3) -> None:
    for _ in range(retries):
        try:
            httpx.get(f"http://localhost:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Agent server on port {port} did not start in time.")


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _MissionResult:
    mission_name: str
    agreement: Optional[Dict[str, str]]
    timedout: bool
    broken: bool
    total_rounds: int
    n_steps: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mission": self.mission_name,
            "agreement": self.agreement,
            "timedout": self.timedout,
            "broken": self.broken,
            "total_rounds": self.total_rounds,
            "n_steps": self.n_steps,
        }


@dataclass
class CallbackEvalResult:
    """Aggregated results from a :class:`CallbackEnvRunner` run."""

    mission_results: List[_MissionResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        total = len(self.mission_results)
        agreed = sum(1 for r in self.mission_results if r.agreement)
        return {
            "total_missions": total,
            "agreement_rate": round(agreed / total, 4) if total else 0.0,
            "timedout": sum(1 for r in self.mission_results if r.timedout),
            "broken": sum(1 for r in self.mission_results if r.broken),
            "missions": [r.as_dict() for r in self.mission_results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CallbackEnvRunner — mirrors test_via_semantic_neg_agents.py
# ─────────────────────────────────────────────────────────────────────────────


class CallbackEnvRunner:
    """Runs all missions by forwarding decisions through the external negotiation server.

    This runner mirrors ``test_via_semantic_neg_agents.py`` exactly:

    1. Starts a local FastAPI agent server (``/decide`` endpoint).
    2. For each mission, POSTs ``/api/negotiate/initiate`` to the negotiation
       server, registering all configured agents by their ``agent_id``.
    3. Drives the turn-by-turn SAO loop via ``/api/negotiate/decide`` until
       the negotiation resolves.
    4. Collects results and returns a :class:`CallbackEvalResult`.

    Args:
        config: Combined :class:`~evaluation.framework.config.EvaluationConfig`.
        neg_server: Base URL of the running negotiation server.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.neg_server = config.environment.neg_server.rstrip("/")

    # ------------------------------------------------------------------

    def _build_initiate_payload(
        self, mission: MissionEntry, run_id: str
    ) -> Dict[str, Any]:
        mission_slug = _slug(mission.name)
        return SSTPNegotiateMessage(
            kind="negotiate",
            message_id=f"init-{run_id}-{mission_slug}",
            dt_created=datetime.now(timezone.utc).isoformat(),
            origin=Origin(actor_id="callback-env-runner", tenant_id="demo"),
            semantic_context=NegotiateSemanticContext(
                session_id=f"sess-{run_id}-{mission_slug}"
            ),
            payload_hash="0" * 64,
            policy_labels=PolicyLabels(
                sensitivity="internal",
                propagation="restricted",
                retention_policy="default",
            ),
            provenance=Provenance(sources=[], transforms=[]),
            payload={
                "content_text": mission.content_text,
                "agents": [
                    {"id": cfg.agent_id, "name": cfg.agent_id}
                    for cfg in self.config.agents
                ],
                # None → server auto-sizes via compute_n_steps(); 0 is treated as None too.
                "n_steps": mission.n_steps if mission.n_steps > 0 else None,
            },
        ).model_dump(mode="json")

    @staticmethod
    def _forward_batch(
        agent_url: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """POST all messages to /decide as a single batch, return the reply list.

        The agent server's ``POST /decide`` accepts and returns
        ``List[SSTPNegotiateMessage]``, so we send the whole round in one call —
        exactly as :class:`~app.agent.batch_callback_runner.BatchCallbackRunner`
        does internally.
        """
        resp = httpx.post(agent_url, json=messages, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    def _run_mission(
        self,
        mission: MissionEntry,
        run_id: str,
        agent_port: int,
    ) -> _MissionResult:
        """Drive one mission through the external negotiation server."""
        initiate_payload = self._build_initiate_payload(mission, run_id)
        session_id: str = (initiate_payload.get("semantic_context") or {}).get(
            "session_id", "unknown"
        )

        print(f"  POST {self.neg_server}/api/negotiate/initiate …")
        resp = httpx.post(
            f"{self.neg_server}/api/negotiate/initiate",
            json=initiate_payload,
            timeout=self.config.environment.neg_timeout,
        )
        resp.raise_for_status()
        init_data = resp.json()
        init_payload = init_data.get("payload") or {}

        if init_payload.get("status") not in (
            "initiated",
            "ongoing",
            "agreed",
            "broken",
            "timeout",
        ):
            print(
                f"  ERROR: unexpected initiate response: {json.dumps(init_data, indent=2)}"
            )
            return _MissionResult(
                mission_name=mission.name,
                agreement=None,
                timedout=False,
                broken=True,
                total_rounds=0,
                n_steps=mission.n_steps,
            )

        agent_url = f"http://localhost:{agent_port}/decide"
        messages: List[Dict[str, Any]] = init_payload.get("messages") or []
        round_idx = 0

        while messages:
            round_idx += 1
            print(f"  round {round_idx}: dispatching {len(messages)} messages …")

            # Forward entire batch to the agent server in one POST
            agent_replies = self._forward_batch(agent_url, messages)

            decide_payload = _build_decide_payload(session_id, agent_replies)
            decide_resp = httpx.post(
                f"{self.neg_server}/api/negotiate/decide",
                json=decide_payload,
                timeout=30.0,
            )
            decide_resp.raise_for_status()
            decide_data = decide_resp.json()

            status = decide_data.get("status", "unknown")
            print(f"  → status={status}")

            if status == "ongoing":
                messages = decide_data.get("messages") or []
            else:
                result = decide_data.get("final_result") or decide_data
                trace = (result.get("payload") or {}).get("trace") or {}
                # final_agreement is excluded from payload.trace (see build_commit_envelope);
                # it lives in semantic_context.final_agreement as [{issue_id, chosen_option}].
                sc_agreement = (result.get("semantic_context") or {}).get("final_agreement")
                if isinstance(sc_agreement, list):
                    agreement_raw = {
                        item["issue_id"]: item["chosen_option"]
                        for item in sc_agreement
                        if "issue_id" in item and "chosen_option" in item
                    }
                else:
                    agreement_raw = sc_agreement or {}
                return _MissionResult(
                    mission_name=mission.name,
                    agreement=agreement_raw or None,
                    timedout=trace.get("timedout", False),
                    broken=trace.get("broken", False),
                    total_rounds=(result.get("payload") or {}).get("total_rounds")
                    or round_idx,
                    n_steps=mission.n_steps,
                )

        return _MissionResult(
            mission_name=mission.name,
            agreement=None,
            timedout=True,
            broken=False,
            total_rounds=round_idx,
            n_steps=mission.n_steps,
        )

    def run(self) -> CallbackEvalResult:
        """Run all missions and return aggregated results.

        Raises:
            RuntimeError: When the negotiation server is not reachable.
        """
        # Verify the negotiation server is up
        try:
            httpx.get(f"{self.neg_server}/openapi.json", timeout=3.0).raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Negotiation server at {self.neg_server} is not reachable: {exc}\n"
                "Start it with:\n"
                "  cd semantic_negotiation && poetry run uvicorn app.main:app "
                "--host 0.0.0.0 --port 8089"
            ) from exc

        missions = self.config.missions.load()
        env = self.config.environment
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build in-process agents and start the shared callback server
        registry: Dict[str, _CallbackAgent] = {
            cfg.agent_id: _CallbackAgent(cfg) for cfg in self.config.agents
        }
        _start_agent_server(registry, env.agent_port)
        _wait_for_server(env.agent_port)
        print(f"Agent server started on :{env.agent_port}")

        results: List[_MissionResult] = []
        for idx, mission in enumerate(missions, 1):
            print(f"\n{'=' * 60}")
            print(f"  Mission {idx}/{len(missions)}: {mission.name}")
            print(f"{'=' * 60}")
            # Reset preference caches between missions
            for a in registry.values():
                a._prefs = {}
            try:
                r = self._run_mission(mission, run_id, env.agent_port)
                results.append(r)
            except Exception as exc:
                print(f"  ERROR in mission '{mission.name}': {exc}")
                results.append(
                    _MissionResult(
                        mission_name=mission.name,
                        agreement=None,
                        timedout=False,
                        broken=True,
                        total_rounds=0,
                        n_steps=mission.n_steps,
                    )
                )

        return CallbackEvalResult(mission_results=results)
