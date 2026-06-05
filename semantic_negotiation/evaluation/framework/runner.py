# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""General evaluation runner.

:class:`EvaluationRunner` orchestrates SAO-style negotiation rounds for every
mission in the dataset using the configured agents and environment.

Architecture
------------
For **in-process agents** (``AgentConfig.url is None``):

1. A :class:`GenericCallbackAgent` is instantiated per configured agent.
2. A single FastAPI server is started in a daemon thread; every agent shares
   the ``POST /decide`` endpoint (dispatched by ``participant_id``).
3. :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` drives the
   SAO rounds by POSTing batched ``List[SSTPNegotiateMessage]`` to that server
   and reading the ``List[SSTPNegotiateMessage]`` response directly.

For **external agents** (all ``AgentConfig.url`` set to the same URL):

* The runner skips the local FastAPI server and passes the shared URL directly
  to ``BatchCallbackRunner``.

Mixed setups (some in-process, some external) are not yet supported.

``/decide`` contract
--------------------
``POST /decide`` receives ``List[SSTPNegotiateMessage]`` (one per participant)
and **must return** ``List[SSTPNegotiateMessage]`` of the same length.
This matches what :meth:`~app.agent.batch_callback_runner.BatchCallbackRunner._post_batch`
expects.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
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

# ── sys.path: ensure repo root and agent root are importable ─────────────────
_repo_root = str(Path(__file__).resolve().parents[3])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_agent_root = str(Path(__file__).resolve().parents[2])  # semantic_negotiation/
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402

_sn_root = str(Path(__file__).resolve().parents[2])
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)
from app.agent.reply_payload_utils import attach_reason  # noqa: E402

from ...app.agent.batch_callback_runner import (
    BatchCallbackRunner,
    compute_n_steps,
)  # noqa: E402
from ...app.agent.negotiation_model import (
    NegotiationParticipant,
    NegotiationResult,
)  # noqa: E402
from .config import AgentConfig, EvaluationConfig, MissionEntry  # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SSTP message helper (mirrors casino/callback_agent._build_sstp_reply)
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
    """Wrap an agent decision in a full ``SSTPNegotiateMessage`` envelope."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Generic in-process callback agent
# ─────────────────────────────────────────────────────────────────────────────


class GenericCallbackAgent:
    """Parameterised concession agent for general negotiations.

    Unlike :class:`~evaluation.casino.callback_agent.CasinoCallbackAgent` this
    agent is not tied to any specific dataset.  It derives its utility function
    from two parameters:

    * **prefer_low** — when ``True`` option at index 0 has utility 1.0 and the
      last option has utility 0.0; when ``False`` the ranking is reversed.
    * **strategy** *(params key, default ``"boulware"``)* — ``"boulware"`` (``aspiration(t) = max(r, 1−t^e)``) or
      ``"linear"`` (``aspiration(t) = max(r, 1−t)``).

    Issues and options are read from ``semantic_context`` in every incoming
    ``SSTPNegotiateMessage`` so the agent works with any negotiation space
    without prior knowledge.

    Args:
        cfg: Agent configuration.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.name = cfg.agent_id
        self.prefer_low = cfg.prefer_low
        self.strategy = str(cfg.params.get("strategy", "boulware"))
        self.exponent = float(cfg.params.get("exponent", 2.0))
        self.min_reservation = float(cfg.params.get("min_reservation", 0.0))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _option_utility(self, option: str, options: List[str]) -> float:
        """Normalised [0, 1] utility for *option* within its ordered *options* list."""
        n = len(options)
        if n <= 1:
            return 1.0
        try:
            idx = options.index(option)
        except ValueError:
            return 0.0
        rank = idx / (n - 1)  # 0.0 at index-0, 1.0 at last index
        return (1.0 - rank) if self.prefer_low else rank

    def utility(
        self,
        offer: Dict[str, str],
        options_per_issue: Dict[str, List[str]],
    ) -> float:
        """Equal-weighted additive utility across all issues in *offer*."""
        issues = list(offer.keys())
        if not issues:
            return 0.0
        w = 1.0 / len(issues)
        return sum(
            w
            * self._option_utility(
                offer[issue],
                options_per_issue.get(issue, [offer[issue]]),
            )
            for issue in issues
        )

    # ------------------------------------------------------------------
    # Aspiration curve
    # ------------------------------------------------------------------

    def _aspiration(self, t: float) -> float:
        """Target utility at relative time *t* ∈ [0, 1]."""
        if self.strategy == "linear":
            return max(self.min_reservation, 1.0 - t)
        # boulware (default)
        return max(self.min_reservation, 1.0 - (t**self.exponent))

    # ------------------------------------------------------------------
    # Outcome enumeration
    # ------------------------------------------------------------------

    def _all_outcomes_sorted(
        self, options_per_issue: Dict[str, List[str]]
    ) -> List[tuple]:
        """Return all possible offers sorted by descending utility."""
        issues = list(options_per_issue.keys())
        results: List[tuple] = []
        for combo in itertools.product(*[options_per_issue[i] for i in issues]):
            offer = dict(zip(issues, combo))
            results.append((offer, round(self.utility(offer, options_per_issue), 6)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: Dict[str, List[str]],
    ) -> tuple:
        """Return ``(offer, aspiration)`` — the least conceding offer that still
        meets the current aspiration threshold (Boulware / linear)."""
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        outcomes = self._all_outcomes_sorted(options_per_issue)
        best = outcomes[0][0]  # fallback = ideal offer
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
        """Return ``"accept"`` or ``"reject"``."""
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        return "accept" if self.utility(offer, options_per_issue) >= asp else "reject"


# ─────────────────────────────────────────────────────────────────────────────
# Shared FastAPI app factory
# ─────────────────────────────────────────────────────────────────────────────


def make_agent_app(agents_registry: Dict[str, GenericCallbackAgent]) -> FastAPI:
    """Return a FastAPI app whose ``POST /decide`` handles all registered agents.

    The endpoint receives ``List[SSTPNegotiateMessage]``, processes one message
    per participant using its registered :class:`GenericCallbackAgent`, and
    returns ``List[SSTPNegotiateMessage]`` of the same length.

    This matches the contract expected by
    :meth:`~app.agent.batch_callback_runner.BatchCallbackRunner._post_batch`.

    The *agents_registry* dict may be mutated between missions without
    restarting the server.

    Args:
        agents_registry: Mutable ``{participant_id: GenericCallbackAgent}`` map.
    """
    app = FastAPI(title="Evaluation Agent Server")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: List[Dict[str, Any]] = await request.json()

        def _process_one(body: Dict[str, Any]) -> Dict[str, Any]:
            payload: Dict[str, Any] = body.get("payload") or {}
            participant_id: str = payload.get("participant_id") or ""

            # Look up the agent; fall back gracefully when id slightly differs
            agent: Optional[GenericCallbackAgent] = agents_registry.get(participant_id)
            if agent is None:
                for pid, a in agents_registry.items():
                    if participant_id in pid or pid in participant_id:
                        agent = a
                        break
            if agent is None:
                agent = next(iter(agents_registry.values()))

            action: str = payload.get("action", "respond")
            round_num: int = payload.get("round", 1)
            n_steps: int = payload.get("n_steps") or 100

            # Issues and options travel in semantic_context (not payload)
            sc: Dict[str, Any] = body.get("semantic_context") or {}
            issues: List[str] = sc.get("issues") or []
            options_per_issue: Dict[str, List[str]] = sc.get("options_per_issue") or {}
            session_id: str = sc.get("session_id") or "unknown"

            sao_state_dict = sc.get("sao_state")
            incoming_sao_state: Optional[SAOState] = (
                SAOState(**sao_state_dict) if sao_state_dict else None
            )

            if action == "propose":
                offer, _asp = agent.decide_propose(
                    round_num, n_steps, options_per_issue
                )
                reply_payload = attach_reason(
                    {
                        "action": "counter_offer",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "offer": offer,
                    },
                    f"{agent.name}: counter-offer.",
                )
                sao_resp = SAOResponse(
                    response=ResponseType.REJECT_OFFER, outcome=offer
                )
                return _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )

            elif action == "respond":
                current_offer: Dict[str, str] = payload.get("current_offer") or {}
                decision = agent.decide_respond(
                    current_offer, round_num, n_steps, options_per_issue
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
                return _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )

            else:
                reply_payload = attach_reason(
                    {"action": "reject", "round": round_num},
                    f"{agent.name}: unknown action; rejecting.",
                )
                sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER)
                return _build_sstp_reply(
                    session_id,
                    agent.name,
                    reply_payload,
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )

        # Expand broadcast messages (participant_id=null) into one copy per agent
        # so BatchCallbackRunner gets exactly N replies (one per participant).
        import copy as _copy

        expanded: List[Dict[str, Any]] = []
        for msg in messages:
            pid = (msg.get("payload") or {}).get("participant_id")
            if pid is None:
                for agent_pid in agents_registry:
                    msg_copy = _copy.deepcopy(msg)
                    msg_copy.setdefault("payload", {})["participant_id"] = agent_pid
                    expanded.append(msg_copy)
            else:
                expanded.append(msg)

        # Process all participants and return a list of SSTP reply dicts
        replies = [_process_one(msg) for msg in expanded]
        return JSONResponse(replies)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Server lifecycle helpers
# ─────────────────────────────────────────────────────────────────────────────


def _start_agent_server(
    agents_registry: Dict[str, GenericCallbackAgent],
    port: int,
) -> tuple:
    """Start the shared agent FastAPI server in a daemon thread.

    Returns ``(server, thread)`` — call ``server.should_exit = True`` to stop.
    """
    app = make_agent_app(agents_registry)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server, t


def _wait_for_server(port: int, retries: int = 40, delay: float = 0.2) -> None:
    """Block until the agent server at *port* is accepting connections."""
    for _ in range(retries):
        try:
            httpx.get(f"http://127.0.0.1:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(
        f"Agent server on port {port} did not start within " f"{retries * delay:.1f}s."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissionResult:
    """Outcome of one negotiation mission.

    Attributes:
        mission_name: :attr:`MissionEntry.name` of the evaluated mission.
        issues: Issues that were negotiated.
        options_per_issue: Option space used during evaluation.
        n_steps_budget: SAO step budget passed to the runner.
        result: Full :class:`~app.agent.negotiation_model.NegotiationResult`
            returned by :class:`~app.agent.batch_callback_runner.BatchCallbackRunner`.
    """

    mission_name: str
    issues: List[str]
    options_per_issue: Dict[str, List[str]]
    n_steps_budget: int
    result: NegotiationResult

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable summary of this mission's outcome."""
        return {
            "mission": self.mission_name,
            "agreement": (
                {o.issue_id: o.chosen_option for o in self.result.agreement}
                if self.result.agreement
                else None
            ),
            "timedout": self.result.timedout,
            "broken": self.result.broken,
            "steps": self.result.steps,
            "n_steps_budget": self.n_steps_budget,
        }


@dataclass
class EvaluationResult:
    """Aggregated results across all missions.

    Attributes:
        mission_results: One :class:`MissionResult` per evaluated mission,
            in the same order as the missions dataset.
    """

    mission_results: List[MissionResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dict suitable for printing or JSON export."""
        total = len(self.mission_results)
        agreed = sum(1 for r in self.mission_results if r.result.agreement is not None)
        timedout = sum(1 for r in self.mission_results if r.result.timedout)
        broken = sum(1 for r in self.mission_results if r.result.broken)
        avg_steps = (
            sum(r.result.steps for r in self.mission_results) / total if total else 0.0
        )
        return {
            "total_missions": total,
            "agreement_rate": round(agreed / total, 4) if total else 0.0,
            "timedout": timedout,
            "broken": broken,
            "avg_steps": round(avg_steps, 1),
            "missions": [r.as_dict() for r in self.mission_results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationRunner:
    """Runs all missions from the dataset with the configured agents.

    Usage::

        config = EvaluationConfig(
            agents=[
                AgentConfig("buyer",  prefer_low=True),
                AgentConfig("seller", prefer_low=False),
            ],
            environment=EnvironmentConfig(agent_port=8094),
            missions=MissionsConfig(path="missions.yaml"),
        )
        results = EvaluationRunner(config).run()
        import json; print(json.dumps(results.summary(), indent=2))

    In-process vs external agents
    ------------------------------
    * **All in-process** (``AgentConfig.url is None`` for every agent): the
      runner starts a local FastAPI server and routes all rounds through it.
    * **All external** (every ``AgentConfig.url`` is set to the same URL): the
      runner skips the local server and uses that URL directly.
    * **Mixed** setups raise :exc:`ValueError`.

    Args:
        config: Combined evaluation configuration.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _resolve_agent_url(self) -> Optional[str]:
        """Return the single agent callback URL, or ``None`` for in-process."""
        urls = {cfg.url for cfg in self.config.agents}

        if urls == {None}:
            return None  # all in-process

        none_count = sum(1 for cfg in self.config.agents if cfg.url is None)
        if none_count > 0:
            raise ValueError(
                "Mixed in-process and external agents are not supported. "
                "Set 'url' for all agents or for none."
            )

        unique_urls = urls - {None}
        if len(unique_urls) > 1:
            raise ValueError(
                "All external agents must share the same callback URL. "
                f"Got: {unique_urls}"
            )
        return next(iter(unique_urls))

    def _build_participants(
        self, options_per_issue: Dict[str, List[str]]
    ) -> List[NegotiationParticipant]:
        """Build :class:`NegotiationParticipant` list from agent configs.

        Preference weights are derived from ``AgentConfig.prefer_low`` so they
        match the in-process agent's utility model.
        """
        participants: List[NegotiationParticipant] = []
        for cfg in self.config.agents:
            prefs: Dict[str, Dict[str, float]] = {}
            for issue, options in options_per_issue.items():
                n = len(options)
                prefs[issue] = {
                    opt: (
                        (1.0 - idx / max(n - 1, 1))
                        if cfg.prefer_low
                        else (idx / max(n - 1, 1))
                    )
                    for idx, opt in enumerate(options)
                }
            participants.append(
                NegotiationParticipant(
                    id=cfg.agent_id,
                    name=cfg.agent_id,
                    preferences=prefs,
                )
            )
        return participants

    # ------------------------------------------------------------------
    # Per-mission helper
    # ------------------------------------------------------------------

    def _resolve_mission_space(self, mission: MissionEntry) -> tuple:
        """Return ``(issues, options_per_issue, n_steps)`` for a mission.

        Raises:
            ValueError: When the mission has no issues or options and no LLM
                discovery is configured (future extension point).
        """
        options_per_issue = dict(mission.options_per_issue)
        issues = (
            list(options_per_issue.keys())
            if options_per_issue
            else list(mission.issues)
        )

        if not issues or not options_per_issue:
            raise ValueError(
                f"Mission '{mission.name}' has no issues/options_per_issue. "
                "Either populate them in the YAML or implement an "
                "IntentDiscovery + OptionsGeneration step."
            )

        n_steps = mission.n_steps
        if n_steps == 0:
            n_steps = compute_n_steps(
                n_agents=len(self.config.agents),
                n_issues=len(issues),
                options_per_issue=options_per_issue,
            )
            logger.info("Mission '%s': auto-computed n_steps=%d", mission.name, n_steps)

        return issues, options_per_issue, n_steps

    def _run_mission(
        self,
        mission: MissionEntry,
        agent_url: str,
        batch_runner: BatchCallbackRunner,
    ) -> MissionResult:
        """Execute one mission and return its :class:`MissionResult`."""
        issues, options_per_issue, n_steps = self._resolve_mission_space(mission)
        participants = self._build_participants(options_per_issue)
        session_id = str(uuid.uuid4())

        # BatchCallbackRunner is re-used across missions; n_steps varies per mission
        batch_runner.n_steps = n_steps

        logger.info(
            "Running mission '%s'  issues=%s  n_steps=%d  session=%s",
            mission.name,
            issues,
            n_steps,
            session_id,
        )

        result: NegotiationResult = batch_runner.run(
            issues=issues,
            options_per_issue=options_per_issue,
            participants=participants,
            session_id=session_id,
            agent_url=agent_url,
        )

        status = (
            "agreement"
            if result.agreement
            else ("timedout" if result.timedout else "broken")
        )
        logger.info("Mission '%s' → %s  steps=%d", mission.name, status, result.steps)

        return MissionResult(
            mission_name=mission.name,
            issues=issues,
            options_per_issue=options_per_issue,
            n_steps_budget=n_steps,
            result=result,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> EvaluationResult:
        """Run all missions in the dataset and return aggregated results.

        The method:

        1. Loads missions from :class:`~evaluation.framework.config.MissionsConfig`.
        2. Starts the in-process agent server (when all agents are in-process).
        3. Iterates over missions, running
           :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` for each.
        4. Shuts down the agent server (when started) before returning.

        Returns:
            :class:`EvaluationResult` with one :class:`MissionResult` per mission.
        """
        missions = self.config.missions.load()
        if not missions:
            logger.warning("No missions to evaluate; returning empty result.")
            return EvaluationResult()

        env = self.config.environment
        external_url = self._resolve_agent_url()

        server: Optional[uvicorn.Server] = None
        server_thread: Optional[threading.Thread] = None
        agents_registry: Dict[str, GenericCallbackAgent] = {}

        try:
            if external_url is None:
                # Build in-process agents and start the shared callback server
                agents_registry = {
                    cfg.agent_id: GenericCallbackAgent(cfg)
                    for cfg in self.config.agents
                }
                server, server_thread = _start_agent_server(
                    agents_registry, env.agent_port
                )
                _wait_for_server(env.agent_port)
                agent_url = f"http://127.0.0.1:{env.agent_port}/decide"
            else:
                agent_url = external_url

            batch_runner = BatchCallbackRunner(
                n_steps=100,  # overridden per mission in _run_mission
                timeout=env.neg_timeout,
            )

            mission_results: List[MissionResult] = []
            for mission in missions:
                try:
                    mission_result = self._run_mission(mission, agent_url, batch_runner)
                    mission_results.append(mission_result)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Mission '%s' failed with error: %s",
                        mission.name,
                        exc,
                        exc_info=True,
                    )

        finally:
            if server is not None:
                server.should_exit = True
            if server_thread is not None:
                server_thread.join(timeout=5.0)

        return EvaluationResult(mission_results=mission_results)
