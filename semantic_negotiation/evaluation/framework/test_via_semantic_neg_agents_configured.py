# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""test_via_semantic_neg_agents_configured.py

LLM negotiation agents with per-mission persona configs loaded from
``datasets/agent_configs.yaml``.

This is the **configured variant** of ``test_via_semantic_neg_agents.py``.
Both scripts are kept; use this one when running Example9 or any dataset that
needs domain-specific agent personas.

Key differences from the base script
--------------------------------------
* ``--agent-configs PATH`` loads per-mission LLM agent personas from YAML.
  Missions with a matching ``mission_agents`` entry get domain-specific
  agents (e.g. DeleteAgent / ArchiveAgent / ComplianceAgent for Example 01).
  Other missions fall back to ``default_agents`` (3 generic agents) or
  the built-in hard-coded trio.
* ``_build_initiate_payload`` uses the actual agent ids/names from config.
* Agent registry is swapped in-place between missions (no server restart).

Usage
-----
All missions live in ``evaluation/framework/missions.yaml``.
Use ``--filter`` to select a subset; omit it to run all 16.

::

    # Terminal 1: start the negotiation server (keep running)
    cd semantic_negotiation
    poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089

    # Terminal 2: run Hard 5 missions only
    cd semantic_negotiation
    poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py \
        --filter hard

    # Run Example9 missions only (domain-specific agents per mission from agent_configs.yaml)
    poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py \
        --filter example

    # Run all 16 missions
    poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py

    # Run a single mission by name substring
    poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py \
        --filter "Hard 01"

    # Override agent config (defaults to agent_configs.yaml next to this script)
    poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py \
        --filter hard \
        --agent-configs /path/to/custom_agent_configs.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import copy
import random
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Add workspace root so protocol.sstp is importable
# File lives at semantic_negotiation/evaluation/framework/ — 3 levels inside the repo root.
_workspace_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

# Add semantic_negotiation/app so config.utils is importable
_sna_app_root = str(Path(__file__).resolve().parent.parent.parent / "app")
if _sna_app_root not in sys.path:
    sys.path.insert(0, _sna_app_root)

from protocol.sstp import SSTPNegotiateMessage  # noqa: E402
from protocol.sstp._base import Origin, PolicyLabels, Provenance  # noqa: E402
from protocol.sstp.negotiate import NegotiateSemanticContext  # noqa: E402
from protocol.sstp.negmas_sao import ResponseType, SAOResponse, SAOState  # noqa: E402
from config.utils import get_llm_provider  # noqa: E402

try:
    import litellm as _litellm  # noqa: E402
except ImportError:
    _litellm = None  # type: ignore[assignment]

NEG_SERVER = "http://localhost:8089"

# ── Per-mission LLM usage accumulator ────────────────────────────────────────
# Tracks all litellm.completion() calls made in THIS process (agent decide calls).
# IntentDiscovery + OptionsGeneration run inside the neg server (separate process)
# and are NOT captured here — they are counted as 2 fixed calls in llm_calls_initiate.
_mission_llm_usage: dict[str, Any] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
}


def _llm_usage_callback(
    kwargs: dict,
    completion_response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """litellm success_callback — accumulates token usage + cost per mission."""
    usage = getattr(completion_response, "usage", None) or {}
    if hasattr(usage, "__dict__"):
        usage = usage.__dict__
    elif not isinstance(usage, dict):
        usage = {}
    _mission_llm_usage["calls"] += 1
    _mission_llm_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    _mission_llm_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    _mission_llm_usage["total_tokens"] += int(usage.get("total_tokens") or 0)
    try:
        cost = _litellm.completion_cost(completion_response=completion_response)
        _mission_llm_usage["estimated_cost_usd"] += float(cost or 0.0)
    except Exception:
        pass


def _reset_mission_llm_usage() -> None:
    """Reset the per-mission accumulator before each mission starts."""
    _mission_llm_usage.update(
        calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )


def _snapshot_mission_llm_usage() -> dict[str, Any]:
    """Return a copy of the current accumulator (decide-phase calls only)."""
    snap = dict(_mission_llm_usage)
    snap["estimated_cost_usd"] = round(snap["estimated_cost_usd"], 6)
    snap["note"] = (
        "decide-phase only (agent LLM calls); initiate-phase runs in neg server"
    )
    return snap


AGENT_PORT = 8092  # single shared server — all agents are reachable here

# ── LLM prompt mode ───────────────────────────────────────────────────────
# "sstp"    — LLM receives the full raw SSTPNegotiateMessage as a JSON block.
#             This is the most information-dense representation and follows
#             the protocol structure exactly.
# "english" — LLM receives a plain-English narrative summarising the message
#             content: session, issues, options, current offer, sao_state,
#             and deadline pressure.  Useful when the model handles prose
#             better than raw JSON.
#
# Can be overridden per-agent via the `prompt_mode` constructor argument.
PROMPT_MODE: str = "sstp"


# ── filesystem helpers ─────────────────────────────────────────────────────


def _slug(name: str) -> str:
    """Convert a display name to a lowercase, underscore-separated filesystem slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_sstp_reply(
    session_id: str,
    agent_name: str,
    reply_payload: dict[str, Any],
    sao_response: SAOResponse | None = None,
    sao_state: SAOState | None = None,
) -> dict[str, Any]:
    """Wrap an agent's reply payload in a full SSTPNegotiateMessage envelope.

    The ``reply_payload`` becomes ``payload`` inside the message.  For a
    propose turn it should be ``{"action": "counter_offer", "offer": {...}}``;
    for a respond turn ``{"action": "accept" | "reject"}``.
    ``sao_response`` encodes the agent's SAO-level decision so the server can
    read the accept/reject/counter from the structured field without parsing
    the raw payload.
    """
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
        origin=Origin(
            actor_id=_slug(agent_name),
            tenant_id=session_id,
        ),
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


# ── agent decision engine ──────────────────────────────────────────────────


class LocalAgent:
    """Stateless decision engine for a callback agent.

    Preferences are private — never sent to the negotiation server.

    Args:
        name: Display name (logged in output).
        prefer_low: If True, prefer index-0 options (cheap side).
                    If False, prefer last-index options (premium side).
        accept_threshold: Minimum utility to accept an offer (0–1).
    """

    def __init__(
        self, name: str, prefer_low: bool, accept_threshold: float = 0.35
    ) -> None:
        self.name = name
        self.prefer_low = prefer_low
        self.accept_threshold = accept_threshold
        # Preferences are built lazily from the first payload that carries
        # options_per_issue — the agent has no prior knowledge of the space.
        self._prefs: dict[str, dict[str, float]] = {}

    def _build_prefs(
        self, options_per_issue: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Derive a utility map from the wire-format options list."""
        prefs: dict[str, dict[str, float]] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            denom = max(n - 1, 1)
            if self.prefer_low:
                prefs[issue] = {
                    o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)
                }
            else:
                prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}
        return prefs

    def _ensure_prefs(
        self, options_per_issue: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Return the cached preference map, rebuilding if issues changed."""
        if set(options_per_issue.keys()) != set(self._prefs.keys()):
            self._prefs = self._build_prefs(options_per_issue)
        return self._prefs

    def utility(
        self, offer: dict[str, str], options_per_issue: dict[str, list[str]]
    ) -> float:
        """Compute mean utility for an offer, deriving the scale from options_per_issue."""
        prefs = self._ensure_prefs(options_per_issue)
        known = [issue for issue in offer if issue in prefs]
        if not known:
            return 0.0
        return sum(prefs[issue].get(offer[issue], 0.0) for issue in known) / len(known)

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> dict[str, str]:
        """Generate a counter-offer using the negotiation space from the payload.

        Starts at the preferred extreme, concedes toward the middle
        as rounds progress.  Returns ``(offer, None)`` for API compatibility
        with :class:`NegMASConcessionAgent` which returns ``(offer, aspiration)``.
        """
        progress = round_num / max(n_steps, 1)  # 0.0 → 1.0
        offer: dict[str, str] = {}
        for issue, opts in options_per_issue.items():
            n = len(opts)
            if self.prefer_low:
                idx = int(progress * (n // 2))
            else:
                idx = (n - 1) - int(progress * (n // 2))
            offer[issue] = opts[max(0, min(n - 1, idx))]
        return offer, None

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Return 'accept' or 'reject' for the incoming offer."""
        u = self.utility(offer, options_per_issue)
        # Threshold shrinks as rounds progress (willing to accept less over time)
        threshold = max(
            self.accept_threshold, 0.8 * (1.0 - round_num / max(n_steps, 1))
        )
        decision = "accept" if u >= threshold else "reject"
        print(
            f"  [{self.name}] respond  round={round_num}  utility={u:.3f}"
            f"  threshold={threshold:.3f}  → {decision}",
            flush=True,
        )
        return decision


class NegMASConcessionAgent(LocalAgent):
    """Agent that uses the NegMAS ``AspirationNegotiator`` concession formula.

    At relative time ``t = round / n_steps`` (0 → 1), the aspiration level is::

        aspiration(t) = 1 - t ** exponent

    Common presets (same as NegMAS time-based negotiators):

    - **Boulware**  ``exponent > 1``  — holds ideal offer long, concedes hard at end.
    - **Linear**    ``exponent = 1``  — steady linear concession.
    - **Conceder**  ``exponent < 1``  — concedes quickly early, then plateaus.

    **Propose**: enumerate every possible outcome, pick the one with utility
    closest to (but ≥) ``aspiration(t)``.  Falls back to the best outcome if
    none qualifies.

    **Respond**: accept if ``utility(offer) ≥ aspiration(t)``.

    Args:
        name: Display name.
        prefer_low: Determines which end of each option list is preferred.
        exponent: Concession exponent.  Default ``2.0`` (Boulware).
        min_reservation: Hard floor — never accept below this utility (0–1).
    """

    def __init__(
        self,
        name: str,
        prefer_low: bool,
        exponent: float = 2.0,
        min_reservation: float = 0.0,
    ) -> None:
        super().__init__(name, prefer_low, accept_threshold=min_reservation)
        self.exponent = exponent
        self.min_reservation = min_reservation

    def _aspiration(self, t: float) -> float:
        """NegMAS aspiration curve.  Returns 1.0 at t=0, approaches 0 at t=1."""
        return max(self.min_reservation, 1.0 - (t**self.exponent))

    def _all_outcomes_sorted(
        self, options_per_issue: dict[str, list[str]]
    ) -> list[tuple[dict, float]]:
        """Return all (offer, utility) pairs sorted by utility descending."""
        prefs = self._ensure_prefs(options_per_issue)
        issues = list(options_per_issue.keys())
        results: list[tuple[dict, float]] = []
        for combo in itertools.product(*[options_per_issue[i] for i in issues]):
            offer = dict(zip(issues, combo))
            u = sum(prefs[issue].get(offer[issue], 0.0) for issue in issues) / len(
                issues
            )
            results.append((offer, round(u, 4)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> dict[str, str]:
        """Propose the outcome with utility closest to (and ≥) aspiration(t)."""
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        outcomes = self._all_outcomes_sorted(options_per_issue)
        # Walk from best → worst; return the last one that is still >= asp.
        # That is: the minimum-utility outcome that still satisfies the aspiration.
        best_qualifying: dict[str, str] = outcomes[0][0]  # fallback = ideal offer
        for offer, u in outcomes:
            if u >= asp:
                best_qualifying = (
                    offer  # keep updating → we want the minimum qualifying
                )
            else:
                break  # sorted desc — no further outcome can qualify
        return best_qualifying, asp

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Accept if utility(offer) ≥ aspiration(t), otherwise reject."""
        u = self.utility(offer, options_per_issue)
        t = round_num / max(n_steps, 1)
        asp = self._aspiration(t)
        decision = "accept" if u >= asp else "reject"
        print(
            f"  [{self.name}] respond  round={round_num}  utility={u:.3f}"
            f"  aspiration={asp:.3f}  → {decision}",
            flush=True,
        )
        return decision


# ── LLM-based negotiation agent ───────────────────────────────────────────


class LLMNegotiationAgent(LocalAgent):
    """Negotiation agent that reads the full SSTPNegotiateMessage and fills
    ``sao_response`` based on its private objective.

    The LLM always receives the **complete** incoming ``SSTPNegotiateMessage``
    (all fields: ``kind``, ``version``, ``origin``, ``semantic_context`` with
    ``sao_state`` / ``sao_response`` / ``issues`` / ``options_per_issue``, full
    ``payload``, ``policy_labels``, ``provenance``, etc.) and must reply by
    filling ``sao_response``:

    - ``response=0`` (ACCEPT_OFFER) + ``outcome`` = accepted offer dict
    - ``response=1`` (REJECT_OFFER) + ``outcome`` dict = counter-offer  (propose)
    - ``response=1`` (REJECT_OFFER) + ``outcome=null``                   (reject)

    The *prompt_mode* argument controls how the message is rendered in the prompt:

    - ``"sstp"``    — the raw message is embedded as a ``json`` fenced block,
                       so the LLM sees the exact protocol structure.
    - ``"english"`` — the same information is rendered as a plain-English
                       narrative paragraph (useful for models that handle prose
                       better than raw JSON).

    Args:
        name: Display name.
        prefer_low: Determines which end of each option list is preferred
                    (used to build the utility weight table sent to the LLM).
        persona: Free-text description of the agent's negotiating style and
                 goals — injected into every LLM prompt.
        prompt_mode: ``"sstp"`` (default) or ``"english"``.
                     Falls back to the module-level :data:`PROMPT_MODE` constant.
    """

    def __init__(
        self,
        name: str,
        prefer_low: bool,
        persona: str,
        prompt_mode: str = PROMPT_MODE,
    ) -> None:
        super().__init__(name, prefer_low, accept_threshold=0.0)
        self.persona = persona
        self.prompt_mode = prompt_mode
        self._llm = get_llm_provider()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _utility_table(self, options_per_issue: dict[str, list[str]]) -> str:
        """Format a human-readable utility weight table for the LLM prompt."""
        prefs = self._ensure_prefs(options_per_issue)
        lines: list[str] = []
        for issue, opts in options_per_issue.items():
            for opt in opts:
                u = prefs[issue].get(opt, 0.0)
                lines.append(f"  [{issue}] '{opt}' → {u:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Return the first balanced JSON object found in *text*.

        Handles nested objects (e.g. ``{"response": 1, "outcome": {"k": "v"}}``).
        Falls back to a direct ``json.loads`` attempt first so clean responses
        (no surrounding prose) are parsed without the brace-counting path.
        """
        # Fast path: the whole text is already valid JSON
        try:
            stripped = text.strip()
            # Strip optional markdown code fences
            if stripped.startswith("```"):
                stripped = re.sub(r"^```[a-z]*\n?", "", stripped).rstrip("`").strip()
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # Slow path: scan for the outermost balanced { … }
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response: {text!r}")
        depth = 0
        in_str = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_str:
                escape_next = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise ValueError(f"Unbalanced braces in LLM response: {text!r}")

    # ------------------------------------------------------------------
    # primary decision interface: read full SSTPNegotiateMessage
    # ------------------------------------------------------------------

    def decide_from_sstp_message(
        self,
        body: dict[str, Any],
    ) -> tuple[str, dict[str, str] | None, SAOResponse]:
        """Read a full SSTPNegotiateMessage and return (action_str, outcome, sao_response).

        The LLM reads ``semantic_context`` (sao_state, issues, options_per_issue)
        and ``payload`` (action, current_offer, round, n_steps) and fills
        ``sao_response`` according to the SSTP/NegMAS SAO protocol.

        Returns:
            action_str: ``"counter_offer"`` | ``"accept"`` | ``"reject"``
            outcome: offer dict for counter_offer/accept, ``None`` for reject
            sao_response: filled :class:`SAOResponse` Pydantic object
        """
        payload: dict[str, Any] = body.get("payload", {})
        semantic_ctx: dict[str, Any] = body.get("semantic_context") or {}
        action = payload.get("action", "respond")
        round_num: int = payload.get("round", 1)
        n_steps: int = payload.get("n_steps") or 200
        current_offer: dict = payload.get("current_offer") or {}
        issues: list[str] = semantic_ctx.get("issues") or []
        options_per_issue: dict[str, list[str]] = (
            semantic_ctx.get("options_per_issue") or {}
        )
        sao_state_raw: dict = semantic_ctx.get("sao_state") or {}
        t = round_num / max(n_steps, 1)
        # True when this agent is the designated next proposer and may counter_offer.
        _pid = payload.get("participant_id", "")
        _nxt = payload.get("next_proposer_id", "")
        is_next_proposer: bool = bool(_pid and _pid == _nxt)

        # Shadow respond with no offer — skip LLM call
        if action == "respond" and not current_offer:
            return (
                "reject",
                None,
                SAOResponse(response=ResponseType.REJECT_OFFER, outcome=None),
            )

        # ── Build the prompt context based on prompt_mode ─────────────────
        if action == "propose":
            task_and_format = (
                "Task: You must PROPOSE a counter-offer for this round.\n"
                "Set response=1 (REJECT_OFFER) and outcome = your proposed offer dict.\n\n"
                "Reply ONLY with this JSON (no explanation, no markdown):\n"
                '{"response": 1, "outcome": {"<issue>": "<chosen_option>", ...}}'
            )
        else:
            if is_next_proposer:
                task_and_format = (
                    "Task: You are the designated proposer this round."
                    " You may ACCEPT, REJECT, or COUNTER-OFFER the current_offer.\n"
                    "If you accept       → response=0, outcome = the offer dict.\n"
                    "If you reject       → response=1, outcome = null.\n"
                    "If you counter-offer → response=1, outcome = your proposed offer dict.\n"
                    "The closer to the deadline (t→1), the more you should be willing to accept.\n\n"
                    "Reply ONLY with this JSON (no explanation, no markdown):\n"
                    '{"response": 0, "outcome": {...}}  OR  {"response": 1, "outcome": null}  OR\n'
                    '{"response": 1, "outcome": {"<issue>": "<chosen_option>", ...}}'
                )
            else:
                task_and_format = (
                    "Task: You must ACCEPT or REJECT the current_offer in the payload.\n"
                    "If you accept → response=0, outcome = the offer dict.\n"
                    "If you reject → response=1, outcome = null.\n"
                    "The closer to the deadline (t→1), the more you should be willing to accept.\n\n"
                    "Reply ONLY with this JSON (no explanation, no markdown):\n"
                    '{"response": 0, "outcome": {...}} or {"response": 1, "outcome": null}'
                )

        if self.prompt_mode == "english":
            # ── English-narrative rendering ───────────────────────────────
            issues_str = ", ".join(issues) if issues else "(none)"
            opts_lines = "\n".join(
                f"  - {iss}: {', '.join(opts)}"
                for iss, opts in options_per_issue.items()
            )
            sao_state_summary = (
                f"Round {sao_state_raw.get('step', round_num)} of "
                f"{sao_state_raw.get('n_steps', n_steps)}, "
                f"running={sao_state_raw.get('running', True)}, "
                f"current standing offer in state: "
                f"{json.dumps(sao_state_raw.get('current_offer')) or '(none)'}, "
                f"agreement so far: {json.dumps(sao_state_raw.get('agreement')) or '(none)'}"
            )
            prev_sao_resp = semantic_ctx.get("sao_response")
            prev_resp_str = (
                f"The server's previous sao_response was: {json.dumps(prev_sao_resp)}"
                if prev_sao_resp
                else "No previous sao_response from the server."
            )
            offer_str = (
                f"The current offer on the table is: {json.dumps(current_offer)}."
                if current_offer
                else "No current offer on the table (you are the first proposer)."
            )
            message_context = (
                f"You have received an SSTPNegotiateMessage with the following content:\n"
                f"  Session ID   : {semantic_ctx.get('session_id', 'unknown')}\n"
                f"  Message kind : {body.get('kind', 'negotiate')}\n"
                f"  Message ID   : {body.get('message_id', '?')}\n"
                f"  Origin       : {json.dumps(body.get('origin'))}\n"
                f"  Issues to negotiate: {issues_str}\n"
                f"  Available options per issue:\n{opts_lines}\n"
                f"  SAO state    : {sao_state_summary}\n"
                f"  {prev_resp_str}\n"
                f"  Payload action: {action}  (round {round_num} / {n_steps})\n"
                f"  {offer_str}\n"
                f"  Policy labels: {json.dumps(body.get('policy_labels'))}\n"
                f"  Provenance   : {json.dumps(body.get('provenance'))}\n"
            )
        else:
            # ── Default: full SSTPNegotiateMessage as JSON ────────────────
            message_context = (
                f"You have received this complete SSTPNegotiateMessage:\n"
                f"```json\n{json.dumps(body, indent=2)}\n```\n"
            )

        prompt = (
            f"You are {self.name} in a multi-party SAO (Stacked Alternating Offers) negotiation.\n"
            f"{self.persona}\n\n"
            f"{message_context}\n"
            f"Your private utility weights (higher = better for you):\n"
            f"{self._utility_table(options_per_issue)}\n\n"
            f"Relative time: t = {t:.2f}  (0=negotiation start, 1=deadline)\n"
            f"Prompt mode  : {self.prompt_mode}\n\n"
            f"ResponseType: 0 = ACCEPT_OFFER, 1 = REJECT_OFFER\n\n"
            f"{task_and_format}"
        )

        try:
            raw = self._llm(prompt)
            data = self._extract_json(raw)
            resp_int = int(data.get("response", 1))
            outcome_raw = data.get("outcome")

            if resp_int == int(ResponseType.ACCEPT_OFFER):
                outcome = current_offer or (
                    outcome_raw if isinstance(outcome_raw, dict) else {}
                )
                sao_resp = SAOResponse(
                    response=ResponseType.ACCEPT_OFFER, outcome=outcome
                )
                print(
                    f"  [{self.name}] {action}  round={round_num}"
                    f"  sao_response=ACCEPT_OFFER",
                    flush=True,
                )
                return "accept", outcome, sao_resp

            else:  # REJECT_OFFER
                if outcome_raw and isinstance(outcome_raw, dict) and options_per_issue:
                    # Validate counter-offer options against known space
                    validated: dict[str, str] = {}
                    for issue, opts in options_per_issue.items():
                        chosen = outcome_raw.get(issue)
                        validated[issue] = (
                            chosen
                            if chosen in opts
                            else (opts[0] if self.prefer_low else opts[-1])
                        )
                    sao_resp = SAOResponse(
                        response=ResponseType.REJECT_OFFER, outcome=validated
                    )
                    print(
                        f"  [{self.name}] {action}  round={round_num}"
                        f"  sao_response=REJECT_OFFER+counter  offer={validated}",
                        flush=True,
                    )
                    return "counter_offer", validated, sao_resp
                else:
                    sao_resp = SAOResponse(
                        response=ResponseType.REJECT_OFFER, outcome=None
                    )
                    print(
                        f"  [{self.name}] {action}  round={round_num}"
                        f"  sao_response=REJECT_OFFER",
                        flush=True,
                    )
                    return "reject", None, sao_resp

        except Exception as exc:
            print(
                f"  [{self.name}] LLM SSTP error — using fallback: {exc}",
                flush=True,
            )
            if action == "propose" or is_next_proposer:
                fallback = {
                    issue: (opts[0] if self.prefer_low else opts[-1])
                    for issue, opts in options_per_issue.items()
                }
                return (
                    "counter_offer",
                    fallback,
                    SAOResponse(response=ResponseType.REJECT_OFFER, outcome=fallback),
                )
            return (
                "reject",
                None,
                SAOResponse(response=ResponseType.REJECT_OFFER, outcome=None),
            )

    # ------------------------------------------------------------------
    # Fallback shims (base-class interface compatibility)
    # ------------------------------------------------------------------

    def decide_propose(
        self,
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> tuple[dict[str, str], None]:
        """Shim: synthesise a minimal SSTP body and delegate to decide_from_sstp_message."""
        body: dict[str, Any] = {
            "payload": {"action": "propose", "round": round_num, "n_steps": n_steps},
            "semantic_context": {
                "issues": list(options_per_issue.keys()),
                "options_per_issue": options_per_issue,
                "sao_state": {"step": round_num, "n_steps": n_steps},
            },
        }
        _, offer, _ = self.decide_from_sstp_message(body)
        return offer or {}, None

    def decide_respond(
        self,
        offer: dict[str, str],
        round_num: int,
        n_steps: int,
        options_per_issue: dict[str, list[str]],
    ) -> str:
        """Shim: synthesise a minimal SSTP body and delegate to decide_from_sstp_message."""
        body: dict[str, Any] = {
            "payload": {
                "action": "respond",
                "round": round_num,
                "n_steps": n_steps,
                "current_offer": offer,
            },
            "semantic_context": {
                "issues": list(options_per_issue.keys()),
                "options_per_issue": options_per_issue,
                "sao_state": {
                    "step": round_num,
                    "n_steps": n_steps,
                    "current_offer": offer,
                },
            },
        }
        action_str, _, _ = self.decide_from_sstp_message(body)
        return "accept" if action_str == "accept" else "reject"


# ── FastAPI mini-app factory ───────────────────────────────────────────────


def make_agent_app(
    agents: dict[str, LocalAgent],
    trace_state: dict[str, Path],
) -> FastAPI:
    """Return a FastAPI app whose single POST /decide endpoint handles ALL agents.

    The endpoint receives and returns **``List[SSTPNegotiateMessage]``**.  The
    :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` sends the whole
    round's batch in one call — one message per participant.  Each message carries
    ``payload.participant_id`` which is used to dispatch to the right
    :class:`LocalAgent` instance.

    ``agents`` is keyed by participant id (as declared in the initiate payload).

    ``trace_state`` is a mutable dict with key ``"trace_dir"`` (a :class:`Path`).
    The caller updates ``trace_state["trace_dir"]`` before each mission so the
    same running server writes traces into the correct per-mission folder.

    Every non-shadow incoming message and the agent's reply are saved under
    ``<trace_dir>/round_{N:04d}/{action}__{agent_slug}__request.json`` and
    ``…__reply.json``.
    """
    app = FastAPI(title="Shared Agent Server")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: list[dict[str, Any]] = await request.json()

        def _process_one(
            body: dict[str, Any], skip_request_trace: bool = False
        ) -> dict[str, Any]:
            payload: dict[str, Any] = body.get("payload", {})
            participant_id: str = payload.get("participant_id", "")
            agent = agents.get(participant_id)
            if agent is None:
                # Fallback: try matching by name slug against any agent
                for pid, a in agents.items():
                    if _slug(a.name) in _slug(participant_id) or participant_id in pid:
                        agent = a
                        break
            if agent is None:
                # Last resort: use first agent
                agent = next(iter(agents.values()))

            action = payload.get("action")  # "propose" or "respond"
            round_num: int = payload.get("round", 1)
            n_steps: int = payload.get("n_steps") or 200
            _semantic_ctx: dict[str, Any] = body.get("semantic_context") or {}
            issues: list[str] = _semantic_ctx.get("issues") or []
            options_per_issue: dict[str, list[str]] = (
                _semantic_ctx.get("options_per_issue") or {}
            )
            session_id: str = _semantic_ctx.get("session_id") or "unknown-session"
            _sao_state_dict = _semantic_ctx.get("sao_state")
            incoming_sao_state: SAOState | None = (
                SAOState(**_sao_state_dict) if _sao_state_dict else None
            )
            # is_shadow_call=True (e.g. round 1 responders with no standing offer):
            # return a valid response but skip trace writing.
            is_shadow: bool = payload.get("is_shadow_call", False)

            # ── inject issues/options block on first real callback ─────────
            if (
                not is_shadow
                and not trace_state["dialogue_context_logged"]
                and issues
                and options_per_issue
            ):
                trace_state["dialogue_log"].append("")
                trace_state["dialogue_log"].append("  ISSUES IDENTIFIED")
                for iss in issues:
                    trace_state["dialogue_log"].append(f"  • {iss}")
                trace_state["dialogue_log"].append("")
                trace_state["dialogue_log"].append("  OPTIONS PER ISSUE")
                for iss, opts in options_per_issue.items():
                    trace_state["dialogue_log"].append(f"  • {iss}: {', '.join(opts)}")
                trace_state["dialogue_context_logged"] = True

            slug = _slug(agent.name)
            round_dir = trace_state["trace_dir"] / f"round_{round_num:04d}"

            if not is_shadow and not skip_request_trace:
                _save_json(round_dir / f"{action}__{slug}__request.json", body)

            # ── LLM agents: read full SSTPNegotiateMessage, fill sao_response ──
            if isinstance(agent, LLMNegotiationAgent):
                action_str, outcome, sao_resp = agent.decide_from_sstp_message(body)

                if action_str == "counter_offer":
                    offer = outcome or {}
                    if not is_shadow:
                        if round_num != trace_state["dialogue_last_round"]:
                            trace_state["dialogue_log"].append("")
                            trace_state["dialogue_log"].append(
                                f"[Round {round_num}]  Proposer: {agent.name}"
                            )
                            trace_state["dialogue_last_round"] = round_num
                        offer_str = "  |  ".join(
                            f"{k}: '{v}'" for k, v in offer.items()
                        )
                        trace_state["dialogue_log"].append(f"  OFFER    : {offer_str}")
                    reply_payload: dict[str, Any] = {
                        "action": "counter_offer",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "offer": offer,
                    }
                elif action_str == "accept":
                    if not is_shadow:
                        if round_num != trace_state["dialogue_last_round"]:
                            trace_state["dialogue_log"].append("")
                            trace_state["dialogue_log"].append(
                                f"[Round {round_num}]  Proposer: server"
                            )
                            _co = payload.get("current_offer") or {}
                            if _co:
                                _os = "  |  ".join(
                                    f"{k}: '{v}'" for k, v in _co.items()
                                )
                                trace_state["dialogue_log"].append(
                                    f"  OFFER    : {_os}"
                                )
                            trace_state["dialogue_last_round"] = round_num
                        trace_state["dialogue_log"].append(
                            f"  [{agent.name:<8}]  ACCEPT ✓"
                        )
                    reply_payload = {
                        "action": "accept",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                    }
                else:  # reject
                    if not is_shadow:
                        if round_num != trace_state["dialogue_last_round"]:
                            trace_state["dialogue_log"].append("")
                            trace_state["dialogue_log"].append(
                                f"[Round {round_num}]  Proposer: server"
                            )
                            _co = payload.get("current_offer") or {}
                            if _co:
                                _os = "  |  ".join(
                                    f"{k}: '{v}'" for k, v in _co.items()
                                )
                                trace_state["dialogue_log"].append(
                                    f"  OFFER    : {_os}"
                                )
                            trace_state["dialogue_last_round"] = round_num
                        trace_state["dialogue_log"].append(
                            f"  [{agent.name:<8}]  REJECT"
                        )
                    reply_payload = {
                        "action": "reject",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                    }

                reply = _build_sstp_reply(
                    session_id,
                    agent.name,
                    {**reply_payload, "participant_id": participant_id},
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )
                if not is_shadow:
                    _save_json(round_dir / f"{action}__{slug}__reply.json", reply)
                return reply

            # ── Rule-based agents (LocalAgent / NegMASConcessionAgent) ───────
            elif isinstance(agent, LocalAgent):
                current_offer: dict[str, str] = payload.get("current_offer") or {}
                allowed_actions: list[str] = payload.get("allowed_actions") or []

                if action == "propose":
                    # Agent is designated proposer — must counter_offer.
                    offer, _asp = agent.decide_propose(
                        round_num, n_steps, options_per_issue
                    )
                    action_str = "counter_offer"
                    sao_resp = SAOResponse(
                        response=ResponseType.REJECT_OFFER,
                        outcome=offer,
                    )
                    if not is_shadow:
                        if round_num != trace_state["dialogue_last_round"]:
                            trace_state["dialogue_log"].append("")
                            trace_state["dialogue_log"].append(
                                f"[Round {round_num}]  Proposer: {agent.name}"
                            )
                            trace_state["dialogue_last_round"] = round_num
                        offer_str = "  |  ".join(
                            f"{k}: '{v}'" for k, v in offer.items()
                        )
                        trace_state["dialogue_log"].append(f"  OFFER    : {offer_str}")
                    reply_payload = {
                        "action": "counter_offer",
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                        "offer": offer,
                    }
                else:
                    # Agent is responding (accept/reject) to an existing offer.
                    action_str = agent.decide_respond(
                        current_offer, round_num, n_steps, options_per_issue
                    )
                    if action_str == "accept":
                        sao_resp = SAOResponse(
                            response=ResponseType.ACCEPT_OFFER,
                            outcome=current_offer or None,
                        )
                        if not is_shadow:
                            if round_num != trace_state["dialogue_last_round"]:
                                trace_state["dialogue_log"].append("")
                                trace_state["dialogue_log"].append(
                                    f"[Round {round_num}]"
                                )
                                if current_offer:
                                    _os = "  |  ".join(
                                        f"{k}: '{v}'" for k, v in current_offer.items()
                                    )
                                    trace_state["dialogue_log"].append(
                                        f"  OFFER    : {_os}"
                                    )
                                trace_state["dialogue_last_round"] = round_num
                            trace_state["dialogue_log"].append(
                                f"  [{agent.name:<8}]  ACCEPT ✓"
                            )
                    else:
                        sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER)
                        if not is_shadow:
                            if round_num != trace_state["dialogue_last_round"]:
                                trace_state["dialogue_log"].append("")
                                trace_state["dialogue_log"].append(
                                    f"[Round {round_num}]"
                                )
                                if current_offer:
                                    _os = "  |  ".join(
                                        f"{k}: '{v}'" for k, v in current_offer.items()
                                    )
                                    trace_state["dialogue_log"].append(
                                        f"  OFFER    : {_os}"
                                    )
                                trace_state["dialogue_last_round"] = round_num
                            trace_state["dialogue_log"].append(
                                f"  [{agent.name:<8}]  REJECT"
                            )
                        action_str = "reject"

                    reply_payload = {
                        "action": action_str,
                        "round": round_num,
                        "issues": issues,
                        "options_per_issue": options_per_issue,
                    }

                reply = _build_sstp_reply(
                    session_id,
                    agent.name,
                    {**reply_payload, "participant_id": participant_id},
                    sao_response=sao_resp,
                    sao_state=incoming_sao_state,
                )
                if not is_shadow:
                    _save_json(round_dir / f"{action_str}__{slug}__reply.json", reply)
                return reply
            reply_payload = {"action": "reject", "round": round_num}
            sao_resp = SAOResponse(response=ResponseType.REJECT_OFFER)
            reply = _build_sstp_reply(
                session_id,
                agent.name,
                reply_payload,
                sao_response=sao_resp,
                sao_state=incoming_sao_state,
            )
            if not is_shadow:
                _save_json(round_dir / f"unknown__{slug}__reply.json", reply)
            return reply

        # Expand broadcast messages (participant_id=null) into per-agent copies
        # so each agent in the registry receives and replies to the message.
        # Targeted messages (participant_id set) pass through unchanged.
        # Expand broadcast messages (participant_id=null) into per-agent copies.
        # Track which copies are broadcast expansions out-of-band so _process_one
        # can skip writing per-agent request trace files (the broadcast file was
        # already saved above).
        expanded: list[tuple[dict[str, Any], bool]] = []
        for msg in messages:
            pid = (msg.get("payload") or {}).get("participant_id")
            if pid is None or pid in ("broadcast", "server"):
                # Save the original broadcast message once
                # so the trace shows 1 file instead of N identical per-agent copies.
                _bc_action = (msg.get("payload") or {}).get("action", "respond")
                _bc_round = (msg.get("payload") or {}).get("round", 1)
                if not (msg.get("payload") or {}).get("is_shadow_call", False):
                    _bc_dir = trace_state["trace_dir"] / f"round_{_bc_round:04d}"
                    _save_json(_bc_dir / f"{_bc_action}__broadcast__request.json", msg)
                for agent_pid in agents:
                    msg_copy = copy.deepcopy(msg)
                    msg_copy["payload"]["participant_id"] = agent_pid
                    expanded.append((msg_copy, True))
            else:
                expanded.append((msg, False))

        # Run all agent decisions concurrently — each _process_one call (including
        # any LLM round-trip) executes in its own thread so the whole batch
        # resolves in parallel.  Results are gathered and returned as a single
        # list in one HTTP response body.
        replies = await asyncio.gather(
            *[asyncio.to_thread(_process_one, msg, skip) for msg, skip in expanded]
        )
        return JSONResponse(replies)

    return app


# ── server thread ──────────────────────────────────────────────────────────


def start_agent_server(
    agents: dict[str, LocalAgent],
    port: int,
    trace_state: dict,
) -> threading.Thread:
    """Start the shared agent FastAPI server (all agents) in a daemon thread.

    The *agents* dict is passed **by reference** into :func:`make_agent_app`.
    Callers can mutate it in-place to swap agents between missions without
    restarting the server thread.  The reference is also stored at
    ``trace_state["_agents_registry"]`` so the per-mission loop can update it.
    """
    # Store the live registry reference so the per-mission loop can swap agents
    # by mutating this dict in-place (clear + update) rather than restarting.
    trace_state["_agents_registry"] = agents
    app = make_agent_app(agents, trace_state)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


def wait_for_server(port: int, retries: int = 20, delay: float = 0.3) -> None:
    """Block until the agent server is accepting connections."""
    for _ in range(retries):
        try:
            httpx.get(f"http://localhost:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(f"Agent server on port {port} did not start in time")


def _forward_to_agent(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Synchronously POST a single SSTPNegotiateMessage to the agent's /decide endpoint.

    Wraps the message in a single-element list and returns ALL replies.
    A broadcast message (participant_id=null) yields N replies (one per agent);
    a targeted message yields 1 reply.  Used with ``asyncio.to_thread``.
    """
    agent_url = f"http://localhost:{AGENT_PORT}/decide"
    resp = httpx.post(agent_url, json=[msg], timeout=180.0)
    resp.raise_for_status()
    return resp.json() or []


def _build_decide_payload(
    initiate_payload: dict[str, Any],
    session_id: str,
    agent_replies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap agent replies in an SSTPNegotiateMessage for POST /api/negotiate/decide."""
    from protocol.sstp import SSTPNegotiateMessage
    from protocol.sstp._base import Origin, PolicyLabels, Provenance
    from protocol.sstp.negotiate import NegotiateSemanticContext
    import hashlib, json as _json, uuid as _uuid
    from datetime import datetime, timezone

    inner: dict[str, Any] = {
        "session_id": session_id,
        "agent_replies": agent_replies,
    }
    payload_str = _json.dumps(inner, sort_keys=True)
    payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    msg = SSTPNegotiateMessage(
        kind="negotiate",
        message_id=str(_uuid.uuid4()),
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="test-runner", tenant_id="demo"),
        semantic_context=NegotiateSemanticContext(session_id=session_id),
        payload_hash=payload_hash,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload=inner,
    )
    return msg.model_dump(mode="json")


# ── mission definitions ───────────────────────────────────────────────────
#
# Missions are loaded from missions.yaml (next to this script).  Add or edit
# missions there — no changes to this file are required.
#
# Folder layout produced under neg_trace/:
#
#   neg_trace/
#     <YYYYMMDD_HHMMSS>/           ← one directory per run
#       project_contract/          ← one sub-folder per mission (slug of name)
#         00_initiate_request.json
#         round_0001/
#           propose__agent_a__request.json
#           propose__agent_a__reply.json
#           …
#         round_<N+1>/
#           commit_final_result.json   ← commit sits alongside round dirs
#       cloud_platform/
#         00_initiate_request.json
#         round_0001/
#         …
#         round_<N+1>/
#           commit_final_result.json

# Default missions file: evaluation/framework/missions.yaml
# Already contains all missions: 2 generic + 5 hard5 + 9 Example.
# No need to pass --missions-file for standard runs.
_MISSIONS_FILE = Path(__file__).resolve().parent / "missions.yaml"
_AGENT_CONFIGS_FILE = Path(__file__).resolve().parent / "agent_configs.yaml"


def _load_missions(path: Path = _MISSIONS_FILE) -> list[dict[str, Any]]:
    """Load and return the missions list from *path* (a YAML file).

    The file must contain a top-level ``missions:`` key whose value is a
    sequence of mission dicts (see ``missions.yaml`` for the schema).
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    missions = data.get("missions") or []
    if not missions:
        raise ValueError(f"No missions found in {path}")
    return missions


def _load_agent_configs(path: Path | None = None) -> dict[str, Any]:
    """Load agent persona configs from *path* (default: datasets/agent_configs.yaml).

    Returns a dict with keys:
    * ``default_agents``  — list of agent config dicts used when no mission-specific
                            agents are defined.
    * ``mission_agents``  — dict mapping mission name slug → list of agent config dicts.

    Each agent entry may use ``persona_file`` (a path relative to the config file)
    instead of an inline ``persona`` string.  Both forms are supported.
    """
    resolved = path or _AGENT_CONFIGS_FILE
    if not resolved.exists():
        return {"default_agents": [], "mission_agents": {}}
    with resolved.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base_dir = resolved.parent

    def _read_parts(parts_list: list[str]) -> str:
        """Concatenate persona text from a list of part file paths."""
        parts = []
        for rel_path in parts_list:
            pf = base_dir / rel_path
            with pf.open(encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                text = next(iter(loaded.values()), "") or ""
            else:
                text = loaded or ""
            parts.append(text.rstrip())
        return "\n\n".join(parts)

    def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
        """Inline persona_file or persona_parts → persona if needed."""
        if "persona_file" in cfg and "persona" not in cfg:
            pf = base_dir / cfg["persona_file"]
            with pf.open(encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "persona_parts" in loaded:
                # persona_file points to a file that itself composes from parts
                persona = _read_parts(loaded["persona_parts"])
            elif isinstance(loaded, dict):
                persona = loaded.get("persona") or ""
            else:
                persona = loaded or ""
            cfg = {**cfg, "persona": persona}
        elif "persona_parts" in cfg and "persona" not in cfg:
            cfg = {**cfg, "persona": _read_parts(cfg["persona_parts"])}
        return cfg

    default_agents = [_resolve(a) for a in (data.get("default_agents") or [])]
    mission_agents = {
        slug: [_resolve(a) for a in agents]
        for slug, agents in (data.get("mission_agents") or {}).items()
    }
    return {
        "default_agents": default_agents,
        "mission_agents": mission_agents,
    }


def _build_agents_for_mission(
    mission: dict[str, Any],
    agent_configs: dict[str, Any],
) -> dict[str, "LocalAgent"]:
    """Construct :class:`LLMNegotiationAgent` instances for *mission*.

    Lookup order:
    1. ``agent_configs["mission_agents"][<mission_slug>]``  — mission-specific personas
    2. ``agent_configs["default_agents"]``                  — dataset-level defaults
    3. Built-in fallback (3 generic agents A/B/C)

    Returns a dict keyed by participant id (matches ``id`` field in each config entry).
    """
    mission_slug = _slug(mission["name"])
    cfg_list: list[dict[str, Any]] = (
        agent_configs.get("mission_agents", {}).get(mission_slug)
        or agent_configs.get("default_agents")
        or []
    )

    if cfg_list:
        agents: dict[str, LocalAgent] = {}
        for cfg in cfg_list:
            pid = cfg["id"]
            agents[pid] = LLMNegotiationAgent(
                name=cfg.get("name", pid),
                prefer_low=bool(cfg.get("prefer_low", True)),
                persona=cfg.get("persona", "You are a pragmatic negotiator.").strip(),
                prompt_mode=PROMPT_MODE,
            )
        source = (
            "mission-specific"
            if agent_configs.get("mission_agents", {}).get(mission_slug)
            else "default"
        )
        print(
            f"  Agent config  : {source} ({len(agents)} agents: {list(agents.keys())})"
        )
        return agents

    # Built-in fallback
    print("  Agent config  : built-in fallback (3 generic agents)")
    return {
        "agent-a": LLMNegotiationAgent(
            "Agent A",
            prefer_low=True,
            persona=(
                "You are a cost-conscious buyer who strongly prefers the cheapest options "
                "on every issue. You hold your ideal position early in the negotiation but "
                "are willing to make measured concessions as the deadline approaches."
            ),
            prompt_mode=PROMPT_MODE,
        ),
        "agent-b": LLMNegotiationAgent(
            "Agent B",
            prefer_low=False,
            persona=(
                "You are a quality-focused buyer who prefers premium options but knows "
                "that complex multi-issue negotiations require compromise to reach a deal. "
                "You start firm, but from the halfway point (t > 0.5) you actively move "
                "toward mid-range options on at least two issues per proposal. "
                "After t > 0.7 you must accept any offer that gives you premium or mid-range "
                "options on the majority of issues — do not hold out for perfection."
            ),
            prompt_mode=PROMPT_MODE,
        ),
        "agent-c": LLMNegotiationAgent(
            "Agent C",
            prefer_low=True,
            persona=(
                "You are a balanced, pragmatic negotiator who leans toward cost-effective "
                "options but values reaching an agreement. You concede more readily than "
                "the other parties and will accept reasonable compromises at any stage."
            ),
            prompt_mode=PROMPT_MODE,
        ),
    }


MISSIONS: list[dict[str, Any]] = _load_missions()


def _build_initiate_payload(
    mission: dict[str, Any],
    run_id: str,
    agents: dict[str, "LocalAgent"] | None = None,
) -> dict[str, Any]:
    """Construct an SSTPNegotiateMessage initiate payload for *mission*.

    ``protocol``, ``version``, ``schema_id``, and ``schema_version`` are
    sourced from the canonical defaults in the protocol package.

    Args:
        mission: Mission dict (from missions YAML).
        run_id: Unique run timestamp string.
        agents: Dict of participant_id → LocalAgent for this mission.  When
                provided, the ``agents`` list in the payload is built from
                the actual participating agents rather than the hardcoded
                defaults.  Falls back to agent-a/b/c if omitted.
    """
    mission_slug = _slug(mission["name"])
    agents_list = (
        [{"id": pid, "name": a.name} for pid, a in agents.items()]
        if agents
        else [
            {"id": "agent-a", "name": "Agent A"},
            {"id": "agent-b", "name": "Agent B"},
            {"id": "agent-c", "name": "Agent C"},
        ]
    )
    return SSTPNegotiateMessage(
        kind="negotiate",
        message_id=f"init-{run_id}-{mission_slug}",
        dt_created=datetime.now(timezone.utc).isoformat(),
        origin=Origin(actor_id="test-runner", tenant_id="demo"),
        semantic_context=NegotiateSemanticContext(
            session_id=f"sess-{run_id}-{mission_slug}",
        ),
        payload_hash="0" * 64,
        policy_labels=PolicyLabels(
            sensitivity="internal",
            propagation="restricted",
            retention_policy="default",
        ),
        provenance=Provenance(sources=[], transforms=[]),
        payload={
            "content_text": mission["content_text"],
            "agents": agents_list,
            "n_steps": mission["n_steps"] or None,
        },
    ).model_dump(mode="json")


async def run(
    neg_server: str,
    missions_file: Path | None = None,
    agent_configs_file: Path | None = None,
    mission_filter: str | None = None,
) -> None:
    # ── load missions ─────────────────────────────────────────────────────
    missions = _load_missions(missions_file) if missions_file else MISSIONS
    if mission_filter:
        needle = mission_filter.lower()
        missions = [m for m in missions if needle in m["name"].lower()]
        if not missions:
            print(f"ERROR: --filter '{mission_filter}' matched no missions. Available:")
            for m in _load_missions(missions_file) if missions_file else MISSIONS:
                print(f"  • {m['name']}")
            return

    # ── load agent persona configs ────────────────────────────────────────
    agent_configs = _load_agent_configs(agent_configs_file)
    if agent_configs["default_agents"] or agent_configs["mission_agents"]:
        _cfg_path = agent_configs_file or _AGENT_CONFIGS_FILE
        print(f"Agent configs : {_cfg_path.resolve()}")
        print(
            f"  {len(agent_configs['default_agents'])} default agents, "
            f"{len(agent_configs['mission_agents'])} mission-specific sets"
        )
    else:
        print("Agent configs : none found — using built-in fallback")

    # ── unique id for this run ────────────────────────────────────────────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_timestamp  # used in session_id / message_id
    run_trace_dir = Path("neg_trace") / run_timestamp
    run_trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run trace root: {run_trace_dir.resolve()}")
    print(f"Missions file : {(missions_file or _MISSIONS_FILE).resolve()}")
    print(f"Missions to negotiate: {len(missions)}")
    for m in missions:
        print(f"  • {m['name']}")
    print()

    # ── build initial agents for the first mission (server must start with
    #    some agents registered; the registry is swapped per-mission below) ──
    initial_agents = _build_agents_for_mission(missions[0], agent_configs)
    agents: dict[str, LocalAgent] = initial_agents

    # ── mutable trace pointer — updated before each mission ───────────────
    trace_state: dict[str, Any] = {
        "trace_dir": run_trace_dir,
        "dialogue_log": [],  # list[str] — one line per event
        "dialogue_last_round": -1,  # tracks round header printing
        "dialogue_context_logged": False,  # True after issues/options injected
    }

    print(f"Starting shared agent server on :{AGENT_PORT}…")
    start_agent_server(agents, AGENT_PORT, trace_state)
    wait_for_server(AGENT_PORT)
    print("Agent server is up.\n")

    # ── verify negotiation server is reachable ─────────────────────────────
    try:
        httpx.get(f"{neg_server}/openapi.json", timeout=3.0).raise_for_status()
    except Exception as exc:
        print(f"ERROR: negotiation server at {neg_server} is not reachable: {exc}")
        print(
            "Start it in another terminal (same Poetry env as this repo), e.g.:\n"
            "  cd semantic_negotiation && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089"
        )
        sys.exit(1)

    # ── register litellm usage callback (once per process) ──────────────
    if _llm_usage_callback not in (_litellm.success_callback or []):
        _litellm.success_callback = [_llm_usage_callback]

    # ── per-run summary log (written to run root at the end) ──────────────
    run_log: list[dict[str, Any]] = []

    # ── iterate over missions ──────────────────────────────────────────────
    for idx, mission in enumerate(missions, start=1):
        _mission_start = time.monotonic()
        mission_slug = _slug(mission["name"])
        mission_trace_dir = run_trace_dir / mission_slug
        mission_trace_dir.mkdir(parents=True, exist_ok=True)

        # Point the running agent server at this mission's trace folder.
        trace_state["trace_dir"] = mission_trace_dir

        # Reset dialogue log and round tracker for this mission.
        _content = mission.get("content_text", "").strip().replace("\n", " ")
        trace_state["dialogue_log"] = [
            "═" * 62,
            f"  NEGOTIATION DIALOGUE: {mission['name']}",
            f"  Session  : sess-{run_id}-{_slug(mission['name'])}",
            f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "═" * 62,
            "",
            "  MISSION",
            f"  {_content}",
        ]
        trace_state["dialogue_last_round"] = -1
        trace_state["dialogue_context_logged"] = False

        # Build (or swap to) the correct agent set for this mission.
        # Each mission may use different domain-specific personas.
        agents = _build_agents_for_mission(mission, agent_configs)
        # Update the running agent server's registry in-place so the same
        # server thread serves all missions without a restart.
        make_agent_app(agents, trace_state)  # rebuilds FastAPI routes
        # Simpler: mutate the registry dict used by the already-started app.
        # The server holds a reference to the same dict, so clearing + updating
        # it live-swaps the agents without restarting the uvicorn thread.
        # (start_agent_server passes `agents` by reference into make_agent_app)
        # We re-assign `agents` here; the live server keeps its original dict ref.
        # To actually swap, we need to update the original dict in-place:
        _live_agents_ref = trace_state.get("_agents_registry")
        if _live_agents_ref is not None:
            _live_agents_ref.clear()
            _live_agents_ref.update(agents)
            agents = _live_agents_ref
        # (trace_state["_agents_registry"] is populated by start_agent_server below
        #  on the first mission; subsequent missions reuse the live ref)

        _reset_mission_llm_usage()

        print(f"{'=' * 62}")
        print(f"  Mission {idx}/{len(missions)}: {mission['name']}")
        print(f"  Trace  : {mission_trace_dir.resolve()}")
        print(f"{'=' * 62}\n")

        initiate_payload = _build_initiate_payload(mission, run_id, agents)

        session_id: str = (initiate_payload.get("semantic_context", {}) or {}).get(
            "session_id", "unknown"
        )

        _save_json(mission_trace_dir / "00_initiate_request.json", initiate_payload)

        print(f"POST {neg_server}/api/negotiate/initiate …")
        resp = httpx.post(
            f"{neg_server}/api/negotiate/initiate",
            json=initiate_payload,
            timeout=120.0,  # only Components 1+2 run here
        )
        resp.raise_for_status()
        init_envelope = resp.json()
        _save_json(mission_trace_dir / "01_initiate_response.json", init_envelope)

        init_payload = init_envelope.get("payload", {})
        if not isinstance(init_payload, dict) or init_payload.get("status") not in (
            "initiated",
            "ongoing",
            "agreed",
            "broken",
            "timeout",
        ):
            print(
                "ERROR: unexpected initiate response:",
                json.dumps(init_envelope, indent=2),
            )
            result = init_envelope
        else:
            # ── Turn-by-turn loop ────────────────────────────────────────
            messages: list[dict] = init_payload.get("messages", [])
            dispatch_count = 0
            current_round = 1
            current_n_steps = 0
            result = {}

            while messages:
                dispatch_count += 1
                # Read the SAO round number and budget from the outgoing message payload.
                _payload = messages[0].get("payload") or {}
                current_round = _payload.get("round", current_round)
                current_n_steps = _payload.get("n_steps", current_n_steps)
                _round_label = (
                    f"{current_round}/{current_n_steps}"
                    if current_n_steps
                    else str(current_round)
                )
                print(
                    f"  round {_round_label}: dispatching {len(messages)} messages to agents …"
                )

                # Forward each message to the agent server.  A broadcast message
                # (participant_id=null) is expanded server-side to all N agents and
                # returns N replies; a targeted message returns 1.  Flatten all
                # per-message reply batches into a single list before sending to
                # /api/negotiate/decide.
                reply_batches = await asyncio.gather(
                    *[asyncio.to_thread(_forward_to_agent, msg) for msg in messages]
                )
                agent_replies = [r for batch in reply_batches for r in batch]

                # POST decisions to server
                decide_payload = _build_decide_payload(
                    initiate_payload, session_id, agent_replies
                )
                decide_resp = httpx.post(
                    f"{neg_server}/api/negotiate/decide",
                    json=decide_payload,
                    timeout=60.0,
                )
                decide_resp.raise_for_status()
                decide_data = decide_resp.json()

                status = decide_data.get("status", "unknown")
                print(f"  → status={status}")

                if status == "ongoing":
                    messages = decide_data.get("messages", [])
                else:
                    # Done — capture agreed round from the response.
                    current_round = decide_data.get("round", current_round)
                    result = decide_data.get("final_result", decide_data)
                    messages = []

        print(f"HTTP done  dispatches={dispatch_count}  final_round={current_round}")

        result_clean = copy.deepcopy(result)
        _commit_trace = result_clean.get("payload", {}).get("trace")
        if isinstance(_commit_trace, dict):
            _commit_trace.pop("sstp_message_trace", None)

        # Save commit as a round-numbered file alongside the other round dirs.
        # total_rounds comes from decide_data["round"] captured as current_round
        # at terminal status; fall back to walking the commit envelope.
        _total_rounds_num = (
            current_round
            or result_clean.get("total_rounds")
            or result_clean.get("payload", {}).get("total_rounds")
        ) or 0
        _save_json(
            mission_trace_dir
            / f"round_{_total_rounds_num + 1:04d}"
            / "commit__final_result.json",
            result_clean,
        )

        print(json.dumps(result_clean, indent=2))

        # ── write dialogue log ────────────────────────────────────────────
        _elapsed = round(time.monotonic() - _mission_start, 1)
        _payload = result_clean.get("payload") or {}
        _trace = _payload.get("trace") or {}
        _status = _payload.get("status", "unknown")
        _timedout: bool = _trace.get("timedout", False)
        _broken: bool = _trace.get("broken", False)
        _agreement = _trace.get("final_agreement") or {}
        # final_agreement may be a list [{issue_id, chosen_option}] or a dict
        if isinstance(_agreement, list):
            _agreement = {
                item["issue_id"]: item["chosen_option"]
                for item in _agreement
                if "issue_id" in item and "chosen_option" in item
            }
        _total_rounds = _payload.get("total_rounds", "?")
        _n_steps = mission.get("n_steps", "?")

        # Determine verdict
        if _agreement:
            _verdict = "CONSENSUS REACHED ✓"
        elif _timedout:
            _verdict = (
                f"TIMED OUT — no agreement after {_total_rounds} / {_n_steps} rounds"
            )
        elif _broken:
            _verdict = "BROKEN — negotiation ended without agreement"
        else:
            _verdict = f"ENDED — status: {_status}"

        trace_state["dialogue_log"].append("")
        trace_state["dialogue_log"].append("═" * 62)
        trace_state["dialogue_log"].append(f"  VERDICT  : {_verdict}")
        if _agreement:
            deal_str = "  |  ".join(f"{k}: '{v}'" for k, v in _agreement.items())
            trace_state["dialogue_log"].append(f"  DEAL     : {deal_str}")
        trace_state["dialogue_log"].append(f"  Rounds   : {_total_rounds} / {_n_steps}")
        trace_state["dialogue_log"].append(f"  Duration : {_elapsed}s")
        trace_state["dialogue_log"].append("═" * 62)
        dialogue_path = mission_trace_dir / "dialogue.log"
        dialogue_path.write_text(
            "\n".join(trace_state["dialogue_log"]) + "\n", encoding="utf-8"
        )
        print(f"Dialogue log   : {dialogue_path.resolve()}")
        run_log.append(
            {
                # ── mission metadata ──────────────────────────────────────────
                "mission": mission["name"],
                "duration_s": _elapsed,
                "mode": "sync",
                "trace_dir": str(mission_trace_dir.resolve()),
                # ── SSTP envelope (top-level SSTPCommitMessage fields) ────────
                "kind": result.get("kind"),
                "protocol": result.get("protocol"),
                "version": result.get("version"),
                "message_id": result.get("message_id"),
                "dt_created": result.get("dt_created"),
                "origin": result.get("origin"),
                "semantic_context": result.get("semantic_context"),
                "payload_hash": result.get("payload_hash"),
                "policy_labels": result.get("policy_labels"),
                "provenance": result.get("provenance"),
                # ── SSTPCommitMessage-specific fields ─────────────────────────
                "state_object_id": result.get("state_object_id"),
                "parent_ids": result.get("parent_ids"),
                "logical_clock": result.get("logical_clock"),
                "confidence_score": result.get("confidence_score"),
                "risk_score": result.get("risk_score"),
                "ttl_seconds": result.get("ttl_seconds"),
                "merge_strategy": result.get("merge_strategy"),
                "payload_refs": result.get("payload_refs"),
                # ── negotiation outcome (from payload) ────────────────────────
                "session_id": session_id,
                "status": _payload.get("status"),
                "total_rounds": _payload.get("total_rounds"),
                "timedout": _trace.get("timedout"),
                "broken": _trace.get("broken"),
                "final_agreement": _trace.get("final_agreement"),
                # ── LLM call accounting ────────────────────────────────────────────────
                # llm_calls_initiate: 2 fixed (IntentDiscovery + OptionsGeneration
                #   run inside the neg server process — token counts not available here)
                # llm_usage_decide: real token/cost data captured via litellm callback
                "n_agents": len(agents),
                "llm_calls_initiate": 2,
                "llm_calls_total": 2 + _mission_llm_usage["calls"],
                "llm_usage_decide": _snapshot_mission_llm_usage(),
            }
        )

        print(f"\nMission {idx} trace saved to: {mission_trace_dir.resolve()}\n")

    # ── write run-level summary ────────────────────────────────────────────
    run_log_path = run_trace_dir / "run_log.json"
    _save_json(run_log_path, {"run_id": run_timestamp, "missions": run_log})
    print(f"Run log written : {run_log_path.resolve()}")
    print(
        f"All {len(missions)} missions complete.  Run trace root: {run_trace_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-agent callback negotiation test")
    parser.add_argument(
        "--neg-server",
        default=NEG_SERVER,
        help=f"Base URL of the negotiation server (default: {NEG_SERVER})",
    )
    parser.add_argument(
        "--missions-file",
        default=None,
        metavar="PATH",
        help=f"Path to a YAML missions file (default: missions.yaml next to this script)",
    )
    parser.add_argument(
        "--agent-configs",
        default=None,
        metavar="PATH",
        help=(
            f"Path to agent_configs.yaml with per-mission LLM agent personas "
            f"(default: agent_configs.yaml next to this script)"
        ),
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="TEXT",
        help=(
            "Case-insensitive substring filter on mission names. "
            "E.g. --filter hard  → only 'Hard 0X' missions; "
            "--filter example  → only 'Example 0X' missions; "
            "--filter 'Hard 01'  → single mission."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.neg_server,
            Path(args.missions_file) if args.missions_file else None,
            Path(args.agent_configs) if args.agent_configs else None,
            mission_filter=args.filter,
        )
    )
