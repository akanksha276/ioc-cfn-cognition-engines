# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""test_via_cfn_service.py

Run negotiation missions against the **CFN service** (ioc-cognition-fabric-node-svc)
rather than the standalone semantic-negotiation app.

The CFN exposes the negotiation API at::

    POST /api/workspaces/{wid}/multi-agentic-systems/{mid}/semantic-negotiation/start
    POST /api/workspaces/{wid}/multi-agentic-systems/{mid}/semantic-negotiation/decide

This script follows the exact call flow documented in
``ioc-cognition-fabric-node-svc/docs/negotiation-example.md``:

1. POST ``/start`` with ``{session_id, content_text, agents, n_steps}``
   → returns ``{status, session_id, round, issues, options_per_issue, messages}``
2. For each message, forward it to the local agent callback server (port 8092)
   to get the agent's decision.
3. Map agent replies to ``{participant_id, action, offer?}`` and POST to ``/decide``
   with ``{session_id, agent_replies}``.
4. Repeat until ``status`` is terminal (``agreed``, ``failed``, ``timedout``).

Prerequisites
-------------
- The CFN service must be running (default ``http://localhost:9002``).
  Start it via ``./localrun.sh`` in ``ioc-cognition-fabric-node-svc/``.
- A valid ``workspace_id`` and ``mas_id`` must exist in the management plane.

Usage
-----
::

    # Terminal 1: start the CFN service
    cd ioc-cognition-fabric-node-svc && ./localrun.sh

    # Terminal 2: run missions against it
    cd ioc-cfn-cognitive-agents/semantic_negotiation
    poetry run python evaluation/framework/test_via_cfn_service.py \\
        --cfn-url http://localhost:9002 \\
        --workspace-id ws1 \\
        --mas-id mas1 \\
        --filter hard
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

# Import everything reusable from the existing configured-agents test script.
# Both files live in the same directory.
from test_via_semantic_neg_agents_configured import (
    MISSIONS,
    _AGENT_CONFIGS_FILE,
    _MISSIONS_FILE,
    LocalAgent,
    NegMASConcessionAgent,
    _build_agents_for_mission,
    _forward_to_agent,
    _llm_usage_callback,
    _load_agent_configs,
    _load_missions,
    _reset_mission_llm_usage,
    _save_json,
    _slug,
    _snapshot_mission_llm_usage,
    start_agent_server,
    wait_for_server,
)

try:
    import litellm as _litellm
except ImportError:
    _litellm = None  # type: ignore[assignment]

# ── defaults ──────────────────────────────────────────────────────────────

CFN_URL = "http://localhost:9002"
_DEFAULT_AGENT_PORT = 8092
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CFN_CONFIG = _THIS_DIR / "cfn_service_config.yaml"


# ── YAML config loader ────────────────────────────────────────────────────


def _load_cfn_config(config_path: Path | None) -> dict[str, Any]:
    """Load cfn_service_config.yaml and return its contents as a dict.

    Resolves *missions_file* and *agent_configs_file* paths relative to the
    config file's directory so callers always get absolute paths back.
    """
    path = config_path or _DEFAULT_CFN_CONFIG
    if not path.exists():
        return {}
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    for key in ("missions_file", "agent_configs_file"):
        if raw.get(key):
            resolved = (base / raw[key]).resolve()
            raw[key] = str(resolved)
    return raw


def _agents_from_config(cfg_agents: list[dict[str, Any]]) -> dict[str, LocalAgent]:
    """Build a ``{participant_id: LocalAgent}`` registry from the YAML agents list."""
    agents: dict[str, LocalAgent] = {}
    for entry in cfg_agents:
        agent_id = entry["id"]
        name = entry.get("name", agent_id)
        prefer_low = bool(entry.get("prefer_low", True))
        exponent = float(entry.get("exponent", 2.0))
        min_reservation = float(entry.get("min_reservation", 0.0))
        agents[agent_id] = NegMASConcessionAgent(
            name=name,
            prefer_low=prefer_low,
            exponent=exponent,
            min_reservation=min_reservation,
        )
    return agents


# ── CFN-specific helpers ──────────────────────────────────────────────────


def _build_start_body(
    mission: dict[str, Any],
    run_id: str,
    agents: dict[str, LocalAgent],
) -> dict[str, Any]:
    """Build the flat ``InitiateNegotiationRequest`` body for CFN ``/start``.

    Shape::

        {
          "session_id": "sess-<run_id>-<slug>",
          "content_text": "...",
          "agents": [{"id": "agent-a", "name": "Agent A"}, ...],
          "n_steps": 20
        }
    """
    mission_slug = _slug(mission["name"])
    return {
        "session_id": f"sess-{run_id}-{mission_slug}",
        "content_text": mission["content_text"],
        "agents": [{"id": pid, "name": a.name} for pid, a in agents.items()],
        "n_steps": mission.get("n_steps") or 20,
    }


def _extract_agent_reply(sstp_reply: dict[str, Any]) -> dict[str, Any]:
    """Map an SSTP reply from the agent server to the CFN ``AgentReply`` shape.

    The agent server returns full SSTPNegotiateMessage dicts.  The CFN
    ``/decide`` endpoint expects::

        {
          "participant_id": "...",
          "action": "accept"|"reject"|"counter_offer",
          "offer": {...},          # counter_offer only
          "reason": "..."           # recommended on every action
        }

    We extract ``participant_id``, ``action``, ``offer``, and ``reason`` from
    the reply payload.
    """
    payload = sstp_reply.get("payload", {})
    reply: dict[str, Any] = {
        "participant_id": payload.get("participant_id", "unknown"),
        "action": payload.get("action", "reject"),
    }
    if payload.get("offer"):
        reply["offer"] = payload["offer"]
    if payload.get("reason"):
        reply["reason"] = payload["reason"]
    return reply


def _cfn_health_check(cfn_url: str) -> None:
    """Verify the CFN service is reachable and healthy, or exit with an error."""
    url = f"{cfn_url}/api/internal/diagnostics/health"
    try:
        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to CFN service at {cfn_url} — is it running?")
        print(
            "Start it in another terminal, e.g.:\n"
            "  cd ioc-cognition-fabric-node-svc && ./localrun.sh"
        )
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"ERROR: CFN health check timed out ({url})")
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(
            f"ERROR: CFN health endpoint returned HTTP {exc.response.status_code} ({url})"
        )
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: CFN health check failed: {exc}")
        sys.exit(1)

    body = r.json()
    status = body.get("status", "").lower()
    if status not in ("healthy", "up"):
        print(
            f"ERROR: CFN service reports status '{body.get('status')}' "
            f"(expected 'healthy' or 'up'). Response: {body}"
        )
        sys.exit(1)


def _build_agents_cfn(
    mission: dict[str, Any],
    agent_configs: dict[str, Any],
    yaml_agents: list[dict[str, Any]] | None,
) -> dict[str, LocalAgent]:
    """Build the agent registry for *mission*, respecting the priority chain:

    1. Mission-specific agents from ``agent_configs["mission_agents"]``
    2. Default agents from YAML config (``yaml_agents``) when provided
    3. Default agents from ``agent_configs["default_agents"]``
    4. Built-in fallback (from ``_build_agents_for_mission``)
    """
    mission_slug = _slug(mission["name"])
    mission_specific = (agent_configs.get("mission_agents") or {}).get(mission_slug)
    if mission_specific:
        # Delegate fully — will use mission-specific LLM personas
        return _build_agents_for_mission(mission, agent_configs)

    # Use YAML-defined rule-based agents as default when provided
    if yaml_agents:
        agents = _agents_from_config(yaml_agents)
        print(
            f"  Agent config  : cfn_service_config.yaml "
            f"({len(agents)} agents: {list(agents.keys())})"
        )
        return agents

    # Fall through to agent_configs defaults or built-in fallback
    return _build_agents_for_mission(mission, agent_configs)


async def run(
    cfn_url: str,
    workspace_id: str,
    mas_id: str,
    agent_port: int = _DEFAULT_AGENT_PORT,
    missions_file: Path | None = None,
    agent_configs_file: Path | None = None,
    yaml_agents: list[dict[str, Any]] | None = None,
    mission_filter: str | None = None,
) -> None:
    # ── load missions ─────────────────────────────────────────────────────
    missions = _load_missions(missions_file) if missions_file else MISSIONS
    if mission_filter:
        needle = mission_filter.lower()
        missions = [m for m in missions if needle in m["name"].lower()]
        if not missions:
            print(f"ERROR: --filter '{mission_filter}' matched no missions. Available:")
            all_m = _load_missions(missions_file) if missions_file else MISSIONS
            for m in all_m:
                print(f"  • {m['name']}")
            return

    # ── load agent persona configs (per-mission overrides) ────────────────
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
    run_id = run_timestamp
    run_trace_dir = Path("neg_trace") / run_timestamp
    run_trace_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run trace root: {run_trace_dir.resolve()}")
    print(f"CFN URL       : {cfn_url}")
    print(f"Workspace     : {workspace_id}")
    print(f"MAS           : {mas_id}")
    print(f"Agent port    : {agent_port}")
    print(f"Missions file : {(missions_file or _MISSIONS_FILE).resolve()}")
    print(f"Missions to negotiate: {len(missions)}")
    for m in missions:
        print(f"  • {m['name']}")
    print()

    # ── CFN API base path ─────────────────────────────────────────────────
    api_base = (
        f"{cfn_url}/api/workspaces/{workspace_id}"
        f"/multi-agentic-systems/{mas_id}/semantic-negotiation"
    )

    # ── build initial agents for the first mission ────────────────────────
    initial_agents = _build_agents_cfn(missions[0], agent_configs, yaml_agents)
    agents: dict[str, LocalAgent] = initial_agents

    # ── mutable trace pointer — updated before each mission ───────────────
    trace_state: dict[str, Any] = {
        "trace_dir": run_trace_dir,
        "dialogue_log": [],
        "dialogue_last_round": -1,
        "dialogue_context_logged": False,
    }

    print(f"Starting shared agent server on :{agent_port}…")
    start_agent_server(agents, agent_port, trace_state)
    wait_for_server(agent_port)
    print("Agent server is up.\n")

    # ── verify CFN service is reachable ────────────────────────────────────
    _cfn_health_check(cfn_url)
    print(f"CFN service is healthy.\n")

    # ── register litellm usage callback (once per process) ──────────────
    if _litellm and _llm_usage_callback not in (_litellm.success_callback or []):
        _litellm.success_callback = list(_litellm.success_callback or []) + [
            _llm_usage_callback
        ]

    run_log: list[dict[str, Any]] = []

    for idx, mission in enumerate(missions, 1):
        _mission_start = time.monotonic()
        mission_slug = _slug(mission["name"])
        mission_trace_dir = run_trace_dir / mission_slug
        trace_state["trace_dir"] = mission_trace_dir

        _content = mission["content_text"]
        if len(_content) > 120:
            _content = _content[:117] + "…"
        trace_state["dialogue_log"] = [
            "═" * 62,
            f"  NEGOTIATION DIALOGUE: {mission['name']}",
            f"  Session  : sess-{run_id}-{mission_slug}",
            f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  CFN      : {cfn_url}",
            "═" * 62,
            "",
            "  MISSION",
            f"  {_content}",
        ]
        trace_state["dialogue_last_round"] = -1
        trace_state["dialogue_context_logged"] = False

        # Build (or swap to) the correct agent set for this mission.
        agents = _build_agents_cfn(mission, agent_configs, yaml_agents)
        _live_agents_ref = trace_state.get("_agents_registry")
        if _live_agents_ref is not None:
            _live_agents_ref.clear()
            _live_agents_ref.update(agents)
            agents = _live_agents_ref

        _reset_mission_llm_usage()

        print(f"{'=' * 62}")
        print(f"  Mission {idx}/{len(missions)}: {mission['name']}")
        print(f"  Trace  : {mission_trace_dir.resolve()}")
        print(f"{'=' * 62}\n")

        # ── Step 1: POST /start ───────────────────────────────────────────
        start_body = _build_start_body(mission, run_id, agents)
        session_id = start_body["session_id"]

        _save_json(mission_trace_dir / "00_start_request.json", start_body)

        print(f"POST {api_base}/start …")
        for _attempt in range(3):
            try:
                resp = httpx.post(
                    f"{api_base}/start",
                    json=start_body,
                    timeout=600.0,
                )
                break
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if _attempt < 2:
                    print(f"  /start error ({type(exc).__name__}), retrying… (attempt {_attempt + 2}/3)")
                    continue
                raise
        resp.raise_for_status()
        start_data = resp.json()

        status = start_data.get("status", "unknown")
        print(f"  → status={status}")

        # Capture issues/options from the start response for the dialogue log
        issues = start_data.get("issues", [])
        options_per_issue = start_data.get("options_per_issue", {})

        _save_json(mission_trace_dir / "00_start_response.json", start_data)

        if issues and options_per_issue:
            trace_state["dialogue_log"].append("")
            trace_state["dialogue_log"].append("  ISSUES IDENTIFIED")
            for iss in issues:
                trace_state["dialogue_log"].append(f"  • {iss}")
            trace_state["dialogue_log"].append("")
            trace_state["dialogue_log"].append("  OPTIONS PER ISSUE")
            for iss, opts in options_per_issue.items():
                trace_state["dialogue_log"].append(f"  • {iss}: {', '.join(opts)}")
            trace_state["dialogue_context_logged"] = True

        messages: list[dict] = start_data.get("messages", [])
        current_round = start_data.get("round", 1)
        n_steps = start_data.get("n_steps", 0)
        dispatch_count = 0
        result: dict[str, Any] = {}

        if status in ("agreed", "failed", "timedout", "broken"):
            result = start_data
            messages = []

        # ── Step 2+: decide loop ──────────────────────────────────────────
        while messages:
            dispatch_count += 1
            _msg_payload = messages[0].get("payload") or {}
            current_round = _msg_payload.get("round", current_round)
            _round_label = (
                f"{current_round}/{n_steps}" if n_steps else str(current_round)
            )
            print(
                f"  round {_round_label}: dispatching {len(messages)} "
                f"messages to agents …"
            )

            # Forward each message to the agent callback server.
            reply_batches = await asyncio.gather(
                *[asyncio.to_thread(_forward_to_agent, msg) for msg in messages]
            )
            all_sstp_replies = [r for batch in reply_batches for r in batch]

            # Map SSTP replies → CFN AgentReply format
            agent_replies = [_extract_agent_reply(r) for r in all_sstp_replies]

            # Log agent decisions to dialogue
            for ar in agent_replies:
                action = ar["action"]
                agent_name = ar["participant_id"]
                if action == "counter_offer":
                    if current_round != trace_state["dialogue_last_round"]:
                        trace_state["dialogue_log"].append("")
                        trace_state["dialogue_log"].append(
                            f"[Round {current_round}]  Proposer: {agent_name}"
                        )
                        trace_state["dialogue_last_round"] = current_round
                    offer = ar.get("offer", {})
                    offer_str = "  |  ".join(f"{k}: '{v}'" for k, v in offer.items())
                    trace_state["dialogue_log"].append(f"  OFFER    : {offer_str}")
                elif action == "accept":
                    if current_round != trace_state["dialogue_last_round"]:
                        trace_state["dialogue_log"].append("")
                        trace_state["dialogue_log"].append(f"[Round {current_round}]")
                        trace_state["dialogue_last_round"] = current_round
                    trace_state["dialogue_log"].append(f"  [{agent_name:<8}]  ACCEPT ✓")
                else:  # reject
                    if current_round != trace_state["dialogue_last_round"]:
                        trace_state["dialogue_log"].append("")
                        trace_state["dialogue_log"].append(f"[Round {current_round}]")
                        trace_state["dialogue_last_round"] = current_round
                    trace_state["dialogue_log"].append(f"  [{agent_name:<8}]  REJECT")

            # Build decide request body
            decide_body: dict[str, Any] = {
                "session_id": session_id,
                "agent_replies": agent_replies,
            }

            _save_json(
                mission_trace_dir
                / f"round_{current_round:04d}"
                / "decide_request.json",
                decide_body,
            )

            # POST /decide
            print(f"  POST {api_base}/decide  ({len(agent_replies)} replies)")
            for _attempt in range(3):
                try:
                    decide_resp = httpx.post(
                        f"{api_base}/decide",
                        json=decide_body,
                        timeout=600.0,
                    )
                    break
                except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                    if _attempt < 2:
                        print(f"  /decide error ({type(exc).__name__}), retrying… (attempt {_attempt + 2}/3)")
                        continue
                    raise
            decide_resp.raise_for_status()
            decide_data = decide_resp.json()

            _save_json(
                mission_trace_dir
                / f"round_{current_round:04d}"
                / "decide_response.json",
                decide_data,
            )

            status = decide_data.get("status", "unknown")
            print(f"  → status={status}")

            if status == "ongoing":
                messages = decide_data.get("messages", [])
                current_round = decide_data.get("round", current_round)
                n_steps = decide_data.get("n_steps", n_steps)
            else:
                # Terminal state
                current_round = decide_data.get("round", current_round)
                result = decide_data
                messages = []

        print(f"  Done  dispatches={dispatch_count}  final_round={current_round}")

        # ── Save final result ─────────────────────────────────────────────
        result_clean = copy.deepcopy(result)

        # The CFN response may have the result nested or flat
        _final_result = result_clean.get("final_result", result_clean)
        _payload = (
            _final_result.get("payload", _final_result)
            if isinstance(_final_result, dict)
            else {}
        )
        _trace = _payload.get("trace", {}) if isinstance(_payload, dict) else {}

        # Remove verbose SSTP message trace from saved output
        if isinstance(_trace, dict):
            _trace.pop("sstp_message_trace", None)

        _total_rounds_num = (
            current_round
            or result_clean.get("total_rounds")
            or _payload.get("total_rounds")
            or 0
        )
        _save_json(
            mission_trace_dir
            / f"round_{_total_rounds_num + 1:04d}"
            / "commit__final_result.json",
            result_clean,
        )

        print(json.dumps(result_clean, indent=2))

        # ── write dialogue log ────────────────────────────────────────────
        _elapsed = round(time.monotonic() - _mission_start, 1)
        _status = _payload.get("status") or result_clean.get("status") or "unknown"
        _timedout = _trace.get("timedout", False) if isinstance(_trace, dict) else False
        _broken = _trace.get("broken", False) if isinstance(_trace, dict) else False

        # final_agreement may be a list [{issue_id, chosen_option}] or a dict
        _agreement = (
            _trace.get("final_agreement")
            or result_clean.get("result", {}).get("agreement", {})
            or {}
        )
        if isinstance(_agreement, list):
            _agreement = {
                item["issue_id"]: item["chosen_option"]
                for item in _agreement
                if "issue_id" in item and "chosen_option" in item
            }

        _total_rounds = (
            _payload.get("total_rounds")
            or result_clean.get("total_rounds")
            or current_round
            or "?"
        )
        _n_steps = mission.get("n_steps", "?")

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
                "mission": mission["name"],
                "duration_s": _elapsed,
                "mode": "cfn-service",
                "cfn_url": cfn_url,
                "workspace_id": workspace_id,
                "mas_id": mas_id,
                "trace_dir": str(mission_trace_dir.resolve()),
                "session_id": session_id,
                "status": _status,
                "total_rounds": _total_rounds,
                "timedout": _timedout,
                "broken": _broken,
                "final_agreement": _agreement or None,
                "n_agents": len(agents),
                "llm_calls_initiate": 2,
                "llm_calls_total": 2 + _snapshot_mission_llm_usage()["calls"],
                "llm_usage_decide": _snapshot_mission_llm_usage(),
            }
        )

        print(f"\nMission {idx} trace saved to: {mission_trace_dir.resolve()}\n")

    # ── write run-level summary ────────────────────────────────────────────
    run_log_path = run_trace_dir / "run_log.json"
    _save_json(run_log_path, {"run_id": run_timestamp, "missions": run_log})
    print(f"Run log written : {run_log_path.resolve()}")
    print(
        f"All {len(missions)} missions complete.  "
        f"Run trace root: {run_trace_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run negotiation missions against the CFN service"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            f"Path to a YAML config file (default: cfn_service_config.yaml next to "
            f"this script). CLI flags override values from the config file."
        ),
    )
    parser.add_argument(
        "--cfn-url",
        default=None,
        help=f"Base URL of the CFN service (default from config: {CFN_URL})",
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Workspace ID for CFN API routes (default from config)",
    )
    parser.add_argument(
        "--mas-id",
        default=None,
        help="Multi-Agentic System ID for CFN API routes (default from config)",
    )
    parser.add_argument(
        "--agent-port",
        default=None,
        type=int,
        help=f"Port for the local agent callback server (default from config: {_DEFAULT_AGENT_PORT})",
    )
    parser.add_argument(
        "--missions-file",
        default=None,
        metavar="PATH",
        help="Path to a YAML missions file (default from config or missions.yaml)",
    )
    parser.add_argument(
        "--agent-configs",
        default=None,
        metavar="PATH",
        help="Path to agent_configs.yaml with per-mission LLM agent personas",
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="TEXT",
        help=(
            "Case-insensitive substring filter on mission names. "
            "E.g. --filter hard → only 'Hard 0X' missions."
        ),
    )
    args = parser.parse_args()

    # Resolution order: YAML config → ENV var → CLI flag (CLI always wins).
    cfg = _load_cfn_config(Path(args.config) if args.config else None)

    cfn_url = (
        args.cfn_url
        or os.getenv("CFN_URL")
        or cfg.get("cfn_url")
        or CFN_URL
    )
    workspace_id = (
        args.workspace_id
        or os.getenv("CFN_WORKSPACE_ID")
        or cfg.get("workspace_id")
    )
    mas_id = (
        args.mas_id
        or os.getenv("CFN_MAS_ID")
        or cfg.get("mas_id")
    )
    agent_port = int(
        args.agent_port
        or os.getenv("CFN_AGENT_PORT")
        or cfg.get("agent_port")
        or _DEFAULT_AGENT_PORT
    )
    # CFN_SESSION_ID overrides the auto-generated run timestamp used as the
    # session-ID prefix (e.g. "replay-001" → sess-replay-001-<slug>).
    session_id_prefix = os.getenv("CFN_SESSION_ID") or None
    yaml_agents: list[dict[str, Any]] | None = cfg.get("agents") or None

    missions_file: Path | None = None
    if args.missions_file:
        missions_file = Path(args.missions_file)
    elif cfg.get("missions_file"):
        missions_file = Path(cfg["missions_file"])

    agent_configs_file: Path | None = None
    if args.agent_configs:
        agent_configs_file = Path(args.agent_configs)
    elif cfg.get("agent_configs_file"):
        agent_configs_file = Path(cfg["agent_configs_file"])

    if not workspace_id:
        parser.error(
            "--workspace-id is required (set via CLI or 'workspace_id' in config YAML)"
        )
    if not mas_id:
        parser.error(
            "--mas-id is required (set via CLI or 'mas_id' in config YAML)"
        )

    asyncio.run(
        run(
            cfn_url=cfn_url,
            workspace_id=workspace_id,
            mas_id=mas_id,
            agent_port=agent_port,
            missions_file=missions_file,
            agent_configs_file=agent_configs_file,
            yaml_agents=yaml_agents,
            mission_filter=args.filter,
        )
    )
