# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""test_direct_agents.py - Direct LLM-to-LLM negotiation loop.

No semantic-negotiation server, no SSTP, no SAO protocol, no NegMAS.
Three agents exchange plain-language messages in a round-robin loop until
they reach an agreement or hit the round limit.

Each turn:
  1. The active agent receives the full conversation history as context.
  2. It replies with a plain-language message that either:
     - Makes / updates a proposal  -> {"status": "propose", "proposal": {...}}
     - Accepts the current proposal -> {"status": "accept"}
     - Rejects and counter-proposes -> {"status": "counter", "proposal": {...}}
  3. Once ALL non-proposing agents reply "accept" in the same round,
     the loop ends with agreement.

Trace layout:
    neg_trace/direct_<run_ts>/<mission_slug>/
        00_mission.json
        round_NNNN.json      # full round record (messages + decisions)
        dialogue.log
        final_result.json

Usage:
    python test_direct_agents.py
    python test_direct_agents.py --missions-file my_missions.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path setup -- LLM provider only
# ---------------------------------------------------------------------------
# File lives at semantic_negotiation/evaluation/framework/ — 3 levels inside the repo root.
_workspace_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

_sna_app_root = str(Path(__file__).resolve().parent.parent.parent / "app")
if _sna_app_root not in sys.path:
    sys.path.insert(0, _sna_app_root)

from config.utils import get_llm_provider  # noqa: E402

# Reuse agent-config loading from the configured-agents test script.
from test_via_semantic_neg_agents_configured import (  # noqa: E402
    _AGENT_CONFIGS_FILE,
    _load_agent_configs,
    _slug,
)

# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------
_MISSIONS_FILE = Path(__file__).resolve().parent.parent.parent.parent / "missions.yaml"


def _load_missions(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("missions", data) if isinstance(data, dict) else data


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Agent personas
# ---------------------------------------------------------------------------

# Hardcoded fallback — used when agent_configs.yaml has no entry for a mission
# and no default_agents are defined.
_FALLBACK_PERSONAS: dict[str, dict[str, Any]] = {
    "agent-a": {
        "name": "Agent A",
        "prefer_low": True,
        "persona": (
            "You are a cost-conscious buyer. You always push for the cheapest, "
            "most minimal options. You resist premium choices unless "
            "absolutely necessary. However, you want a deal — as the deadline "
            "approaches, you concede more readily on secondary issues."
        ),
    },
    "agent-b": {
        "name": "Agent B",
        "prefer_low": False,
        "persona": (
            "You are a quality-focused buyer. You prefer premium, high-end options "
            "and are willing to pay more for better service. You are firm early, but "
            "from the halfway point (t > 0.5) you actively move toward mid-range options "
            "on at least two issues. After t > 0.7 you must accept any offer that gives "
            "you mid-range or better on the majority of issues — do not hold out for perfection."
        ),
    },
    "agent-c": {
        "name": "Agent C",
        "prefer_low": True,
        "persona": (
            "You are a pragmatic, balanced negotiator. You lean toward cost-effective "
            "choices and care most about reaching a deal. You concede readily "
            "and look for middle-ground options that everyone can live with."
        ),
    },
    "agent-d": {
        "name": "Agent D",
        "prefer_low": False,
        "persona": (
            "You are a risk-averse, compliance-focused stakeholder. You refuse any "
            "option unless it explicitly meets regulatory and audit requirements. "
            "You are slow to accept and introduce new objections late in the process. "
            "You only concede when all other parties have already agreed on a "
            "proposal that fully satisfies compliance requirements."
        ),
    },
    "agent-e": {
        "name": "Agent E",
        "prefer_low": False,
        "persona": (
            "You are an aggressive, adversarial negotiator who represents a minority "
            "stakeholder with veto power. You frequently reopen issues that others "
            "thought were settled, demand side-payments, and will only accept a deal "
            "that gives you a demonstrably better outcome than the opening position. "
            "You make concessions only when the other parties offer tangible gains "
            "on your key interests."
        ),
    },
}

# Keep the old name as an alias so any external code that references AGENT_PERSONAS
# still works against the fallback table.
AGENT_PERSONAS = _FALLBACK_PERSONAS


def _build_direct_personas(
    mission: dict[str, Any],
    agent_configs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve per-mission agent personas from *agent_configs*.

    Lookup order (mirrors ``_build_agents_for_mission`` in the configured script):
    1. ``agent_configs["mission_agents"][<mission_slug>]`` — mission-specific personas
    2. ``agent_configs["default_agents"]``                 — dataset-level defaults
    3. ``_FALLBACK_PERSONAS``                              — hardcoded built-in fallback

    Returns a ``{participant_id: {name, persona, prefer_low}}`` dict compatible
    with ``_call_agent``.
    """
    mission_slug = _slug(mission["name"])
    cfg_list: list[dict[str, Any]] = (
        agent_configs.get("mission_agents", {}).get(mission_slug)
        or agent_configs.get("default_agents")
        or []
    )

    if cfg_list:
        source = (
            "mission-specific"
            if agent_configs.get("mission_agents", {}).get(mission_slug)
            else "default"
        )
        personas: dict[str, dict[str, Any]] = {}
        for cfg in cfg_list:
            pid = cfg["id"]
            personas[pid] = {
                "name": cfg.get("name", pid),
                "prefer_low": bool(cfg.get("prefer_low", True)),
                "persona": cfg.get("persona", "You are a pragmatic negotiator.").strip(),
            }
        print(
            f"  Agent config  : {source} "
            f"({len(personas)} agents: {list(personas.keys())})"
        )
        return personas

    print("  Agent config  : built-in fallback")
    return _FALLBACK_PERSONAS

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_llm = get_llm_provider()


def _preference_hint(prefer_low: bool) -> str:
    """Return a natural-language utility direction hint, mirroring the utility
    weight table injected into LLMNegotiationAgent prompts in the semantic
    negotiation path.  Keeps the information signal symmetric across both
    test harnesses."""
    if prefer_low:
        return (
            "Your preference direction: you favor lower-cost, simpler, more minimal "
            "options. Among the terms on the table, choices that are cheaper, more basic, "
            "or less resource-intensive are better for you (higher utility). Premium, "
            "high-end, or expansive choices are worse for you (lower utility)."
        )
    return (
        "Your preference direction: you favor higher-quality, premium, more expansive "
        "options. Among the terms on the table, choices that are more comprehensive, "
        "higher-end, or feature-rich are better for you (higher utility). Cheap, "
        "minimal, or cut-rate choices are worse for you (lower utility)."
    )


def _stochastic_nudge(t: float) -> str:
    """Return a random concession nudge with probability that rises with t.

    At t=0 the probability is 0.05 (rare).  At t=1 it reaches 0.50.
    When triggered, one of several qualitatively different nudges is chosen
    so the signal stays unpredictable and avoids gaming.
    """
    p = 0.05 + 0.45 * t  # linear ramp: 5% at start → 50% at deadline
    if random.random() >= p:
        return ""
    nudges = [
        "Hint: consider making a small but genuine concession on at least one issue this round.",
        "Hint: the other parties are watching for flexibility — softening one term could unlock agreement.",
        "Hint: try accepting the least important contested term and counter only on what matters most to you.",
        "Hint: a deal with minor compromises is better than no deal — look for one issue you can yield on.",
        "Hint: if the current proposal is close to acceptable, consider accepting it rather than counter-proposing.",
    ]
    return random.choice(nudges)


def _call_agent(
    agent_id: str,
    mission: str,
    history: list[dict],
    round_num: int,
    n_steps: int,
    role: str,  # "proposer" or "responder"
    current_proposal: dict | None,
    personas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Ask one agent for its next move. Returns a structured decision dict."""
    info = personas[agent_id]

    history_text = "\n".join(f"  [{m['from']}] {m['text']}" for m in history[-20:])

    if role == "proposer":
        task = (
            "Your turn to PROPOSE (or update your proposal). "
            "State your proposal clearly as a set of key-value terms.\n"
            "Reply ONLY with JSON:\n"
            '{"status": "propose", "text": "<your message>", "proposal": {"<term>": "<value>", ...}}'
        )
    else:
        if current_proposal:
            prop_str = json.dumps(current_proposal, indent=2)
            t = round_num / max(n_steps, 1)
            nudge = _stochastic_nudge(t)
            nudge_line = f"\n{nudge}" if nudge else ""
            task = (
                f"The current proposal on the table is:\n{prop_str}\n\n"
                "You must ACCEPT or COUNTER-PROPOSE.\n"
                '- To accept: {"status": "accept", "text": "<your message>"}\n'
                '- To counter: {"status": "counter", "text": "<your message>", '
                '"proposal": {"<term>": "<value>", ...}}\n'
                f"Relative time: t = {t:.2f}  (0=negotiation start, 1=deadline). "
                "The closer to 1, the more you should be willing to accept a reasonable offer."
                f"{nudge_line}"
            )
        else:
            t = round_num / max(n_steps, 1)
            task = (
                "No proposal on the table yet. Make your opening position statement.\n"
                f"Relative time: t = {t:.2f}  (0=negotiation start, 1=deadline)\n"
                'Reply with: {"status": "propose", "text": "<your message>", "proposal": {"<term>": "<value>", ...}}'
            )

    prompt = (
        f"You are {info['name']} in a multi-party negotiation.\n"
        f"{info['persona']}\n\n"
        f"Your preference profile: {_preference_hint(info['prefer_low'])}\n\n"
        f"Mission: {mission}\n\n"
        f"Conversation so far (most recent last):\n{history_text or '(none yet)'}\n\n"
        f"{task}\n\n"
        "Reply ONLY with the JSON object. No explanation, no markdown fences."
    )

    raw = _llm(prompt)

    # Extract JSON
    try:
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-z]*\n?", "", stripped).rstrip("`").strip()
        result = json.loads(stripped)
        if isinstance(result, dict) and "status" in result:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Scans for first balanced { }
    start = raw.find("{")
    if start != -1:
        depth, in_str, escape = 0, False, False
        for i, ch in enumerate(raw[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
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
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    print(f"  [{agent_id}] WARNING: could not parse LLM response, using fallback")
    return {
        "status": "counter" if role == "responder" else "propose",
        "text": raw[:200],
        "proposal": current_proposal or {},
    }


# ---------------------------------------------------------------------------
# Direct negotiation loop
# ---------------------------------------------------------------------------


def _run_direct(
    session_id: str,
    mission: dict[str, Any],
    trace_dir: Path,
    agent_ids: list[str] | None = None,
    personas: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    content = mission.get("content_text", "").strip().replace("\n", " ")
    n_steps: int = mission.get("n_steps", 30)
    _personas = personas or _FALLBACK_PERSONAS
    if agent_ids is None:
        agent_ids = list(_personas.keys())

    history: list[dict] = []
    message_trace: list[dict] = []
    current_proposal: dict | None = None
    agreement: dict | None = None
    timedout = False
    total_rounds = 0
    round_records: list[dict] = []

    dialogue_log: list[str] = [
        "=" * 62,
        f"  DIRECT NEGOTIATION: {mission['name']}",
        f"  Session : {session_id}",
        f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 62,
        "",
        "  MISSION",
        f"  {content}",
        "",
    ]

    for round_num in range(1, n_steps + 1):
        total_rounds = round_num
        proposer_id = agent_ids[(round_num - 1) % len(agent_ids)]
        round_record: dict[str, Any] = {
            "round": round_num,
            "proposer": proposer_id,
            "messages": [],
        }

        dialogue_log.append(f"[Round {round_num}]  Turn: {proposer_id}")

        # -- Proposer ----------------------------------------------------------
        decision = _call_agent(
            agent_id=proposer_id,
            mission=content,
            history=history,
            round_num=round_num,
            n_steps=n_steps,
            role="proposer",
            current_proposal=current_proposal,
            personas=_personas,
        )

        proposal_text = decision.get("text", "")
        if decision.get("proposal"):
            current_proposal = decision["proposal"]

        msg = {
            "round": round_num,
            "from": proposer_id,
            "role": "proposer",
            "status": decision.get("status", "propose"),
            "text": proposal_text,
            "proposal": current_proposal,
        }
        history.append({"from": proposer_id, "text": proposal_text, "round": round_num})
        message_trace.append(msg)
        round_record["messages"].append(msg)

        prop_str = "  |  ".join(
            f"{k}: '{v}'" for k, v in (current_proposal or {}).items()
        )
        dialogue_log.append(f"  [{proposer_id:<10}]  PROPOSE  {prop_str}")
        dialogue_log.append(f'                    "{proposal_text[:120]}"')
        print(f"  [r{round_num}] {proposer_id} proposes: {prop_str}", flush=True)

        if not current_proposal:
            round_records.append(round_record)
            _save_json(trace_dir / f"round_{round_num:04d}.json", round_record)
            continue

        # -- Responders --------------------------------------------------------
        n_accepts = 0
        for responder_id in agent_ids:
            if responder_id == proposer_id:
                continue

            resp = _call_agent(
                agent_id=responder_id,
                mission=content,
                history=history,
                round_num=round_num,
                n_steps=n_steps,
                role="responder",
                current_proposal=current_proposal,
                personas=_personas,
            )

            status = resp.get("status", "counter")
            resp_text = resp.get("text", "")
            history.append(
                {"from": responder_id, "text": resp_text, "round": round_num}
            )

            resp_msg = {
                "round": round_num,
                "from": responder_id,
                "role": "responder",
                "status": status,
                "text": resp_text,
                "proposal": resp.get("proposal"),
            }
            message_trace.append(resp_msg)
            round_record["messages"].append(resp_msg)

            if status == "accept":
                n_accepts += 1
                dialogue_log.append(f"  [{responder_id:<10}]  ACCEPT")
                print(f"  [r{round_num}] {responder_id} ACCEPTS", flush=True)
            else:
                counter = resp.get("proposal")
                if counter:
                    current_proposal = counter
                c_str = "  |  ".join(f"{k}: '{v}'" for k, v in (counter or {}).items())
                dialogue_log.append(f"  [{responder_id:<10}]  COUNTER  {c_str}")
                dialogue_log.append(f'                    "{resp_text[:120]}"')
                print(f"  [r{round_num}] {responder_id} counters: {c_str}", flush=True)

        round_records.append(round_record)
        _save_json(trace_dir / f"round_{round_num:04d}.json", round_record)

        if n_accepts >= len(agent_ids) - 1:
            agreement = current_proposal
            dialogue_log.append(f"  -> AGREEMENT at round {round_num}")
            print(
                f"\n  AGREEMENT at round {round_num}: {current_proposal}\n", flush=True
            )
            break
    else:
        timedout = True
        dialogue_log.append(f"  -> TIMED OUT after {n_steps} rounds")
        print(f"\n  TIMED OUT after {n_steps} rounds\n", flush=True)

    _save_json(trace_dir / "message_trace.json", message_trace)

    status_str = "agreed" if agreement else ("timeout" if timedout else "no_agreement")
    result = {
        "session_id": session_id,
        "status": status_str,
        "total_rounds": total_rounds,
        "agreement": agreement,
        "timedout": timedout,
        "rounds": round_records,
        "message_trace": message_trace,
    }

    dialogue_log.append("")
    dialogue_log.append("=" * 62)
    if agreement:
        deal_str = "  |  ".join(f"{k}: '{v}'" for k, v in agreement.items())
        dialogue_log.append("  VERDICT  : CONSENSUS REACHED")
        dialogue_log.append(f"  DEAL     : {deal_str}")
    elif timedout:
        dialogue_log.append(f"  VERDICT  : TIMED OUT after {total_rounds} rounds")
    else:
        dialogue_log.append("  VERDICT  : NO AGREEMENT")
    dialogue_log.append(f"  Rounds   : {total_rounds} / {n_steps}")
    dialogue_log.append("=" * 62)

    (trace_dir / "dialogue.log").write_text(
        "\n".join(dialogue_log) + "\n", encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    missions_file: Path | None = None,
    n_agents: int = 3,
    agent_configs_file: Path | None = None,
) -> None:
    missions = _load_missions(missions_file or _MISSIONS_FILE)

    agent_configs = _load_agent_configs(agent_configs_file)
    cfg_path = agent_configs_file or _AGENT_CONFIGS_FILE
    if agent_configs["default_agents"] or agent_configs["mission_agents"]:
        print(
            f"Agent configs : {cfg_path.resolve()}\n"
            f"  {len(agent_configs['default_agents'])} default agents, "
            f"{len(agent_configs['mission_agents'])} mission-specific sets"
        )
    else:
        print("Agent configs : none found — using built-in fallback")

    # Resolve a sample persona set to validate n_agents against the first mission.
    sample_personas = _build_direct_personas(missions[0], agent_configs)
    _all_ids = list(sample_personas.keys())
    if n_agents < 2 or n_agents > len(_all_ids):
        raise ValueError(
            f"--n-agents must be between 2 and {len(_all_ids)} (got {n_agents})"
        )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_trace_dir = Path("neg_trace") / f"direct_{run_timestamp}_{n_agents}ag"
    run_trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run trace root : {run_trace_dir.resolve()}")
    print(f"Missions       : {len(missions)}")
    for m in missions:
        print(f"  * {m['name']}")
    print()

    run_log: list[dict] = []

    for idx, mission in enumerate(missions, start=1):
        t_start = time.monotonic()
        mission_slug = _slug(mission["name"])
        session_id = f"sess-direct-{run_timestamp}-{mission_slug}"
        trace_dir = run_trace_dir / mission_slug
        trace_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'=' * 62}")
        print(f"  Mission {idx}/{len(missions)}: {mission['name']}")
        print(f"  Session : {session_id}")
        print(f"{'=' * 62}\n")

        personas = _build_direct_personas(mission, agent_configs)
        active_agent_ids = list(personas.keys())[:n_agents]

        _save_json(
            trace_dir / "00_mission.json",
            {
                "session_id": session_id,
                "content_text": mission.get("content_text", ""),
                "n_steps": mission.get("n_steps", 30),
                "n_agents": len(active_agent_ids),
                "agents": [
                    {
                        "id": aid,
                        "name": personas[aid]["name"],
                        "persona": personas[aid]["persona"],
                    }
                    for aid in active_agent_ids
                ],
            },
        )

        result = _run_direct(
            session_id=session_id,
            mission=mission,
            trace_dir=trace_dir,
            agent_ids=active_agent_ids,
            personas=personas,
        )
        _save_json(trace_dir / "final_result.json", result)

        elapsed = round(time.monotonic() - t_start, 1)
        print(f"  Dialogue : {(trace_dir / 'dialogue.log').resolve()}")
        print(
            f"  Status   : {result['status']}  rounds={result['total_rounds']}  ({elapsed}s)"
        )
        print(f"\nMission {idx} trace : {trace_dir.resolve()}\n")

        run_log.append(
            {
                "mission": mission["name"],
                "session_id": session_id,
                "status": result["status"],
                "total_rounds": result["total_rounds"],
                "agreement": result.get("agreement"),
                "duration_s": elapsed,
                "trace_dir": str(trace_dir.resolve()),
            }
        )

    _save_json(
        run_trace_dir / "run_log.json", {"run_id": run_timestamp, "missions": run_log}
    )
    print(f"Run log : {(run_trace_dir / 'run_log.json').resolve()}")
    print(f"All {len(missions)} missions complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct LLM-to-LLM negotiation (no server, no protocol)"
    )
    parser.add_argument(
        "--missions-file",
        default=None,
        metavar="PATH",
        help="Path to a YAML missions file (default: missions.yaml)",
    )
    parser.add_argument(
        "--agent-configs",
        default=None,
        metavar="PATH",
        help=(
            "Path to agent_configs.yaml with per-mission LLM agent personas "
            f"(default: {_AGENT_CONFIGS_FILE})"
        ),
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of agents to include (default 3). "
            "Agents are drawn in order from the resolved persona set. "
            "More agents = more LLM calls per round AND a harder consensus condition."
        ),
    )
    args = parser.parse_args()
    run(
        missions_file=Path(args.missions_file) if args.missions_file else None,
        n_agents=args.n_agents,
        agent_configs_file=Path(args.agent_configs) if args.agent_configs else None,
    )
