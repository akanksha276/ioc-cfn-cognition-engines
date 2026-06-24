#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0
"""
Convert neg_trace run_log.json into SAVInput fixtures for integration tests.

Each mission produces up to 3 snapshot fixtures depending on total rounds:
  rounds < 10  → 1 snapshot (final round)
  rounds 10-19 → 2 snapshots (round 10, final round)
  rounds >= 20 → 3 snapshots (1/3, 2/3, final round)

Each snapshot fixture contains:
  - latest_message     — snapshot round's last agent reply (string with reason)
  - interaction_history — all agent replies from rounds before snapshot round
  - final_decision     — ONLY in the final snapshot (set when snapshot is last round)
  - expected           — ONLY in the final snapshot (taken from run_log validation)

Usage:
    python generate_fixtures.py --run-log <path/to/run_log.json> --out <fixtures_dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────

def _mission_text(trace_dir: str) -> str:
    dlg = Path(trace_dir) / "dialogue.log"
    if not dlg.exists():
        return ""
    content = dlg.read_text()
    m = re.search(r"MISSION\s*\n\s*(.*?)(?:\n\s*\n)", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _final_agreement(trace_dir: str) -> dict:
    commits = sorted(glob.glob(f"{trace_dir}/*/commit__final_result.json"))
    if not commits:
        return {}
    d = json.load(open(commits[-1]))
    fr_payload = (d.get("final_result") or {}).get("payload") or {}
    trace = fr_payload.get("trace") or {}
    rounds = trace.get("rounds") or []
    if rounds:
        return rounds[-1].get("offer") or {}
    return {}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _list_round_dirs(trace_dir: str) -> List[Path]:
    """Return round directories sorted by round number."""
    round_dirs = sorted(Path(trace_dir).glob("round_[0-9]*"))
    return [d for d in round_dirs if d.is_dir()]


def _last_round_with_replies(round_dirs: List[Path]) -> int:
    """Return the highest round number that has at least one agent reply file.

    Skips commit-only rounds (e.g. the agreed round in many traces, where
    server commits without further agent replies).
    """
    for d in reversed(round_dirs):
        if list(d.glob("respond__*__reply.json")):
            # Exclude broadcast — we want actual agent replies
            for f in d.glob("respond__*__reply.json"):
                if "broadcast" not in f.name:
                    return _round_num(d)
    return 0


def _round_num(round_dir: Path) -> int:
    m = re.search(r"round_(\d+)", round_dir.name)
    return int(m.group(1)) if m else 0


def _agent_replies_for_round(round_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(agent_name, payload), ...] in the order files were created."""
    reply_files = sorted(round_dir.glob("respond__*__reply.json"), key=lambda p: p.stat().st_ctime)
    out = []
    for f in reply_files:
        if "broadcast" in f.name:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        payload = r.get("payload") or {}
        agent_name = payload.get("participant_id", f.stem)
        out.append((str(agent_name), payload))
    return out


def _round_current_offer(round_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the active offer agents are voting on this round.

    Stored by the server in respond__broadcast__request.json:current_offer.
    """
    f = round_dir / "respond__broadcast__request.json"
    if not f.exists():
        return None
    try:
        d = json.load(open(f))
        return (d.get("payload") or {}).get("current_offer")
    except Exception:
        return None


def _format_reply(
    agent_name: str,
    payload: Dict[str, Any],
    round_num: int,
    current_offer: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a single agent reply as a string.

    For counter_offer actions the payload carries its own ``offer`` field.
    For accept/reject actions the offer being voted on is the round-level
    ``current_offer`` (from the broadcast request) — without it, the entry
    just says "agent accepts" with no context for what was accepted.
    """
    action = payload.get("action", "?")
    reason = (payload.get("reason") or "").strip()
    offer = payload.get("offer") or {}

    pieces = [f"[round {round_num}] [{agent_name}] action={action}"]
    if offer:
        offer_summary = "; ".join(f"{k}: {v}" for k, v in offer.items())
        pieces.append(f"offer={{{offer_summary}}}")
    elif action in ("accept", "reject") and current_offer:
        offer_summary = "; ".join(f"{k}: {v}" for k, v in current_offer.items())
        pieces.append(f"voting_on={{{offer_summary}}}")
    if reason:
        pieces.append(f"reason: {reason}")
    return " | ".join(pieces)


def _snapshot_rounds(total_rounds: int) -> List[int]:
    """Pick up to 3 snapshot rounds based on the rules."""
    if total_rounds < 1:
        return []
    if total_rounds < 10:
        return [total_rounds]
    if total_rounds < 20:
        return sorted({10, total_rounds})
    return sorted({
        max(1, total_rounds // 3),
        max(1, total_rounds * 2 // 3),
        total_rounds,
    })


# ── snapshot fixture builder ─────────────────────────────────────────────────

def _build_snapshot_fixture(
    mission_name: str,
    mission_text: str,
    agents: List[Dict[str, Any]],
    round_dirs: List[Path],
    snapshot_round: int,
    is_final: bool,
    final_decision_str: Optional[str],
    expected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a single snapshot fixture."""

    # interaction_history: all replies from rounds < snapshot_round.
    # Dedup voting_on across entries — only render the active offer when it
    # differs from the most recent offer the reader has seen (either from a
    # prior round's voting_on or a counter_offer that changed it).
    history: List[str] = []
    snapshot_dir: Optional[Path] = None
    last_shown_offer: Optional[Dict[str, Any]] = None
    for d in round_dirs:
        n = _round_num(d)
        if n < snapshot_round:
            round_offer = _round_current_offer(d)
            round_voting_shown = False
            for agent_name, payload in _agent_replies_for_round(d):
                action = payload.get("action")
                show_offer: Optional[Dict[str, Any]] = None
                if action in ("accept", "reject"):
                    if (not round_voting_shown and round_offer is not None
                            and round_offer != last_shown_offer):
                        show_offer = round_offer
                        last_shown_offer = round_offer
                        round_voting_shown = True
                history.append(_format_reply(agent_name, payload, n, show_offer))
                if action == "counter_offer" and payload.get("offer"):
                    last_shown_offer = payload.get("offer")
        elif n == snapshot_round:
            snapshot_dir = d
            break

    # Snapshot round: all replies except the last are appended to history;
    # the last reply (by file creation order) becomes latest_message.
    # This way SAV sees every agent's action in the snapshot round, not
    # just whichever one happened to be written last.
    latest_message: Optional[str] = None
    if snapshot_dir is not None:
        snap_offer = _round_current_offer(snapshot_dir)
        replies = _agent_replies_for_round(snapshot_dir)
        if replies:
            round_voting_shown = False
            for agent_name, payload in replies[:-1]:
                action = payload.get("action")
                show_offer: Optional[Dict[str, Any]] = None
                if action in ("accept", "reject"):
                    if (not round_voting_shown and snap_offer is not None
                            and snap_offer != last_shown_offer):
                        show_offer = snap_offer
                        last_shown_offer = snap_offer
                        round_voting_shown = True
                history.append(_format_reply(agent_name, payload, snapshot_round, show_offer))
                if action == "counter_offer" and payload.get("offer"):
                    last_shown_offer = payload.get("offer")
            # Last reply becomes latest_message
            agent_name, payload = replies[-1]
            action = payload.get("action")
            show_offer: Optional[Dict[str, Any]] = None
            if action in ("accept", "reject"):
                if (not round_voting_shown and snap_offer is not None
                        and snap_offer != last_shown_offer):
                    show_offer = snap_offer
            latest_message = _format_reply(agent_name, payload, snapshot_round, show_offer)

    fixture: Dict[str, Any] = {
        "name": mission_name,
        "snapshot_round": snapshot_round,
        "is_final_snapshot": is_final,
        "sav_input": {
            "mission": mission_text,
            "participants": agents,
            "context": None,
            "final_decision": final_decision_str if is_final else None,
            "latest_message": latest_message,
            "interaction_history": history if history else None,
        },
    }
    if is_final and expected is not None:
        fixture["expected"] = expected
    return fixture


# ── main conversion ──────────────────────────────────────────────────────────

def convert(run_log_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.load(open(run_log_path))
    missions = data if isinstance(data, list) else data.get("missions", [])

    total_written = 0
    for m in missions:
        trace_dir = m.get("trace_dir", "")
        name = m.get("mission", "unknown")

        mission_text = _mission_text(trace_dir)
        if not mission_text:
            print(f"  SKIP {name}: no mission text found")
            continue

        start_req_path = Path(trace_dir) / "00_start_request.json"
        agents = []
        if start_req_path.exists():
            sr = json.load(open(start_req_path))
            agents = [
                {"id": a.get("id", ""), "name": a.get("name"), "role": None}
                for a in sr.get("agents", [])
            ]

        # Final decision + expected from mission-level data
        final_agreement = _final_agreement(trace_dir)
        final_decision_str = json.dumps(final_agreement, indent=2) if final_agreement else None

        val = m.get("validation") or {}
        expected = {
            "needs_intervention": bool(val.get("needs_intervention", False)),
            "severity": val.get("severity", "low"),
        }

        # Round-by-round structure — base snapshots on rounds that actually
        # have agent replies (skip commit-only rounds so every snapshot has
        # a latest_message for PM evaluation).
        round_dirs = _list_round_dirs(trace_dir)
        total_rounds = _last_round_with_replies(round_dirs)
        if total_rounds == 0:
            print(f"  SKIP {name}: no rounds with agent replies")
            continue

        snapshot_rounds = _snapshot_rounds(total_rounds)
        slug = _slug(name)

        for snap_round in snapshot_rounds:
            is_final = (snap_round == total_rounds)
            fixture = _build_snapshot_fixture(
                mission_name=name,
                mission_text=mission_text,
                agents=agents,
                round_dirs=round_dirs,
                snapshot_round=snap_round,
                is_final=is_final,
                final_decision_str=final_decision_str,
                expected=expected,
            )

            tag = "final" if is_final else f"snap{snap_round:03d}"
            out_path = out / f"{slug}__{tag}.json"
            out_path.write_text(json.dumps(fixture, indent=2))
            history_len = len(fixture["sav_input"]["interaction_history"] or [])
            print(
                f"  wrote {out_path.name}  "
                f"(round={snap_round}/{total_rounds}, history_len={history_len}, "
                f"final={is_final})"
            )
            total_written += 1

    print(f"\nTotal fixtures written: {total_written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--out", default="semantic_validation/tests/fixtures")
    args = parser.parse_args()
    convert(args.run_log, args.out)
