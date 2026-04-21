#!/usr/bin/env python3

# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
test_two_agents.py — Simulate two rational agents stepping through a negotiation.

Flow:
  1. POST /api/negotiate/initiate  → server runs components 1→2→3, returns full trace
  2. Agents A and B take turns evaluating each offer in the trace:
       - If the offer's utility is above their acceptance threshold → "accept"
       - Otherwise → "continue" (counter / keep going)
  3. The loop ends when one agent accepts, or NegMAS reaches its own agreement,
     or the trace is exhausted.

Agent preferences are derived using the same gradient the server applies:
  Agent A (index 0, even) — prefers low-index options  (utility = 1 - i/(n-1))
  Agent B (index 1, odd)  — prefers high-index options (utility = i/(n-1))

Usage:
    poetry run python test_two_agents.py
    poetry run python test_two_agents.py --threshold-a 0.4 --threshold-b 0.3
    poetry run python test_two_agents.py --url http://localhost:8089
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

# ── SSTP envelope ─────────────────────────────────────────────────────────────

def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _envelope(session_id: str, payload: dict, *, actor_id: str = "test-runner") -> dict:
    return {
        "kind": "negotiate",
        "protocol": "SSTP",
        "version": "0",
        "message_id": str(uuid.uuid4()),
        "dt_created": _now(),
        "origin": {"actor_id": actor_id, "tenant_id": "ws-demo"},
        "semantic_context": {
            "schema_id": "urn:ioc:schema:negotiate:negmas-sao:v1",
            "schema_version": "1.0",
            "session_id": session_id,
            "sao_state": None,
        },
        "payload_hash": _payload_hash(payload),
        "policy_labels": {
            "sensitivity": "internal",
            "propagation": "restricted",
            "retention_policy": "default",
        },
        "provenance": {"sources": [], "transforms": []},
        "payload": payload,
    }


# ── Agent model ───────────────────────────────────────────────────────────────

# Stub options exactly as returned by OptionsGeneration — must mirror the server.
KNOWN_OPTIONS: dict[str, list[str]] = {
    "budget":   ["minimal", "low", "medium", "high", "uncapped"],
    "timeline": ["express", "short", "standard", "extended", "long"],
    "scope":    ["core", "standard", "extended", "full"],
    "quality":  ["basic", "standard", "premium"],
}


class Agent:
    """A rational negotiating agent with a utility function and accept threshold."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_idx: int,           # 0 → prefers low-index options, 1 → high
        options: dict[str, list[str]],
        accept_threshold: float = 0.55,
    ) -> None:
        self.id = agent_id
        self.name = name
        self.accept_threshold = accept_threshold
        # Build utility function: {issue: {option: utility}}
        self.prefs: dict[str, dict[str, float]] = {}
        for issue, opts in options.items():
            n = len(opts)
            denom = max(n - 1, 1)
            if agent_idx % 2 == 0:
                self.prefs[issue] = {o: round(1.0 - i / denom, 3) for i, o in enumerate(opts)}
            else:
                self.prefs[issue] = {o: round(i / denom, 3) for i, o in enumerate(opts)}

    def utility(self, offer: dict[str, str]) -> float:
        """Average utility across all issues for a given offer dict."""
        if not offer:
            return 0.0
        total = sum(
            self.prefs.get(issue, {}).get(option, 0.0)
            for issue, option in offer.items()
        )
        return round(total / len(offer), 3)

    def decide(self, offer: dict[str, str]) -> tuple[str, float]:
        """Return (action, utility) — 'accept' if utility >= threshold, else 'continue'."""
        u = self.utility(offer)
        action = "accept" if u >= self.accept_threshold else "continue"
        return action, u


# ── Main ──────────────────────────────────────────────────────────────────────

SEP = "─" * 65

CONTENT_TEXT = (
    "We need to negotiate a software development contract between two parties. "
    "Key areas include budget constraints, delivery timeline, scope of work, "
    "and quality standards."
)


def main(base_url: str, n_steps: int, threshold_a: float, threshold_b: float) -> None:
    session_id = str(uuid.uuid4())

    agent_a = Agent("agent-a", "Agent A", agent_idx=0, options=KNOWN_OPTIONS, accept_threshold=threshold_a)
    agent_b = Agent("agent-b", "Agent B", agent_idx=1, options=KNOWN_OPTIONS, accept_threshold=threshold_b)
    agents_by_id = {agent_a.id: agent_a, agent_b.id: agent_b}

    print(SEP)
    print("Two-Agent Negotiation Simulation")
    print(f"  session  : {session_id}")
    print(f"  Agent A threshold : {threshold_a}  (prefers cheap/fast options — low index)")
    print(f"  Agent B threshold : {threshold_b}  (prefers premium/slow options — high index)")
    print(SEP)

    # ── Step 1: Initiate ─────────────────────────────────────────────────────
    print("[1] Initiating negotiation (components 1 → 2 → 3) …")

    initiate_payload: dict[str, Any] = {
        "content_text": CONTENT_TEXT,
        "agents": [{"id": agent_a.id, "name": agent_a.name},
                   {"id": agent_b.id, "name": agent_b.name}],
        "n_steps": n_steps,
    }

    resp = requests.post(
        f"{base_url}/api/negotiate/initiate",
        json=_envelope(session_id, initiate_payload),
        timeout=30,
    )
    resp.raise_for_status()
    init_data = resp.json()

    status       = init_data.get("status", "unknown")
    total_rounds = init_data.get("total_rounds", 0)
    trace        = init_data.get("trace", {})

    print(f"    status         : {status}")
    print(f"    total rounds   : {total_rounds}")

    if status != "ongoing":
        _print_final(status, init_data)
        return

    print()

    # ── Step 2: Detect the NegMAS-agreed round ──────────────────────────────
    #
    # BoulwareTBNegotiator is very conservative: offers barely move for the
    # first ~80% of steps, so client-side utility would show 0.000 for most
    # rounds.  NegMAS already ran the full SAO and recorded its own agreement
    # in trace["final_agreement"].  We mirror that outcome:
    #   • find the first trace round whose offer matches final_agreement
    #     (that is the round NegMAS accepted on),
    #   • force "accept" at that round — otherwise use the utility threshold.
    #
    rounds_list: list = trace.get("rounds", [])
    final_agreement: list | None = trace.get("final_agreement")

    negmas_accept_round: int | None = None
    if final_agreement:
        agreed_map = {item["issue_id"]: item["chosen_option"] for item in final_agreement}
        # Search backwards — acceptance usually happens on the last matching bid
        for i in range(len(rounds_list) - 1, -1, -1):
            if rounds_list[i].get("offer") == agreed_map:
                negmas_accept_round = i + 1  # 1-based
                break

    print("[2] Agents stepping through trace …")
    if negmas_accept_round:
        print(f"    NegMAS reached agreement at round {negmas_accept_round}/{total_rounds}")
    else:
        print("    NegMAS timed out or broke — client will try to accept on threshold")
    print()

    # ── Collect unique offer transitions per proposer ────────────────────────
    # Boulware holds its extreme offer for ~80% of rounds then slowly concedes.
    # Instead of POSTing all 339 rounds, we:
    #   1. Show the distinct offer snapshots as each agent gradually concedes.
    #   2. Jump straight to the accept round (or threshold-trigger round).
    # We identify "interesting" rounds: when a proposer's offer changes vs their
    # own previous offer.
    last_offer_by_proposer: dict[str, dict[str, str]] = {}
    interesting: list[int] = []            # 1-based round numbers worth showing
    for i, r in enumerate(rounds_list):
        pid = r["proposer_id"]
        off = r.get("offer", {})
        if off != last_offer_by_proposer.get(pid, {}):
            interesting.append(i + 1)      # new concession from this proposer
            last_offer_by_proposer[pid] = off

    # Always include the accept/threshold round
    accept_round = negmas_accept_round
    if accept_round is None:
        # Find first round where any agent crosses threshold
        for i, r in enumerate(rounds_list):
            pid = r["proposer_id"]
            off = r.get("offer", {})
            responder = next((a for a in agents_by_id.values() if a.id != pid), None)
            if responder and responder.utility(off) >= responder.accept_threshold:
                accept_round = i + 1
                break

    if accept_round and accept_round not in interesting:
        interesting.append(accept_round)
    interesting = sorted(set(interesting))

    print(f"  ({len(interesting)} distinct offer snapshot(s) out of {total_rounds} rounds)\n")

    # ── Print interesting rounds + post accept at the right round ────────────
    shown_prev = 0
    for round_num in interesting:
        current = rounds_list[round_num - 1]
        proposer_id: str = current["proposer_id"]
        offer: dict[str, str] = current.get("offer", {})

        proposer  = agents_by_id.get(proposer_id)
        responder = next((a for a in agents_by_id.values() if a.id != proposer_id), None)

        u_proposer  = proposer.utility(offer)  if proposer  else 0.0
        u_responder = responder.utility(offer) if responder else 0.0

        is_accept = (accept_round is not None and round_num >= accept_round)
        action    = "accept" if is_accept else "continue"

        proposer_name  = proposer.name  if proposer  else proposer_id
        responder_name = responder.name if responder else "?"

        if shown_prev > 0 and round_num > shown_prev + 1:
            print(f"  ... ({round_num - shown_prev - 1} unchanged rounds skipped)")

        print(
            f"  round {round_num:>3}/{total_rounds}  "
            f"proposer={proposer_name:<9}  "
            f"responder={responder_name:<9}  "
            f"u(proposer)={u_proposer:.3f}  u(responder)={u_responder:.3f}  "
            f"→ {action.upper()}"
        )
        if offer:
            offer_str = ", ".join(f"{k}:{v}" for k, v in offer.items())
            print(f"           offer: {offer_str}")
        shown_prev = round_num

        # Only POST for the accept round; skip intermediate API calls
        if is_accept:
            actor = responder.id if responder else "system"
            offer_payload: dict[str, Any] = {
                "round": round_num,
                "action": action,
                "trace": trace,
            }
            or_resp = requests.post(
                f"{base_url}/api/negotiate/offer-response",
                json=_envelope(session_id, offer_payload, actor_id=actor),
                timeout=30,
            )
            or_resp.raise_for_status()
            or_data = or_resp.json()
            print()
            _print_final(or_data.get("status", "unknown"), or_data)
            return

    # No accept round found — NegMAS timed out; report final state
    print()
    _print_final("timeout", init_data)


def _print_final(status: str, data: dict) -> None:
    print(SEP)
    print(f"Outcome : {status.upper()}")
    agreement = data.get("agreement") or (data.get("trace") or {}).get("final_agreement")
    if agreement:
        print("Agreement:")
        for item in agreement:
            print(f"  {item['issue_id']:<12} → {item['chosen_option']}")
    elif status == "timeout":
        print("Negotiation timed out — no agreement.")
    elif status == "broken":
        print("Negotiation broken off — no agreement.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-agent negotiation simulation")
    parser.add_argument("--url", default="http://localhost:8089")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--threshold-a", type=float, default=0.35,
                        help="Agent A accept threshold (default 0.35)")
    parser.add_argument("--threshold-b", type=float, default=0.35,
                        help="Agent B accept threshold (default 0.35)")
    args = parser.parse_args()
    main(args.url, args.n_steps, args.threshold_a, args.threshold_b)
