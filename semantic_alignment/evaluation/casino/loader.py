# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""CaSiNo dataset loader.

Converts ``casino.json`` dialogues into pipeline-native types so every
evaluation phase can share a single, tested data-access layer.

Dataset facts
-------------
* 1030 dialogues, all result in a deal.
* Three issues: Food, Water, Firewood — 3 packages each, split between agents.
* Each agent has one High, one Medium and one Low priority issue.
* Scoring: High=5 pts/pkg, Medium=4 pts/pkg, Low=3 pts/pkg.

NegMAS framing
--------------
All options are expressed from **Agent A** (mturk_agent_1) perspective:
``"0"``–``"3"`` = number of packages Agent A receives.  Agent B's utility is
therefore the *complement*: if A gets ``q``, B gets ``3 - q``.

This framing makes every NegMAS agreement automatically a valid allocation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ensure semantic_alignment root is importable ─────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from ...app.agent.negotiation_model import NegotiationParticipant  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CaSiNo domain constants
# ─────────────────────────────────────────────────────────────────────────────

#: The three negotiable issues, lower-cased.
ISSUES: List[str] = ["food", "water", "firewood"]

#: Total packages available per issue to split between both agents.
TOTAL_PACKAGES: int = 3

#: CaSiNo point values per package by priority level.
PRIORITY_SCORES: Dict[str, int] = {"High": 5, "Medium": 4, "Low": 3}

#: Issue weights for ``LinearAdditiveUtilityFunction`` — normalised from
#: CaSiNo point values so they sum to 1.0.
_WEIGHT_SUM: int = sum(PRIORITY_SCORES.values())  # = 12
PRIORITY_WEIGHTS: Dict[str, float] = {
    k: v / _WEIGHT_SUM for k, v in PRIORITY_SCORES.items()
}
# High ≈ 0.417, Medium ≈ 0.333, Low = 0.250

#: Options per issue in the NegMAS outcome space (Agent A's quantity).
CASINO_OPTIONS: List[str] = ["0", "1", "2", "3"]

# ─────────────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentData:
    """Per-agent data extracted from a CaSiNo dialogue entry."""

    agent_id: str
    #: Maps priority level → issue name, e.g. ``{"High": "Firewood", ...}``.
    value2issue: Dict[str, str]
    #: Maps priority level → free-text reason, e.g. ``{"High": "Need warmth..."}``.
    value2reason: Dict[str, str]
    points_scored: int
    satisfaction: str
    opponent_likeness: str


@dataclass
class CasinoDialogue:
    """A single CaSiNo negotiation dialogue with both agents and the outcome."""

    dialogue_id: int
    #: Raw agent id strings from the dataset (``"mturk_agent_1"`` etc.).
    agent1_id: str
    agent2_id: str
    agent1: AgentData
    agent2: AgentData
    #: Agent 1's allocation per issue (number of packages); ``None`` = no deal.
    deal_agent1: Optional[Dict[str, int]]
    has_deal: bool
    chat_logs: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[List[str]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_casino(path: str | Path) -> List[CasinoDialogue]:
    """Load and parse ``casino.json``.

    Args:
        path: Absolute or relative path to ``casino.json``.

    Returns:
        List of :class:`CasinoDialogue`, one per entry in the JSON file.
    """
    with open(path, encoding="utf-8") as fh:
        raw: List[Dict[str, Any]] = json.load(fh)

    dialogues: List[CasinoDialogue] = []
    for entry in raw:
        agent1_id, agent2_id = _infer_agent_ids(entry)
        agent1 = _parse_agent_data(entry["participant_info"][agent1_id], agent1_id)
        agent2 = _parse_agent_data(entry["participant_info"][agent2_id], agent2_id)

        deal_agent1 = _extract_deal(entry["chat_logs"], agent1_id)
        dialogues.append(
            CasinoDialogue(
                dialogue_id=entry["dialogue_id"],
                agent1_id=agent1_id,
                agent2_id=agent2_id,
                agent1=agent1,
                agent2=agent2,
                deal_agent1=deal_agent1,
                has_deal=deal_agent1 is not None,
                chat_logs=entry.get("chat_logs", []),
                annotations=entry.get("annotations", []),
            )
        )
    return dialogues


def to_negotiation_participant(
    agent_data: AgentData,
    *,
    is_agent_a: bool,
) -> NegotiationParticipant:
    """Convert a :class:`AgentData` into a :class:`NegotiationParticipant`.

    The outcome space is expressed from **Agent A's** perspective: each option
    value is the number of packages (``"0"``–``"3"``) that Agent A receives.

    * **Agent A** utility for option ``q``: ``q / 3``  (more = better).
    * **Agent B** utility for option ``q``: ``(3 - q) / 3``  (less for A = more for B).

    Issue weights follow the CaSiNo scoring rule (High=5/12, Medium=4/12,
    Low=3/12).

    Args:
        agent_data: Parsed agent data from the dataset.
        is_agent_a: Whether this agent is framed as the "A" side (Agent 1).

    Returns:
        A :class:`NegotiationParticipant` ready for :class:`NegotiationModel.run`.
    """
    # Build issue → priority level (lower-cased issue names)
    issue2priority: Dict[str, str] = {
        v.lower(): k for k, v in agent_data.value2issue.items()
    }

    preferences: Dict[str, Dict[str, float]] = {}
    issue_weights: Dict[str, float] = {}

    for issue in ISSUES:
        priority = issue2priority.get(issue, "Low")
        issue_weights[issue] = PRIORITY_WEIGHTS[priority]

        if is_agent_a:
            # More packages for A → higher utility
            preferences[issue] = {
                str(q): round(q / TOTAL_PACKAGES, 6)
                for q in range(TOTAL_PACKAGES + 1)
            }
        else:
            # Fewer packages for A → higher utility for B (B gets the rest)
            preferences[issue] = {
                str(q): round((TOTAL_PACKAGES - q) / TOTAL_PACKAGES, 6)
                for q in range(TOTAL_PACKAGES + 1)
            }

    return NegotiationParticipant(
        id=agent_data.agent_id,
        name=agent_data.agent_id,
        preferences=preferences,
        issue_weights=issue_weights,
    )


def build_agent_summary(agent_data: AgentData) -> str:
    """Build a natural-language reason paragraph for a single agent.

    Joins the agent's ``value2reason`` texts into a single paragraph.
    Priority labels (High / Medium / Low) and issue names are intentionally
    excluded so the text is a blind, realistic input to
    :class:`~app.agent.intent_discovery.IntentDiscovery`.

    Args:
        agent_data: A parsed :class:`AgentData` from the dataset.

    Returns:
        A sentence-joined string of all the agent's reasons, e.g.::

            "We have a larger group than normal and therefore require extra
            firewood to keep everyone warm. Extra food will be needed to feed
            our larger than normal-sized group. Our group has sufficient water."
    """
    sentences: List[str] = []
    for reason in agent_data.value2reason.values():
        r = reason.strip()
        if r:
            sentences.append(r if r[-1] in ".,!?" else r + ".")
    return " ".join(sentences)


def build_content_text(agent1: AgentData, agent2: AgentData) -> str:
    """Build a joint natural-language context for :class:`IntentDiscovery`.

    Combines both agents' reason summaries (via :func:`build_agent_summary`)
    into a single passage without exposing priority labels or issue names.
    This is the realistic input a system would receive — only the agents'
    stated reasons, not their ranked preferences.

    Args:
        agent1: First participant's data.
        agent2: Second participant's data.

    Returns:
        A multi-line string suitable for ``IntentDiscovery.discover(content_text=...)``.
    """
    s1 = build_agent_summary(agent1)
    s2 = build_agent_summary(agent2)
    return f"Participant 1: {s1}\n\nParticipant 2: {s2}"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _infer_agent_ids(entry: Dict[str, Any]) -> tuple[str, str]:
    """Return ``(agent1_id, agent2_id)`` from the participant_info keys."""
    ids = sorted(entry["participant_info"].keys())
    if len(ids) != 2:  # pragma: no cover
        raise ValueError(f"Dialogue {entry['dialogue_id']}: expected 2 participants, got {ids}")
    return ids[0], ids[1]


def _parse_agent_data(info: Dict[str, Any], agent_id: str) -> AgentData:
    outcomes = info.get("outcomes", {})
    return AgentData(
        agent_id=agent_id,
        value2issue=info.get("value2issue", {}),
        value2reason=info.get("value2reason", {}),
        points_scored=int(outcomes.get("points_scored", 0)),
        satisfaction=outcomes.get("satisfaction", ""),
        opponent_likeness=outcomes.get("opponent_likeness", ""),
    )


def _extract_deal(
    chat_logs: List[Dict[str, Any]],
    agent1_id: str,
) -> Optional[Dict[str, int]]:
    """Return agent1's allocation from the Submit-Deal message, or ``None``.

    The deal is always expressed from the *submitter's* perspective using
    ``issue2youget`` (what I get) and ``issue2theyget`` (what the other gets).
    We normalise to Agent 1's allocation regardless of who submitted.
    """
    for msg in chat_logs:
        if msg.get("text") == "Submit-Deal":
            td = msg.get("task_data", {})
            submitter = msg.get("id", "")
            if submitter == agent1_id:
                raw = td.get("issue2youget", {})
            else:
                raw = td.get("issue2theyget", {})
            return {k.lower(): int(v) for k, v in raw.items()}
    return None
