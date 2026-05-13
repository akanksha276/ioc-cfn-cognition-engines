# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Data models for the ACSE semantic alignment evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GoalSpec:
    """Structured representation of what the negotiation is supposed to resolve."""

    original_goal: str
    issues: List[str]
    options_per_issue: Dict[str, List[str]]
    mission_constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class RoundRecord:
    round_index: int
    proposer_id: str
    offer: Dict[str, str]


@dataclass
class NegotiationTrace:
    rounds: List[RoundRecord]
    final_agreement: Dict[str, str]
    timedout: bool
    broken: bool
    total_rounds: int


@dataclass
class TraceState:
    """Summarized per-issue state extracted from the negotiation trace."""

    issue_focus_counts: Dict[str, int]
    issue_positions_by_agent: Dict[str, Dict[str, List[str]]]
    all_mentions_by_agent: Dict[str, List[Dict[str, str]]]
    final_agreement: Dict[str, str]
    total_rounds: int


@dataclass
class InteractionSignals:
    """
    Supporting instability signals extracted from trace dynamics.
    These are NOT the main semantic validator outputs.
    """

    positional_instability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_issue_divergence: Dict[str, float] = field(default_factory=dict)
    oscillation_rate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_instability_score: float = 0.0


@dataclass
class IssueEvaluation:
    issue_id: str
    resolution_quality: float
    constraint_fit: float
    consistency: float
    focus_retention: float
    final_choice: str
    notes: List[str] = field(default_factory=list)


@dataclass
class AlignmentEvaluation:
    aligned: bool
    alignment_score: float
    issue_scores: Dict[str, IssueEvaluation]
    cognitive_alignment: float
    failure_modes: List[str]
    needs_intervention: bool
    severity: Severity
    reasoning: str
    raw_llm: Optional[Dict[str, Any]] = None
    agreement_coherence: Optional[float] = None
    cross_issue_conflicts: List[str] = field(default_factory=list)
