# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


class SeverityType(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Severity:
    type: SeverityType
    high: float = 0.4
    medium: float = 0.7

    def evaluate(self, score: float) -> SeverityType:
        if score <= self.high:
            return SeverityType.HIGH
        elif score <= self.medium:
            return SeverityType.MEDIUM
        return SeverityType.LOW


class SMType(Enum):
    """Agreement-level failure modes — require final_decision."""
    SM_0 = "Unclassified"
    SM_1 = "Vague Consensus"
    SM_2 = "Constraint Violation"
    SM_3 = "Asymmetric Adoption"
    SM_4 = "Mission Drift"
    SM_5 = "Pressure Capitulation"


class PMType(Enum):
    """Process-level failure modes — require latest_message or interaction_history."""
    PM_0 = "Unclassified"
    PM_1 = "Persistent Divergence"
    PM_2 = "Dominant Narrative"
    PM_3 = "Repetition"
    PM_4 = "Reasoning Breakdown"
    PM_5 = "Task Deviation"
    PM_6 = "Constraint Violation"
    PM_7 = "Ambiguity"


# ── Failure mode descriptions (single source of truth for taxonomy) ─────────

SM_DESCRIPTIONS: dict = {
    SMType.SM_0: "The failure does not clearly fit SM-1 to SM-5.",
    SMType.SM_1: (
        "The final decision is too ambiguous to operationalize — agents agreed "
        "on a label without establishing shared meaning. The resolution exists "
        "but cannot be acted on without further clarification."
    ),
    SMType.SM_2: (
        "The final decision violates an explicit mission requirement or stated "
        "constraint. The decision is inappropriate for the mission."
    ),
    SMType.SM_3: (
        "One agent's position was adopted without genuine consensus from the "
        "others. The decision reflects one party's position, not a shared "
        "understanding."
    ),
    SMType.SM_4: (
        "The agents resolved something different from what the mission required, "
        "or the final decision does not address the mission goal."
    ),
    SMType.SM_5: (
        "Agreement was reached under pressure rather than genuine alignment. "
        "An agent abandoned stated requirements without a principled reason."
    ),
}

PM_DESCRIPTIONS: dict = {
    PMType.PM_0: "The process issue does not clearly fit PM-1 to PM-7.",
    PMType.PM_1: (
        "Agents hold opposing positions across multiple rounds without converging. "
        "Repeated rejection without movement, conflicting offers that do not narrow."
    ),
    PMType.PM_2: (
        "One agent's position dominates the discussion; other agents passively accept "
        "without genuine engagement. Asymmetric airtime or contributions."
    ),
    PMType.PM_3: (
        "Agents repeat similar offers or arguments without adding new information. "
        "Dialogue stagnates in a loop."
    ),
    PMType.PM_4: (
        "An agent's reasoning is internally inconsistent, contradicts previous "
        "statements, or makes unsupported claims."
    ),
    PMType.PM_5: (
        "An agent introduces topics or actions unrelated to the mission goal. "
        "Off-task tangents that delay progress."
    ),
    PMType.PM_6: (
        "An agent proposes or accepts an option that violates a mission constraint "
        "or hard rule, mid-process."
    ),
    PMType.PM_7: (
        "An agent's message is unclear, underspecified, or its intent is unresolvable "
        "from the context."
    ),
}


def description_for(fm_type: Union[SMType, PMType]) -> str:
    """Return the static description for a failure mode type."""
    if isinstance(fm_type, SMType):
        return SM_DESCRIPTIONS.get(fm_type, "")
    return PM_DESCRIPTIONS.get(fm_type, "")


@dataclass
class FailureMode:
    type: Union[SMType, PMType]
    score: float
    severity: Severity
    description: str
    reasoning: str


@dataclass
class Participant:
    id: str
    name: Optional[str] = None
    role: Optional[str] = None


@dataclass
class SAVInput:
    mission: str
    participants: List[Participant]
    context: Optional[str] = None
    latest_message: Optional[str] = None
    final_decision: Optional[str] = None
    interaction_history: Optional[List[str]] = None


@dataclass
class SAVOutput:
    failure_modes: List[FailureMode] = field(default_factory=list)
