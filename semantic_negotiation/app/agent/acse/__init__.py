# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
acse — ACSE v2 Semantic Alignment Evaluator subpackage.

Components:
  models     — data models (GoalSpec, NegotiationTrace, AlignmentEvaluation, …)
  goal_spec  — GoalSpecExtractor: extracts mission constraints from mission text
  trace_state — TraceStateBuilder + InteractionSignalExtractor
  evaluator  — SemanticAlignmentEvaluator: LLM-backed evaluation with heuristic fallback
"""

from .models import (
    AlignmentEvaluation,
    GoalSpec,
    InteractionSignals,
    IssueEvaluation,
    NegotiationTrace,
    RoundRecord,
    Severity,
    TraceState,
)
from .goal_spec import GoalSpecExtractor
from .trace_state import InteractionSignalExtractor, TraceStateBuilder
from .evaluator import SemanticAlignmentEvaluator

__all__ = [
    "AlignmentEvaluation",
    "GoalSpec",
    "GoalSpecExtractor",
    "InteractionSignalExtractor",
    "InteractionSignals",
    "IssueEvaluation",
    "NegotiationTrace",
    "RoundRecord",
    "SemanticAlignmentEvaluator",
    "Severity",
    "TraceState",
    "TraceStateBuilder",
]
