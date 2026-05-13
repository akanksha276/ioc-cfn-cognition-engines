# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
SemanticAlignmentEvaluator — evaluates whether a negotiation trace reached
genuine semantic alignment.  Uses an LLM when available, falls back to
deterministic heuristics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .models import (
    AlignmentEvaluation,
    GoalSpec,
    InteractionSignals,
    IssueEvaluation,
    Severity,
    TraceState,
)
from .prompts import ALIGNMENT_PROMPT
from .utils import clamp01, is_vague_option, mean, safe_json_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

_SCORE_ALIGNED = 0.75          # alignment_score + cognitive_alignment floor for "aligned"
_SCORE_INTERVENTION = 0.6      # alignment_score below this → needs_intervention
_COGNITIVE_INTERVENTION = 0.5  # cognitive_alignment below this → needs_intervention
_SCORE_HIGH_SEVERITY = 0.4     # alignment_score threshold for HIGH severity
_SCORE_MEDIUM_SEVERITY = 0.75  # alignment_score threshold for MEDIUM severity
_CONSTRAINT_FIT_CRITICAL = 0.25  # constraint_fit below this → critical issue penalty
_HIGH_STAKES_KEYWORDS = frozenset([
    "enterprise", "critical", "infrastructure", "sla", "production", "multi-year",
    "high-stakes",
])


class SemanticAlignmentEvaluator:
    def __init__(self, llm_provider: Optional[Callable[[str], str]] = None) -> None:
        self._llm = llm_provider

    def evaluate(
        self,
        goal_spec: GoalSpec,
        trace_state: TraceState,
        interaction_signals: InteractionSignals,
    ) -> AlignmentEvaluation:
        if self._llm:
            try:
                llm_eval = self._evaluate_with_llm(goal_spec, trace_state)
                if llm_eval is not None:
                    return self._postprocess_llm_eval(
                        goal_spec, trace_state, interaction_signals, llm_eval
                    )
            except Exception:
                logger.warning(
                    "SemanticAlignmentEvaluator LLM call failed — using heuristic fallback",
                    exc_info=True,
                )

        return self._evaluate_heuristically(goal_spec, trace_state, interaction_signals)

    # ── LLM path ──────────────────────────────────────────────────────────────

    def _evaluate_with_llm(
        self,
        goal_spec: GoalSpec,
        trace_state: TraceState,
    ) -> Optional[Dict[str, Any]]:
        trace_summary = {
            "issue_focus_counts": trace_state.issue_focus_counts,
            "issue_positions_by_agent": trace_state.issue_positions_by_agent,
            "total_rounds": trace_state.total_rounds,
        }
        prompt = (
            ALIGNMENT_PROMPT
            .replace("{mission_goal}", goal_spec.original_goal or "(none provided)")
            .replace("{mission_constraints}", json.dumps(goal_spec.mission_constraints, indent=2))
            .replace("{success_criteria}", json.dumps(goal_spec.success_criteria, indent=2))
            .replace("{issues}", json.dumps(goal_spec.issues, indent=2))
            .replace("{options_per_issue}", json.dumps(goal_spec.options_per_issue, indent=2))
            .replace("{trace_summary}", json.dumps(trace_summary, indent=2))
            .replace("{final_agreement}", json.dumps(trace_state.final_agreement, indent=2))
        )
        raw = self._llm(prompt)
        return safe_json_parse(str(raw))

    def _apply_critical_issue_penalty(
        self,
        alignment_score: float,
        issue_scores: Dict[str, IssueEvaluation],
    ) -> float:
        """Cap alignment score when any issue has critically low constraint_fit."""
        critical_failures = [
            ev for ev in issue_scores.values()
            if ev.constraint_fit < _CONSTRAINT_FIT_CRITICAL
        ]
        if critical_failures:
            n = len(critical_failures)
            cap = 0.50 if n == 1 else 0.40
            return min(alignment_score, cap)
        return alignment_score

    def _postprocess_llm_eval(
        self,
        goal_spec: GoalSpec,
        trace_state: TraceState,
        interaction_signals: InteractionSignals,
        llm_eval: Dict[str, Any],
    ) -> AlignmentEvaluation:
        issue_scores: Dict[str, IssueEvaluation] = {}
        llm_issue_scores = llm_eval.get("issue_scores") or {}
        mission_text = (goal_spec.original_goal or "").lower()
        high_stakes = any(kw in mission_text for kw in _HIGH_STAKES_KEYWORDS)

        for issue in goal_spec.issues:
            s = llm_issue_scores.get(issue, {})
            final_choice = trace_state.final_agreement.get(issue, "")
            issue_scores[issue] = IssueEvaluation(
                issue_id=issue,
                resolution_quality=float(s.get("resolution_quality", 0.5)),
                constraint_fit=float(s.get("constraint_fit", 0.5)),
                consistency=float(s.get("consistency", 0.5)),
                focus_retention=float(s.get("focus_retention", 0.5)),
                final_choice=final_choice,
                notes=list(s.get("notes") or []),
            )

            if final_choice and is_vague_option(final_choice):
                issue_scores[issue].resolution_quality = min(
                    issue_scores[issue].resolution_quality, 0.3
                )
                issue_scores[issue].notes.append(
                    "Final choice is vague or non-operational, so resolution quality is capped."
                )

            if high_stakes and issue_scores[issue].constraint_fit <= 0.2:
                issue_scores[issue].resolution_quality = min(
                    issue_scores[issue].resolution_quality, 0.6
                )
                issue_scores[issue].notes.append(
                    "Resolution quality capped because the final choice is mission-inadequate "
                    "in a high-stakes setting."
                )

        per_issue_scores = [
            0.4 * ev.resolution_quality
            + 0.3 * ev.constraint_fit
            + 0.2 * ev.consistency
            + 0.1 * ev.focus_retention
            for ev in issue_scores.values()
        ]
        per_issue_avg = mean(per_issue_scores)
        per_issue_avg = self._apply_critical_issue_penalty(per_issue_avg, issue_scores)

        agreement_coherence = float(llm_eval.get("agreement_coherence", 1.0))
        alignment_score = round(0.9 * per_issue_avg + 0.1 * agreement_coherence, 4)
        cognitive_alignment = float(llm_eval.get("cognitive_alignment", 0.5))

        raw_failure_modes = list(llm_eval.get("failure_modes") or [])
        failure_modes = []
        for fm in raw_failure_modes:
            fm_str = str(fm).strip()
            if not fm_str:
                continue
            failure_modes.append(fm_str if fm_str.startswith("SM-") else f"SM-0: unclassified: {fm_str}")

        cross_issue_conflicts = [str(c) for c in (llm_eval.get("cross_issue_conflicts") or [])]
        reasoning = str(llm_eval.get("reasoning") or "").strip()

        aligned = alignment_score >= _SCORE_ALIGNED and cognitive_alignment >= _SCORE_ALIGNED
        needs_intervention = (
            alignment_score < _SCORE_INTERVENTION or cognitive_alignment < _COGNITIVE_INTERVENTION
        )

        if alignment_score < _SCORE_HIGH_SEVERITY and cognitive_alignment < _COGNITIVE_INTERVENTION:
            severity = Severity.HIGH
        elif alignment_score < _SCORE_MEDIUM_SEVERITY or cognitive_alignment < _COGNITIVE_INTERVENTION:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        return AlignmentEvaluation(
            aligned=aligned,
            alignment_score=alignment_score,
            issue_scores=issue_scores,
            cognitive_alignment=cognitive_alignment,
            failure_modes=failure_modes,
            needs_intervention=needs_intervention,
            severity=severity,
            reasoning=reasoning or "Alignment evaluation completed.",
            raw_llm=llm_eval,
            agreement_coherence=agreement_coherence,
            cross_issue_conflicts=cross_issue_conflicts,
        )

    # ── Heuristic path ────────────────────────────────────────────────────────

    def _evaluate_heuristically(
        self,
        goal_spec: GoalSpec,
        trace_state: TraceState,
        interaction_signals: InteractionSignals,
    ) -> AlignmentEvaluation:
        issue_scores: Dict[str, IssueEvaluation] = {}
        failure_modes: List[str] = []
        mission_text = (goal_spec.original_goal or "").lower()
        high_stakes = any(kw in mission_text for kw in _HIGH_STAKES_KEYWORDS)

        for issue in goal_spec.issues:
            final_choice = trace_state.final_agreement.get(issue, "")
            options = goal_spec.options_per_issue.get(issue, [])

            # 1. Resolution quality
            if not final_choice:
                resolution_quality = 0.0
                notes = ["No final choice for this issue."]
            elif is_vague_option(final_choice):
                resolution_quality = 0.2
                notes = ["Final choice is overly vague and may not resolve ambiguity."]
            elif final_choice not in options:
                resolution_quality = 0.3
                notes = ["Final choice is outside the known candidate option set."]
            else:
                resolution_quality = 0.85
                notes = ["Final choice is specific and within candidate options."]

            # 2. Constraint fit
            if high_stakes and final_choice and is_vague_option(final_choice):
                constraint_fit = 0.1
                notes.append("Mission appears high-stakes; vague outcome is mission-inappropriate.")
            elif final_choice:
                constraint_fit = 0.8
            else:
                constraint_fit = 0.0

            # 3. Consistency
            by_agent = trace_state.issue_positions_by_agent.get(issue, {})
            per_agent_consistency = []
            for positions in by_agent.values():
                if len(positions) <= 1:
                    per_agent_consistency.append(1.0)
                    continue
                changes = sum(
                    1 for i in range(1, len(positions)) if positions[i] != positions[i - 1]
                )
                reversals = sum(
                    1
                    for i in range(2, len(positions))
                    if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]
                )
                base = 1.0 - min(1.0, 0.25 * changes + 0.35 * reversals)
                per_agent_consistency.append(clamp01(base))
            consistency = round(mean(per_agent_consistency), 4)

            # 4. Focus retention
            focus_count = trace_state.issue_focus_counts.get(issue, 0)
            focus_retention = clamp01(focus_count / max(1, trace_state.total_rounds))

            issue_eval = IssueEvaluation(
                issue_id=issue,
                resolution_quality=round(resolution_quality, 4),
                constraint_fit=round(constraint_fit, 4),
                consistency=round(consistency, 4),
                focus_retention=round(focus_retention, 4),
                final_choice=final_choice,
                notes=notes,
            )
            issue_scores[issue] = issue_eval

            # Failure mode extraction
            if not final_choice:
                failure_modes.append(f"SM-4: {issue}: issue left unresolved — no final choice made")
            elif resolution_quality < 0.35:
                failure_modes.append(f"SM-1: {issue}: agreed option is too vague to operationalize")
            if consistency < 0.4:
                failure_modes.append(
                    f"SM-5: {issue}: high position inconsistency suggests pressure capitulation"
                )
            if focus_retention < 0.25:
                failure_modes.append(
                    f"SM-4: {issue}: low focus retention — agents drifted from this issue"
                )
            if high_stakes and final_choice and is_vague_option(final_choice):
                failure_modes.append(
                    f"SM-2: {issue}: vague option violates high-stakes mission constraint"
                )
            if constraint_fit < 0.3:
                failure_modes.append(
                    f"SM-2: {issue}: agreed option incompatible with mission constraints"
                )

        # Cognitive alignment: combine final agreement specificity + cross-agent divergence inverse
        divergence_scores = [
            1.0 - interaction_signals.per_issue_divergence.get(issue, 0.0)
            for issue in goal_spec.issues
        ]
        cognitive_alignment = round(mean(divergence_scores), 4)

        per_issue_scores = [
            0.4 * ev.resolution_quality
            + 0.3 * ev.constraint_fit
            + 0.2 * ev.consistency
            + 0.1 * ev.focus_retention
            for ev in issue_scores.values()
        ]
        alignment_score = round(mean(per_issue_scores), 4)
        alignment_score = self._apply_critical_issue_penalty(alignment_score, issue_scores)

        aligned = alignment_score >= _SCORE_ALIGNED and cognitive_alignment >= _SCORE_ALIGNED
        needs_intervention = (
            alignment_score < _SCORE_INTERVENTION or cognitive_alignment < _COGNITIVE_INTERVENTION
        )

        if alignment_score < _SCORE_HIGH_SEVERITY and cognitive_alignment < _COGNITIVE_INTERVENTION:
            severity = Severity.HIGH
        elif alignment_score < _SCORE_MEDIUM_SEVERITY:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        reasoning_parts = []
        if not failure_modes:
            reasoning_parts.append(
                "The negotiation appears to have resolved the intended ambiguity adequately."
            )
        else:
            reasoning_parts.append("The negotiation shows signs of residual semantic misalignment.")
            if any("SM-1" in f for f in failure_modes):
                reasoning_parts.append("At least one final choice remains too vague to operationalize.")
            if any("SM-2" in f for f in failure_modes):
                reasoning_parts.append("Some outcomes appear inconsistent with mission constraints.")
            if any("SM-3" in f for f in failure_modes):
                reasoning_parts.append("One agent's interpretation was adopted asymmetrically.")
            if any("SM-4" in f for f in failure_modes):
                reasoning_parts.append(
                    "Key issues were left unresolved or agents drifted from mission intent."
                )
            if any("SM-5" in f for f in failure_modes):
                reasoning_parts.append(
                    "At least one agent exhibited unstable positions suggesting pressure capitulation."
                )

        return AlignmentEvaluation(
            aligned=aligned,
            alignment_score=alignment_score,
            issue_scores=issue_scores,
            cognitive_alignment=cognitive_alignment,
            failure_modes=failure_modes,
            needs_intervention=needs_intervention,
            severity=severity,
            reasoning=" ".join(reasoning_parts).strip(),
            raw_llm=None,
            cross_issue_conflicts=[],
        )
