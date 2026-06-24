# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
LLM prompts for the ACSE (Alignment Check for Semantic Evaluation) pipeline.

- ALIGNMENT_PROMPT    — SemanticAlignmentEvaluator: per-issue scoring + failure mode taxonomy
- GOAL_SPEC_PROMPT    — GoalSpecExtractor: mission constraints + success criteria extraction
"""

# ---------------------------------------------------------------------------
# SemanticAlignmentEvaluator
# ---------------------------------------------------------------------------

ALIGNMENT_PROMPT = """
You are an ACSE semantic alignment evaluator.

Your job is to determine whether a semantic negotiation actually resolved the
intended ambiguity and preserved mission appropriateness.

Return only valid JSON with this schema:
{
  "issue_scores": {
    "<issue_id>": {
      "resolution_quality": 0.0,
      "constraint_fit": 0.0,
      "consistency": 0.0,
      "focus_retention": 0.0,
      "notes": ["..."]
    }
  },
  "agreement_coherence": 0.0,
  "cognitive_alignment": 0.0,
  "failure_modes": ["SM-X: <issue_id>: <one-line description>"],
  "cross_issue_conflicts": ["<issue_a> and <issue_b>: <one-line reason>"],
  "reasoning": "...",
  "confidence": 0.0
}

Definitions:
- resolution_quality: Did the final choice clearly resolve the ambiguity?
- constraint_fit: Is the final choice compatible with mission constraints?
  IMPORTANT: Evaluate whether the choice is APPROPRIATE for the mission, not just
  technically valid. A specific option that is too minimal for the mission context
  (e.g. basic tier for enterprise infrastructure, self-service for critical systems)
  should score LOW on constraint_fit even if it is unambiguous.
  Also consider whether any agent appears to have CAPITULATED under deadline pressure
  rather than genuinely converging on a shared meaning.
- consistency: Were agent meaning positions reasonably consistent over time?
- focus_retention: Did the trace remain focused on resolving the intended issue?
- agreement_coherence (top-level, not per-issue): Do the final chosen options from
  DIFFERENT issues directly contradict or jointly undermine each other?
  Score 1.0 if the final choices across all issues are mutually compatible.
  Score 0.0-0.3 only if two choices from two distinct issues are mutually exclusive
  (e.g. two different issues both claim the same time slot, or one issue requires
  resource R while another forbids it).
  Score 0.5-0.7 if choices from two distinct issues are practically incompatible —
  i.e., satisfying both simultaneously would require violating one of them in practice
  (e.g. cheapest option chosen for budget and premium option chosen for a linked
  resource, or maximum throughput chosen alongside minimum cost when they are
  jointly unachievable at the scale described).
  Do NOT penalize for: vagueness, missing thresholds, intra-issue inconsistency,
  mission-goal misalignment, or capitulation dynamics — those are captured by the
  per-issue dimensions above.
  If the final_agreement is empty (no choices were made), score 0.0 — absence of
  agreement is not coherence.
  List only direct or practically incompatible cross-issue contradictions in cross_issue_conflicts.
- failure_modes: Classify each failure using this taxonomy. Format each as "SM-X: <issue_id>: <description>".
  SM-1 Vague consensus: the agreed option is too ambiguous to operationalize —
       agents agreed on a label without establishing shared meaning. The resolution
       exists but cannot be acted on without further clarification.
  SM-2 Constraint violation: the agreed option violates an explicit mission
       requirement or stated constraint. The agreement is internally valid but
       externally inappropriate for the mission.
  SM-3 Asymmetric adoption: one agent's interpretation was adopted without genuine
       consensus from the others. The agreement reflects one party's position, not
       a shared understanding.
  SM-4 Goal drift: agents negotiated a different issue than what the mission
       required, or key issues were left entirely unresolved. The negotiation
       lost track of the mission intent.
  SM-5 Pressure capitulation: agreement was reached under deadline pressure rather
       than genuine semantic alignment. An agent abandoned stated requirements
       without a principled reason.
  If a failure does not clearly fit SM-1 to SM-5, use SM-0 for unclassified.
  Only include failure modes that are clearly evidenced in the trace.

- cognitive_alignment: Overall confidence that the agents reached genuine shared understanding aligned with the mission. This must reflect the cumulative weight of your failure_modes: if you identified 2+ substantive failure modes (constraint violations, role collapse, capitulation, unresolved ambiguity), cognitive_alignment should be 0.45 or below. If the agreement is mission-appropriate with only minor issues, score above 0.7.

MISSION GOAL:
{mission_goal}

MISSION CONSTRAINTS:
{mission_constraints}

SUCCESS CRITERIA:
{success_criteria}

ISSUES:
{issues}

OPTIONS PER ISSUE:
{options_per_issue}

TRACE SUMMARY:
{trace_summary}

FINAL AGREEMENT:
{final_agreement}
"""


# ---------------------------------------------------------------------------
# GoalSpecExtractor
# ---------------------------------------------------------------------------

GOAL_SPEC_PROMPT = """
You are extracting a structured goal specification for a semantic negotiation.

A semantic negotiation tries to resolve the meaning of ambiguous terms or concepts
so that agents can continue a mission with a shared interpretation.

Return only valid JSON with this schema:
{
  "mission_constraints": ["..."],
  "success_criteria": ["..."]
}

Guidelines:
- "mission_constraints" should capture requirements that the final agreed meaning
  should preserve.
- "success_criteria" should capture what counts as successful resolution.
- Keep both lists short and concrete.
- If mission context is weak, infer sensible generic criteria.

MISSION GOAL:
{mission_goal}

ISSUES:
{issues}

OPTIONS PER ISSUE:
{options_per_issue}
"""
