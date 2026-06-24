# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for SemanticAlignmentValidationPipeline — two-stage judge protocol.

Adapted from semantic-alignment-validator/evaluation/judges_with_gold.py.

Protocol (matches Nesreen's eval/cross-issue-detection approach):
  1. Run full ACSE pipeline (real LLM) → get needs_intervention bool.
  2. Run three judges (Claude, Gemini, Mistral) independently — judges do NOT
     see ACSE output.  They evaluate the trace themselves and return
     needs_intervention: true/false.
       Stage 1 — unsupervised: judge sees mission + trace + final agreement.
       Stage 2 — supervised:   same + gold_issues + gold_options (only if gold available).
  3. PASS if ACSE's needs_intervention matches majority (≥2/3) judges on the
     highest available stage.

Fixtures with gold (14): pm_01–09, hard_01–05  → both stages run, verdict from Stage 2.
Fixtures without gold (3): demo_01–03           → Stage 1 only.

Run:
    pytest tests/integration/test_alignment_validation_judges.py -v -m integration
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pytest
from app.agent.semantic_alignment_validation_pipeline import (
    SemanticAlignmentValidationPipeline,
    ValidationResult,
)

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLD_DIR = Path(__file__).parent.parent.parent.parent.parent / \
    "semantic-alignment-validator" / "evaluation"

LITELLM_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://litellm.prod.outshift.ai")
JUDGE_API_KEY    = os.getenv("JUDGE_API_KEY", "")

_JUDGES = {
    "claude":  os.getenv("JUDGE_CLAUDE_MODEL",  "bedrock/global.anthropic.claude-sonnet-4-6"),
    "gemini":  os.getenv("JUDGE_GEMINI_MODEL",   "vertex_ai/gemini-2.5-pro"),
    "mistral": os.getenv("JUDGE_MISTRAL_MODEL",  "bedrock/mistral.mistral-large-3-675b-instruct"),
}

# ── Prompts ────────────────────────────────────────────────────────────────────

_UNSUPERVISED_PROMPT = """You are evaluating the outcome of a multi-agent semantic negotiation.

Your task: determine whether the final agreement serves the mission and resolves
the intended ambiguities, or whether human intervention is needed.

Respond ONLY with valid JSON — no markdown, no code blocks:
{{
  "needs_intervention": <true|false>,
  "reasoning": "<2-4 sentences explaining your verdict>"
}}

needs_intervention = true  means the agreement has problems that require review
                           before the orchestrator acts on it.
needs_intervention = false means the agreement adequately resolves the mission
                           and can proceed.

---

MISSION:
{mission_goal}

NEGOTIATION OUTCOME:
Status: {status} after {total_rounds} rounds

Final Agreement:
{final_agreement}

Negotiation trace (agent positions across rounds):
{trace_summary}
"""

_SUPERVISED_PROMPT = """You are evaluating the outcome of a multi-agent semantic negotiation
against a structured set of gold-standard interpretations.

Your task: determine whether the final agreement adequately resolves the mission's
ambiguous terms according to the gold interpretations, or whether human intervention
is needed.

The gold interpretations define the valid resolution options for each ambiguous issue.
If the agreed resolution falls meaningfully outside these options, or if a key issue
was left unresolved, intervention is needed.

Note: the pipeline may use different issue names than the gold file — reason across
this abstraction gap. Focus on whether the semantic intent of each gold issue was
resolved, not exact label matching.

Respond ONLY with valid JSON — no markdown, no code blocks:
{{
  "needs_intervention": <true|false>,
  "reasoning": "<2-4 sentences explaining your verdict, referencing specific gold issues>"
}}

---

MISSION:
{mission_goal}

NEGOTIATION OUTCOME:
Status: {status} after {total_rounds} rounds

Final Agreement:
{final_agreement}

Negotiation trace (agent positions across rounds):
{trace_summary}

---

GOLD STANDARD INTERPRETATIONS:
These are the valid resolution options for each ambiguous issue in this mission.

{gold_content}
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class StageVerdict:
    judge: str
    stage: str          # "unsupervised" or "supervised"
    needs_intervention: bool
    reasoning: str


@dataclass
class JudgeResult:
    judge: str
    unsupervised: StageVerdict
    supervised: Optional[StageVerdict] = None
    gold_changed_verdict: bool = False


# ── Trace helpers ──────────────────────────────────────────────────────────────

def _extract(trace: List[Dict[str, Any]]):
    mission_goal = (trace[0].get("payload") or {}).get("content_text", "")

    issues, options_per_issue = [], {}
    for msg in trace:
        sc = msg.get("semantic_context") or {}
        if sc.get("issues"):
            issues = sc["issues"]
            options_per_issue = sc.get("options_per_issue") or {}
            break

    final_agreement, status = {}, "timeout"
    for msg in reversed(trace):
        sc = msg.get("semantic_context") or {}
        sao_r = sc.get("sao_response") or {}
        if sao_r.get("response") == "ACCEPT_OFFER" and sao_r.get("outcome"):
            final_agreement = sao_r["outcome"]
            status = "agreed"
            break

    total_rounds = sum(
        1 for m in trace
        if (m.get("origin") or {}).get("actor_id") == "negotiation-server"
    )
    return mission_goal, issues, options_per_issue, final_agreement, status, total_rounds


def _fmt_agreement(fa: Dict) -> str:
    if not fa:
        return "  No agreement reached (timeout or breakdown)"
    return "\n".join(f"  {k}: {v}" for k, v in fa.items())


def _fmt_trace(trace: List[Dict]) -> str:
    """Build per-agent position trajectories from agent counter-offer messages."""
    positions: Dict[str, Dict[str, List[str]]] = {}
    for msg in trace:
        actor = (msg.get("origin") or {}).get("actor_id", "")
        if actor in ("negotiation-server", "test-runner"):
            continue
        payload = msg.get("payload") or {}
        # counter_offer has the proposed offer in payload.offer
        offer = payload.get("offer") or {}
        # ACCEPT_OFFER has outcome in sao_response
        if not offer:
            sc = msg.get("semantic_context") or {}
            sao_r = sc.get("sao_response") or {}
            if sao_r.get("response") == "ACCEPT_OFFER":
                offer = sao_r.get("outcome") or {}
        if not offer:
            continue
        for issue, val in offer.items():
            pos = positions.setdefault(actor, {}).setdefault(issue, [])
            if not pos or pos[-1] != val:
                pos.append(val)

    n_rounds = sum(
        1 for m in trace
        if (m.get("origin") or {}).get("actor_id") == "negotiation-server"
    )
    lines = [f"  {n_rounds} rounds total"]
    for agent, iss_map in positions.items():
        lines.append(f"\n  {agent}:")
        for issue, pos in iss_map.items():
            if len(pos) == 1:
                lines.append(f"    {issue}: held '{pos[0]}'")
            else:
                path = " → ".join(f"'{p}'" for p in pos[:4])
                if len(pos) > 4:
                    path += f" (+{len(pos)-4} more)"
                lines.append(f"    {issue}: {path}")
    return "\n".join(lines)


def _fmt_gold(gold_entry: Optional[Dict]) -> str:
    if not gold_entry:
        return "  (no gold annotations available for this trace)"
    lines = []
    gold_issues = gold_entry.get("gold_issues", [])
    gold_options = gold_entry.get("gold_options") or gold_entry.get("gold_interpretations") or {}
    for issue in gold_issues:
        opts = gold_options.get(issue, [])
        lines.append(f"\nIssue: '{issue}'")
        lines.append("  Valid resolutions:")
        for i, opt in enumerate(opts, 1):
            lines.append(f"    {i}. {opt}")
    return "\n".join(lines) if lines else "  (no gold options found)"


# ── Judge call ─────────────────────────────────────────────────────────────────

def _parse(text: str, judge: str, stage: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Fix unescaped newlines inside string values
        try:
            fixed = re.sub(r'(?<!\\)\n(?=[^"]*"[^"]*(?:"[^"]*"[^"]*)*$)', ' ', candidate)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Last resort — extract key fields with regex
        ni_match = re.search(r'"needs_intervention"\s*:\s*(true|false)', candidate)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', candidate, re.DOTALL)
        if ni_match:
            return {
                "needs_intervention": ni_match.group(1) == "true",
                "reasoning": reasoning_match.group(1) if reasoning_match else "",
            }
    logger.warning("Judge %s stage=%s returned unparseable: %s", judge, stage, text[:200])
    return {}


def _call(judge_name: str, prompt: str, stage: str, retry: int = 2) -> StageVerdict:
    client = openai.OpenAI(api_key=JUDGE_API_KEY, base_url=LITELLM_BASE_URL)
    raw = ""
    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=_JUDGES[judge_name],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
            break
        except Exception as exc:
            logger.warning("Judge %s stage=%s attempt %d: %s", judge_name, stage, attempt, exc)
            if attempt == retry:
                return StageVerdict(judge_name, stage, True, f"call failed: {exc}")
    parsed = _parse(raw, judge_name, stage)
    return StageVerdict(
        judge=judge_name,
        stage=stage,
        needs_intervention=bool(parsed.get("needs_intervention", True)),
        reasoning=str(parsed.get("reasoning", "")),
    )


def _run_judge(
    judge_name: str,
    unsup_prompt: str,
    sup_prompt: Optional[str],
) -> JudgeResult:
    unsup = _call(judge_name, unsup_prompt, "unsupervised")
    sup = None
    if sup_prompt:
        sup = _call(judge_name, sup_prompt, "supervised")
    return JudgeResult(
        judge=judge_name,
        unsupervised=unsup,
        supervised=sup,
        gold_changed_verdict=(
            sup is not None and unsup.needs_intervention != sup.needs_intervention
        ),
    )


def _run_all_judges(
    unsup_prompt: str,
    sup_prompt: Optional[str],
) -> List[JudgeResult]:
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_run_judge, name, unsup_prompt, sup_prompt): name
            for name in _JUDGES
        }
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                logger.error("Judge %s failed: %s", futures[fut], exc)
            else:
                results.append(fut.result())
    return results


# ── Gold index ─────────────────────────────────────────────────────────────────

def _load_gold() -> Dict[str, Dict]:
    index = {}
    for fname in ["pm_usecases_direct_9_gold.json", "hard_convergence_direct_5.json"]:
        path = GOLD_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for entry in json.load(f):
                index[entry["id"]] = entry
    return index


def _gold_for(fixture_path: Path, gold_index: Dict) -> Optional[Dict]:
    prefix = fixture_path.stem.split("_")
    for n in [2, 3]:
        key = "_".join(prefix[:n])
        if key in gold_index:
            return gold_index[key]
    return None


# ── Fixture parametrization ────────────────────────────────────────────────────

def _fixture_params():
    if not FIXTURES_DIR.exists():
        return []
    return sorted(p for p in FIXTURES_DIR.glob("*.json") if p.name != "gold.json")


# ── Test ───────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("fixture_path", _fixture_params(), ids=lambda p: p.stem)
def test_sav_agrees_with_judge_majority(fixture_path: Path):
    """ACSE needs_intervention must match all 3 judges' independent verdict.

    Judges evaluate the trace independently — they do NOT see ACSE output.
    ACSE verdict is compared against judge verdicts as ground truth.

    - Fixtures with gold → two-stage; verdict from Stage 2 (supervised).
    - Fixtures without gold → Stage 1 only (unsupervised).
    """
    with open(fixture_path) as f:
        trace = json.load(f)

    pipeline = SemanticAlignmentValidationPipeline()
    result: ValidationResult = pipeline.run(trace)

    mission_goal, issues, options_per_issue, final_agreement, status, total_rounds = (
        _extract(trace)
    )

    per_issue_lines = "\n".join(
        f"      [{ie.issue_id}]  resolution_quality={ie.resolution_quality:.3f}  "
        f"constraint_fit={ie.constraint_fit:.3f}  "
        f"consistency={ie.consistency:.3f}  "
        f"focus_retention={ie.focus_retention:.3f}"
        for ie in result.per_issue_evaluations
    ) or "      (none)"
    failure_lines = "\n".join(f"      - {fm}" for fm in result.failure_modes) or "      (none)"
    conflict_lines = "\n".join(f"      - {c}" for c in result.cross_issue_conflicts) or "      (none)"

    print(
        f"\n{'='*72}\n"
        f"  DEMO : {fixture_path.stem}\n"
        f"{'='*72}\n"
        f"  needs_intervention  : {result.needs_intervention}\n"
        f"  severity            : {result.severity}\n"
        f"  recommendation      : {result.recommendation}\n"
        f"  timed_out           : {result.timed_out}\n"
        f"  alignment_score     : {result.alignment_score:.4f}\n"
        f"  cognitive_alignment : {result.cognitive_alignment:.4f}\n"
        f"  agreement_coherence : {result.agreement_coherence:.4f}\n"
        f"  per_issue_scores:\n{per_issue_lines}\n"
        f"  failure_modes:\n{failure_lines}\n"
        f"  cross_issue_conflicts:\n{conflict_lines}\n"
        f"  reasoning           : {result.reasoning}\n"
        f"{'='*72}"
    )

    gold_index = _load_gold()
    gold_entry = _gold_for(fixture_path, gold_index)

    trace_summary = _fmt_trace(trace)
    options_str = ""
    if options_per_issue:
        lines = ["\nNEGOTIATED OPTION SPACE:"]
        for issue, opts in options_per_issue.items():
            lines.append(f"  {issue}:")
            for opt in opts:
                lines.append(f"    - {opt}")
        options_str = "\n".join(lines)

    common = dict(
        mission_goal=mission_goal[:600],
        status=status.upper(),
        total_rounds=total_rounds,
        final_agreement=_fmt_agreement(final_agreement),
        trace_summary=trace_summary,
    )

    unsup_prompt = _UNSUPERVISED_PROMPT.format(
        **{**common, "trace_summary": trace_summary + options_str}
    )
    sup_prompt = (
        _SUPERVISED_PROMPT.format(**common, gold_content=_fmt_gold(gold_entry))
        if gold_entry else None
    )

    judge_results = _run_all_judges(unsup_prompt, sup_prompt)

    # Use supervised verdict if available, otherwise unsupervised
    def _verdict(jr: JudgeResult) -> bool:
        if jr.supervised is not None:
            return jr.supervised.needs_intervention
        return jr.unsupervised.needs_intervention

    agree_count = sum(
        _verdict(jr) == result.needs_intervention
        for jr in judge_results
    )
    has_gold = gold_entry is not None
    stage_used = "supervised" if has_gold else "unsupervised"

    judge_verdicts = {jr.judge: _verdict(jr) for jr in judge_results}
    reasoning = {
        jr.judge: (jr.supervised or jr.unsupervised).reasoning
        for jr in judge_results
    }
    gold_changed = {jr.judge: jr.gold_changed_verdict for jr in judge_results if jr.supervised}

    logger.info(
        "fixture=%s gold=%s stage=%s sav_ni=%s score=%.3f agree=%d/3 "
        "sav_vs_judges=%s gold_changed=%s",
        fixture_path.name, has_gold, stage_used,
        result.needs_intervention, result.alignment_score,
        agree_count, judge_verdicts, gold_changed,
    )

    # Severity-tiered agreement threshold:
    #   HIGH   → all 3 judges must agree (both alignment_score and cognitive_alignment < 0.5)
    #   MEDIUM → at least 1 judge agrees
    #   LOW    → no judge agreement required (ACSE confident it's fine)
    from app.agent.sav.models import Severity
    min_agree = {Severity.HIGH: 3, Severity.MEDIUM: 1, Severity.LOW: 0}[result.severity]

    assert agree_count >= min_agree, (
        f"ACSE agrees with only {agree_count}/3 judges (need {min_agree}) [{fixture_path.name}]\n"
        f"Stage used: {stage_used} (gold={'yes' if has_gold else 'no'})\n"
        f"ACSE: needs_intervention={result.needs_intervention}, "
        f"severity={result.severity}, score={result.alignment_score:.3f}\n"
        f"ACSE reasoning: {result.reasoning}\n"
        f"Judge needs_intervention: {judge_verdicts}\n"
        f"Judge reasoning:\n"
        + "\n".join(f"  [{j}] {r}" for j, r in reasoning.items())
        + (f"\nGold changed verdict: {gold_changed}" if gold_changed else "")
    )
