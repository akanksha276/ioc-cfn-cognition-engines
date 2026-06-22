# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""PM (process-level) integration tests — three-judge protocol.

Protocol:
  1. Run SAVEvaluator on all snapshot fixtures (latest_message / interaction_history).
  2. Run three independent judges (Claude, Gemini, Mistral) evaluating
     process-level interaction quality — NOT outcome alignment.
  3. PASS if SAVEvaluator's PM verdict matches majority (≥2/3) of judges,
     subject to a severity-tiered threshold:
        HIGH   → all 3 judges must agree
        MEDIUM → at least 1 judge must agree
        LOW    → no agreement required

Both final-snapshot and mid-snapshot fixtures are used here. Mid snapshots
have no ground-truth `expected` — only judge majority is used.

Run:
    pytest semantic_validation/tests/integration/test_pm_judges.py -v -m integration
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import openai
import pytest

from semantic_validation.app.agent.evaluator import SAVEvaluator
from semantic_validation.app.agent.models import (
    Participant,
    PMType,
    SAVInput,
    SeverityType,
)
from semantic_validation.app.config.utils import get_llm_provider

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

LITELLM_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://litellm.prod.outshift.ai")
JUDGE_API_KEY    = os.getenv("JUDGE_API_KEY", "")

_JUDGES = {
    "claude":  os.getenv("JUDGE_CLAUDE_MODEL",  "bedrock/global.anthropic.claude-sonnet-4-6"),
    "gemini":  os.getenv("JUDGE_GEMINI_MODEL",   "vertex_ai/gemini-2.5-pro"),
    "mistral": os.getenv("JUDGE_MISTRAL_MODEL",  "bedrock/mistral.mistral-large-3-675b-instruct"),
}

_PM_JUDGE_PROMPT = """You are evaluating the PROCESS quality of a multi-agent dialogue
(NOT the final outcome — focus on how the agents interact).

Your task: determine whether the interaction process between agents shows
problematic patterns that would require human intervention BEFORE a final
decision is reached. Examples of problematic patterns include:
- agents persistently diverging without progress
- one agent dominating, others passively accepting
- agents repeating same offers/arguments without new information
- reasoning that is internally inconsistent or unsupported
- agents drifting off-topic from the mission
- an agent violating a mission constraint mid-process
- messages that are unclear/ambiguous

A healthy negotiation may include disagreement, counter-offers, and revisions —
these are NOT problems. Only flag observable process dysfunctions.

Respond ONLY with valid JSON — no markdown, no code blocks.
IMPORTANT: Put "needs_intervention" FIRST, then "reasoning". Keep reasoning to 1-2 sentences max.
{{
  "needs_intervention": <true|false>,
  "reasoning": "<1-2 sentences>"
}}

needs_intervention = true  means the process shows problems requiring review.
needs_intervention = false means the process looks productive.

---

MISSION:
{mission}

PARTICIPANTS:
{participants}

LATEST MESSAGE:
{latest_message}

INTERACTION HISTORY:
{interaction_history}
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    judge: str
    needs_intervention: bool
    reasoning: str


# ── Judge helpers ──────────────────────────────────────────────────────────────

def _parse(text: str, judge: str) -> Dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    ni = re.search(r'"needs_intervention"\s*:\s*(true|false)', text)
    reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
    if ni:
        return {
            "needs_intervention": ni.group(1) == "true",
            "reasoning": reasoning.group(1) if reasoning else "(truncated)",
        }
    logger.warning("Judge %s returned unparseable: %s", judge, text[:200])
    return {}


def _call_judge(judge_name: str, prompt: str, retry: int = 2) -> JudgeVerdict:
    client = openai.OpenAI(api_key=JUDGE_API_KEY, base_url=LITELLM_BASE_URL)
    raw = ""
    for attempt in range(retry + 1):
        try:
            resp = client.chat.completions.create(
                model=_JUDGES[judge_name],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content or ""
            break
        except Exception as exc:
            logger.warning("Judge %s attempt %d: %s", judge_name, attempt, exc)
            if attempt == retry:
                return JudgeVerdict(judge_name, True, f"call failed: {exc}")
    parsed = _parse(raw, judge_name)
    return JudgeVerdict(
        judge=judge_name,
        needs_intervention=bool(parsed.get("needs_intervention", True)),
        reasoning=str(parsed.get("reasoning", "")),
    )


def _run_all_judges(prompt: str) -> List[JudgeVerdict]:
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_call_judge, name, prompt): name
            for name in _JUDGES
        }
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                logger.error("Judge %s failed: %s", futures[fut], exc)
            else:
                results.append(fut.result())
    return results


# ── severity derivation ───────────────────────────────────────────────────────

def _overall_pm_severity(failure_modes) -> SeverityType:
    """Worst severity across PM failure modes only."""
    pm_modes = [fm for fm in failure_modes if isinstance(fm.type, PMType)]
    if any(fm.severity.type == SeverityType.HIGH for fm in pm_modes):
        return SeverityType.HIGH
    if any(fm.severity.type == SeverityType.MEDIUM for fm in pm_modes):
        return SeverityType.MEDIUM
    return SeverityType.LOW


def _severity_to_intervention(severity: SeverityType) -> bool:
    """Test-level mapping: SAV severity → binary intervention signal,
    used only to compare with judges' binary needs_intervention output.
    SAV's public API exposes severity per failure mode, not a binary verdict —
    callers decide their own intervention threshold."""
    return severity in (SeverityType.HIGH, SeverityType.MEDIUM)


# ── Fixture parametrization ────────────────────────────────────────────────────

def _pm_fixture_params():
    """Fixtures with a latest_message (PM evaluation requires it).
    interaction_history is optional context."""
    if not FIXTURES_DIR.exists():
        return []
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        try:
            fixture = json.loads(path.read_text())
            sav_input = fixture.get("sav_input", {})
            if sav_input.get("latest_message"):
                fixtures.append(path)
        except Exception:
            continue
    return fixtures


# ── Test ───────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.parametrize("fixture_path", _pm_fixture_params(), ids=lambda p: p.stem)
def test_pm_agrees_with_judge_majority(fixture_path: Path, write_result):
    """SAV's PM verdict must match majority of judges, severity-tiered."""
    fixture = json.loads(fixture_path.read_text())
    sav_input_data = fixture["sav_input"]

    participants = [
        Participant(id=p["id"], name=p.get("name"), role=p.get("role"))
        for p in sav_input_data.get("participants", [])
    ]
    # PM-only: pass only latest_message + interaction_history (omit final_decision)
    sav_input = SAVInput(
        mission=sav_input_data["mission"],
        participants=participants,
        context=sav_input_data.get("context"),
        latest_message=sav_input_data.get("latest_message"),
        interaction_history=sav_input_data.get("interaction_history"),
    )

    evaluator = SAVEvaluator(llm_provider=get_llm_provider())
    sav_output = evaluator.evaluate(sav_input)

    severity = _overall_pm_severity(sav_output.failure_modes)
    sav_ni = _severity_to_intervention(severity)

    participants_str = ", ".join(
        f"{p.name or p.id}" + (f" ({p.role})" if p.role else "")
        for p in participants
    )
    history_str = (
        "\n".join(sav_input.interaction_history)
        if sav_input.interaction_history
        else "(not provided)"
    )
    judge_prompt = _PM_JUDGE_PROMPT.format(
        mission=sav_input.mission,
        participants=participants_str or "(not specified)",
        latest_message=sav_input.latest_message or "(not provided)",
        interaction_history=history_str,
    )

    judge_results = _run_all_judges(judge_prompt)
    agree_count = sum(jr.needs_intervention == sav_ni for jr in judge_results)

    # When SAV says intervene (HIGH or MEDIUM), require ≥1/3 judges to agree.
    # When SAV says no intervene (LOW), no judge verification is needed.
    min_agree = 1 if sav_ni else 0

    pm_modes = [fm for fm in sav_output.failure_modes if isinstance(fm.type, PMType)]
    passed = agree_count >= min_agree

    write_result(
        f"pm__{fixture_path.stem}",
        {
            "fixture": fixture_path.stem,
            "kind": "pm",
            "mission": sav_input.mission[:200],
            "sav_severity": severity.value,
            "sav_needs_intervention": sav_ni,
            "sav_failure_modes": [
                {
                    "type": fm.type.value,
                    "code": fm.type.name.replace("_", "-"),
                    "score": fm.score,
                    "severity": fm.severity.type.value,
                    "reasoning": fm.reasoning,
                }
                for fm in pm_modes
            ],
            "judge_verdicts": {
                jr.judge: {
                    "needs_intervention": jr.needs_intervention,
                    "reasoning": jr.reasoning,
                }
                for jr in judge_results
            },
            "agree_count": agree_count,
            "min_agree": min_agree,
            "passed": passed,
        },
    )

    print(
        f"\n{'='*72}\n"
        f"  FIXTURE : {fixture_path.stem}\n"
        f"{'='*72}\n"
        f"  SAV needs_intervention : {sav_ni}\n"
        f"  SAV severity (PM)      : {severity.value}\n"
        f"  PM failure_modes       : {len(pm_modes)}\n"
        + "\n".join(f"    - [{fm.type.value}] score={fm.score:.2f}  {fm.description[:60]}"
                    for fm in pm_modes) + "\n"
        f"  judge verdicts         : { {jr.judge: jr.needs_intervention for jr in judge_results} }\n"
        f"  agree_count            : {agree_count}/3 (need {min_agree})\n"
        f"{'='*72}"
    )

    for jr in judge_results:
        logger.info("  [%s] needs_intervention=%s  reasoning: %s", jr.judge, jr.needs_intervention, jr.reasoning[:120])

    assert passed, (
        f"SAV(PM) agrees with only {agree_count}/3 judges (need {min_agree}) [{fixture_path.stem}]\n"
        f"SAV: needs_intervention={sav_ni}, severity={severity.value}\n"
        f"Judges: { {jr.judge: jr.needs_intervention for jr in judge_results} }\n"
        + "\n".join(f"  [{jr.judge}] {jr.reasoning}" for jr in judge_results)
    )
