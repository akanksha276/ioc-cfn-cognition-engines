#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare SAV (PM) evaluation vs Peter's derailment detection across the
existing fixtures.

The two evaluators look at different things:
  - SAV PM:             whole-process state from latest_message + interaction_history
  - Peter's derailment: per-utterance verdict against task_goal

This script runs both on each fixture and emits a side-by-side comparison
to a JSON/CSV pair plus a short summary. The intent is a paper-trail
artefact for the IOCCF-130 ticket — not a performance benchmark, because
the two evaluators answer different questions.

Usage:
    PYTHONPATH=. .venv/bin/python semantic_validation/tests/scripts/compare_sav_vs_peter.py \
        --fixtures semantic_validation/tests/fixtures \
        --out semantic_validation/tests/results/comparison_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_validation.app.agent.evaluator import SAVEvaluator
from semantic_validation.app.agent.models import (
    Participant,
    PMType,
    SAVInput,
    SeverityType,
)
from semantic_validation.app.config.utils import get_llm_provider

logger = logging.getLogger(__name__)

# ── Peter's derailment prompt (refined with concrete definitions) ────────────

PETERS_DERAILMENT_PROMPT = """You are evaluating a single utterance from a multi-agent dialogue
against the mission task goal.

Determine whether the utterance is a derailment from the mission, and classify
the cause if so. Return ONLY valid JSON with this schema:
{
  "derailed": <true|false>,
  "derailment_cause": "<one of: blatant_error, blatant_error_network, policy_tangent, topic_shift, data_drift, none>",
  "ambiguous": <true|false>,
  "ambiguity_score": <float 0..1>,
  "alignment_score": <float 0..1>,
  "aligned": <true|false>,
  "judge_confidence": <float 0..1>,
  "critique": "<one or two sentences explaining the verdict>"
}

Derailment cause definitions:
  blatant_error          — utterance proposes/accepts an option that violates an
                           explicit safety, mission, or constraint requirement.
  blatant_error_network  — utterance bypasses coverage / contract / scoping
                           constraints (e.g. routes outside the agreed network).
  policy_tangent         — utterance defers the mission task to discuss
                           internal policy, governance, or wording details.
  topic_shift            — utterance abandons the mission task for unrelated
                           operational/process topics (KPIs, staffing, etc.).
  data_drift             — utterance shifts focus to historical analytics /
                           telemetry / dashboards unrelated to the task.
  none                   — utterance is on-task; no derailment.

Ambiguity:
  An utterance is ambiguous if its intent is unclear, key details are missing
  (timeframes, quantities, responsibilities), or its connection to the task
  goal is unresolvable from the wording alone.

Alignment:
  alignment_score reflects how well the utterance advances the mission task.
  1.0 = fully on-task and constructive, 0.0 = clearly off-task or counter-productive.

Bias toward "not derailed" and high alignment_score when uncertain. A healthy
negotiation includes disagreement, counter-offers, and revisions; those are
NOT derailments.

---

MISSION (task_goal):
{mission}

PARTICIPANTS:
{participants}

RECENT HISTORY (most recent first; may be empty):
{history}

UTTERANCE TO EVALUATE:
{utterance}
"""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PeterVerdict:
    derailed: bool
    derailment_cause: Optional[str]
    ambiguous: bool
    ambiguity_score: float
    alignment_score: float
    aligned: bool
    judge_confidence: float
    critique: str
    raw: str = ""


@dataclass
class FixtureComparison:
    fixture: str
    snapshot_round: Optional[int]
    is_final: bool
    mission: str

    # SAV PM verdict
    sav_pm_severity: str
    sav_pm_intervene: bool
    sav_pm_failure_modes: List[Dict[str, Any]] = field(default_factory=list)

    # Peter's derailment verdict
    peter_derailed: bool = False
    peter_cause: Optional[str] = None
    peter_ambiguous: bool = False
    peter_alignment: float = 1.0
    peter_critique: str = ""

    # Agreement: did both flag a process-level issue?
    both_flagged: bool = False
    only_sav_flagged: bool = False
    only_peter_flagged: bool = False
    neither_flagged: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
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
            return None
    return None


def _format_participants(participants: List[Dict[str, Any]]) -> str:
    return ", ".join(
        f"{p.get('name') or p.get('id')}"
        + (f" ({p.get('role')})" if p.get("role") else "")
        for p in participants
    )


def _format_history(history: Optional[List[str]], limit: int = 20) -> str:
    if not history:
        return "(no prior history)"
    # Most recent first, capped at `limit`
    recent = list(reversed(history[-limit:]))
    return "\n".join(recent)


def _call_peter(
    llm_provider,
    mission: str,
    participants: str,
    history: str,
    utterance: str,
) -> PeterVerdict:
    prompt = (
        PETERS_DERAILMENT_PROMPT
        .replace("{mission}", mission)
        .replace("{participants}", participants)
        .replace("{history}", history)
        .replace("{utterance}", utterance)
    )
    try:
        raw = llm_provider(prompt)
    except Exception as exc:
        logger.warning("Peter prompt LLM call failed: %s", exc)
        return PeterVerdict(False, None, False, 0.0, 1.0, True, 0.0, f"LLM error: {exc}")

    parsed = _safe_json_parse(raw) or {}
    cause = parsed.get("derailment_cause")
    if cause == "none":
        cause = None
    return PeterVerdict(
        derailed=bool(parsed.get("derailed", False)),
        derailment_cause=cause,
        ambiguous=bool(parsed.get("ambiguous", False)),
        ambiguity_score=float(parsed.get("ambiguity_score", 0.0)),
        alignment_score=float(parsed.get("alignment_score", 1.0)),
        aligned=bool(parsed.get("aligned", True)),
        judge_confidence=float(parsed.get("judge_confidence", 0.0)),
        critique=str(parsed.get("critique", "")),
        raw=raw[:500],
    )


# ── Main comparison ──────────────────────────────────────────────────────────

def _has_pm_signal(fixture: Dict[str, Any]) -> bool:
    """Peter and SAV PM both require a current utterance. Fixtures without a
    latest_message (e.g. final snapshots where the agreed round has no agent
    reply) are not comparable — Peter has no SM-equivalent."""
    sav_input = fixture.get("sav_input", {})
    return bool(sav_input.get("latest_message"))


def _evaluate_fixture(fixture_path: Path, llm_provider) -> Optional[FixtureComparison]:
    fixture = json.loads(fixture_path.read_text())
    if not _has_pm_signal(fixture):
        return None

    sav_input_data = fixture["sav_input"]
    participants_list = sav_input_data.get("participants", [])
    participants = [
        Participant(id=p["id"], name=p.get("name"), role=p.get("role"))
        for p in participants_list
    ]

    # ── SAV PM evaluation (no final_decision so only PM path runs) ───────────
    sav_input = SAVInput(
        mission=sav_input_data["mission"],
        participants=participants,
        context=sav_input_data.get("context"),
        latest_message=sav_input_data.get("latest_message"),
        interaction_history=sav_input_data.get("interaction_history"),
    )
    evaluator = SAVEvaluator(llm_provider=llm_provider)
    sav_output = evaluator.evaluate(sav_input)

    pm_modes = [fm for fm in sav_output.failure_modes if isinstance(fm.type, PMType)]
    if any(fm.severity.type == SeverityType.HIGH for fm in pm_modes):
        sav_severity = SeverityType.HIGH
    elif any(fm.severity.type == SeverityType.MEDIUM for fm in pm_modes):
        sav_severity = SeverityType.MEDIUM
    else:
        sav_severity = SeverityType.LOW
    sav_intervene = sav_severity in (SeverityType.HIGH, SeverityType.MEDIUM)

    # ── Peter's derailment evaluation (per-utterance) ────────────────────────
    utterance = sav_input.latest_message or "(no latest message)"
    history = _format_history(sav_input.interaction_history)
    participants_str = _format_participants(participants_list)
    peter = _call_peter(
        llm_provider,
        mission=sav_input.mission,
        participants=participants_str or "(not specified)",
        history=history,
        utterance=utterance,
    )

    # ── Comparison ───────────────────────────────────────────────────────────
    peter_flagged = peter.derailed or peter.ambiguous
    both = sav_intervene and peter_flagged
    only_sav = sav_intervene and not peter_flagged
    only_peter = peter_flagged and not sav_intervene
    neither = not sav_intervene and not peter_flagged

    return FixtureComparison(
        fixture=fixture_path.stem,
        snapshot_round=fixture.get("snapshot_round"),
        is_final=fixture.get("is_final_snapshot", False),
        mission=sav_input.mission[:200],
        sav_pm_severity=sav_severity.value,
        sav_pm_intervene=sav_intervene,
        sav_pm_failure_modes=[
            {
                "type": fm.type.value,
                "code": fm.type.name.replace("_", "-"),
                "score": fm.score,
                "severity": fm.severity.type.value,
                "reasoning": fm.reasoning,
            }
            for fm in pm_modes
        ],
        peter_derailed=peter.derailed,
        peter_cause=peter.derailment_cause,
        peter_ambiguous=peter.ambiguous,
        peter_alignment=peter.alignment_score,
        peter_critique=peter.critique,
        both_flagged=both,
        only_sav_flagged=only_sav,
        only_peter_flagged=only_peter,
        neither_flagged=neither,
    )


def run_comparison(fixtures_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = sorted(fixtures_dir.glob("*.json"))
    llm_provider = get_llm_provider()

    results: List[FixtureComparison] = []
    print(f"Running comparison on {len(fixture_paths)} fixtures...")

    with ThreadPoolExecutor(max_workers=4) as pool:
        future_to_path = {
            pool.submit(_evaluate_fixture, fp, llm_provider): fp
            for fp in fixture_paths
        }
        for fut in as_completed(future_to_path):
            fp = future_to_path[fut]
            try:
                result = fut.result()
            except Exception as exc:
                logger.warning("Fixture %s raised: %s", fp.name, exc)
                continue
            if result is None:
                continue
            results.append(result)
            print(
                f"  done {result.fixture}  "
                f"sav={result.sav_pm_severity}/intervene={result.sav_pm_intervene}  "
                f"peter=derailed={result.peter_derailed}/cause={result.peter_cause}/amb={result.peter_ambiguous}"
            )

    # ── Write per-fixture JSON ───────────────────────────────────────────────
    per_fixture_path = out_dir / "per_fixture.json"
    per_fixture_path.write_text(
        json.dumps(
            [r.__dict__ for r in results],
            indent=2,
            default=str,
        )
    )

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fixture", "snapshot_round", "is_final",
            "sav_severity", "sav_intervene", "sav_pm_modes",
            "peter_derailed", "peter_cause", "peter_ambiguous", "peter_alignment",
            "agreement",
        ])
        for r in results:
            sav_modes = ";".join(fm["code"] for fm in r.sav_pm_failure_modes)
            if r.both_flagged:
                agreement = "both_flagged"
            elif r.only_sav_flagged:
                agreement = "only_sav"
            elif r.only_peter_flagged:
                agreement = "only_peter"
            else:
                agreement = "neither"
            writer.writerow([
                r.fixture, r.snapshot_round, r.is_final,
                r.sav_pm_severity, r.sav_pm_intervene, sav_modes,
                r.peter_derailed, r.peter_cause or "", r.peter_ambiguous, f"{r.peter_alignment:.2f}",
                agreement,
            ])

    # ── Write summary ────────────────────────────────────────────────────────
    total = len(results)
    both = sum(1 for r in results if r.both_flagged)
    only_sav = sum(1 for r in results if r.only_sav_flagged)
    only_peter = sum(1 for r in results if r.only_peter_flagged)
    neither = sum(1 for r in results if r.neither_flagged)

    peter_cause_counts: Dict[str, int] = {}
    for r in results:
        c = r.peter_cause or "none"
        peter_cause_counts[c] = peter_cause_counts.get(c, 0) + 1

    sav_pm_type_counts: Dict[str, int] = {}
    for r in results:
        for fm in r.sav_pm_failure_modes:
            t = fm["code"]
            sav_pm_type_counts[t] = sav_pm_type_counts.get(t, 0) + 1

    summary = {
        "total_fixtures": total,
        "agreement": {
            "both_flagged": both,
            "only_sav_flagged": only_sav,
            "only_peter_flagged": only_peter,
            "neither_flagged": neither,
        },
        "agreement_rate_when_either_flags": round(
            both / (both + only_sav + only_peter), 4
        ) if (both + only_sav + only_peter) else None,
        "peter_cause_distribution": peter_cause_counts,
        "sav_pm_type_distribution": sav_pm_type_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nResults written to {out_dir}/")
    print(f"  per_fixture.json")
    print(f"  comparison.csv")
    print(f"  summary.json")
    print(f"\nAgreement: both={both}  only_sav={only_sav}  only_peter={only_peter}  neither={neither}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="semantic_validation/tests/fixtures")
    parser.add_argument(
        "--out",
        default=f"semantic_validation/tests/results/comparison_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    run_comparison(Path(args.fixtures), Path(args.out))
