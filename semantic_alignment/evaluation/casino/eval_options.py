# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 evaluation — Options Generation against CaSiNo.

Tests whether :class:`~app.agent.options_generation.OptionsGeneration` produces
options that are valid for the CaSiNo domain and cover the ground-truth deal
values from the dataset.

What we measure
---------------
For each dialogue we call ``OptionsGeneration.generate(["food","water","firewood"])``
and check the returned ``{issue: [option, ...]}`` mapping against two criteria:

1. **Syntactic validity** — are the options integer strings in ``{"0","1","2","3"}``?
   CaSiNo allocations must be whole-package counts.  The stub returns label strings
   (e.g. ``"option_a"``) so this will score 0 % until a real implementation exists.

2. **Deal coverage** — does each issue's option list contain the quantity that was
   actually agreed in the dataset?  This is a *necessary* condition for NegMAS to
   even be able to reproduce a human deal.

Metrics (per-dialogue, then averaged):
  * Validity rate   — fraction of generated options that are valid quantities
  * Per-issue coverage — whether the ground-truth quantity appears in the options
  * Overall coverage rate — mean coverage across all three issues

Usage::

    # from semantic_alignment/
    python -m evaluation.casino.eval_options \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --limit 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# ── sys.path: ensure agent root is importable ─────────────────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

_eval_env = Path(__file__).resolve().parents[1] / ".env"
if _eval_env.exists():
    load_dotenv(_eval_env, override=True)

from ...app.agent.intent_discovery import IntentDiscovery  # noqa: E402
from ...app.agent.options_generation import OptionsGeneration  # noqa: E402
from ..casino.loader import CASINO_OPTIONS, ISSUES, build_agent_summary, load_casino  # noqa: E402

#: Gold option set — the integers 0-3 as strings.
_GOLD_OPTIONS: set = set(CASINO_OPTIONS)  # {"0", "1", "2", "3"}


def _extract_int(option_str: str) -> Optional[str]:
    """Pull the first integer out of a verbose option string.

    e.g. "2 packages of firewood" -> "2", "0" -> "0", "none" -> None
    """
    m = re.search(r"\b(\d+)\b", str(option_str))
    return m.group(1) if m else None


# CaSiNo-specific context string fed to both components.
_CASINO_CONTEXT = (
    "Two campsite neighbors are negotiating packages of exactly three resources: "
    "Food, Water, and Firewood. These are the only negotiable issues in this domain. "
    "Always identify negotiable entities using these exact canonical names \u2014 "
    "'food', 'water', 'firewood' — regardless of how they are phrased in the sentence."
)

def _load_summaries(
    summaries_path: Optional[str],
    casino_path: Optional[str],
    limit: Optional[int],
) -> List[Dict]:
    """Load or generate per-agent summaries, flattened to one dict per (dialogue, agent)."""
    if summaries_path:
        raw: List[Dict] = json.loads(Path(summaries_path).read_text())
    elif casino_path:
        dialogues = load_casino(casino_path)
        raw = [
            {
                "dialogue_id": dlg.dialogue_id,
                "summaries": {
                    dlg.agent1_id: build_agent_summary(dlg.agent1),
                    dlg.agent2_id: build_agent_summary(dlg.agent2),
                },
            }
            for dlg in dialogues
            if dlg.agent1.value2reason or dlg.agent2.value2reason
        ]
    else:
        raise ValueError("Provide either --summaries-path or --casino-path.")

    if limit:
        dialogue_ids = sorted({rec["dialogue_id"] for rec in raw})[:limit]
        raw = [rec for rec in raw if rec["dialogue_id"] in set(dialogue_ids)]

    flat: List[Dict] = []
    for rec in raw:
        for agent_id, summary in rec.get("summaries", {}).items():
            flat.append({"dialogue_id": rec["dialogue_id"], "agent_id": agent_id, "summary": summary})
    return flat


# ─────────────────────────────────────────────────────────────────────────────


def evaluate_options(
    summaries_path: Optional[str] = None,
    casino_path: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run Phase 2: IntentDiscovery -> OptionsGeneration per agent summary."""
    samples = _load_summaries(summaries_path, casino_path, limit)
    intent_discovery = IntentDiscovery()
    options_gen = OptionsGeneration()
    per_sample: List[dict] = []

    for i, s in enumerate(samples):
        if i % 50 == 0:
            print(f"  ... {i}/{len(samples)} samples processed", flush=True)

        summary = s["summary"]
        entities: List[str] = intent_discovery.discover(summary, context=_CASINO_CONTEXT)
        found_set = {e.lower() for e in entities}
        issue_recall = len(found_set & set(ISSUES)) / len(ISSUES)

        options_map: Dict[str, List[str]] = options_gen.generate_options(
            entities, summary, context=_CASINO_CONTEXT
        )

        found_canonical = found_set & set(ISSUES)
        issue_metrics: Dict[str, dict] = {}
        for issue in ISSUES:
            if issue not in found_canonical:
                issue_metrics[issue] = {"skipped": True}
                continue
            matched_key = next((k for k in options_map if k.lower() == issue), None)
            raw_opts = options_map.get(matched_key, []) if matched_key else []
            extracted = {_extract_int(o) for o in raw_opts} - {None}
            tp = len(extracted & _GOLD_OPTIONS)
            fp = len(extracted - _GOLD_OPTIONS)
            fn = len(_GOLD_OPTIONS - extracted)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            issue_metrics[issue] = {
                "skipped": False,
                "extracted": sorted(extracted),
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "full_coverage": extracted >= _GOLD_OPTIONS,
            }

        scored = [v for v in issue_metrics.values() if not v.get("skipped")]
        avg_f1 = sum(v["f1"] for v in scored) / len(scored) if scored else None
        all_covered = all(v["full_coverage"] for v in scored) if scored else None

        row = {
            "dialogue_id": s["dialogue_id"],
            "agent_id": s["agent_id"],
            "found_issues": sorted(found_canonical),
            "issue_recall": round(issue_recall, 4),
            "issue_metrics": issue_metrics,
            "avg_options_f1": round(avg_f1, 4) if avg_f1 is not None else None,
            "all_issues_fully_covered": all_covered,
        }
        per_sample.append(row)

        if verbose:
            for issue, m in issue_metrics.items():
                if m.get("skipped"):
                    print(f"  dlg {s['dialogue_id']:4d} / {s['agent_id']:<18} | {issue:<8} | skipped")
                else:
                    print(
                        f"  dlg {s['dialogue_id']:4d} / {s['agent_id']:<18} | {issue:<8} | "
                        f"extracted={m['extracted']} | "
                        f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
                        f"{'ok' if m['full_coverage'] else 'x'}"
                    )

    return _aggregate_options(per_sample)


def _aggregate_options(per_sample: List[dict]) -> dict:
    n = len(per_sample)
    if n == 0:
        return {}
    avg_intent_recall = sum(r["issue_recall"] for r in per_sample) / n

    scored_samples = [r for r in per_sample if r["avg_options_f1"] is not None]
    n_scored = len(scored_samples)
    avg_options_f1 = sum(r["avg_options_f1"] for r in scored_samples) / n_scored if n_scored else None
    full_coverage_rate = (
        sum(1 for r in scored_samples if r["all_issues_fully_covered"]) / n_scored
        if n_scored else None
    )

    per_issue: Dict[str, dict] = {}
    for issue in ISSUES:
        issue_scored = [r for r in per_sample if not r["issue_metrics"][issue].get("skipped")]
        ni = len(issue_scored)
        per_issue[issue] = {
            "n_scored": ni,
            "avg_precision": round(sum(r["issue_metrics"][issue]["precision"] for r in issue_scored) / ni, 4) if ni else None,
            "avg_recall":    round(sum(r["issue_metrics"][issue]["recall"]    for r in issue_scored) / ni, 4) if ni else None,
            "avg_f1":        round(sum(r["issue_metrics"][issue]["f1"]        for r in issue_scored) / ni, 4) if ni else None,
            "full_coverage_rate": round(sum(1 for r in issue_scored if r["issue_metrics"][issue]["full_coverage"]) / ni, 4) if ni else None,
        }
    return {
        "n_samples": n,
        "n_scored_samples": n_scored,
        "avg_intent_recall": round(avg_intent_recall, 4),
        "avg_options_f1_conditioned": round(avg_options_f1, 4) if avg_options_f1 is not None else None,
        "all_issues_full_coverage_rate_conditioned": round(full_coverage_rate, 4) if full_coverage_rate is not None else None,
        "per_issue": per_issue,
    }


def _print_results(results: dict) -> None:
    print("\n" + "=" * 70)
    print("PHASE 2 — Options Generation Results (conditioned on discovery)")
    print("=" * 70)
    print(f"  Total samples              : {results['n_samples']}")
    print(f"  Samples with >=1 issue found: {results['n_scored_samples']} "
          f"({results['n_scored_samples']/results['n_samples']:.1%})")
    print(f"  Avg intent recall (Phase 1): {results['avg_intent_recall']:.1%}")
    f1 = results["avg_options_f1_conditioned"]
    cov = results["all_issues_full_coverage_rate_conditioned"]
    print(f"  Avg options F1 | given discovered  : {f1:.1%}" if f1 is not None else "  Avg options F1: N/A")
    print(f"  Full coverage  | given discovered  : {cov:.1%}" if cov is not None else "  Full coverage: N/A")
    print()
    print("  Per-issue (gold = {0,1,2,3}, scored only over discovered issues):")
    print(f"    {'issue':<10}  {'n_scored':>8}  {'P':>6}  {'R':>6}  {'F1':>6}  {'full cov':>8}")
    print(f"    {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
    for issue in ISSUES:
        m = results["per_issue"][issue]
        if m["avg_f1"] is None:
            print(f"    {issue:<10}  {'0':>8}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>8}")
        else:
            print(
                f"    {issue:<10}  {m['n_scored']:>8}  {m['avg_precision']:>6.1%}  "
                f"{m['avg_recall']:>6.1%}  {m['avg_f1']:>6.1%}  {m['full_coverage_rate']:>8.1%}"
            )
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OptionsGeneration (Phase 2) against CaSiNo."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--summaries-path",
        metavar="PATH",
        help="Path to casino_summaries.json (pre-generated by generate_summaries).",
    )
    source.add_argument(
        "--casino-path",
        metavar="PATH",
        help="Path to casino.json — summaries built on-the-fly (no LLM).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N dialogues (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample results.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write JSON results to FILE.",
    )
    args = parser.parse_args(argv)

    src = args.summaries_path or args.casino_path
    print(f"Loading summaries from {src} ...")

    results = evaluate_options(
        summaries_path=args.summaries_path,
        casino_path=args.casino_path,
        limit=args.limit,
        verbose=args.verbose,
    )
    _print_results(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
