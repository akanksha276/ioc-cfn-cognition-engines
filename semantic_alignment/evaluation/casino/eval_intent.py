# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 evaluation — Intent Discovery against CaSiNo.

Tests whether :class:`~app.agent.intent_discovery.IntentDiscovery` correctly
identifies ``{"food", "water", "firewood"}`` as the negotiable issues when
given each agent's natural-language reason summary.

Input
-----
A pre-generated ``casino_summaries.json`` (produced by
``evaluation.casino.generate_summaries``).  Each dialogue contributes **two**
independent evaluation samples — one per agent — since each agent's summary
is a realistic standalone input to intent discovery.

As a fallback, ``--casino-path`` can be passed instead; summaries will be
generated on-the-fly from the raw dataset without saving them.

What we measure
---------------
For each (dialogue_id, agent_id, summary) triple we call
``IntentDiscovery.discover(content_text=summary)`` and compare the result
against the gold set ``{"food", "water", "firewood"}``.

Metrics (per sample, then averaged):
  * Per-issue Precision / Recall / F1
  * Micro-F1  (global TP / FP / FN)
  * Macro-F1  (mean of per-issue F1s)
  * Exact Match rate  (per agent summary)
  * Both-agents Exact Match rate  (both agents correct for the same dialogue)

Usage::

    # Step 1 — generate the summaries dataset once:
    python -m evaluation.casino.generate_summaries \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --output ../../CaSiNo/data/casino_summaries.json

    # Step 2 — run Phase 1:
    python -m evaluation.casino.eval_intent \\
        --summaries-path ../../CaSiNo/data/casino_summaries.json

    # Or generate summaries on-the-fly (no pre-generated file needed):
    python -m evaluation.casino.eval_intent \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── sys.path: ensure agent root is importable ─────────────────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from dotenv import load_dotenv

_eval_env = Path(__file__).resolve().parents[1] / ".env"
if _eval_env.exists():
    load_dotenv(_eval_env, override=True)

from ...app.agent.intent_discovery import IntentDiscovery  # noqa: E402
from ..casino.loader import ISSUES, build_agent_summary, load_casino  # noqa: E402
from ..casino.metrics import aggregate_intent_metrics, issue_f1_metrics  # noqa: E402

_CASINO_CONTEXT = (
    "Two campsite neighbors are negotiating packages of exactly three resources: "
    "Food, Water, and Firewood. These are the only negotiable issues in this domain. "
    "Always identify negotiable entities using these exact canonical names — "
    "'food', 'water', 'firewood' — regardless of how they are phrased in the sentence."
)

# ─────────────────────────────────────────────────────────────────────────────


def _load_summaries(
    summaries_path: Optional[str],
    casino_path: Optional[str],
    limit: Optional[int],
) -> List[Dict]:
    """Load or generate summaries, flatten to one dict per (dialogue, agent)."""
    if summaries_path:
        raw: List[Dict] = json.loads(Path(summaries_path).read_text())
    elif casino_path:
        # On-the-fly: build raw summaries directly from the loader (no LLM)
        dialogues = load_casino(casino_path)
        raw = [
            {
                "dialogue_id": dlg.dialogue_id,
                "gold_issues": list(ISSUES),
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

    # Apply dialogue-level limit before flattening
    if limit:
        dialogue_ids = sorted({rec["dialogue_id"] for rec in raw})[:limit]
        raw = [rec for rec in raw if rec["dialogue_id"] in set(dialogue_ids)]

    # Flatten: one entry per agent per dialogue
    flat: List[Dict] = []
    for rec in raw:
        for agent_id, summary in rec.get("summaries", {}).items():
            flat.append(
                {
                    "dialogue_id": rec["dialogue_id"],
                    "agent_id": agent_id,
                    "summary": summary,
                    "gold_issues": rec.get("gold_issues", ["food", "water", "firewood"]),
                }
            )
    return flat


def evaluate_intent(
    summaries_path: Optional[str] = None,
    casino_path: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run Phase 1 evaluation over per-agent summaries.

    Provide *either* ``summaries_path`` (pre-generated JSON) *or*
    ``casino_path`` (raw dataset; summaries generated on-the-fly).

    Args:
        summaries_path: Path to ``casino_summaries.json``.
        casino_path: Path to ``casino.json`` (fallback if no summaries file).
        limit: Cap to the first N dialogues (each contributing 2 samples).
        verbose: Print per-sample results to stdout.

    Returns:
        Aggregate metrics dict from
        :func:`~evaluation.casino.metrics.aggregate_intent_metrics`.
    """
    samples = _load_summaries(summaries_path, casino_path, limit)
    intent_discovery = IntentDiscovery()

    per_sample: List[dict] = []

    for s in samples:
        predicted: List[str] = intent_discovery.discover(s["summary"], context=_CASINO_CONTEXT)
        m = issue_f1_metrics(predicted)
        m["dialogue_id"] = s["dialogue_id"]
        m["agent_id"] = s["agent_id"]

        if verbose:
            em = "✓" if m["exact_match"] else "✗"
            preview = s["summary"][:80].replace("\n", " ") + ("…" if len(s["summary"]) > 80 else "")
            print(
                f"  [{em}] dlg {s['dialogue_id']:4d} / {s['agent_id']:<18} | "
                f"predicted={predicted} | micro_f1={m['micro_f1']:.2f}\n"
                f"       \"{preview}\""
            )

        per_sample.append(m)

    agg = aggregate_intent_metrics(per_sample)

    # Dialogue-level: did BOTH agents' summaries yield a correct prediction?
    n_dialogues = len({s["dialogue_id"] for s in per_sample})
    dialogue_results: Dict[int, List[bool]] = {}
    for m in per_sample:
        dialogue_results.setdefault(m["dialogue_id"], []).append(m["exact_match"])
    both_rate = (
        sum(1 for v in dialogue_results.values() if all(v)) / n_dialogues
        if n_dialogues else 0.0
    )

    agg["n_samples"] = len(per_sample)
    agg["n_dialogues"] = n_dialogues
    agg["both_agents_exact_match_rate"] = round(both_rate, 4)
    return agg


def _print_results(results: dict) -> None:
    print("\n" + "=" * 65)
    print("PHASE 1 — Intent Discovery Results")
    print("=" * 65)
    print(f"  Dialogues evaluated : {results.get('n_dialogues', '?')}")
    print(f"  Samples (per-agent) : {results.get('n_samples', results['n_dialogues'])}")
    print(f"  Exact Match (per agent)    : {results['exact_match_rate']:.1%}")
    print(f"  Exact Match (both agents)  : {results.get('both_agents_exact_match_rate', 0):.1%}")
    print(f"  Micro-F1            : {results['avg_micro_f1']:.4f}")
    print(f"  Macro-F1            : {results['avg_macro_f1']:.4f}")
    print()
    print("  Per-issue averages:")
    for issue, m in sorted(results.get("per_issue", {}).items()):
        print(
            f"    {issue:<10}  P={m['avg_precision']:.3f}  "
            f"R={m['avg_recall']:.3f}  F1={m['avg_f1']:.3f}"
        )
    print()
    if results["avg_micro_f1"] == 0.0:
        print(
            "  ⚠  All F1 scores are 0 — IntentDiscovery is using the stub.\n"
            "     Replace the stub body to see real scores."
        )
    print("=" * 65)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IntentDiscovery (Phase 1) against CaSiNo agent summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Path to casino.json — summaries generated on-the-fly.",
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
    print(f"Loading summaries from {src} …")

    results = evaluate_intent(
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
