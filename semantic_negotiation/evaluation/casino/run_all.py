# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Master CLI runner — evaluates all three phases against CaSiNo.

Runs Phase 1 (Intent Discovery), Phase 2 (Options Generation) and Phase 3
(Negotiation Model) in sequence and prints a consolidated summary.

Usage::

    # from semantic_negotiation/
    python -m evaluation.casino.run_all \\
        --casino-path ../../CaSiNo/data/casino.json

    # Quick smoke test with 20 dialogues:
    python -m evaluation.casino.run_all \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --limit 20 --verbose

    # Run only Phase 3 with a specific strategy:
    python -m evaluation.casino.run_all \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --phase negotiation \\
        --strategy ConcederTBNegotiator \\
        --n-steps 100 \\
        --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── sys.path: ensure agent root is importable ─────────────────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

# Re-import printers from eval modules
from ..casino.eval_intent import _print_results as _print_intent  # noqa: E402
from ..casino.eval_intent import evaluate_intent  # noqa: E402
from ..casino.eval_negotiation import (  # noqa: E402
    _print_results as _print_neg,
)
from ..casino.eval_negotiation import (
    evaluate_negotiation,
)
from ..casino.eval_options import _print_results as _print_opts  # noqa: E402
from ..casino.eval_options import evaluate_options  # noqa: E402
from ..casino.loader import load_casino  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────

PHASES = ("intent", "options", "negotiation", "all")


def run_all(
    casino_path: str,
    *,
    summaries_path: Optional[str] = None,
    phases: str = "all",
    limit: Optional[int] = None,
    strategy: str = "BoulwareTBNegotiator",
    n_steps: int = 50,
    verbose: bool = False,
    output: Optional[str] = None,
) -> dict:
    """Run the selected evaluation phases and return the combined results dict.

    Args:
        casino_path: Path to ``casino.json``.
        summaries_path: Optional path to ``casino_summaries.json`` for Phase 1.
            If omitted, summaries are generated on-the-fly from *casino_path*.
        phases: Which phases to run: ``"intent"``, ``"options"``,
            ``"negotiation"``, or ``"all"``.
        limit: Cap the number of dialogues (``None`` = all 1030).
        strategy: NegMAS negotiator class name for Phase 3.
        n_steps: Maximum SAO rounds for Phase 3.
        verbose: Print per-dialogue details for every phase.
        output: Optional path to write consolidated JSON results.

    Returns:
        Dict keyed by phase name containing that phase's metrics.
    """
    print(f"\nLoading CaSiNo from {casino_path} …")
    dialogues = load_casino(casino_path)
    n = len(dialogues[:limit]) if limit else len(dialogues)
    print(f"Loaded {len(dialogues)} dialogues — evaluating {n}.")
    if summaries_path:
        print(f"Phase 1 summaries : {summaries_path}")
    print()

    combined: dict = {"casino_path": str(casino_path), "n_dialogues": n}
    t0 = time.perf_counter()

    # ── Phase 1 — Intent Discovery ────────────────────────────────────────────
    if phases in ("all", "intent"):
        print("─" * 60)
        print("Running Phase 1 — Intent Discovery …")
        t1 = time.perf_counter()
        results_intent = evaluate_intent(
            summaries_path=summaries_path,
            casino_path=casino_path,
            limit=limit,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t1
        _print_intent(results_intent)
        print(f"  (completed in {elapsed:.1f}s)")
        combined["intent"] = results_intent

    # ── Phase 2 — Options Generation ─────────────────────────────────────────
    if phases in ("all", "options"):
        print("─" * 60)
        print("Running Phase 2 — Options Generation …")
        t2 = time.perf_counter()
        results_options = evaluate_options(
            summaries_path=summaries_path,
            casino_path=casino_path,
            limit=limit,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t2
        _print_opts(results_options)
        print(f"  (completed in {elapsed:.1f}s)")
        combined["options"] = results_options

    # ── Phase 3 — Negotiation Model ──────────────────────────────────────────
    if phases in ("all", "negotiation"):
        print("─" * 60)
        print("Running Phase 3 — Negotiation Model …")
        t3 = time.perf_counter()
        results_neg = evaluate_negotiation(
            dialogues,
            strategy=strategy,
            n_steps=n_steps,
            limit=limit,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t3
        _print_neg(results_neg)
        print(f"  (completed in {elapsed:.1f}s)")
        combined["negotiation"] = results_neg

    total_elapsed = time.perf_counter() - t0
    combined["total_elapsed_s"] = round(total_elapsed, 2)

    # ── Consolidated summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CONSOLIDATED SUMMARY")
    print("=" * 65)
    if "intent" in combined:
        r = combined["intent"]
        print(
            f"  Phase 1 Intent     | Exact Match (agent): {r['exact_match_rate']:.1%} "
            f"| Both agents: {r.get('both_agents_exact_match_rate', 0):.1%} "
            f"| Micro-F1: {r['avg_micro_f1']:.3f}"
        )
    if "options" in combined:
        r = combined["options"]
        f1 = r.get("avg_options_f1_conditioned")
        cov = r.get("all_issues_full_coverage_rate_conditioned")
        print(
            f"  Phase 2 Options    | F1 (cond.): {f1:.1%} " if f1 is not None else "  Phase 2 Options    | F1: N/A ",
            end=""
        )
        print(f"| Full coverage (cond.): {cov:.1%}" if cov is not None else "| Full coverage: N/A")
    if "negotiation" in combined:
        r = combined["negotiation"]
        pareto = f"{r['pareto_rate']:.1%}" if r.get("pareto_rate") is not None else "N/A"
        print(
            f"  Phase 3 Negotiation| Agreement: {r['agreement_rate']:.1%} "
            f"| Pareto: {pareto} "
            f"| Joint score: {r.get('avg_joint_score', 'N/A')}"
        )
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 65)

    if output:
        Path(output).write_text(json.dumps(combined, indent=2))
        print(f"\nResults written to {output}")

    return combined


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run CaSiNo evaluation for all three pipeline phases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--casino-path",
        required=True,
        metavar="PATH",
        help="Path to casino.json",
    )
    parser.add_argument(
        "--summaries-path",
        default=None,
        metavar="PATH",
        help="Path to casino_summaries.json for Phase 1 (generated by generate_summaries). "
             "If omitted, summaries are built on-the-fly from casino.json.",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        metavar="PHASE",
        help=f"Which phase to run: {PHASES}. Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N dialogues (default: all 1030).",
    )
    parser.add_argument(
        "--strategy",
        default="BoulwareTBNegotiator",
        metavar="CLASS",
        help="NegMAS negotiator class for Phase 3 (default: BoulwareTBNegotiator).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        metavar="N",
        help="Max SAO rounds for Phase 3 (default: 50).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-dialogue details.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write consolidated JSON results to FILE.",
    )
    args = parser.parse_args(argv)

    run_all(
        casino_path=args.casino_path,
        summaries_path=args.summaries_path,
        phases=args.phase,
        limit=args.limit,
        strategy=args.strategy,
        n_steps=args.n_steps,
        verbose=args.verbose,
        output=args.output,
    )


if __name__ == "__main__":
    main()
