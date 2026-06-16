# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 evaluation — Negotiation Model against CaSiNo.

This is the only phase that is **immediately runnable** with the current
codebase — the NegMAS SAO mechanism in :class:`~app.agent.negotiation_model.NegotiationModel`
is a real implementation, not a stub.

Setup
-----
For each CaSiNo dialogue we:

1. Map each agent's ``value2issue`` priorities into a
   :class:`~app.agent.negotiation_model.NegotiationParticipant` whose
   preferences and issue weights follow the CaSiNo scoring rule
   (High=5/12, Medium=4/12, Low=3/12 as weights; options ``"0"``–``"3"``
   are Agent A's quantity for each issue, Agent B's utility is the complement).

2. Run :class:`~app.agent.negotiation_model.NegotiationModel` with those two
   participants and the canonical CaSiNo options.

3. Compare the NegMAS outcome to the ground-truth human deal.

Metrics (per-dialogue, then averaged):
  * **Agreement rate** — did NegMAS reach an agreement?
  * **Steps** — how many SAO rounds to agreement (vs. ~11.6 utterances in dataset)
  * **Score A / B** — CaSiNo points from the NegMAS outcome
  * **GT Score A / B** — CaSiNo points from the human deal
  * **Mean Absolute Delta** — |NegMAS_score - GT_score| per agent
  * **Pareto efficiency** — is the NegMAS outcome Pareto-optimal?
  * **Nash efficiency** — nash_product / best_possible_nash_product

Usage::

    # from semantic_alignment/
    python -m evaluation.casino.eval_negotiation \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --limit 100 \\
        --strategy BoulwareTBNegotiator \\
        --n-steps 50
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# ── sys.path: ensure agent root is importable ─────────────────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from ...app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationParticipant,
)  # noqa: E402
from ..casino.callback_agent import (  # noqa: E402
    CasinoCallbackAgent,
    start_casino_agent_server,
    wait_for_server,
)
from ..casino.loader import (  # noqa: E402
    CASINO_OPTIONS,
    ISSUES,
    CasinoDialogue,
    load_casino,
    to_negotiation_participant,
)
from ..casino.metrics import (  # noqa: E402
    aggregate_negotiation_metrics,
    best_possible_nash,
    casino_points,
    nash_product,
    pareto_efficient,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default options available to NegMAS for each issue.
_OPTIONS_PER_ISSUE: Dict[str, List[str]] = {issue: CASINO_OPTIONS for issue in ISSUES}


# ─────────────────────────────────────────────────────────────────────────────


def evaluate_negotiation(
    dialogues: List[CasinoDialogue],
    *,
    strategy: str = "BoulwareTBNegotiator",
    n_steps: int = 50,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run Phase 3 evaluation over *dialogues*.

    Args:
        dialogues: Loaded CaSiNo dialogues.
        strategy: NegMAS negotiator class name passed to :class:`NegotiationModel`.
        n_steps: Maximum SAO rounds per negotiation session.
        limit: Cap the number of dialogues evaluated (``None`` = all).
        verbose: Print per-dialogue results to stdout.

    Returns:
        Aggregate metrics dict from
        :func:`~evaluation.casino.metrics.aggregate_negotiation_metrics`.
    """
    subset = dialogues[:limit] if limit else dialogues
    model = NegotiationModel(n_steps=n_steps, strategy=strategy)

    per_dialogue: List[dict] = []
    n_errors = 0

    for i, dlg in enumerate(subset):
        # Progress heartbeat every 100 dialogues
        if (i + 0) % 100 == 0:
            print(f"  … {i}/{len(subset)} dialogues processed", flush=True)

        # Build NegotiationParticipants from CaSiNo preference data
        participant_a: NegotiationParticipant = to_negotiation_participant(
            dlg.agent1, is_agent_a=True
        )
        participant_b: NegotiationParticipant = to_negotiation_participant(
            dlg.agent2, is_agent_a=False
        )

        # Ground-truth CaSiNo scores
        gt_alloc = dlg.deal_agent1 or {issue: 0 for issue in ISSUES}
        gt_score_a, gt_score_b = casino_points(
            gt_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
        )
        best_nash = best_possible_nash(dlg.agent1.value2issue, dlg.agent2.value2issue)

        row: dict = {
            "dialogue_id": dlg.dialogue_id,
            "agreed": False,
            "steps": None,
            "score_a": None,
            "score_b": None,
            "gt_score_a": gt_score_a,
            "gt_score_b": gt_score_b,
            "pareto": None,
            "nash_product": None,
            "best_nash": best_nash,
        }

        try:
            result = model.run(
                issues=ISSUES,
                options_per_issue=_OPTIONS_PER_ISSUE,
                participants=[participant_a, participant_b],
                session_id=f"eval-casino-{dlg.dialogue_id}",
            )

            row["steps"] = result.steps

            if result.agreement is not None:
                # Convert NegMAS agreement → allocation dict
                negmas_alloc: Dict[str, int] = {
                    outcome.issue_id: int(outcome.chosen_option)
                    for outcome in result.agreement
                }
                score_a, score_b = casino_points(
                    negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                )
                is_pareto = pareto_efficient(
                    negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                )
                np_val = nash_product(
                    negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                )

                row.update(
                    {
                        "agreed": True,
                        "score_a": score_a,
                        "score_b": score_b,
                        "pareto": is_pareto,
                        "nash_product": np_val,
                        "negmas_allocation": negmas_alloc,
                        "gt_allocation": gt_alloc,
                    }
                )

                if verbose:
                    pe = "✓ Pareto" if is_pareto else "✗ Pareto"
                    print(
                        f"  ✓ [{pe}] dialogue {dlg.dialogue_id:4d} | "
                        f"steps={result.steps:3d} | "
                        f"NegMAS: A={score_a:.0f} B={score_b:.0f} | "
                        f"GT: A={gt_score_a:.0f} B={gt_score_b:.0f} | "
                        f"alloc={negmas_alloc}"
                    )
            else:
                if verbose:
                    reason = "timeout" if result.timedout else "broken"
                    print(
                        f"  ✗ [{reason}] dialogue {dlg.dialogue_id:4d} | "
                        f"steps={result.steps:3d} | "
                        f"GT: A={gt_score_a:.0f} B={gt_score_b:.0f}"
                    )

        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            if verbose:
                print(f"  ERROR dialogue {dlg.dialogue_id}: {exc}")
                traceback.print_exc()

        per_dialogue.append(row)

    if n_errors:
        print(f"  ⚠  {n_errors} dialogue(s) raised exceptions and were skipped.")

    agg = aggregate_negotiation_metrics(per_dialogue)
    agg["strategy"] = strategy
    agg["n_steps"] = n_steps
    agg["n_errors"] = n_errors
    return agg


def evaluate_negotiation_callback(
    dialogues: List[CasinoDialogue],
    *,
    n_steps: int = 50,
    port: int = 8093,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run Phase 3 evaluation over the **production callback path** (BatchCallbackRunner).

    Two :class:`~evaluation.casino.callback_agent.CasinoCallbackAgent` instances
    are created per dialogue from the CaSiNo preference data.  They are swapped
    into a shared in-process FastAPI server (running on *port*) which the
    ``BatchCallbackRunner`` contacts on every SAO round via real HTTP calls.

    The agent server calls :func:`~app.agent.batch_callback_runner.store_decisions`
    directly instead of POSTing to a negotiation server, so no external process
    is required.

    Metrics are identical to :func:`evaluate_negotiation` and the JSON output
    format is the same — results are directly comparable.

    Args:
        dialogues: Loaded CaSiNo dialogues.
        n_steps: Maximum SAO rounds per negotiation session.
        port: TCP port for the mock agent server (default 8093).
        limit: Cap the number of dialogues evaluated (``None`` = all).
        verbose: Print per-dialogue results to stdout.

    Returns:
        Aggregate metrics dict (same keys as :func:`evaluate_negotiation`).
    """
    subset = dialogues[:limit] if limit else dialogues

    # Shared mutable registry — swapped per dialogue, no server restart needed.
    agents_registry: dict[str, CasinoCallbackAgent] = {}
    server, _ = start_casino_agent_server(agents_registry, port=port)
    wait_for_server(port)
    callback_url = f"http://localhost:{port}/decide"
    print(f"  CaSiNo agent server listening on {callback_url}", flush=True)

    model = NegotiationModel(n_steps=n_steps, strategy=None)
    per_dialogue: List[dict] = []
    n_errors = 0

    try:
        for i, dlg in enumerate(subset):
            if i % 100 == 0:
                print(f"  … {i}/{len(subset)} dialogues processed", flush=True)

            # Build CaSiNo-weighted agents and register them
            agent_a = CasinoCallbackAgent(dlg.agent1, is_agent_a=True)
            agent_b = CasinoCallbackAgent(dlg.agent2, is_agent_a=False)
            agents_registry.clear()
            agents_registry[dlg.agent1.agent_id] = agent_a
            agents_registry[dlg.agent2.agent_id] = agent_b

            participant_a = NegotiationParticipant(
                id=dlg.agent1.agent_id,
                name=dlg.agent1.agent_id,
            )
            participant_b = NegotiationParticipant(
                id=dlg.agent2.agent_id,
                name=dlg.agent2.agent_id,
            )

            gt_alloc = dlg.deal_agent1 or {issue: 0 for issue in ISSUES}
            gt_score_a, gt_score_b = casino_points(
                gt_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
            )
            best_nash = best_possible_nash(
                dlg.agent1.value2issue, dlg.agent2.value2issue
            )

            row: dict = {
                "dialogue_id": dlg.dialogue_id,
                "agreed": False,
                "steps": None,
                "score_a": None,
                "score_b": None,
                "gt_score_a": gt_score_a,
                "gt_score_b": gt_score_b,
                "pareto": None,
                "nash_product": None,
                "best_nash": best_nash,
            }

            try:
                from ...app.agent.batch_callback_runner import BatchCallbackRunner as _BCR

                _runner = _BCR(n_steps=n_steps)
                result = _runner.run(
                    issues=ISSUES,
                    options_per_issue=_OPTIONS_PER_ISSUE,
                    participants=[participant_a, participant_b],
                    session_id=f"eval-casino-cb-{dlg.dialogue_id}",
                    agent_url=callback_url,
                )

                row["steps"] = result.steps

                if result.agreement is not None:
                    negmas_alloc: Dict[str, int] = {
                        outcome.issue_id: int(outcome.chosen_option)
                        for outcome in result.agreement
                    }
                    score_a, score_b = casino_points(
                        negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                    )
                    is_pareto = pareto_efficient(
                        negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                    )
                    np_val = nash_product(
                        negmas_alloc, dlg.agent1.value2issue, dlg.agent2.value2issue
                    )
                    row.update(
                        {
                            "agreed": True,
                            "score_a": score_a,
                            "score_b": score_b,
                            "pareto": is_pareto,
                            "nash_product": np_val,
                            "negmas_allocation": negmas_alloc,
                            "gt_allocation": gt_alloc,
                            "round_decisions": result.round_decisions,
                        }
                    )
                    if verbose:
                        pe = "✓ Pareto" if is_pareto else "✗ Pareto"
                        print(
                            f"  ✓ [{pe}] dialogue {dlg.dialogue_id:4d} | "
                            f"steps={result.steps:3d} | "
                            f"NegMAS: A={score_a:.0f} B={score_b:.0f} | "
                            f"GT: A={gt_score_a:.0f} B={gt_score_b:.0f} | "
                            f"alloc={negmas_alloc}"
                        )
                else:
                    if verbose:
                        reason = "timeout" if result.timedout else "broken"
                        print(
                            f"  ✗ [{reason}] dialogue {dlg.dialogue_id:4d} | "
                            f"steps={result.steps:3d} | "
                            f"GT: A={gt_score_a:.0f} B={gt_score_b:.0f}"
                        )

            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                if verbose:
                    print(f"  ERROR dialogue {dlg.dialogue_id}: {exc}")
                    traceback.print_exc()

            per_dialogue.append(row)

    finally:
        server.should_exit = True

    if n_errors:
        print(f"  ⚠  {n_errors} dialogue(s) raised exceptions and were skipped.")

    agg = aggregate_negotiation_metrics(per_dialogue)
    agg["strategy"] = "CasinoCallbackAgent(Boulware)"
    agg["n_steps"] = n_steps
    agg["n_errors"] = n_errors
    agg["callback_port"] = port
    return agg


def _print_results(results: dict) -> None:
    n = results["n_dialogues"]
    n_agreed = results["n_agreed"]
    print("\n" + "=" * 65)
    print("PHASE 3 — Negotiation Model Results")
    print("=" * 65)
    print(f"  Strategy            : {results.get('strategy', 'N/A')}")
    print(f"  Max steps           : {results.get('n_steps', 'N/A')}")
    print(f"  Dialogues evaluated : {n}")
    print(f"  Errors              : {results.get('n_errors', 0)}")
    print()
    print(f"  Agreement rate      : {results['agreement_rate']:.1%}  ({n_agreed}/{n})")
    if results.get("avg_steps") is not None:
        print(
            f"  Avg steps           : {results['avg_steps']:.1f}  (dataset avg ~11.6 utterances)"
        )
    print()
    print("  Scores (agreed dialogues only):")
    if results.get("avg_score_a") is not None:
        print(
            f"    NegMAS  A={results['avg_score_a']:.1f}  "
            f"B={results['avg_score_b']:.1f}  "
            f"joint={results['avg_joint_score']:.1f}"
        )
    print(
        f"    GT      A={results['gt_avg_score_a']:.1f}  "
        f"B={results['gt_avg_score_b']:.1f}  "
        f"joint={results['gt_avg_joint_score']:.1f}"
    )
    if results.get("mean_abs_delta_a") is not None:
        print(
            f"    MAD     A={results['mean_abs_delta_a']:.1f}  "
            f"B={results['mean_abs_delta_b']:.1f}"
        )
    print()
    if results.get("pareto_rate") is not None:
        print(f"  Pareto efficiency   : {results['pareto_rate']:.1%}")
    if results.get("avg_nash_efficiency") is not None:
        print(f"  Avg Nash efficiency : {results['avg_nash_efficiency']:.1%}")
    print("=" * 65)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NegotiationModel (Phase 3) against CaSiNo."
    )
    parser.add_argument(
        "--casino-path",
        required=True,
        metavar="PATH",
        help="Path to casino.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N dialogues (default: all).",
    )
    parser.add_argument(
        "--strategy",
        default="BoulwareTBNegotiator",
        metavar="CLASS",
        help="NegMAS SAO negotiator class name (default: BoulwareTBNegotiator).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        metavar="N",
        help="Maximum SAO rounds per session (default: 50).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-dialogue results.",
    )
    parser.add_argument(
        "--callback",
        action="store_true",
        default=False,
        help=(
            "Use the production BatchCallbackRunner path with mock CaSiNo SSTP agents "
            "instead of the internal NegMAS path.  Tests the real HTTP callback flow."
        ),
    )
    parser.add_argument(
        "--callback-port",
        type=int,
        default=8093,
        metavar="PORT",
        help="TCP port for the mock agent server when --callback is set (default: 8093).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write JSON results to FILE.",
    )
    args = parser.parse_args(argv)

    print(f"Loading CaSiNo from {args.casino_path} …")
    dialogues = load_casino(args.casino_path)
    print(f"Loaded {len(dialogues)} dialogues.")

    if args.callback:
        print(
            f"Mode: callback (BatchCallbackRunner)  |  n_steps: {args.n_steps}  |  port: {args.callback_port}"
        )
        print("Starting mock SSTP agent server …")
        results = evaluate_negotiation_callback(
            dialogues,
            n_steps=args.n_steps,
            port=args.callback_port,
            limit=args.limit,
            verbose=args.verbose,
        )
    else:
        print(
            f"Mode: internal NegMAS  |  strategy: {args.strategy}  |  n_steps: {args.n_steps}"
        )
        print("Running NegMAS negotiations …")
        results = evaluate_negotiation(
            dialogues,
            strategy=args.strategy,
            n_steps=args.n_steps,
            limit=args.limit,
            verbose=args.verbose,
        )
    _print_results(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
