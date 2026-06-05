# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Metric helpers shared across all three CaSiNo evaluation phases.

All functions are pure (no I/O, no side effects) so they can be unit-tested
independently and composed freely in the eval scripts.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Set, Tuple

from ..casino.loader import ISSUES, PRIORITY_SCORES, TOTAL_PACKAGES

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Intent Discovery metrics
# ─────────────────────────────────────────────────────────────────────────────

_GOLD_ISSUES: Set[str] = {"food", "water", "firewood"}


def issue_f1_metrics(predicted: List[str]) -> Dict:
    """Compute per-issue and aggregate F1 for a single dialogue prediction.

    Both *predicted* and the gold set (always ``{"food","water","firewood"}``)
    are lower-cased before comparison.

    Returns a dict with keys:
    ``per_issue``, ``micro_f1``, ``macro_f1``, ``exact_match``,
    ``precision``, ``recall``.
    """
    pred_set: Set[str] = {p.lower().strip() for p in predicted}
    gold_set: Set[str] = _GOLD_ISSUES

    per_issue: Dict[str, Dict[str, float]] = {}
    for issue in sorted(gold_set):
        tp = 1 if issue in pred_set else 0
        fp = 1 if issue in pred_set and issue not in gold_set else 0
        fn = 1 if issue not in pred_set and issue in gold_set else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_issue[issue] = {"precision": prec, "recall": rec, "f1": f1}

    # Micro-F1: aggregate TP/FP/FN across all issues
    tp_total = len(pred_set & gold_set)
    fp_total = len(pred_set - gold_set)
    fn_total = len(gold_set - pred_set)
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    # Macro-F1: mean of per-issue F1s
    macro_f1 = sum(v["f1"] for v in per_issue.values()) / len(per_issue)

    return {
        "per_issue": per_issue,
        "precision": round(micro_prec, 4),
        "recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match": pred_set == gold_set,
    }


def aggregate_intent_metrics(per_dialogue: List[Dict]) -> Dict:
    """Average intent metrics across dialogues.

    Args:
        per_dialogue: List of dicts returned by :func:`issue_f1_metrics`.

    Returns:
        Dict with ``avg_micro_f1``, ``avg_macro_f1``, ``exact_match_rate``
        and per-issue averaged precision/recall/F1.
    """
    n = len(per_dialogue)
    if n == 0:
        return {}

    avg_micro = sum(d["micro_f1"] for d in per_dialogue) / n
    avg_macro = sum(d["macro_f1"] for d in per_dialogue) / n
    em_rate = sum(1 for d in per_dialogue if d["exact_match"]) / n

    per_issue: Dict[str, Dict[str, float]] = {}
    for issue in sorted(_GOLD_ISSUES):
        per_issue[issue] = {
            "avg_precision": sum(d["per_issue"][issue]["precision"] for d in per_dialogue) / n,
            "avg_recall": sum(d["per_issue"][issue]["recall"] for d in per_dialogue) / n,
            "avg_f1": sum(d["per_issue"][issue]["f1"] for d in per_dialogue) / n,
        }

    return {
        "n_dialogues": n,
        "avg_micro_f1": round(avg_micro, 4),
        "avg_macro_f1": round(avg_macro, 4),
        "exact_match_rate": round(em_rate, 4),
        "per_issue": per_issue,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Options Generation metrics
# ─────────────────────────────────────────────────────────────────────────────


def is_valid_casino_option(opt: str) -> bool:
    """Return True if *opt* is an integer string in ``{"0","1","2","3"}``."""
    try:
        return 0 <= int(opt) <= TOTAL_PACKAGES
    except (ValueError, TypeError):
        return False


def options_validity_rate(options_per_issue: Dict[str, List[str]]) -> float:
    """Fraction of generated options that are valid CaSiNo quantities (0–3).

    Args:
        options_per_issue: ``{issue: [option, ...]}`` returned by
            :class:`OptionsGeneration.generate`.

    Returns:
        Value in ``[0.0, 1.0]``.
    """
    all_opts = [opt for opts in options_per_issue.values() for opt in opts]
    if not all_opts:
        return 0.0
    return sum(is_valid_casino_option(o) for o in all_opts) / len(all_opts)


def deal_coverage(
    deal_agent1: Dict[str, int],
    options_per_issue: Dict[str, List[str]],
) -> Dict[str, bool]:
    """Check whether each ground-truth deal value is present in the generated options.

    Args:
        deal_agent1: ``{issue: quantity}`` for Agent 1 (from the dataset).
        options_per_issue: ``{issue: [option, ...]}`` from OptionsGeneration.

    Returns:
        ``{issue: True/False}`` — True means the agreed quantity for that issue
        is covered by the generated options.
    """
    coverage: Dict[str, bool] = {}
    for issue in ISSUES:
        qty = deal_agent1.get(issue)
        opts = options_per_issue.get(issue, [])
        # Match as string OR integer representation
        coverage[issue] = str(qty) in opts or qty in opts  # type: ignore[operator]
    return coverage


def aggregate_options_metrics(per_dialogue: List[Dict]) -> Dict:
    """Average options metrics across dialogues.

    Each element of *per_dialogue* should have keys:
    ``validity_rate``, ``coverage`` (dict per issue), and optionally
    ``has_deal`` (bool).  Coverage is only computed over dialogues that
    have a ground-truth deal; no-deal dialogues are excluded from that
    denominator so they don't artificially drag the rate down.
    """
    n = len(per_dialogue)
    if n == 0:
        return {}

    avg_validity = sum(d["validity_rate"] for d in per_dialogue) / n

    # Coverage: only dialogues with an actual deal are meaningful
    deal_dialogues = [d for d in per_dialogue if d.get("has_deal", True)]
    n_deal = len(deal_dialogues)

    per_issue_coverage: Dict[str, float] = {}
    for issue in ISSUES:
        if n_deal:
            per_issue_coverage[issue] = (
                sum(1 for d in deal_dialogues if d["coverage"].get(issue, False)) / n_deal
            )
        else:
            per_issue_coverage[issue] = 0.0

    overall_coverage = (
        sum(per_issue_coverage.values()) / len(per_issue_coverage)
        if per_issue_coverage else 0.0
    )

    return {
        "n_dialogues": n,
        "n_deal_dialogues": n_deal,
        "avg_validity_rate": round(avg_validity, 4),
        "per_issue_coverage": {k: round(v, 4) for k, v in per_issue_coverage.items()},
        "overall_coverage_rate": round(overall_coverage, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Negotiation Model metrics
# ─────────────────────────────────────────────────────────────────────────────


def casino_points(
    allocation_a: Dict[str, int],
    value2issue_a: Dict[str, str],
    value2issue_b: Dict[str, str],
) -> Tuple[float, float]:
    """Compute CaSiNo points for both agents given Agent A's allocation.

    Args:
        allocation_a: ``{issue: quantity}`` for Agent A (0–3 per issue).
        value2issue_a: Agent A's ``{priority: issue}`` mapping from the dataset.
        value2issue_b: Agent B's ``{priority: issue}`` mapping from the dataset.

    Returns:
        ``(points_a, points_b)`` — raw CaSiNo point totals.
    """
    issue2priority_a: Dict[str, str] = {v.lower(): k for k, v in value2issue_a.items()}
    issue2priority_b: Dict[str, str] = {v.lower(): k for k, v in value2issue_b.items()}

    pts_a = sum(
        PRIORITY_SCORES.get(issue2priority_a.get(issue, "Low"), 3) * qty
        for issue, qty in allocation_a.items()
    )
    pts_b = sum(
        PRIORITY_SCORES.get(issue2priority_b.get(issue, "Low"), 3) * (TOTAL_PACKAGES - qty)
        for issue, qty in allocation_a.items()
    )
    return float(pts_a), float(pts_b)


def pareto_efficient(
    allocation_a: Dict[str, int],
    value2issue_a: Dict[str, str],
    value2issue_b: Dict[str, str],
) -> bool:
    """Return True if *allocation_a* is Pareto-efficient.

    Enumerates all ``4^3 = 64`` feasible allocations (each issue independently
    0–3 for Agent A) and checks whether any alternative weakly dominates on
    both agents' scores with at least one strict improvement.

    Args:
        allocation_a: Agent A's quantities (``{issue: int}``).
        value2issue_a: Agent A's ``{priority: issue}`` map.
        value2issue_b: Agent B's ``{priority: issue}`` map.
    """
    pts_a, pts_b = casino_points(allocation_a, value2issue_a, value2issue_b)

    for combo in product(range(TOTAL_PACKAGES + 1), repeat=len(ISSUES)):
        alt_a = dict(zip(ISSUES, combo))
        alt_pts_a, alt_pts_b = casino_points(alt_a, value2issue_a, value2issue_b)
        # Does this alternative dominate?
        if alt_pts_a >= pts_a and alt_pts_b >= pts_b and (alt_pts_a > pts_a or alt_pts_b > pts_b):
            return False
    return True


def nash_product(
    allocation_a: Dict[str, int],
    value2issue_a: Dict[str, str],
    value2issue_b: Dict[str, str],
    disagreement: Tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the Nash bargaining product for an allocation.

    ``(pts_a - d_a) * (pts_b - d_b)`` where ``(d_a, d_b)`` is the
    disagreement point (default 0,0).  Higher is better.
    """
    pts_a, pts_b = casino_points(allocation_a, value2issue_a, value2issue_b)
    return max(0.0, pts_a - disagreement[0]) * max(0.0, pts_b - disagreement[1])


def best_possible_nash(
    value2issue_a: Dict[str, str],
    value2issue_b: Dict[str, str],
) -> float:
    """Return the maximum Nash product over all 64 feasible allocations."""
    best = 0.0
    for combo in product(range(TOTAL_PACKAGES + 1), repeat=len(ISSUES)):
        alloc = dict(zip(ISSUES, combo))
        best = max(best, nash_product(alloc, value2issue_a, value2issue_b))
    return best


def aggregate_negotiation_metrics(per_dialogue: List[Dict]) -> Dict:
    """Summarise Phase 3 metrics across dialogues.

    Each element of *per_dialogue* should have keys: ``agreed``,
    ``steps`` (Optional[int]), ``score_a``, ``score_b``, ``gt_score_a``,
    ``gt_score_b``, ``pareto`` (Optional[bool]), ``nash_product`` (Optional[float]),
    ``best_nash`` (Optional[float]).
    """
    n = len(per_dialogue)
    if n == 0:
        return {}

    agreed = [d for d in per_dialogue if d["agreed"]]
    n_agreed = len(agreed)
    agreement_rate = n_agreed / n

    avg_steps: Optional[float] = (
        sum(d["steps"] for d in agreed) / n_agreed if agreed else None
    )
    avg_score_a: Optional[float] = (
        sum(d["score_a"] for d in agreed) / n_agreed if agreed else None
    )
    avg_score_b: Optional[float] = (
        sum(d["score_b"] for d in agreed) / n_agreed if agreed else None
    )
    avg_joint: Optional[float] = (
        sum(d["score_a"] + d["score_b"] for d in agreed) / n_agreed if agreed else None
    )
    gt_avg_joint = sum(d["gt_score_a"] + d["gt_score_b"] for d in per_dialogue) / n
    gt_avg_score_a = sum(d["gt_score_a"] for d in per_dialogue) / n
    gt_avg_score_b = sum(d["gt_score_b"] for d in per_dialogue) / n

    pareto_rate: Optional[float] = (
        sum(1 for d in agreed if d.get("pareto")) / n_agreed if agreed else None
    )

    # Mean absolute score delta vs ground truth (only for agreed dialogues)
    mad_a: Optional[float] = (
        sum(abs(d["score_a"] - d["gt_score_a"]) for d in agreed) / n_agreed if agreed else None
    )
    mad_b: Optional[float] = (
        sum(abs(d["score_b"] - d["gt_score_b"]) for d in agreed) / n_agreed if agreed else None
    )

    # Nash efficiency = nash_product / best_possible_nash
    nash_efficiencies = [
        d["nash_product"] / d["best_nash"]
        for d in agreed
        if d.get("nash_product") is not None and d.get("best_nash", 0) > 0
    ]
    avg_nash_efficiency: Optional[float] = (
        sum(nash_efficiencies) / len(nash_efficiencies) if nash_efficiencies else None
    )

    return {
        "n_dialogues": n,
        "n_agreed": n_agreed,
        "agreement_rate": round(agreement_rate, 4),
        "avg_steps": round(avg_steps, 2) if avg_steps is not None else None,
        "avg_score_a": round(avg_score_a, 2) if avg_score_a is not None else None,
        "avg_score_b": round(avg_score_b, 2) if avg_score_b is not None else None,
        "avg_joint_score": round(avg_joint, 2) if avg_joint is not None else None,
        "gt_avg_score_a": round(gt_avg_score_a, 2),
        "gt_avg_score_b": round(gt_avg_score_b, 2),
        "gt_avg_joint_score": round(gt_avg_joint, 2),
        "mean_abs_delta_a": round(mad_a, 2) if mad_a is not None else None,
        "mean_abs_delta_b": round(mad_b, 2) if mad_b is not None else None,
        "pareto_rate": round(pareto_rate, 4) if pareto_rate is not None else None,
        "avg_nash_efficiency": round(avg_nash_efficiency, 4) if avg_nash_efficiency is not None else None,
    }
