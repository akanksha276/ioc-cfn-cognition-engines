# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""eval_pipeline.py — LLM pipeline evaluation + negotiation outcome benchmarking.

Covers two complementary evaluation dimensions:

1. **Pipeline quality** — evaluates IntentDiscovery + OptionsGeneration against
   a gold dataset using an LLM judge.
2. **Negotiation outcomes** — aggregates agreement rates, round counts, and
   real LLM token/cost metrics from a ``run_log.json`` produced by GOAL A.

Both dimensions can be run together or independently.

Gold dataset schema
-------------------
.. code-block:: json

    [
      {
        "id": "sample_01",
        "sentence": "...",
        "context": "...",
        "domain": "...",
        "gold_issues": ["issue1", "issue2", ...],
        "gold_options": {
          "issue1": ["interp_a", "interp_b", "interp_c"],
          "issue2": [...]
        },
        "metadata": {"difficulty": "hard", ...}
      }
    ]

.. note::
    ``gold_interpretations`` is accepted as an alias for ``gold_options`` for
    backward compatibility with older dataset versions.

Compatible datasets
-------------------
* ``evaluation/framework/ground_truth/hard_convergence_direct_5.json``
* ``evaluation/framework/ground_truth/example_usecases_direct_9_gold.json``
* Any future dataset using this schema.

Pipeline evaluation (``--dataset``)
-------------------------------------
**Phase 1 — Intent Discovery**
  ``IntentDiscovery.discover(sentence, context)`` → ``predicted_entities``.
  LLM judge checks each gold issue against predicted entities.
  Metrics: ``pipeline_recall``, ``intent_precision``, micro/macro F1.

**Phase 2 — Options Generation**
  For each covered gold issue, ``OptionsGeneration.generate_options_llm_only(...)``
  → ``options_per_issue``.  LLM judge per covered issue:

  * For each gold option: covered by any generated option? → recall
  * For each generated option: semantically useful? → precision

  Metrics: per-issue P/R/F1, micro/macro P/R/F1, full-coverage rate.

Pipeline evaluation modes
--------------------------
**Online** (default): calls IntentDiscovery + OptionsGeneration LLMs live.
  Requires full service stack.

**Offline** (``--trace-dir PATH``): reads ``01_initiate_response.json`` files
  from a prior GOAL A run.  Only the judge LLM is called.

Negotiation outcome evaluation (``--run-log``)
------------------------------------------------
Reads ``run_log.json`` written by
``test_via_semantic_neg_agents_configured.py``. No additional LLM calls.

Metrics computed:

* ``agreement_rate`` / ``timeout_rate`` / ``broken_rate``
* ``avg_rounds_to_agreement``, ``avg_rounds_all``, ``avg_duration_s``
* ``llm_calls_total`` (initiate fixed at 2/mission + actual decide calls)
* ``decide_prompt_tokens``, ``decide_completion_tokens``,
  ``decide_estimated_cost_usd`` — captured via ``litellm.success_callback``
  in the test script; only decide-phase (agent) calls are measured;
  initiate-phase calls run in the neg server process and are estimated.

Output formats
--------------
**Pipeline JSON** (``--output FILE``)::

    {
      "scorer": "llm",
      "judge_model": "...",
      "generator_model": "...",
      "overall": { "n_samples": ..., "micro_precision": ..., ... },
      "by_domain":     { "<domain>": { ... }, ... },
      "by_difficulty": { "<level>":  { ... }, ... },
      "negotiation": {
        "agreement_rate": 0.80,
        "avg_rounds_to_agreement": 47.5,
        "llm_calls_total": 722,
        "decide_total_tokens": 198400,
        "decide_estimated_cost_usd": 0.61,
        "missions": [ ... ]
      },
      "samples": [ ... ]
    }

**Pipeline CSV** (``--csv FILE``): one row per sample + OVERALL row.

**Negotiation CSV** (``--neg-csv FILE``): one row per mission + OVERALL row.
Columns: ``mission, agreed, verdict, status, total_rounds, n_agents,
llm_calls_initiate, llm_calls_decide, llm_calls_total, prompt_tokens,
completion_tokens, total_tokens, estimated_cost_usd, duration_s, session_id,
deal_issue_<n>, deal_option_<n>``.

Usage
-----
::

    # Online pipeline eval:
    poetry run python -m evaluation.eval_pipeline \\
        --dataset evaluation/framework/ground_truth/hard_convergence_direct_5.json \\
        --output  results/eval_hard5.json --verbose

    # Offline pipeline eval from a prior run:
    poetry run python -m evaluation.eval_pipeline \\
        --dataset   evaluation/framework/ground_truth/hard_convergence_direct_5.json \\
        --trace-dir neg_trace/20260403_143022 \\
        --output    results/eval_hard5_offline.json \\
        --csv       results/eval_hard5_offline.csv --verbose

    # Negotiation outcomes only (no judge LLM calls):
    poetry run python -m evaluation.eval_pipeline \\
        --run-log  neg_trace/20260403_143022/run_log.json \\
        --output   results/neg_report.json \\
        --neg-csv  results/neg_report.csv

    # Combined pipeline + negotiation:
    poetry run python -m evaluation.eval_pipeline \\
        --dataset   evaluation/framework/ground_truth/hard_convergence_direct_5.json \\
        --trace-dir neg_trace/20260403_143022 \\
        --run-log   neg_trace/20260403_143022/run_log.json \\
        --output    results/full_report.json \\
        --csv       results/pipeline_report.csv \\
        --neg-csv   results/neg_report.csv --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ── sys.path: repo root + semantic_negotiation/app must be importable ────────
_repo_root = str(Path(__file__).resolve().parents[2])  # ioc-cfn-cognitive-agents/
_sn_root = str(Path(__file__).resolve().parents[1])    # semantic_negotiation/
_sn_app = str(Path(__file__).resolve().parents[1] / "app")  # semantic_negotiation/app
for _p in (_repo_root, _sn_root, _sn_app):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from semantic_negotiation/ before importing settings/litellm
_env_file = Path(__file__).resolve().parents[1] / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)

import litellm  # noqa: E402

litellm.drop_params = True  # allow model-specific unsupported params (e.g. temperature for gpt-5)

from app.agent.intent_discovery import IntentDiscovery  # noqa: E402
from app.agent.options_generation import OptionsGeneration  # noqa: E402
from app.config.settings import settings  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("eval_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a gold-schema JSON dataset and normalise field names.

    Accepts both ``gold_options`` (canonical) and ``gold_interpretations``
    (legacy alias) — ``gold_options`` takes precedence when both are present.

    Each returned dict is guaranteed to have:
    ``id``, ``sentence``, ``context``, ``domain``, ``difficulty``,
    ``gold_issues``, ``gold_options``.
    """
    raw: List[Dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    normalised: List[Dict[str, Any]] = []
    for entry in raw:
        # Prefer gold_options; fall back to gold_interpretations (legacy)
        gi = entry.get("gold_options") or entry.get("gold_interpretations") or {}
        meta = entry.get("metadata") or {}
        normalised.append(
            {
                "id": entry.get("id", ""),
                "sentence": entry.get("sentence", ""),
                "context": entry.get("context", ""),
                "domain": entry.get("domain", "unknown"),
                "difficulty": meta.get("difficulty") or entry.get("difficulty", "unknown"),
                "gold_issues": list(entry.get("gold_issues", [])),
                "gold_options": {k: list(v) for k, v in gi.items()},
            }
        )
    return normalised


# ─────────────────────────────────────────────────────────────────────────────# Offline mode — load saved run traces
# ─────────────────────────────────────────────────────────────────────────


def load_trace_dir(trace_dir: str) -> Dict[str, Dict[str, Any]]:
    """Scan a ``test_via_semantic_neg_agents.py`` run directory for initiate responses.

    Walks ``<trace_dir>/*/01_initiate_response.json`` (SSTP format) and
    ``<trace_dir>/*/00_start_response.json`` (CFN /start format) and returns:
    ``{ mission_slug: {"issues": [...], "options_per_issue": {...}} }``.

    The mission slug is the subdirectory name (already slugified by the script).
    """
    base = Path(trace_dir)
    traces: Dict[str, Dict[str, Any]] = {}
    # Format 1: SSTP initiate responses (payload-wrapped)
    for json_file in sorted(base.glob("*/01_initiate_response.json")):
        mission_slug = json_file.parent.name
        try:
            envelope = json.loads(json_file.read_text(encoding="utf-8"))
            payload = envelope.get("payload") or {}
            traces[mission_slug] = {
                "issues": payload.get("issues") or [],
                "options_per_issue": payload.get("options_per_issue") or {},
                "source_file": str(json_file),
            }
        except Exception as exc:
            logger.warning("Could not read %s: %s", json_file, exc)
    # Format 2: CFN /start responses (top-level keys)
    for json_file in sorted(base.glob("*/00_start_response.json")):
        mission_slug = json_file.parent.name
        if mission_slug in traces:
            continue  # prefer 01_initiate_response if both exist
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            traces[mission_slug] = {
                "issues": data.get("issues") or [],
                "options_per_issue": data.get("options_per_issue") or {},
                "source_file": str(json_file),
            }
        except Exception as exc:
            logger.warning("Could not read %s: %s", json_file, exc)
    return traces


def _entry_to_slug(entry_id: str) -> str:
    """Convert a gold dataset entry ``id`` (e.g. ``hard_01``) to the trace slug
    format used by ``test_via_semantic_neg_agents.py``.

    Tries common patterns:
    * Direct match
    * ``<prefix>_<nn>`` → ``hard_0n_*`` prefix scan
    """
    import re
    return re.sub(r"[^a-z0-9]+", "_", entry_id.lower()).strip("_")


def _match_trace(
    entry: Dict[str, Any],
    traces: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the trace entry that corresponds to *entry* from the gold dataset.

    Matching strategy (first match wins):
    1. Exact slug match:   ``entry["id"]`` slug == trace key
    2. Prefix match:       trace key starts with entry id slug
    3. Sentence substring: entry sentence appears in trace source filename path
    """
    entry_slug = _entry_to_slug(entry["id"])
    # 1. exact
    if entry_slug in traces:
        return traces[entry_slug]
    # 2. prefix: entry slug is a prefix of the trace key
    for key, trace in traces.items():
        if key.startswith(entry_slug):
            return trace
    # 3. partial: entry slug contained in trace key (handles hard_01 vs hard_01_ai_model...)
    for key, trace in traces.items():
        if entry_slug in key:
            return trace
    return None


def evaluate_sample_offline(
    entry: Dict[str, Any],
    trace: Dict[str, Any],
    judge_model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Score a single gold dataset entry against a *saved* initiate response trace.

    Unlike :func:`evaluate_sample`, this function skips all generator LLM calls
    (IntentDiscovery + OptionsGeneration) and reads the already-produced
    ``issues`` and ``options_per_issue`` from the trace instead.

    Phase 1 precision/recall are computed by matching trace issues against gold
    issues via the LLM judge (same as online mode).  Phase 2 is scored
    identically to online mode.

    Args:
        entry: Normalised gold-schema dict (from :func:`load_dataset`).
        trace: Dict with ``issues`` and ``options_per_issue`` from the trace.
        judge_model: LiteLLM model string for the judge.
        verbose: Print per-issue progress to stdout.

    Returns:
        Same schema as :func:`evaluate_sample`, with
        ``eval_mode="offline"`` added.
    """
    sample_id = entry["id"]
    sentence = entry["sentence"]
    gold_issues: List[str] = entry["gold_issues"]
    gold_options: Dict[str, List[str]] = entry.get("gold_options") or entry.get("gold_interpretations", {})

    predicted_entities: List[str] = trace.get("issues") or []
    gen_options_map: Dict[str, List[str]] = trace.get("options_per_issue") or {}

    if verbose:
        print(f"\n  [{sample_id}] {sentence[:90]}{'\u2026' if len(sentence) > 90 else ''}")
        print(f"    Trace issues ({len(predicted_entities)}): {predicted_entities}")

    # Phase 1: judge issue coverage against gold
    issue_coverage: Dict[str, Dict[str, Any]] = {}
    for gold_issue in gold_issues:
        cov = _judge_issue_coverage(gold_issue, predicted_entities, judge_model)
        issue_coverage[gold_issue] = cov

    n_covered = sum(1 for v in issue_coverage.values() if v["covered"])
    pipeline_recall = round(n_covered / len(gold_issues), 4) if gold_issues else 0.0

    matched_entities = {
        v["matched_entity"]
        for v in issue_coverage.values()
        if v["covered"] and v["matched_entity"]
    }
    intent_precision = (
        round(len(matched_entities) / len(predicted_entities), 4)
        if predicted_entities
        else 0.0
    )

    if verbose:
        print(
            f"    Pipeline recall: {pipeline_recall:.0%}  "
            f"Intent precision: {intent_precision:.0%}  "
            f"({n_covered}/{len(gold_issues)} gold issues covered)"
        )

    # Phase 2: score trace options against gold options per covered issue
    per_issue_scores: Dict[str, Any] = {}
    for gold_issue in gold_issues:
        cov = issue_coverage[gold_issue]
        gold_opts = gold_options.get(gold_issue, [])

        if not cov["covered"]:
            per_issue_scores[gold_issue] = {"skipped": True, "reason": "not_covered_by_intent_discovery"}
            continue
        if not gold_opts:
            per_issue_scores[gold_issue] = {"skipped": True, "reason": "no_gold_options"}
            continue

        matched_entity = cov.get("matched_entity") or ""
        generated_opts: List[str] = gen_options_map.get(matched_entity, [])
        if not generated_opts and matched_entity:
            me_lower = matched_entity.lower()
            gi_lower = gold_issue.lower()
            for key, opts in gen_options_map.items():
                if me_lower in key.lower() or key.lower() in me_lower or gi_lower in key.lower():
                    generated_opts = opts
                    break

        if not generated_opts:
            per_issue_scores[gold_issue] = {
                "skipped": True,
                "reason": "no_generated_options_found",
                "matched_entity": matched_entity,
            }
            continue

        judge_result = _judge_options_coverage(gold_issue, gold_opts, generated_opts, judge_model)
        gold_covered = judge_result["gold_covered"]
        gen_useful = judge_result["gen_useful"]

        tp_recall = sum(1 for v in gold_covered if v)
        tp_precision = sum(1 for v in gen_useful if v)
        fp = sum(1 for v in gen_useful if not v)
        fn = sum(1 for v in gold_covered if not v)
        precision = round(tp_precision / len(gen_useful), 4) if gen_useful else 0.0
        recall = round(tp_recall / len(gold_covered), 4) if gold_covered else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )

        per_issue_scores[gold_issue] = {
            "skipped": False,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "full_coverage": all(gold_covered),
            "tp_recall": tp_recall,
            "tp_precision": tp_precision,
            "fp": fp,
            "fn": fn,
            "gold_covered": gold_covered,
            "gen_useful": gen_useful,
            "rationale": judge_result["rationale"],
            "skipped_reason": None,
            "predicted_entity": matched_entity,
            "generated_options": generated_opts,
            "gold_options": gold_opts,
        }

        if verbose:
            fc = "✓" if all(gold_covered) else "✗"
            print(
                f"    {gold_issue:<40}  "
                f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}  full={fc}"
            )

    scored = [v for v in per_issue_scores.values() if not v.get("skipped")]
    n_scored = len(scored)
    micro_tp_p = sum(v["tp_precision"] for v in scored)
    micro_tp_r = sum(v["tp_recall"] for v in scored)
    micro_fp = sum(v["fp"] for v in scored)
    micro_fn = sum(v["fn"] for v in scored)
    micro_precision = (
        round(micro_tp_p / (micro_tp_p + micro_fp), 4) if (micro_tp_p + micro_fp) > 0 else 0.0
    )
    micro_recall = (
        round(micro_tp_r / (micro_tp_r + micro_fn), 4) if (micro_tp_r + micro_fn) > 0 else 0.0
    )
    micro_f1 = (
        round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 4)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    avg_precision = round(sum(v["precision"] for v in scored) / n_scored, 4) if n_scored else None
    avg_recall = round(sum(v["recall"] for v in scored) / n_scored, 4) if n_scored else None
    avg_f1 = round(sum(v["f1"] for v in scored) / n_scored, 4) if n_scored else None

    return {
        "id": sample_id,
        "domain": entry["domain"],
        "difficulty": entry["difficulty"],
        "sentence": sentence,
        "gold_issues": gold_issues,
        "predicted_entities": predicted_entities,
        "pipeline_recall": pipeline_recall,
        "intent_precision": intent_precision,
        "n_discovered": len(predicted_entities),
        "n_gold_issues": len(gold_issues),
        "n_covered": n_covered,
        "n_scored": n_scored,
        "issue_coverage": issue_coverage,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_issue": per_issue_scores,
        "eval_mode": "offline",
        "trace_source": trace.get("source_file", ""),
    }


# ─────────────────────────────────────────────────────────────────────────# LLM judge helpers
# ─────────────────────────────────────────────────────────────────────────────

_ISSUE_COVERAGE_PROMPT = """\
You are an evaluation judge for a semantic negotiation pipeline.

A negotiation sentence contains this gold issue — a key negotiable concept:
  Gold issue: "{gold_issue}"

An LLM predicted the following negotiable entities from the same sentence:
  Predicted entities: {predicted_entities}

Question: Does any predicted entity semantically cover or correspond to the gold issue?
"Cover" means the predicted entity refers to the same underlying concept, even if
worded differently or more verbosely.

Answer with a single JSON object only — no prose, no markdown fences:
{{"covered": true_or_false, "matched_entity": "the best matching entity string, or null if none"}}"""

_OPTIONS_SCORING_PROMPT = """\
You are an evaluation judge for a semantic negotiation pipeline.

Gold issue: "{gold_issue}"

Gold interpretations (ground-truth options the pipeline should discover):
{gold_block}

Generated options (what the LLM pipeline actually produced):
{gen_block}

Tasks:
1. For each gold interpretation [0], [1], [2] …: is it semantically covered by any generated option?
   "Covered" means at least one generated option expresses the same idea, constraint, or meaning.
2. For each generated option [0], [1], [2] …: is it semantically useful?
   "Useful" means it is relevant to the gold issue and represents a plausible interpretation
   that could appear in a real negotiation (quality comparable to the gold interpretations).

Answer with a single JSON object only — no prose, no markdown fences:
{{
  "gold_covered": [true_or_false, ...],
  "gen_useful":   [true_or_false, ...],
  "rationale": "one-sentence explanation"
}}"""


def _litellm_kwargs(judge_model: str) -> Dict[str, Any]:
    """Build litellm completion kwargs with credentials from settings."""
    kwargs: Dict[str, Any] = {
        "model": judge_model,
        "temperature": 0.0,
    }
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return kwargs


def _parse_judge_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response, stripping prose/fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (```json or ```)
        stripped = stripped.split("```", 2)[1]
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rstrip("`").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: find first { … } balanced block
    start = text.find("{")
    if start == -1:
        return {}
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _call_judge(prompt: str, judge_model: str) -> Dict[str, Any]:
    from app.config.utils import litellm_completion_compat

    kwargs = _litellm_kwargs(judge_model)
    kwargs["messages"] = [{"role": "user", "content": prompt}]
    try:
        resp = litellm_completion_compat(**kwargs)
        text = resp.choices[0].message.content or ""
        return _parse_judge_json(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge call failed: %s", exc)
        return {}


def _judge_issue_coverage(
    gold_issue: str,
    predicted_entities: List[str],
    judge_model: str,
) -> Dict[str, Any]:
    """Ask the judge if any predicted entity covers the gold issue.

    Returns ``{"covered": bool, "matched_entity": str | None}``.
    """
    prompt = _ISSUE_COVERAGE_PROMPT.format(
        gold_issue=gold_issue,
        predicted_entities=json.dumps(predicted_entities),
    )
    result = _call_judge(prompt, judge_model)
    return {
        "covered": bool(result.get("covered", False)),
        "matched_entity": result.get("matched_entity") or None,
    }


def _judge_options_coverage(
    gold_issue: str,
    gold_interpretations: List[str],
    generated_options: List[str],
    judge_model: str,
) -> Dict[str, Any]:
    """Score generated options against gold interpretations via LLM judge.

    Returns ``{"gold_covered": List[bool], "gen_useful": List[bool], "rationale": str}``.
    """
    gold_block = "\n".join(
        f"  [{i}] {interp}" for i, interp in enumerate(gold_interpretations)
    )
    gen_block = "\n".join(
        f"  [{i}] {opt}" for i, opt in enumerate(generated_options)
    )
    prompt = _OPTIONS_SCORING_PROMPT.format(
        gold_issue=gold_issue,
        gold_block=gold_block,
        gen_block=gen_block,
    )
    result = _call_judge(prompt, judge_model)

    n_gold = len(gold_interpretations)
    n_gen = len(generated_options)
    gold_covered = list(result.get("gold_covered") or [])
    gen_useful = list(result.get("gen_useful") or [])

    # Pad to expected length if judge returned a short list
    gold_covered = (gold_covered + [False] * n_gold)[:n_gold]
    gen_useful = (gen_useful + [False] * n_gen)[:n_gen]

    return {
        "gold_covered": [bool(v) for v in gold_covered],
        "gen_useful": [bool(v) for v in gen_useful],
        "rationale": str(result.get("rationale", "")),
        "raw_response": result.get("raw_response", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_sample(
    entry: Dict[str, Any],
    intent_discovery: IntentDiscovery,
    options_gen: OptionsGeneration,
    judge_model: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full pipeline on one sample and return a scored result dict.

    Args:
        entry: A normalised gold-schema dict (from :func:`load_dataset`).
        intent_discovery: Shared :class:`~app.agent.intent_discovery.IntentDiscovery` instance.
        options_gen: Shared :class:`~app.agent.options_generation.OptionsGeneration` instance.
        judge_model: LiteLLM model string for the judge LLM.
        verbose: Print per-issue progress to stdout.

    Returns:
        A dict containing all Phase 1 and Phase 2 metrics for this sample,
        plus the raw predicted entities and per-issue scored data.
    """
    sample_id = entry["id"]
    sentence = entry["sentence"]
    context = entry["context"] or None
    gold_issues: List[str] = entry["gold_issues"]
    gold_options: Dict[str, List[str]] = entry.get("gold_options") or entry.get("gold_interpretations", {})

    if verbose:
        print(f"\n  [{sample_id}] {sentence[:90]}{'…' if len(sentence) > 90 else ''}")

    # ── Phase 1: Intent Discovery ────────────────────────────────────────────
    discovery = intent_discovery.discover(sentence, context=context)
    predicted_entities: List[str] = discovery.negotiable_entities

    if verbose:
        print(f"    Predicted ({len(predicted_entities)}): {predicted_entities}")

    # For each gold issue: ask the judge whether any predicted entity covers it
    issue_coverage: Dict[str, Dict[str, Any]] = {}
    for gold_issue in gold_issues:
        cov = _judge_issue_coverage(gold_issue, predicted_entities, judge_model)
        issue_coverage[gold_issue] = cov

    n_covered = sum(1 for v in issue_coverage.values() if v["covered"])
    pipeline_recall = round(n_covered / len(gold_issues), 4) if gold_issues else 0.0

    # Intent-level precision: what fraction of predicted entities matched a gold issue?
    matched_entities = {
        v["matched_entity"]
        for v in issue_coverage.values()
        if v["covered"] and v["matched_entity"]
    }
    intent_precision = (
        round(len(matched_entities) / len(predicted_entities), 4)
        if predicted_entities
        else 0.0
    )

    if verbose:
        print(
            f"    Pipeline recall: {pipeline_recall:.0%}  "
            f"Intent precision: {intent_precision:.0%}  "
            f"({n_covered}/{len(gold_issues)} gold issues covered)"
        )

    # ── Phase 2: Options Generation ──────────────────────────────────────────
    # Generate options for ALL discovered entities in one LLM call
    options_output = options_gen.generate_options_llm_only(
        predicted_entities, sentence, context
    )
    gen_options_map: Dict[str, List[str]] = options_output.options_per_issue

    # ── Score options for each gold issue ────────────────────────────────────
    per_issue_scores: Dict[str, Any] = {}

    for gold_issue in gold_issues:
        cov = issue_coverage[gold_issue]
        gold_interps = gold_options.get(gold_issue, [])

        if not cov["covered"]:
            per_issue_scores[gold_issue] = {
                "skipped": True,
                "reason": "not_covered_by_intent_discovery",
            }
            continue

        if not gold_interps:
            per_issue_scores[gold_issue] = {
                "skipped": True,
                "reason": "no_gold_interpretations",
            }
            continue

        # Look up generated options for the matched entity
        matched_entity = cov.get("matched_entity") or ""
        generated_opts: List[str] = gen_options_map.get(matched_entity, [])

        # Fuzzy fallback: search gen_options_map keys by substring match
        if not generated_opts and matched_entity:
            me_lower = matched_entity.lower()
            gi_lower = gold_issue.lower()
            for key, opts in gen_options_map.items():
                if me_lower in key.lower() or key.lower() in me_lower or gi_lower in key.lower():
                    generated_opts = opts
                    break

        if not generated_opts:
            per_issue_scores[gold_issue] = {
                "skipped": True,
                "reason": "no_generated_options_found",
                "matched_entity": matched_entity,
            }
            continue

        # LLM judge: score options coverage
        judge_result = _judge_options_coverage(
            gold_issue, gold_interps, generated_opts, judge_model
        )
        gold_covered = judge_result["gold_covered"]
        gen_useful = judge_result["gen_useful"]

        tp_recall = sum(1 for v in gold_covered if v)
        tp_precision = sum(1 for v in gen_useful if v)
        fp = sum(1 for v in gen_useful if not v)
        fn = sum(1 for v in gold_covered if not v)

        precision = round(tp_precision / len(gen_useful), 4) if gen_useful else 0.0
        recall = round(tp_recall / len(gold_covered), 4) if gold_covered else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )

        per_issue_scores[gold_issue] = {
            "skipped": False,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "full_coverage": all(gold_covered),
            "tp_recall": tp_recall,
            "tp_precision": tp_precision,
            "fp": fp,
            "fn": fn,
            "gold_covered": gold_covered,
            "gen_useful": gen_useful,
            "rationale": judge_result["rationale"],
            "skipped_reason": None,
            "predicted_entity": matched_entity,
            "generated_options": generated_opts,
            "gold_interpretations": gold_interps,
        }

        if verbose:
            fc = "✓" if all(gold_covered) else "✗"
            print(
                f"    {gold_issue:<40}  "
                f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}  full={fc}"
            )

    # ── Aggregate this sample ────────────────────────────────────────────────
    scored = [v for v in per_issue_scores.values() if not v.get("skipped")]
    n_scored = len(scored)

    micro_tp_p = sum(v["tp_precision"] for v in scored)
    micro_tp_r = sum(v["tp_recall"] for v in scored)
    micro_fp = sum(v["fp"] for v in scored)
    micro_fn = sum(v["fn"] for v in scored)

    micro_precision = (
        round(micro_tp_p / (micro_tp_p + micro_fp), 4)
        if (micro_tp_p + micro_fp) > 0
        else 0.0
    )
    micro_recall = (
        round(micro_tp_r / (micro_tp_r + micro_fn), 4)
        if (micro_tp_r + micro_fn) > 0
        else 0.0
    )
    micro_f1 = (
        round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 4)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    avg_precision = round(sum(v["precision"] for v in scored) / n_scored, 4) if n_scored else None
    avg_recall = round(sum(v["recall"] for v in scored) / n_scored, 4) if n_scored else None
    avg_f1 = round(sum(v["f1"] for v in scored) / n_scored, 4) if n_scored else None

    return {
        "id": sample_id,
        "domain": entry["domain"],
        "difficulty": entry["difficulty"],
        "sentence": sentence,
        "gold_issues": gold_issues,
        "predicted_entities": predicted_entities,
        "pipeline_recall": pipeline_recall,
        "intent_precision": intent_precision,
        "n_discovered": len(predicted_entities),
        "n_gold_issues": len(gold_issues),
        "n_covered": n_covered,
        "n_scored": n_scored,
        "issue_coverage": issue_coverage,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_issue": per_issue_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _aggregate(samples: List[Dict[str, Any]], label: str = "overall") -> Dict[str, Any]:
    """Compute aggregate metrics over a list of per-sample result dicts."""
    n = len(samples)
    if n == 0:
        return {"label": label, "n_samples": 0}

    scored_samples = [s for s in samples if s.get("n_scored", 0) > 0]
    n_scored_samples = len(scored_samples)

    # Global micro TP/FP/FN across all issues in all samples
    micro_tp_p = sum(
        v["tp_precision"]
        for s in samples
        for v in s["per_issue"].values()
        if not v.get("skipped")
    )
    micro_tp_r = sum(
        v["tp_recall"]
        for s in samples
        for v in s["per_issue"].values()
        if not v.get("skipped")
    )
    micro_fp = sum(
        v["fp"]
        for s in samples
        for v in s["per_issue"].values()
        if not v.get("skipped")
    )
    micro_fn = sum(
        v["fn"]
        for s in samples
        for v in s["per_issue"].values()
        if not v.get("skipped")
    )
    micro_prec = (
        round(micro_tp_p / (micro_tp_p + micro_fp), 4) if (micro_tp_p + micro_fp) > 0 else 0.0
    )
    micro_rec = (
        round(micro_tp_r / (micro_tp_r + micro_fn), 4) if (micro_tp_r + micro_fn) > 0 else 0.0
    )
    micro_f1 = (
        round(2 * micro_prec * micro_rec / (micro_prec + micro_rec), 4)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    # Macro: mean of per-sample micro F1s (over scored samples only)
    macro_prec = (
        round(sum(s["micro_precision"] for s in scored_samples) / n_scored_samples, 4)
        if n_scored_samples
        else 0.0
    )
    macro_rec = (
        round(sum(s["micro_recall"] for s in scored_samples) / n_scored_samples, 4)
        if n_scored_samples
        else 0.0
    )
    macro_f1 = (
        round(sum(s["micro_f1"] for s in scored_samples) / n_scored_samples, 4)
        if n_scored_samples
        else 0.0
    )

    avg_pipeline_recall = round(sum(s["pipeline_recall"] for s in samples) / n, 4)

    # Full coverage: sample where ALL scored issues are fully covered
    full_coverage_rate = round(
        sum(
            1
            for s in samples
            if s.get("n_scored", 0) > 0
            and all(
                not v.get("skipped") and v.get("full_coverage", False)
                for v in s["per_issue"].values()
                if not v.get("skipped")
            )
        )
        / n,
        4,
    )

    return {
        "label": label,
        "n_samples": n,
        "n_scored_samples": n_scored_samples,
        "avg_pipeline_recall": avg_pipeline_recall,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "full_coverage_rate": full_coverage_rate,
    }


def build_report(
    samples: List[Dict[str, Any]],
    judge_model: str,
    generator_model: str,
) -> Dict[str, Any]:
    """Build the full report dict from per-sample results.

    The structure mirrors ``eval_options_llm_*_scheduling_200.json``.
    """
    overall = _aggregate(samples, "overall")

    domains = sorted({s["domain"] for s in samples})
    by_domain = {
        d: _aggregate([s for s in samples if s["domain"] == d], d) for d in domains
    }

    difficulties = sorted({s["difficulty"] for s in samples})
    by_difficulty = {
        d: _aggregate([s for s in samples if s["difficulty"] == d], d) for d in difficulties
    }

    return {
        "scorer": "llm",
        "judge_model": judge_model,
        "generator_model": generator_model,
        "overall": overall,
        "by_domain": by_domain,
        "by_difficulty": by_difficulty,
        "samples": samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────


def _print_report(report: Dict[str, Any]) -> None:
    ov = report["overall"]
    n = ov["n_samples"]
    print("\n" + "=" * 72)
    print(f"  PIPELINE EVALUATION  —  {n} samples")
    print(f"  Generator : {report['generator_model']}")
    print(f"  Judge     : {report['judge_model']}")
    print("=" * 72)
    print(
        f"  Phase 1 — avg pipeline recall   : {ov['avg_pipeline_recall']:.1%}  "
        f"({ov['n_scored_samples']}/{n} samples had scored issues)"
    )
    print(
        f"  Phase 2 — micro  P / R / F1     : "
        f"{ov['micro_precision']:.1%} / {ov['micro_recall']:.1%} / {ov['micro_f1']:.1%}"
    )
    print(
        f"           macro  P / R / F1     : "
        f"{ov['macro_precision']:.1%} / {ov['macro_recall']:.1%} / {ov['macro_f1']:.1%}"
    )
    print(f"           full-coverage rate   : {ov['full_coverage_rate']:.1%}")

    if report["by_domain"]:
        print()
        print("  By domain:")
        for domain, m in sorted(report["by_domain"].items()):
            print(
                f"    {domain:<38}  n={m['n_samples']:>3}"
                f"  recall={m['avg_pipeline_recall']:.0%}"
                f"  micro-F1={m['micro_f1']:.0%}"
            )

    if report["by_difficulty"]:
        print()
        print("  By difficulty:")
        for diff, m in sorted(report["by_difficulty"].items()):
            print(
                f"    {diff:<15}  n={m['n_samples']:>3}"
                f"  recall={m['avg_pipeline_recall']:.0%}"
                f"  micro-F1={m['micro_f1']:.0%}"
            )

    print("=" * 72 + "\n")


def write_csv_report(report: Dict[str, Any], csv_path: str) -> None:
    """Write a flat CSV report — one row per sample — from the full JSON report.

    Columns:
    * Sample metadata: id, domain, difficulty, sentence (truncated)
    * Phase 1: pipeline_recall, intent_precision, n_gold_issues, n_discovered, n_covered
    * Phase 2 aggregate: micro_precision, micro_recall, micro_f1,
                         avg_precision, avg_recall, avg_f1, n_scored
    * Per-issue detail: for each issue index 0..N-1:
        issue_<n>_name, issue_<n>_precision, issue_<n>_recall, issue_<n>_f1,
        issue_<n>_full_coverage, issue_<n>_skipped
    * Overall aggregates row also written at the end (id="OVERALL")
    """
    import csv as _csv

    samples = report.get("samples", [])
    if not samples:
        logger.warning("No samples to write to CSV.")
        return

    # Determine max number of issues across all samples for dynamic columns
    max_issues = max((s.get("n_gold_issues", 0) for s in samples), default=0)

    base_fields = [
        "id", "domain", "difficulty", "sentence", "eval_mode",
        "pipeline_recall", "intent_precision",
        "n_gold_issues", "n_discovered", "n_covered",
        "micro_precision", "micro_recall", "micro_f1",
        "avg_precision", "avg_recall", "avg_f1",
        "n_scored",
    ]
    issue_fields: List[str] = []
    for n in range(max_issues):
        issue_fields += [
            f"issue_{n}_name",
            f"issue_{n}_precision",
            f"issue_{n}_recall",
            f"issue_{n}_f1",
            f"issue_{n}_full_coverage",
            f"issue_{n}_skipped",
        ]
    fieldnames = base_fields + issue_fields

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for s in samples:
            row: Dict[str, Any] = {
                "id": s.get("id", ""),
                "domain": s.get("domain", ""),
                "difficulty": s.get("difficulty", ""),
                "sentence": (s.get("sentence", "")[:120] + "…"
                             if len(s.get("sentence", "")) > 120
                             else s.get("sentence", "")),
                "eval_mode": s.get("eval_mode", "online"),
                "pipeline_recall": s.get("pipeline_recall"),
                "intent_precision": s.get("intent_precision"),
                "n_gold_issues": s.get("n_gold_issues"),
                "n_discovered": s.get("n_discovered"),
                "n_covered": s.get("n_covered"),
                "micro_precision": s.get("micro_precision"),
                "micro_recall": s.get("micro_recall"),
                "micro_f1": s.get("micro_f1"),
                "avg_precision": s.get("avg_precision"),
                "avg_recall": s.get("avg_recall"),
                "avg_f1": s.get("avg_f1"),
                "n_scored": s.get("n_scored"),
            }
            for n, (issue_name, issue_data) in enumerate(
                s.get("per_issue", {}).items()
            ):
                row[f"issue_{n}_name"] = issue_name
                row[f"issue_{n}_skipped"] = issue_data.get("skipped", True)
                if not issue_data.get("skipped"):
                    row[f"issue_{n}_precision"] = issue_data.get("precision")
                    row[f"issue_{n}_recall"] = issue_data.get("recall")
                    row[f"issue_{n}_f1"] = issue_data.get("f1")
                    row[f"issue_{n}_full_coverage"] = issue_data.get("full_coverage")
                else:
                    row[f"issue_{n}_precision"] = ""
                    row[f"issue_{n}_recall"] = ""
                    row[f"issue_{n}_f1"] = ""
                    row[f"issue_{n}_full_coverage"] = ""
            writer.writerow(row)

        # Write an OVERALL summary row
        ov = report.get("overall", {})
        writer.writerow(
            {
                "id": "OVERALL",
                "domain": "",
                "difficulty": "",
                "sentence": f"n_samples={ov.get('n_samples')}",
                "eval_mode": "",
                "pipeline_recall": ov.get("avg_pipeline_recall"),
                "intent_precision": "",
                "n_gold_issues": "",
                "n_discovered": "",
                "n_covered": "",
                "micro_precision": ov.get("micro_precision"),
                "micro_recall": ov.get("micro_recall"),
                "micro_f1": ov.get("micro_f1"),
                "avg_precision": ov.get("macro_precision"),
                "avg_recall": ov.get("macro_recall"),
                "avg_f1": ov.get("macro_f1"),
                "n_scored": ov.get("n_scored_samples"),
            }
        )

    print(f"CSV report written to {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Negotiation outcome evaluation (run_log.json)
# ─────────────────────────────────────────────────────────────────────────────


def load_run_log(path: str) -> Dict[str, Any]:
    """Load a ``run_log.json`` produced by ``test_via_semantic_neg_agents*.py``.

    Returns the raw dict with ``run_id`` and ``missions`` keys.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_negotiation(run_log: Dict[str, Any]) -> Dict[str, Any]:
    """Compute negotiation outcome metrics from a ``run_log.json`` dict.

    Metrics computed:

    * ``agreement_rate`` — fraction of missions that reached consensus
    * ``avg_rounds_to_agreement`` — mean ``total_rounds`` across agreed missions only
    * ``avg_rounds_all`` — mean ``total_rounds`` across all missions
    * ``avg_duration_s`` — mean wall-clock duration per mission
    * ``timeout_rate`` — fraction of missions that timed out without agreement
    * ``broken_rate`` — fraction of missions that ended in a broken state
    * ``missions`` — per-mission detail list
    """
    missions = run_log.get("missions", [])
    n = len(missions)
    if n == 0:
        return {"n_missions": 0}

    per_mission: List[Dict[str, Any]] = []
    for m in missions:
        # Normalise final_agreement to a plain dict {issue: chosen_option}
        fa = m.get("final_agreement") or {}
        if isinstance(fa, list):
            fa = {
                item["issue_id"]: item["chosen_option"]
                for item in fa
                if "issue_id" in item and "chosen_option" in item
            }

        agreed = bool(fa) or (m.get("status") == "agreed")
        timedout = bool(m.get("timedout")) or (m.get("status") == "timeout")
        broken = bool(m.get("broken")) or (m.get("status") == "broken")

        if agreed:
            verdict = "CONSENSUS REACHED"
        elif timedout:
            verdict = f"TIMED OUT after {m.get('total_rounds', '?')} rounds"
        elif broken:
            verdict = "BROKEN — ended without agreement"
        else:
            verdict = f"ENDED — status: {m.get('status', 'unknown')}"

        per_mission.append({
            "mission": m.get("mission", ""),
            "verdict": verdict,
            "agreed": agreed,
            "timedout": timedout,
            "broken": broken,
            "status": m.get("status", "unknown"),
            "total_rounds": m.get("total_rounds"),
            "n_steps": m.get("n_steps"),
            "duration_s": m.get("duration_s"),
            "deal": fa,
            "session_id": m.get("session_id"),
            "trace_dir": m.get("trace_dir"),
            # LLM call accounting
            "n_agents": m.get("n_agents"),
            "llm_calls_initiate": m.get("llm_calls_initiate", 2),
            "llm_calls_total": m.get("llm_calls_total"),
            "llm_usage_decide": m.get("llm_usage_decide"),
        })

    n_agreed = sum(1 for m in per_mission if m["agreed"])
    n_timedout = sum(1 for m in per_mission if m["timedout"])
    n_broken = sum(1 for m in per_mission if m["broken"])

    agreed_rounds = [
        m["total_rounds"]
        for m in per_mission
        if m["agreed"] and isinstance(m["total_rounds"], (int, float))
    ]
    all_rounds = [
        m["total_rounds"]
        for m in per_mission
        if isinstance(m["total_rounds"], (int, float))
    ]
    all_durations = [
        m["duration_s"]
        for m in per_mission
        if isinstance(m["duration_s"], (int, float))
    ]

    total_llm_calls = sum(
        m["llm_calls_total"] for m in per_mission
        if isinstance(m.get("llm_calls_total"), (int, float))
    )
    total_initiate_calls = sum(
        m["llm_calls_initiate"] for m in per_mission
        if isinstance(m.get("llm_calls_initiate"), (int, float))
    )
    # Real token/cost from litellm callback (decide-phase only)
    decide_usages = [
        m["llm_usage_decide"] for m in per_mission
        if isinstance(m.get("llm_usage_decide"), dict)
    ]
    total_decide_calls = sum(u.get("calls", 0) for u in decide_usages)
    total_prompt_tokens = sum(u.get("prompt_tokens", 0) for u in decide_usages)
    total_completion_tokens = sum(u.get("completion_tokens", 0) for u in decide_usages)
    total_tokens = sum(u.get("total_tokens", 0) for u in decide_usages)
    total_cost_usd = round(sum(u.get("estimated_cost_usd", 0.0) for u in decide_usages), 6)
    avg_llm_calls_per_mission = (
        round(total_llm_calls / n, 1) if total_llm_calls and n else None
    )

    return {
        "run_id": run_log.get("run_id", ""),
        "n_missions": n,
        "n_agreed": n_agreed,
        "n_timedout": n_timedout,
        "n_broken": n_broken,
        "agreement_rate": round(n_agreed / n, 4),
        "timeout_rate": round(n_timedout / n, 4),
        "broken_rate": round(n_broken / n, 4),
        "avg_rounds_to_agreement": (
            round(sum(agreed_rounds) / len(agreed_rounds), 1) if agreed_rounds else None
        ),
        "avg_rounds_all": (
            round(sum(all_rounds) / len(all_rounds), 1) if all_rounds else None
        ),
        "avg_duration_s": (
            round(sum(all_durations) / len(all_durations), 1) if all_durations else None
        ),
        "llm_calls_total": total_llm_calls,
        "llm_calls_initiate": total_initiate_calls,
        "llm_calls_decide": total_decide_calls,
        "avg_llm_calls_per_mission": avg_llm_calls_per_mission,
        # Token/cost totals (decide-phase only; initiate-phase in neg server)
        "decide_prompt_tokens": total_prompt_tokens,
        "decide_completion_tokens": total_completion_tokens,
        "decide_total_tokens": total_tokens,
        "decide_estimated_cost_usd": total_cost_usd,
        "missions": per_mission,
    }


def _print_negotiation_report(neg: Dict[str, Any]) -> None:
    """Print a human-readable negotiation outcome summary to stdout."""
    n = neg["n_missions"]
    print("\n" + "=" * 72)
    print(f"  NEGOTIATION OUTCOMES  —  {n} missions  (run: {neg.get('run_id', '?')})")
    print("=" * 72)
    print(f"  Agreement rate         : {neg['agreement_rate']:.1%}  ({neg['n_agreed']}/{n})")
    print(f"  Timeout rate           : {neg['timeout_rate']:.1%}  ({neg['n_timedout']}/{n})")
    print(f"  Broken rate            : {neg['broken_rate']:.1%}  ({neg['n_broken']}/{n})")
    if neg.get("avg_rounds_to_agreement") is not None:
        print(f"  Avg rounds (agreed)    : {neg['avg_rounds_to_agreement']}")
    if neg.get("avg_rounds_all") is not None:
        print(f"  Avg rounds (all)       : {neg['avg_rounds_all']}")
    if neg.get("avg_duration_s") is not None:
        print(f"  Avg duration/mission   : {neg['avg_duration_s']}s")
    if neg.get("llm_calls_total") is not None:
        print()
        print(f"  LLM calls (total)      : {neg['llm_calls_total']}")
        print(f"    initiate (IntentDisc + OptsGen) : {neg['llm_calls_initiate']}  (2 per mission, neg server)")
        print(f"    decide   (agent calls, actual)  : {neg['llm_calls_decide']}")
        print(f"    avg per mission                 : {neg['avg_llm_calls_per_mission']}")
    if neg.get("decide_total_tokens"):
        print()
        print("  Token usage (decide-phase)")
        print(f"    prompt tokens      : {neg['decide_prompt_tokens']:,}")
        print(f"    completion tokens  : {neg['decide_completion_tokens']:,}")
        print(f"    total tokens       : {neg['decide_total_tokens']:,}")
        print(f"    estimated cost     : ${neg['decide_estimated_cost_usd']:.4f} USD")
    print()
    for m in neg["missions"]:
        icon = "\u2713" if m["agreed"] else "\u2717"
        rounds = m["total_rounds"] if m["total_rounds"] is not None else "?"
        name = m["mission"]
        print(f"  {icon}  {name:<50}  rounds={rounds}")
        if m["agreed"] and m["deal"]:
            deal_str = "  |  ".join(f"{k}: '{v}'" for k, v in m["deal"].items())
            print(f"       Deal : {deal_str}")
    print("=" * 72 + "\n")


def write_negotiation_csv(neg: Dict[str, Any], csv_path: str) -> None:
    """Write a flat negotiation outcome CSV — one row per mission + OVERALL row.

    Columns: mission, agreed, verdict, status, total_rounds, duration_s,
    session_id, deal_issue_<n>, deal_option_<n>.
    """
    import csv as _csv

    max_deal = max(
        (len(m.get("deal") or {}) for m in neg.get("missions", [])), default=0
    )
    fieldnames: List[str] = [
        "mission", "agreed", "verdict", "status",
        "total_rounds", "n_agents", "llm_calls_initiate", "llm_calls_decide",
        "llm_calls_total", "prompt_tokens", "completion_tokens", "total_tokens",
        "estimated_cost_usd", "duration_s", "session_id",
    ]
    for i in range(max_deal):
        fieldnames += [f"deal_issue_{i}", f"deal_option_{i}"]

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in neg.get("missions", []):
            _usage = m.get("llm_usage_decide") or {}
            row: Dict[str, Any] = {
                "mission": m["mission"],
                "agreed": m["agreed"],
                "verdict": m["verdict"],
                "status": m["status"],
                "total_rounds": m["total_rounds"],
                "n_agents": m.get("n_agents", ""),
                "llm_calls_initiate": m.get("llm_calls_initiate", ""),
                "llm_calls_decide": _usage.get("calls", ""),
                "llm_calls_total": m.get("llm_calls_total", ""),
                "prompt_tokens": _usage.get("prompt_tokens", ""),
                "completion_tokens": _usage.get("completion_tokens", ""),
                "total_tokens": _usage.get("total_tokens", ""),
                "estimated_cost_usd": _usage.get("estimated_cost_usd", ""),
                "duration_s": m["duration_s"],
                "session_id": m.get("session_id", ""),
            }
            for i, (issue, option) in enumerate((m.get("deal") or {}).items()):
                row[f"deal_issue_{i}"] = issue
                row[f"deal_option_{i}"] = option
            writer.writerow(row)
        # OVERALL summary row
        writer.writerow({
            "mission": "OVERALL",
            "agreed": neg.get("agreement_rate"),
            "verdict": f"{neg['n_agreed']}/{neg['n_missions']} agreed",
            "status": "",
            "total_rounds": neg.get("avg_rounds_all"),
            "n_agents": "",
            "llm_calls_initiate": neg.get("llm_calls_initiate"),
            "llm_calls_decide": neg.get("llm_calls_decide"),
            "llm_calls_total": neg.get("llm_calls_total"),
            "prompt_tokens": neg.get("decide_prompt_tokens"),
            "completion_tokens": neg.get("decide_completion_tokens"),
            "total_tokens": neg.get("decide_total_tokens"),
            "estimated_cost_usd": neg.get("decide_estimated_cost_usd"),
            "duration_s": neg.get("avg_duration_s"),
            "session_id": "",
        })
    print(f"Negotiation CSV written to {out_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    dataset_path: Optional[str] = None,
    *,
    judge_model: Optional[str] = None,
    limit: Optional[int] = None,
    verbose: bool = False,
    trace_dir: Optional[str] = None,
    run_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline and return the report dict.

    Args:
        dataset_path: Path to the gold-schema JSON dataset.  Optional when
            *run_log_path* is provided and pipeline evaluation is not needed.
        judge_model: LiteLLM model string for the judge.  Defaults to
            ``JUDGE_MODEL`` env var, then ``OPENAI_MODEL``, then
            ``settings.llm_model``.
        limit: Evaluate only the first *N* samples.
        verbose: Print per-issue scores to stdout while running.
        trace_dir: Path to a ``test_via_semantic_neg_agents.py`` run directory
            containing ``*/01_initiate_response.json`` files.  When provided,
            the pipeline runs in **offline mode**: generator LLM calls are
            skipped and the saved issues/options are used instead.  Only the
            judge LLM is called.
        run_log_path: Path to a ``run_log.json`` produced by
            ``test_via_semantic_neg_agents_configured.py``.  When provided,
            negotiation outcome metrics (agreement rate, rounds, deal terms)
            are computed and added to the report under ``"negotiation"``.

    Returns:
        Report dict (see module docstring for schema).
    """
    effective_judge = (
        judge_model
        or os.environ.get("JUDGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or settings.llm_model
    )
    generator_model = (
        "offline (trace)"
        if trace_dir
        else (os.environ.get("OPENAI_MODEL") or settings.llm_model)
    )

    # ── Negotiation-only mode (no pipeline evaluation) ────────────────────────
    if run_log_path and not dataset_path:
        print(f"Loading run log : {run_log_path}")
        run_log = load_run_log(run_log_path)
        neg = evaluate_negotiation(run_log)
        return {
            "scorer": "negotiation_outcomes",
            "judge_model": effective_judge,
            "generator_model": generator_model,
            "negotiation": neg,
        }

    if not dataset_path:
        raise ValueError("Either --dataset or --run-log (or both) must be provided.")

    print(f"Loading dataset : {dataset_path}")
    samples_data = load_dataset(dataset_path)
    if limit is not None:
        samples_data = samples_data[:limit]

    mode_label = f"offline (trace_dir={trace_dir})" if trace_dir else "online"
    print(
        f"Samples         : {len(samples_data)}\n"
        f"Mode            : {mode_label}\n"
        f"Generator model : {generator_model}\n"
        f"Judge model     : {effective_judge}"
    )

    results: List[Dict[str, Any]] = []

    if trace_dir:
        # ── Offline mode ─────────────────────────────────────────────────
        traces = load_trace_dir(trace_dir)
        print(f"Trace files found: {len(traces)}  ({list(traces.keys())})")
        for i, entry in enumerate(samples_data):
            print(f"  [{i + 1:>3}/{len(samples_data)}] {entry['id']} …", flush=True)
            trace = _match_trace(entry, traces)
            if trace is None:
                print(f"    WARNING: no matching trace found for {entry['id']} — skipping")
                continue
            result = evaluate_sample_offline(
                entry,
                trace,
                effective_judge,
                verbose=verbose,
            )
            results.append(result)
    else:
        # ── Online mode ──────────────────────────────────────────────────
        intent_discovery = IntentDiscovery()
        options_gen = OptionsGeneration()
        for i, entry in enumerate(samples_data):
            print(f"  [{i + 1:>3}/{len(samples_data)}] {entry['id']} …", flush=True)
            result = evaluate_sample(
                entry,
                intent_discovery,
                options_gen,
                effective_judge,
                verbose=verbose,
            )
            results.append(result)

    report = build_report(
        results,
        judge_model=effective_judge,
        generator_model=generator_model,
    )

    # ── Attach negotiation outcome section if run_log provided ───────────────
    if run_log_path:
        run_log = load_run_log(run_log_path)
        report["negotiation"] = evaluate_negotiation(run_log)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate IntentDiscovery + OptionsGeneration against a gold dataset. "
            "Works with hard5, pm9, or any dataset following the gold_issues / "
            "gold_interpretations schema."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=False,
        default=None,
        metavar="PATH",
        help=(
            "Path to gold-schema JSON file (hard5 or pm9 or compatible). "
            "Required unless --run-log is used standalone."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help=(
            "LiteLLM model for the judge (default: JUDGE_MODEL env var, "
            "then OPENAI_MODEL, then settings.llm_model)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N samples.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-issue scores while running.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write JSON report to FILE.",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        metavar="DIR",
        help=(
            "Offline mode: path to a test_via_semantic_neg_agents.py run directory "
            "containing <mission>/01_initiate_response.json files. "
            "Skips generator LLM calls; only the judge LLM is called."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="FILE",
        help="Write a flat CSV report (one row per sample) to FILE.",
    )
    parser.add_argument(
        "--run-log",
        default=None,
        metavar="FILE",
        help=(
            "Path to run_log.json from test_via_semantic_neg_agents_configured.py. "
            "Adds negotiation outcome metrics (agreement rate, rounds, deal terms) "
            "to the report. Can be used alone (without --dataset) for "
            "negotiation-only evaluation."
        ),
    )
    parser.add_argument(
        "--neg-csv",
        default=None,
        metavar="FILE",
        help="Write a flat negotiation outcomes CSV (one row per mission) to FILE.",
    )
    args = parser.parse_args(argv)

    report = run_evaluation(
        args.dataset,
        judge_model=args.judge_model,
        limit=args.limit,
        verbose=args.verbose,
        trace_dir=args.trace_dir,
        run_log_path=args.run_log,
    )

    # Print pipeline report only when dataset was evaluated
    if args.dataset:
        _print_report(report)

    # Always print negotiation report when present
    if "negotiation" in report:
        _print_negotiation_report(report["negotiation"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {out_path.resolve()}")

    if args.csv and args.dataset:
        write_csv_report(report, args.csv)

    if args.neg_csv and "negotiation" in report:
        write_negotiation_csv(report["negotiation"], args.neg_csv)


if __name__ == "__main__":
    main()
