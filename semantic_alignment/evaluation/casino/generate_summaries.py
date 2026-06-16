# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a CaSiNo intent-discovery summaries dataset.

Reads ``casino.json`` and emits ``casino_summaries.json`` — one record per
dialogue with a natural-language reason summary for each participant.

Priority labels (High / Medium / Low) and issue names are intentionally
excluded from the summaries.  Each raw reason paragraph is condensed into a
clean 1–3 sentence summary via an LLM before being saved.

Requires ``LLM_API_KEY`` or ``LLM_BASE_URL`` to be set in
``semantic_alignment/evaluation/.env`` (see ``.env.example`` in the repo root).

Output format
-------------
.. code-block:: json

    [
      {
        "dialogue_id": 0,
        "gold_issues": ["food", "water", "firewood"],
        "summaries": {
          "mturk_agent_1": "We have a larger group than normal ...",
          "mturk_agent_2": "my dog has fleas, the fire repels them. ..."
        }
      },
      ...
    ]

Usage::

    # from semantic_alignment/
    python -m evaluation.casino.generate_summaries \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --output ../../CaSiNo/data/casino_summaries.json

    # preview the first 2 records without saving
    python -m evaluation.casino.generate_summaries \\
        --casino-path ../../CaSiNo/data/casino.json \\
        --output /tmp/preview.json --preview
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# ── sys.path: ensure agent root is importable ─────────────────────────────────
_agent_root = str(Path(__file__).resolve().parents[2])
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from ..casino.loader import ISSUES, build_agent_summary, load_casino  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def _make_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes to *log_path* and to stdout."""
    logger = logging.getLogger("generate_summaries")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — full detail
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger



_SUMMARISE_SYSTEM = (
    "You are a concise writing assistant. "
    "Summarise the following negotiation reasons into a single, coherent paragraph "
    "of 1-3 sentences. "
    "Do NOT mention high, medium, or low priorities. "
    "Do NOT name specific items like food, water, or firewood. "
    "Write in the first person from the perspective of the participant. "
    "Return only the summary, nothing else."
)


def _build_llm_client() -> tuple[dict, str] | tuple[None, None]:
    """Return ``(creds_dict, model)`` from the evaluation ``.env`` file.

    Reads ``LLM_MODEL``, ``LLM_API_KEY``, and ``LLM_BASE_URL`` from
    ``semantic_alignment/evaluation/.env``.

    Returns ``(None, None)`` if neither LLM_API_KEY nor LLM_BASE_URL is set.
    """
    # Load the .env that lives one level above this package (evaluation/)
    _env_file = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_env_file, override=True)

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "openai/gpt-4o")

    if not api_key and not base_url:
        return None, None

    creds: dict = {}
    if api_key:
        creds["api_key"] = api_key
    if base_url:
        creds["base_url"] = base_url

    return creds, model


def _summarise_paragraph(raw: str, creds: dict, model: str) -> str:
    """Call the LLM to condense *raw* into a clean 1–3 sentence paragraph.

    Args:
        raw: The joined ``value2reason`` text produced by :func:`build_agent_summary`.
        creds: litellm credential kwargs (api_key, base_url).
        model: litellm model string (e.g. ``openai/gpt-4o``).

    Returns:
        Summarised paragraph, or *raw* unchanged if the call fails.
    """
    try:
        from app.config.utils import litellm_completion_compat

        response = litellm_completion_compat(
            model=model,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": raw},
            ],
            temperature=0.3,
            max_tokens=150,
            **creds,
        )
        return (response.choices[0].message.content or "").strip() or raw
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠  LLM summarisation failed ({exc}); keeping raw text.", file=sys.stderr)
        return raw


# ─────────────────────────────────────────────────────────────────────────────


def generate_summaries(
    casino_path: str | Path,
    creds: dict,
    model: str,
    *,
    log_path: Path | None = None,
) -> List[Dict]:
    """Load CaSiNo and build a per-dialogue LLM-summarised summaries list.

    Each agent's raw ``value2reason`` reasons are joined and condensed into a
    clean 1–3 sentence paragraph via :func:`_summarise_paragraph` before
    being stored.  Priority labels and issue names are never exposed.

    Args:
        casino_path: Path to ``casino.json``.
        creds: litellm credential kwargs (api_key, base_url).
        model: litellm model string (e.g. ``openai/gpt-4o``).
        log_path: Optional path to write a progress log file.  Each completed
            dialogue is logged so you can track progress and resume after a
            crash.

    Returns:
        List of dicts, one per dialogue, with keys:
        ``dialogue_id``, ``gold_issues``, ``summaries``.
        Dialogues where both agents have no ``value2reason`` data are skipped.
    """
    logger = _make_logger(log_path) if log_path else logging.getLogger("generate_summaries")

    dialogues = load_casino(casino_path)
    total = len(dialogues)
    records: List[Dict] = []
    n_skipped = 0

    logger.info("Starting  —  %d dialogues to process  (model: %s)", total, model)

    for i, dlg in enumerate(dialogues, start=1):
        # Skip dialogues with missing participant_info
        if not dlg.agent1.value2reason and not dlg.agent2.value2reason:
            n_skipped += 1
            logger.debug("dialogue %4d / %d  SKIPPED  (no value2reason data)", i, total)
            continue

        summary1 = _summarise_paragraph(build_agent_summary(dlg.agent1), creds, model)
        summary2 = _summarise_paragraph(build_agent_summary(dlg.agent2), creds, model)

        records.append(
            {
                "dialogue_id": dlg.dialogue_id,
                "gold_issues": list(ISSUES),
                "summaries": {
                    dlg.agent1_id: summary1,
                    dlg.agent2_id: summary2,
                },
            }
        )

        logger.debug(
            "dialogue %4d / %d  OK  id=%-4s  [%s] %d chars  [%s] %d chars",
            i, total, dlg.dialogue_id,
            dlg.agent1_id, len(summary1),
            dlg.agent2_id, len(summary2),
        )

        # INFO-level heartbeat every 50 dialogues
        if i % 50 == 0:
            pct = i / total * 100
            logger.info("Progress  %4d / %d  (%.0f%%)  —  %d records so far",
                        i, total, pct, len(records))

    if n_skipped:
        logger.warning("Skipped %d dialogue(s) with no participant_info.", n_skipped)

    logger.info("Done  —  %d records  (%d agent summaries)", len(records), len(records) * 2)
    return records


def _print_sample(records: List[Dict], n: int = 2) -> None:
    """Print the first *n* records as a preview."""
    print(f"\nSample ({min(n, len(records))} of {len(records)} records):")
    for rec in records[:n]:
        print(f"\n  dialogue_id: {rec['dialogue_id']}")
        for agent_id, summary in rec["summaries"].items():
            preview = summary[:120] + ("…" if len(summary) > 120 else "")
            print(f"  [{agent_id}]: {preview}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate casino_summaries.json from casino.json.",
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
        "--output",
        required=True,
        metavar="FILE",
        help="Output path for casino_summaries.json",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the first 2 generated records to stdout.",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        help=(
            "Path to the progress log file.  Defaults to <output>.log "
            "(alongside the output JSON)."
        ),
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")

    creds, model = _build_llm_client()
    if creds is None:
        print(
            "ERROR: LLM_API_KEY or LLM_BASE_URL must be set in evaluation/.env.",
            file=sys.stderr,
        )
        sys.exit(1)

    logger = _make_logger(log_path)
    logger.info("=" * 60)
    logger.info("generate_summaries  started  %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("casino-path : %s", args.casino_path)
    logger.info("output      : %s", out_path)
    logger.info("log-file    : %s", log_path)
    logger.info("model       : %s", model)
    logger.info("=" * 60)

    records = generate_summaries(args.casino_path, creds, model, log_path=log_path)
    logger.info("Generated %d dialogue records  (%d agent summaries).", len(records), len(records) * 2)

    if args.preview:
        _print_sample(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
