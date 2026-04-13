# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""run_evaluation.py — Single entry point for the evaluation framework.

Loads an :class:`EvaluationConfig` from a YAML file, then dispatches to the
correct mechanism runner based on ``config.mechanism``:

* ``direct``   — self-contained, no external server needed
                 (:class:`~evaluation.framework.runner.EvaluationRunner`)
* ``callback`` — requires the negotiation server to be running at
                 ``config.environment.neg_server``
                 (:class:`~evaluation.framework.mechanisms.callback_env.CallbackEnvRunner`)

Usage::

    # From the repo root:
    .venv/bin/python semantic_negotiation/evaluation/framework/run_evaluation.py

    # Custom config:
    .venv/bin/python semantic_negotiation/evaluation/framework/run_evaluation.py \\
        --config path/to/evaluation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── sys.path: make semantic_negotiation + app importable ────────────────────
for _p in (
    str(Path(__file__).resolve().parents[3]),  # repo root
    str(Path(__file__).resolve().parents[3] / "semantic_negotiation" / "app"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantic_negotiation.evaluation.framework.config import (  # noqa: E402
    EvaluationConfig,
    MissionsConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "evaluation.yaml"


def _print_summary(summary: dict) -> None:
    mechanism = summary.get("mechanism", "")
    print("\n" + "=" * 62)
    print(
        f"  EVALUATION COMPLETE — {summary['total_missions']} missions  [{mechanism}]"
    )
    print("=" * 62)
    for m in summary["missions"]:
        agreement = m.get("agreement")
        timedout = m.get("timedout") or m.get("timedout", False)
        steps_key = "steps" if "steps" in m else "total_rounds"
        budget_key = "n_steps_budget" if "n_steps_budget" in m else "n_steps"
        status = "AGREEMENT ✓" if agreement else ("TIMED OUT" if timedout else "BROKEN")
        print(
            f"  {m['mission']:<30}  {status:<12}  steps={m[steps_key]}/{m[budget_key]}"
        )
        if agreement:
            deal = "  |  ".join(f"{k}: '{v}'" for k, v in agreement.items())
            print(f"    Deal: {deal}")
    print("=" * 62)
    print(f"  Agreement rate : {summary['agreement_rate']:.0%}")
    if "avg_steps" in summary:
        print(f"  Avg steps      : {summary['avg_steps']}")
    print("=" * 62 + "\n")


def main(config_path: Path, n_missions: int | None = None) -> None:
    logger.info("Loading config: %s", config_path)
    config = EvaluationConfig.from_yaml(str(config_path))

    missions = config.missions.load()
    if n_missions is not None:
        missions = missions[:n_missions]
        config.missions = MissionsConfig(missions=missions)
    logger.info(
        "Mechanism=%s  agents=%s  missions=%d",
        config.mechanism,
        [a.agent_id for a in config.agents],
        len(missions),
    )

    if config.mechanism == "callback":
        from semantic_negotiation.evaluation.framework.mechanisms.callback_env import (  # noqa: E402
            CallbackEnvRunner,
        )

        results = CallbackEnvRunner(config).run()
        summary = results.summary()
        summary["mechanism"] = "callback"
    else:
        from semantic_negotiation.evaluation.framework.runner import (  # noqa: E402
            EvaluationRunner,
        )

        results = EvaluationRunner(config).run()
        summary = results.summary()
        summary["mechanism"] = "direct"

    _print_summary(summary)

    out_path = Path("neg_trace") / "run_evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", out_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the evaluation framework from a YAML config file."
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        metavar="PATH",
        help=f"YAML config file (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--n-missions",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N missions (default: all)",
    )
    args = parser.parse_args()
    main(Path(args.config), n_missions=args.n_missions)
