# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Sample environment — Direct mode (no external negotiation server required).

This mirrors the approach of the **self-contained** evaluation path where
:class:`~evaluation.framework.runner.EvaluationRunner` drives the SAO rounds
itself via :class:`~app.agent.batch_callback_runner.BatchCallbackRunner`.

Comparison with ``test_callback_agents.py``
-------------------------------------------
``test_callback_agents.py`` requires a running negotiation server (port 8089).
It POSTs ``/api/negotiate/initiate``, then drives turn-by-turn loops via
``/api/negotiate/decide``.  Issues and options are **discovered** by the
negotiation server's LLM pipeline.

This sample needs **no external server** — agents and the runner live in the
same process.  Issues and options must be declared explicitly in the mission
definitions (see :data:`SAMPLE_MISSIONS` below).

Architecture::

    ┌────────────────────────────────────────────────────────────┐
    │  direct_env.py  (everything runs in one Python process)     │
    │                                                             │
    │  GenericCallbackAgent "buyer"                               │
    │  GenericCallbackAgent "seller"                              │
    │           │                                                 │
    │           │  POST List[SSTPNegotiateMessage]                │
    │           ▼                                                 │
    │  FastAPI /decide  (port 8094, daemon thread)                │
    │           │                                                 │
    │           │  SAO rounds                                     │
    │           ▼                                                 │
    │  BatchCallbackRunner  →  NegotiationResult                  │
    │           │                                                 │
    │           ▼                                                 │
    │  EvaluationResult.summary()                                 │
    └────────────────────────────────────────────────────────────┘

Usage::

    # from semantic_negotiation/
    python -m evaluation.framework.mechanisms.direct_env

    # or with a custom missions YAML:
    python -m evaluation.framework.mechanisms.direct_env --missions-file path/to/missions.yaml

    # YAML missions must include issues and options_per_issue for this sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── sys.path: ensure repo root + semantic_negotiation/ are importable ────────
_repo_root = str(Path(__file__).resolve().parents[4])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_sn_root = str(Path(__file__).resolve().parents[3])  # semantic_negotiation/
if _sn_root not in sys.path:
    sys.path.insert(0, _sn_root)

from evaluation.framework.config import (  # noqa: E402
    AgentConfig,
    EnvironmentConfig,
    EvaluationConfig,
    MissionEntry,
    MissionsConfig,
)
from evaluation.framework.runner import EvaluationRunner  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Agents
# ─────────────────────────────────────────────────────────────────────────────
#
# Two agents: a cost-conscious buyer and a premium-oriented seller.
# Tune exponent and min_reservation to control concession behaviour:
#   exponent > 1 : Boulware — holds ideal position long, concedes hard near deadline
#   exponent = 1 : linear — steady concession throughout
#   exponent < 1 : conceder — concedes quickly early, then plateaus
# aspiration(t) = max(min_reservation, 1 - t^exponent)

AGENTS = [
    AgentConfig(
        agent_id="buyer",
        prefer_low=True,  # prefers index-0 (cheapest) options
        params={"exponent": 2.0, "min_reservation": 0.05},
    ),
    AgentConfig(
        agent_id="seller",
        prefer_low=False,  # prefers last-index (premium) options
        params={"exponent": 2.0, "min_reservation": 0.05},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Environment
# ─────────────────────────────────────────────────────────────────────────────
#
# agent_port: TCP port for the in-process FastAPI callback server.
# neg_timeout: per-round HTTP request timeout in seconds.

ENVIRONMENT = EnvironmentConfig(
    agent_port=8094,
    neg_timeout=30.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Missions dataset
# ─────────────────────────────────────────────────────────────────────────────
#
# Direct mode requires issues and options_per_issue to be declared here
# (no LLM discovery step).  n_steps=0 triggers automatic budget sizing.

SAMPLE_MISSIONS = MissionsConfig(
    missions=[
        MissionEntry(
            name="Quick Deal",
            content_text=(
                "Two parties need to agree on price and delivery speed "
                "for an urgent supply order."
            ),
            issues=["price", "delivery_speed"],
            options_per_issue={
                "price": ["$100", "$120", "$150", "$200"],
                "delivery_speed": ["same-day", "3-day", "1-week", "2-week"],
            },
            n_steps=40,
        ),
        MissionEntry(
            name="Cloud Platform",
            content_text=(
                "Teams need to agree on compute tier, storage quota, support level, "
                "deployment region, and contract length for multi-year cloud "
                "infrastructure."
            ),
            issues=["compute", "storage", "support", "region", "contract"],
            options_per_issue={
                "compute": ["small", "medium", "large", "xl"],
                "storage": ["100GB", "500GB", "1TB", "5TB"],
                "support": ["basic", "standard", "premium", "enterprise"],
                "region": ["us-east", "us-west", "eu-west", "ap-south"],
                "contract": ["1-year", "2-year", "3-year", "5-year"],
            },
            # n_steps=0 → auto-sized by compute_n_steps()
            n_steps=0,
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Combined config
# ─────────────────────────────────────────────────────────────────────────────


def build_config(missions_file: str | None = None) -> EvaluationConfig:
    """Return :class:`EvaluationConfig` for the direct environment.

    Args:
        missions_file: Optional path to a YAML missions file.  When provided it
            overrides the inline :data:`SAMPLE_MISSIONS`.  The YAML must include
            ``issues`` and ``options_per_issue`` for every mission.
    """
    missions = MissionsConfig(path=missions_file) if missions_file else SAMPLE_MISSIONS
    return EvaluationConfig(
        agents=AGENTS,
        environment=ENVIRONMENT,
        missions=missions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Direct evaluation environment — no external server required.",
    )
    parser.add_argument(
        "--missions-file",
        default=None,
        metavar="PATH",
        help="YAML missions file (must include issues + options_per_issue).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8094,
        help="TCP port for the in-process agent server (default: 8094).",
    )
    args = parser.parse_args(argv)

    cfg = build_config(args.missions_file)
    cfg.environment.agent_port = args.port

    print("=" * 60)
    print("  Direct Evaluation Environment")
    print("  Mode   : self-contained (BatchCallbackRunner, no external server)")
    print(f"  Agents : {[a.agent_id for a in cfg.agents]}")
    print(f"  Port   : {cfg.environment.agent_port}")
    missions = cfg.missions.load()
    print(f"  Missions ({len(missions)}):")
    for m in missions:
        print(f"    • {m.name}  issues={m.issues}  n_steps={m.n_steps or 'auto'}")
    print("=" * 60)
    print()

    results = EvaluationRunner(cfg).run()
    summary = results.summary()

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print()
    print(
        f"  Agreement rate : {summary['agreement_rate']:.0%}  "
        f"({sum(1 for m in summary['missions'] if m['agreement'])} / "
        f"{summary['total_missions']} missions)"
    )
    print(f"  Avg steps      : {summary['avg_steps']}")


if __name__ == "__main__":
    main()
