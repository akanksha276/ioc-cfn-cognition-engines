# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for persisting integration test results to disk.

Each test run writes one JSON file per fixture under
``semantic_validation/tests/results/<run_id>/`` and an aggregate
``_summary.json`` once the session finishes.

A test calls ``write_result(name, data)`` to record its outcome. The summary
file is generated automatically from all per-test files at session teardown.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

_RESULTS_BASE = Path(__file__).parent.parent / "results"


def _run_id() -> str:
    """Generate a session-wide run id based on current time."""
    return time.strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="session")
def results_dir() -> Path:
    """Per-session directory for this test run."""
    run_id = os.environ.get("SAV_RESULTS_RUN_ID") or _run_id()
    run_dir = _RESULTS_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def write_result(results_dir: Path) -> Callable[[str, Dict[str, Any]], None]:
    """Return a callable that writes a per-test result JSON."""
    def _write(name: str, data: Dict[str, Any]) -> None:
        path = results_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
    return _write


@pytest.fixture(scope="session", autouse=True)
def write_summary(results_dir: Path):
    """At session end, aggregate per-test JSON files into a summary."""
    yield  # let tests run first

    files = sorted(p for p in results_dir.glob("*.json") if p.name != "_summary.json")
    if not files:
        return

    results = []
    for f in files:
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            continue

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed

    by_kind: Dict[str, Dict[str, int]] = {}
    for r in results:
        k = r.get("kind", "unknown")
        bucket = by_kind.setdefault(k, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if r.get("passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    severity_counts: Dict[str, int] = {}
    for r in results:
        sev = r.get("sav_severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary = {
        "run_id": results_dir.name,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_kind": by_kind,
        "by_sav_severity": severity_counts,
        "fixtures": [r.get("fixture", "?") for r in results],
    }
    (results_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
