# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for ioc-cognition-fabric-node-svc#16 (same class of bug in CE routes).

After IngestDataService.ingest became async, callers must await before passing the
result to KnowledgeProcessor.process (which expects a dict).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTES = _REPO_ROOT / "app" / "api" / "routes.py"

# (async function name in routes.py that must await ingest_service.ingest)
_INGEST_CALL_SITES = (
    "knowledge_extraction",
    "extract_concepts_and_relationships_from_file",
)


def _ingest_calls_in_function(tree: ast.Module, func_name: str) -> list[ast.Call]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "ingest"
            ]
    return []


def _call_is_awaited(call: ast.Call, tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and node.value is call:
            return True
    return False


@pytest.mark.parametrize("func_name", _INGEST_CALL_SITES)
def test_ingest_service_ingest_is_awaited_in_routes(func_name: str) -> None:
    tree = ast.parse(_ROUTES.read_text(encoding="utf-8"), filename=str(_ROUTES))

    calls = _ingest_calls_in_function(tree, func_name)
    assert calls, f"expected ingest_service.ingest() in routes.py::{func_name}"

    unawaited = [c.lineno for c in calls if not _call_is_awaited(c, tree)]
    assert not unawaited, (
        f"routes.py::{func_name}: ingest() must be awaited "
        f"(see ioc-cognition-fabric-node-svc#16); unawaited at line(s) {unawaited}"
    )
