# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evidence routes.py — all endpoints routed through EvidenceCognitionEngine."""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from evidence.app.agent.evidence_ce import EvidenceCognitionEngine
from evidence.app.dependencies import get_evidence_cognition_engine
from evidence.app.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_engine(run_return=None) -> EvidenceCognitionEngine:
    engine = MagicMock(spec=EvidenceCognitionEngine)
    engine.run = AsyncMock(return_value=run_return or {})
    return engine


_REASONING_BODY = {
    "header": {"workspace_id": "ws-1", "mas_id": "mas-1", "agent_id": "ag-1"},
    "request_id": "req-1",
    "payload": {
        "intent": "what does the orchestrator do?",
        "metadata": {},
        "additional_context": [],
    },
}

_REASON_RESULT = {
    "status": "OK",
    "records": [],
    "header": {"workspace_id": "ws-1", "mas_id": "mas-1", "agent_id": "ag-1"},
    "response_id": "req-1",
    "meta": {},
}


# ---------------------------------------------------------------------------
# POST /api/knowledge-mgmt/reasoning/evidence
# ---------------------------------------------------------------------------


def test_reasoning_evidence_delegates_to_ce():
    stub = _make_stub_engine(_REASON_RESULT)
    app.dependency_overrides[get_evidence_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post(
            "/api/knowledge-mgmt/reasoning/evidence", json=_REASONING_BODY
        )
        assert resp.status_code == 200
        stub.run.assert_awaited_once()
        action, payload = stub.run.call_args[0]
        assert action.value == "reason"
        assert payload["request_id"] == "req-1"
        assert payload["header"]["workspace_id"] == "ws-1"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/knowledge-mgmt/graph/paths
# ---------------------------------------------------------------------------


def test_graph_paths_delegates_to_ce():
    stub = _make_stub_engine(
        {"status": "success", "paths": [{"edges": [], "symbolic": "c1→c2"}]}
    )
    app.dependency_overrides[get_evidence_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post(
            "/api/knowledge-mgmt/graph/paths",
            json={"source_id": "c1", "target_id": "c2"},
        )
        assert resp.status_code == 200
        action, payload = stub.run.call_args[0]
        assert action.value == "graph_paths"
        assert payload["source_id"] == "c1"
        assert payload["target_id"] == "c2"
        assert resp.json()["status"] == "success"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/knowledge-mgmt/graph/neighbors/{concept_id}
# ---------------------------------------------------------------------------


def test_graph_neighbors_delegates_to_ce():
    stub = _make_stub_engine({"records": [{"id": "c2"}]})
    app.dependency_overrides[get_evidence_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get("/api/knowledge-mgmt/graph/neighbors/c1")
        assert resp.status_code == 200
        action, payload = stub.run.call_args[0]
        assert action.value == "neighbors"
        assert payload["node_id"] == "c1"
        assert resp.json()["records"] == [{"id": "c2"}]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/knowledge-mgmt/graph/concepts/by_ids
# ---------------------------------------------------------------------------


def test_concepts_by_ids_delegates_to_ce():
    stub = _make_stub_engine(
        {"concepts": [{"id": "c1", "name": "n", "type": "t", "description": "d"}]}
    )
    app.dependency_overrides[get_evidence_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post(
            "/api/knowledge-mgmt/graph/concepts/by_ids",
            json={"ids": ["c1"]},
        )
        assert resp.status_code == 200
        action, payload = stub.run.call_args[0]
        assert action.value == "concepts_by_ids"
        assert payload["ids"] == ["c1"]
        assert resp.json()["concepts"][0]["id"] == "c1"
    finally:
        app.dependency_overrides.clear()
