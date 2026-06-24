# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for semantic_alignment routes.py — all endpoints routed through NegotiationCognitionEngine."""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from semantic_alignment.app.agent.semantic_alignment_ce import NegotiationCognitionEngine
from semantic_alignment.app.dependencies import get_negotiation_cognition_engine
from semantic_alignment.app.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_engine(run_return=None) -> NegotiationCognitionEngine:
    engine = MagicMock(spec=NegotiationCognitionEngine)
    engine.run = AsyncMock(return_value=run_return or {})
    return engine


_INITIATE_BODY = {
    "kind": "negotiate",
    "message_id": "msg-1",
    "dt_created": "2026-04-09T10:00:00Z",
    "origin": {"actor_id": "mas-1", "tenant_id": "ws-1"},
    "semantic_context": {
        "session_id": "sess-1",
        "schema_id": "urn:ioc:schema:negotiate:v1",
        "schema_version": "1.0",
        "encoding": "json",
    },
    "payload_hash": "sha256:deadbeef",
    "policy_labels": {
        "sensitivity": "internal",
        "propagation": "forward",
        "retention_policy": "default",
    },
    "provenance": {"sources": [], "transforms": []},
    "payload": {
        "content_text": "Negotiate email archival policy",
        "agents": [{"id": "agent-a", "name": "Agent A"}, {"id": "agent-b", "name": "Agent B"}],
        "n_steps": 10,
    },
}

_DECIDE_BODY = {
    "kind": "negotiate",
    "message_id": "msg-2",
    "dt_created": "2026-04-09T10:00:00Z",
    "origin": {"actor_id": "mas-1", "tenant_id": "ws-1"},
    "semantic_context": {
        "session_id": "sess-1",
        "schema_id": "urn:ioc:schema:negotiate:v1",
        "schema_version": "1.0",
        "encoding": "json",
    },
    "payload_hash": "sha256:deadbeef",
    "policy_labels": {
        "sensitivity": "internal",
        "propagation": "forward",
        "retention_policy": "default",
    },
    "provenance": {"sources": [], "transforms": []},
    "payload": {
        "session_id": "sess-1",
        "agent_replies": [
            {"agent_id": "agent-a", "action": "counter_offer", "offer": {"issue1": "opt1"}},
        ],
    },
}

_INITIATE_RESULT = {
    "session_id": "sess-1",
    "issues": ["archive_age"],
    "options_per_issue": {"archive_age": ["30d", "1y"]},
    "n_steps": 10,
    "messages": [],
    "status": "ongoing",
}

_DECIDE_ONGOING_RESULT = {
    "status": "ongoing",
    "session_id": "sess-1",
    "round": 2,
    "messages": [],
}

_DECIDE_TERMINAL_RESULT = {
    "status": "agreed",
    "session_id": "sess-1",
    "round": 5,
    "final_result": MagicMock(model_dump=lambda mode=None: {"outcome": {"archive_age": "30d"}}),
}


# ---------------------------------------------------------------------------
# POST /api/negotiate/initiate
# ---------------------------------------------------------------------------


def test_initiate_delegates_to_ce():
    stub = _make_stub_engine(_INITIATE_RESULT)
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/initiate", json=_INITIATE_BODY)
        assert resp.status_code == 200
        stub.run.assert_awaited_once()
        action, payload = stub.run.call_args[0]
        assert action.value == "initiate"
        assert payload["session_id"] == "sess-1"
        assert payload["content_text"] == "Negotiate email archival policy"
        assert len(payload["agents"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_initiate_returns_400_on_value_error():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=ValueError("missing agents"))
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/initiate", json=_INITIATE_BODY)
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_initiate_returns_500_on_unexpected_error():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=RuntimeError("pipeline failure"))
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/initiate", json=_INITIATE_BODY)
        assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/negotiate/decide
# ---------------------------------------------------------------------------


def test_decide_ongoing_returns_messages():
    stub = _make_stub_engine(_DECIDE_ONGOING_RESULT)
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/decide", json=_DECIDE_BODY)
        assert resp.status_code == 200
        stub.run.assert_awaited_once()
        action, payload = stub.run.call_args[0]
        assert action.value == "decide"
        assert payload["session_id"] == "sess-1"
        assert len(payload["agent_replies"]) == 1
        body = resp.json()
        assert body["status"] == "ongoing"
        assert body["round"] == 2
    finally:
        app.dependency_overrides.clear()


def test_decide_terminal_returns_final_result():
    stub = _make_stub_engine(_DECIDE_TERMINAL_RESULT)
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/decide", json=_DECIDE_BODY)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "agreed"
        assert "final_result" in body
    finally:
        app.dependency_overrides.clear()


def test_decide_returns_404_on_missing_session():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=KeyError("sess-missing"))
    app.dependency_overrides[get_negotiation_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/negotiate/decide", json=_DECIDE_BODY)
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()
