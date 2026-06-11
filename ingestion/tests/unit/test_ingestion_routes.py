# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ingestion routes.py — all endpoints routed through IngestionCognitionEngine."""

import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from ingestion.app.api import routes as api_routes
from ingestion.app.main import app
from ingestion.app.dependencies import (
    get_ingestion_cognition_engine,
)
from ingestion.app.agent.ingestion_ce import IngestionCognitionEngine

# ---------------------------------------------------------------------------
# Shared stub results
# ---------------------------------------------------------------------------

_STUB_INGEST_RESULT = {
    "knowledge_cognition_request_id": "req-stub",
    "concepts": [{"id": "c1", "name": "agent_a", "type": "concept", "description": "d", "attributes": {}}],
    "relations": [],
    "descriptor": "observe-sdk-otel",
    "meta": {"records_processed": 1},
    "rag_chunks": [],
}

_STUB_METRICS_RESULT = {
    "records_processed": 5,
    "records_sent": 4,
    "records_failed": 1,
    "last_run_timestamp": None,
    "last_run_duration_seconds": 0.5,
    "recent_errors": [],
}

_STUB_FILE_RESULT = {
    "concepts": [{"id": "c2", "name": "worker_b", "type": "concept"}],
    "relations": [],
}

_EXTRACTION_BODY = {
    "header": {"workspace_id": "ws-1", "mas_id": "mas-1"},
    "request_id": "req-1",
    "payload": {
        "metadata": {"format": "observe-sdk-otel"},
        "data": [{"SpanId": "s1", "SpanName": "agent.call"}],
    },
}


def _make_stub_engine(run_return=None) -> IngestionCognitionEngine:
    engine = MagicMock(spec=IngestionCognitionEngine)
    engine.run = AsyncMock(return_value=run_return or _STUB_INGEST_RESULT)
    return engine


# ---------------------------------------------------------------------------
# POST /api/knowledge-mgmt/extraction
# ---------------------------------------------------------------------------


def test_extraction_endpoint_delegates_to_ce():
    stub = _make_stub_engine(_STUB_INGEST_RESULT)
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/knowledge-mgmt/extraction", json=_EXTRACTION_BODY)
        assert resp.status_code == 200
        stub.run.assert_awaited_once()
        action, payload = stub.run.call_args[0]
        assert action.value == "ingest"
        assert payload["records"] == _EXTRACTION_BODY["payload"]["data"]
        assert payload["format"] == "observe-sdk-otel"
        assert payload["request_id"] == "req-1"
        body = resp.json()
        assert body["response_id"] == "req-1"
        assert body["concepts"] == _STUB_INGEST_RESULT["concepts"]
    finally:
        app.dependency_overrides.clear()


def test_otel_trace_span_rows_are_normalized_for_extraction():
    records = api_routes._normalize_records_for_format(
        [
            {
                "start_time": "2026-06-03T19:00:00.000000123Z",
                "trace_id": "trace-1",
                "span_id": "span-1",
                "parent_span_id": "",
                "name": "agent.reply",
                "service_name": "openclaw",
                "kind": 1,
                "duration_nano": 2_500,
                "status_code": 1,
                "attributes": {"ioa_observe.span.kind": "agent"},
                "events": [],
                "links": [],
                "resource": {"service.name": "openclaw"},
            }
        ],
        "otel-trace",
    )
    assert records[0]["traceId"] == "trace-1"
    assert records[0]["spanId"] == "span-1"
    assert records[0]["startTime"][0] != 0


def test_normalize_records_rejects_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        api_routes._normalize_records_for_format([], "unknown-format")


@pytest.mark.asyncio
async def test_extraction_task_callback_payload(monkeypatch):
    calls = []

    class _FakeResponse:
        status_code = 200
        text = ""

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            calls.append({"url": url, "json": json})
            return _FakeResponse()

    monkeypatch.setattr(api_routes.httpx, "AsyncClient", _FakeAsyncClient)

    request = api_routes.ExtractionRequest.model_validate(
        {
            **_EXTRACTION_BODY,
            "taskCallbackUrl": "http://cfn-svc/api/internal/tasks/callback",
        }
    )
    response = api_routes.ExtractionResponseModel(
        header=request.header,
        response_id=request.request_id,
        concepts=_STUB_INGEST_RESULT["concepts"],
        relations=[],
    )

    await api_routes._post_task_callback(
        body=request,
        status="success",
        response=response,
    )

    assert len(calls) == 1
    callback = calls[0]["json"]
    assert calls[0]["url"] == "http://cfn-svc/api/internal/tasks/callback"
    assert callback["workspace_id"] == "ws-1"
    assert callback["mas_id"] == "mas-1"
    assert callback["task_name"] == "otel-task"
    assert callback["status"] == "success"
    result = json.loads(callback["result"])
    assert result["response_id"] == "req-1"
    assert result["concepts"] == _STUB_INGEST_RESULT["concepts"]


def test_run_otel_task_endpoint_removed():
    assert all("runOtelTask" not in getattr(route, "path", "") for route in app.routes)


def test_extraction_endpoint_returns_500_on_engine_error():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=RuntimeError("boom"))
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/knowledge-mgmt/extraction", json=_EXTRACTION_BODY)
        assert resp.status_code == 500
    finally:
        app.dependency_overrides.clear()


def test_extraction_endpoint_bad_request_on_value_error():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=ValueError("bad format"))
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.post("/api/knowledge-mgmt/extraction", json=_EXTRACTION_BODY)
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint_delegates_to_ce():
    stub = _make_stub_engine(_STUB_METRICS_RESULT)
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        stub.run.assert_awaited_once()
        action, payload = stub.run.call_args[0]
        assert action.value == "metrics"
        assert payload == {}
        assert resp.json()["records_processed"] == 5
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/extract/entities_and_relations/from_file
# ---------------------------------------------------------------------------


def test_extract_from_file_endpoint_delegates_to_ce():
    stub = _make_stub_engine(_STUB_FILE_RESULT)
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": "/tmp/test.json", "save_output": "false"},
        )
        assert resp.status_code == 200
        action, payload = stub.run.call_args[0]
        assert action.value == "extract_from_file"
        assert payload["file_path"] == "/tmp/test.json"
        assert payload["save_output"] is False
    finally:
        app.dependency_overrides.clear()


def test_extract_from_file_returns_404_on_not_found():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=FileNotFoundError("not found"))
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": "/missing.json"},
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/extract/concepts_and_relationships/from_file
# ---------------------------------------------------------------------------


def test_ingest_from_file_endpoint_delegates_to_ce():
    stub = _make_stub_engine(_STUB_FILE_RESULT)
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/extract/concepts_and_relationships/from_file",
            params={"file_path": "/tmp/test.json", "save_output": "true"},
        )
        assert resp.status_code == 200
        action, payload = stub.run.call_args[0]
        assert action.value == "ingest_from_file"
        assert payload["file_path"] == "/tmp/test.json"
        assert payload["save_output"] is True
    finally:
        app.dependency_overrides.clear()


def test_ingest_from_file_returns_400_on_value_error():
    stub = _make_stub_engine()
    stub.run = AsyncMock(side_effect=ValueError("bad path"))
    app.dependency_overrides[get_ingestion_cognition_engine] = lambda: stub

    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/extract/concepts_and_relationships/from_file",
            params={"file_path": "/tmp/test.json"},
        )
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()
