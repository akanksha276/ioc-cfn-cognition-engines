# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests / API smoke tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from ingestion.app.api import routes as api_routes
from ingestion.app.dependencies import get_ingest_data_service, get_vector_store
from ingestion.app.main import app
from ingestion.tests.conftest import build_extraction_request


class _StubIngestService:
    def __init__(self):
        self.calls = []

    def ingest(self, payload_data, request_id=None, format_descriptor=None):
        self.calls.append(
            {
                "payload_data": payload_data,
                "request_id": request_id,
                "format_descriptor": format_descriptor,
            }
        )
        return {
            "knowledge_cognition_request_id": request_id or "stub-id",
            "concepts": [
                {
                    "id": "c1",
                    "name": "agent_a",
                    "description": "Agent A",
                    "type": "concept",
                    "attributes": {"concept_type": "agent"},
                }
            ],
            "relations": [],
            "descriptor": format_descriptor or "observe-sdk-otel",
            "meta": {
                "records_processed": len(payload_data),
                "concepts_extracted": 1,
                "relations_extracted": 0,
            },
            "rag_chunks": [{"text": "chunk", "embedding": [[0.1]], "metadata": {"chunk_index": 0}}],
        }


class _StubVectorStore:
    def __init__(self):
        self.stored = None

    def store_concepts(self, concepts):
        self.stored = concepts


@pytest.fixture
def stub_ingest_service():
    return _StubIngestService()


@pytest.fixture
def stub_vector_store():
    return _StubVectorStore()


@pytest.fixture
def client(monkeypatch, stub_ingest_service, stub_vector_store):
    """Create a test client with ingestion/vector dependencies stubbed."""
    class _IdentityProcessor:
        def process(self, result):
            return result

    monkeypatch.setattr(
        api_routes, "get_knowledge_processor", lambda: _IdentityProcessor()
    )
    app.dependency_overrides[get_ingest_data_service] = lambda: stub_ingest_service
    app.dependency_overrides[get_vector_store] = lambda: stub_vector_store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


SAMPLE_OTEL_DATA = [
    {
        "TraceId": "test_trace_001",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "test.agent",
        "SpanKind": "Server",
        "ServiceName": "test.service",
        "SpanAttributes": {
            "agent_id": "test_agent",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Tell me about testing.",
        },
        "Duration": 1000000,
    },
    {
        "TraceId": "test_trace_001",
        "SpanId": "span_002",
        "ParentSpanId": "span_001",
        "SpanName": "child.agent",
        "SpanKind": "Client",
        "ServiceName": "test.service",
        "SpanAttributes": {
            "agent_id": "child_agent",
            "gen_ai.request.model": "gpt-4o",
        },
        "Duration": 500000,
    },
]


@pytest.fixture
def sample_request_body():
    """Build a valid ExtractionRequest envelope."""
    return build_extraction_request(otel_records=SAMPLE_OTEL_DATA, request_id="test_request_id")


@pytest.fixture
def sample_request_body_with_id():
    """Build a valid ExtractionRequest envelope with a client-supplied request_id."""
    return build_extraction_request(
        SAMPLE_OTEL_DATA,
        request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert data["status"] == "running"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "metrics" in data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        assert client.get("/api/v1/metrics").status_code == 200

    def test_metrics_returns_expected_fields(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "records_processed" in data
        assert "records_sent" in data
        assert "records_failed" in data
        assert "recent_errors" in data


# ---------------------------------------------------------------------------
# POST /api/knowledge-mgmt/extraction  (primary endpoint)
# ---------------------------------------------------------------------------

class TestKnowledgeExtractionEndpoint:
    """Tests for the unified knowledge extraction endpoint."""

    def test_returns_200(self, client, sample_request_body):
        resp = client.post(
            "/api/knowledge-mgmt/extraction",
            json=sample_request_body,
        )
        assert resp.status_code == 200

    def test_response_has_header(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "header" in data
        assert data["header"]["workspace_id"] == "test-ws"
        assert data["header"]["mas_id"] == "test-mas"

    def test_response_has_response_id(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "response_id" in data
        assert len(data["response_id"]) > 0

    def test_echoes_request_id(self, client, sample_request_body_with_id):
        data = client.post(
            "/api/knowledge-mgmt/extraction", json=sample_request_body_with_id
        ).json()
        assert data["response_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_returns_concepts(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "concepts" in data
        assert isinstance(data["concepts"], list)

    def test_returns_relations(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "relations" in data
        assert isinstance(data["relations"], list)

    def test_returns_descriptor(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert data["descriptor"] == "observe-sdk-otel"

    def test_returns_metadata(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "metadata" in data
        meta = data["metadata"]
        assert "records_processed" in meta
        assert "concepts_extracted" in meta
        assert "relations_extracted" in meta

    def test_returns_rag_chunks(self, client, sample_request_body):
        data = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body).json()
        assert "rag_chunks" in data
        assert isinstance(data["rag_chunks"], list)
        assert data["rag_chunks"][0]["metadata"]["chunk_index"] == 0

    def test_unsupported_format_returns_422(self, client):
        body = build_extraction_request(SAMPLE_OTEL_DATA, data_format="unknown-format")
        resp = client.post("/api/knowledge-mgmt/extraction", json=body)
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data

        errors = data["detail"]

        # Assert format enum validation error exists
        format_errors = [e for e in errors if
            e["loc"] == ["body", "payload", "metadata", "format"]]

        assert format_errors, "Expected validation error for unsupported format"
        assert "observe-sdk-otel" in format_errors[0]["msg"]
        assert "openclaw" in format_errors[0]["msg"]

    def test_empty_data_returns_400(self, client):
        body = build_extraction_request([], request_id="test_request_id")
        resp = client.post("/api/knowledge-mgmt/extraction", json=body)
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_missing_header_returns_422(self, client):
        """Pydantic rejects a body without the required header."""
        body = {"payload": {"metadata": {"format": "observe-sdk-otel"}, "data": [{}]}}
        resp = client.post("/api/knowledge-mgmt/extraction", json=body)
        assert resp.status_code == 422

    def test_missing_request_id_returns_422(self, client):
        body = {
            "header": {"workspace_id": "ws", "mas_id": "mas"},
            "payload": {"metadata": {"format": "observe-sdk-otel"}, "data": [{}]},
        }
        resp = client.post("/api/knowledge-mgmt/extraction", json=body)
        assert resp.status_code == 422

    def test_missing_format_returns_422(self, client):
        body = {
            "header": {"workspace_id": "ws", "mas_id": "mas"},
            "request_id": "r1",
            "payload": {"metadata": {}, "data": [{}]},
        }
        resp = client.post("/api/knowledge-mgmt/extraction", json=body)
        assert resp.status_code == 422

    def test_persists_concepts_to_faiss(self, client, sample_request_body, stub_vector_store):
        resp = client.post("/api/knowledge-mgmt/extraction", json=sample_request_body)
        assert resp.status_code == 200
        assert isinstance(stub_vector_store.stored, list)
        assert stub_vector_store.stored[0]["name"] == "agent_a"

    def test_faiss_storage_error_is_non_fatal(self, monkeypatch, stub_ingest_service, sample_request_body):
        class _FailingVectorStore:
            def store_concepts(self, concepts):
                raise RuntimeError("vector store unavailable")

        class _IdentityProcessor:
            def process(self, result):
                return result

        monkeypatch.setattr(
            api_routes, "get_knowledge_processor", lambda: _IdentityProcessor()
        )
        app.dependency_overrides[get_ingest_data_service] = lambda: stub_ingest_service
        app.dependency_overrides[get_vector_store] = lambda: _FailingVectorStore()
        with TestClient(app) as local_client:
            resp = local_client.post("/api/knowledge-mgmt/extraction", json=sample_request_body)
            assert resp.status_code == 200
            assert "error" not in resp.json()
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# File-based endpoints (unchanged format, kept for dev/testing)
# ---------------------------------------------------------------------------

class TestFileExtractionEndpoint:
    def test_file_extraction_missing_file_returns_404(self, client):
        response = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": "/nonexistent/file.json"},
        )
        assert response.status_code == 404

    def test_file_extraction_invalid_extension_returns_400(self, client, tmp_path):
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("[]")
        response = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": str(invalid_file)},
        )
        assert response.status_code == 400

    def test_concept_file_extraction_missing_file_returns_404(self, client):
        response = client.get(
            "/api/v1/extract/concepts_and_relationships/from_file",
            params={"file_path": "/nonexistent/file.json"},
        )
        assert response.status_code == 404
