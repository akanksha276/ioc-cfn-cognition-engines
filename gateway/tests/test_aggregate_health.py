# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gateway's aggregating /api/internal/diagnostics/health endpoint."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


def _make_sub_app(
    status: str,
    checks: dict,
    http_status: int = 200,
    dependencies_checks: dict | None = None,
) -> FastAPI:
    """Tiny mock sub-app that returns a controlled health response.

    When ``dependencies_checks`` is provided, it is returned instead of
    ``checks`` when the caller passes ``?dependencies=true``.
    """
    mock = FastAPI()

    @mock.get("/api/internal/diagnostics/health")
    async def health(dependencies: bool = False):
        effective_checks = (
            dependencies_checks
            if dependencies and dependencies_checks is not None
            else checks
        )
        return JSONResponse(
            content={"status": status, "checks": effective_checks},
            status_code=http_status,
        )

    return mock


@pytest.fixture(scope="module")
def gateway_app():
    """Import the gateway app with all heavy sub-app dependencies mocked out."""
    # Stub out modules that require external dependencies (dotenv, fastembed, etc.)
    stubs = {
        "dotenv": MagicMock(),
        "ingestion": MagicMock(),
        "ingestion.app": MagicMock(),
        "ingestion.app.main": MagicMock(app=FastAPI()),
        "ingestion.app.api": MagicMock(),
        "ingestion.app.api.routes": MagicMock(
            extraction_router=FastAPI().router,
        ),
        "ingestion.app.agent": MagicMock(),
        "ingestion.app.agent.knowledge_processor": MagicMock(EmbeddingManager=MagicMock()),
        "evidence": MagicMock(),
        "evidence.app": MagicMock(),
        "evidence.app.main": MagicMock(app=FastAPI()),
        "evidence.app.api": MagicMock(),
        "evidence.app.api.routes": MagicMock(router=FastAPI().router),
        "semantic_alignment": MagicMock(),
        "semantic_alignment.app": MagicMock(),
        "semantic_alignment.app.main": MagicMock(app=FastAPI()),
        "semantic_alignment.app.api": MagicMock(),
        "semantic_alignment.app.api.routes": MagicMock(router=FastAPI().router),
        "distill": MagicMock(),
        "distill.app": MagicMock(),
        "distill.app.main": MagicMock(app=FastAPI()),
        "distill.app.api": MagicMock(),
        "distill.app.api.routes": MagicMock(router=FastAPI().router),
        "caching": MagicMock(),
        "caching.app": MagicMock(),
        "caching.app.agent": MagicMock(),
        "caching.app.agent.caching_layer": MagicMock(CachingLayer=MagicMock()),
    }

    # Remove any previously imported gateway modules so we get a fresh import
    for key in list(sys.modules.keys()):
        if key.startswith("gateway"):
            del sys.modules[key]

    with patch.dict(sys.modules, stubs):
        import gateway.app.main as gm
        # Patch register_cognition_engines so lifespan doesn't call out
        gm_registration = MagicMock()
        gm_registration.register_cognition_engines = MagicMock(return_value=None)
        with patch.dict(sys.modules, {"gateway.app.registration": gm_registration}):
            pass
        return gm


@pytest.fixture()
def client(gateway_app):
    return TestClient(gateway_app.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_cache_layer(gateway_app):
    yield
    if hasattr(gateway_app.app.state, "cache_layer"):
        del gateway_app.app.state.cache_layer


@pytest.fixture(autouse=True)
def patch_distill_sub_app_health(gateway_app, monkeypatch):
    """distill stub mount has no diagnostics route unless patched."""
    monkeypatch.setattr(gateway_app, "_distill_app", _make_sub_app("UP", {}))


class TestAggregateHealth:
    def test_all_up(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {"embedding_model": True}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {"data_layer": True}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "UP"
        assert set(body["services"]) == {"gateway", "ingestion", "evidence", "distill", "semantic_alignment"}

    def test_ingestion_critical_down(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("DOWN", {"embedding_model": False}, http_status=500))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {"data_layer": True}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 500
        assert resp.json()["status"] == "DOWN"

    def test_evidence_degraded(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {"embedding_model": True}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("DEGRADED", {"data_layer": False}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "DEGRADED"

    def test_down_takes_priority_over_degraded(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("DOWN", {"embedding_model": False}, http_status=500))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("DEGRADED", {"data_layer": False}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 500
        assert resp.json()["status"] == "DOWN"

    def test_distill_sub_app_http_500_drives_overall_down(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {"embedding_model": True}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {"data_layer": True}))
        monkeypatch.setattr(gateway_app, "_distill_app", _make_sub_app("DOWN", {}, http_status=500))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 500
        body = resp.json()
        assert body["status"] == "DOWN"
        assert body["services"]["distill"]["status"] == "DOWN"

    def test_response_shape(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        body = client.get("/api/internal/diagnostics/health").json()
        assert "status" in body
        assert "services" in body
        assert set(body["services"]) == {"gateway", "ingestion", "evidence", "distill", "semantic_alignment"}

    def test_sub_app_exception_gives_unknown(self, client, gateway_app, monkeypatch):
        gateway_app.app.state.cache_layer = object()

        broken = FastAPI()

        @broken.get("/api/internal/diagnostics/health")
        async def boom():
            raise RuntimeError("connection refused")

        monkeypatch.setattr(gateway_app, "_ingestion_app", broken)
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}))

        resp = client.get("/api/internal/diagnostics/health")
        assert resp.status_code == 500
        body = resp.json()
        assert body["status"] == "DOWN"
        assert body["services"]["ingestion"]["status"] == "UNKNOWN"

    def test_excluded_from_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/internal/diagnostics/health" not in schema.get("paths", {})

    def test_light_mode_no_external_checks(self, client, gateway_app, monkeypatch):
        """Without ?dependencies, sub-apps are called without the param — no external checks."""
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {"embedding_model": True}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {}, dependencies_checks={"cognition_fabric_node": True}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}, dependencies_checks={"cognition_fabric_node": True}))

        body = client.get("/api/internal/diagnostics/health").json()
        assert body["status"] == "UP"
        assert body["services"]["evidence"]["checks"] == {}
        assert body["services"]["semantic_alignment"]["checks"] == {}

    def test_dependencies_param_forwarded_to_sub_apps(self, client, gateway_app, monkeypatch):
        """With ?dependencies=true, the param is forwarded and external checks appear."""
        gateway_app.app.state.cache_layer = object()
        monkeypatch.setattr(gateway_app, "_ingestion_app", _make_sub_app("UP", {"embedding_model": True}))
        monkeypatch.setattr(gateway_app, "_evidence_app", _make_sub_app("UP", {}, dependencies_checks={"cognition_fabric_node": True}))
        monkeypatch.setattr(gateway_app, "_semantic_alignment_app", _make_sub_app("UP", {}, dependencies_checks={"cognition_fabric_node": True}))

        body = client.get("/api/internal/diagnostics/health?dependencies=true").json()
        assert body["status"] == "UP"
        assert body["services"]["evidence"]["checks"]["cognition_fabric_node"] is True
        assert body["services"]["semantic_alignment"]["checks"]["cognition_fabric_node"] is True
