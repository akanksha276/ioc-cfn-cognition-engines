# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared diagnostics router factory."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.diagnostics.router import HealthCheck, HealthState, make_diagnostics_router




def _make_app(health_checks=None) -> FastAPI:
    """Minimal FastAPI app with the diagnostics router mounted."""
    app = FastAPI()
    app.include_router(
        make_diagnostics_router(
            service_name="test-service",
            version="1.2.3",
            description="A test service",
            health_checks=health_checks,
        ),
        prefix="/api/internal/diagnostics",
        include_in_schema=False,
    )
    return app


@pytest.fixture
def client():
    return TestClient(_make_app())




class TestHealth:
    def test_up_when_no_checks(self, client):
        r = client.get("/api/internal/diagnostics/health")
        assert r.status_code == 200
        assert r.json()["status"] == "UP"

    def test_response_shape(self, client):
        data = client.get("/api/internal/diagnostics/health").json()
        assert "status" in data
        assert "service_name" in data
        assert "service_state" in data
        assert "last_updated" in data

    def test_down_when_critical_check_fails(self):
        app = _make_app(health_checks=[
            HealthCheck("db", lambda: False, critical=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 500
            data = r.json()
            assert data["status"] == "DOWN"
            assert data["checks"]["db"] is False

    def test_degraded_when_optional_check_fails(self):
        app = _make_app(health_checks=[
            HealthCheck("cache", lambda: False, critical=False),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 200
            assert r.json()["status"] == "DEGRADED"

    def test_down_takes_priority_over_degraded(self):
        app = _make_app(health_checks=[
            HealthCheck("optional", lambda: False, critical=False),
            HealthCheck("critical", lambda: False, critical=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 500
            assert r.json()["status"] == "DOWN"

    def test_check_exception_treated_as_failure(self):
        def _boom():
            raise RuntimeError("unreachable")

        app = _make_app(health_checks=[HealthCheck("bad", _boom, critical=True)])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 500
            assert r.json()["checks"]["bad"] is False

    def test_passing_checks_included_in_response(self):
        app = _make_app(health_checks=[HealthCheck("db", lambda: True)])
        with TestClient(app) as c:
            data = c.get("/api/internal/diagnostics/health").json()
            assert data["checks"]["db"] is True

    def test_no_checks_key_when_no_checks_registered(self, client):
        data = client.get("/api/internal/diagnostics/health").json()
        assert "checks" not in data

    def test_external_check_skipped_by_default(self):
        app = _make_app(health_checks=[
            HealthCheck("cfn", lambda: False, critical=True, external=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 200
            assert r.json()["status"] == "UP"
            assert "cfn" not in r.json().get("checks", {})

    def test_external_check_included_with_dependencies_param(self):
        app = _make_app(health_checks=[
            HealthCheck("cfn", lambda: True, critical=True, external=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health?dependencies=true")
            assert r.status_code == 200
            assert r.json()["checks"]["cfn"] is True

    def test_external_check_failure_hidden_without_param(self):
        app = _make_app(health_checks=[
            HealthCheck("cfn", lambda: False, critical=True, external=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health")
            assert r.status_code == 200
            assert r.json()["status"] == "UP"

    def test_external_check_failure_visible_with_param(self):
        app = _make_app(health_checks=[
            HealthCheck("cfn", lambda: False, critical=True, external=True),
        ])
        with TestClient(app) as c:
            r = c.get("/api/internal/diagnostics/health?dependencies=true")
            assert r.status_code == 500
            assert r.json()["status"] == "DOWN"
            assert r.json()["checks"]["cfn"] is False




class TestInfo:
    def test_returns_200(self, client):
        assert client.get("/api/internal/diagnostics/info").status_code == 200

    def test_service_fields(self, client):
        data = client.get("/api/internal/diagnostics/info").json()
        assert data["service"] == "test-service"
        assert data["version"] == "1.2.3"
        assert data["description"] == "A test service"

    def test_git_block_present(self, client):
        data = client.get("/api/internal/diagnostics/info").json()
        assert "git" in data
        assert "commit" in data["git"]
        assert "id" in data["git"]["commit"]
        assert "time" in data["git"]["commit"]
        assert "branch" in data["git"]

    def test_git_reads_env_vars(self, monkeypatch, client):
        monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
        monkeypatch.setenv("GIT_COMMIT_TIME", "2026-01-01T00:00:00Z")
        monkeypatch.setenv("GIT_BRANCH", "main")
        # Re-create app so the endpoint re-reads env at call time
        with TestClient(_make_app()) as c:
            data = c.get("/api/internal/diagnostics/info").json()
        assert data["git"]["commit"]["id"] == "abc123"
        assert data["git"]["commit"]["time"] == "2026-01-01T00:00:00Z"
        assert data["git"]["branch"] == "main"

    def test_git_defaults_to_unknown(self, monkeypatch):
        monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
        monkeypatch.delenv("GIT_COMMIT_TIME", raising=False)
        monkeypatch.delenv("GIT_BRANCH", raising=False)
        with TestClient(_make_app()) as c:
            data = c.get("/api/internal/diagnostics/info").json()
        assert data["git"]["commit"]["id"] == "unknown"
        assert data["git"]["branch"] == "unknown"




class TestMetrics:
    def test_returns_200(self, client):
        assert client.get("/api/internal/diagnostics/metrics").status_code == 200

    def test_uptime_present_and_positive(self, client):
        data = client.get("/api/internal/diagnostics/metrics").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_threads_present(self, client):
        data = client.get("/api/internal/diagnostics/metrics").json()
        assert "threads" in data
        assert data["threads"] >= 1




class TestGetLoggers:
    def test_returns_200(self, client):
        assert client.get("/api/internal/diagnostics/loggers").status_code == 200

    def test_top_level_log_level_present(self, client):
        data = client.get("/api/internal/diagnostics/loggers").json()
        assert "log-level" in data

    def test_loggers_dict_present(self, client):
        data = client.get("/api/internal/diagnostics/loggers").json()
        assert "loggers" in data
        assert isinstance(data["loggers"], dict)

    def test_root_logger_in_loggers(self, client):
        data = client.get("/api/internal/diagnostics/loggers").json()
        assert "root" in data["loggers"]

    def test_logger_entry_has_configured_and_effective_level(self, client):
        # Ensure at least the root entry has both fields
        data = client.get("/api/internal/diagnostics/loggers").json()
        root = data["loggers"]["root"]
        assert "configured_level" in root
        assert "effective_level" in root




class TestPutLoggers:
    def test_returns_204_on_success(self, client):
        r = client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "DEBUG"},
        )
        assert r.status_code == 204

    def test_no_body_on_success(self, client):
        r = client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "INFO"},
        )
        assert r.content == b""

    def test_sets_root_logger_level(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "WARNING"},
        )
        assert logging.getLogger().level == logging.WARNING
        # restore
        logging.getLogger().setLevel(logging.INFO)

    def test_ROOT_alias_sets_root_logger(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "ROOT", "log-level": "ERROR"},
        )
        assert logging.getLogger().level == logging.ERROR
        logging.getLogger().setLevel(logging.INFO)

    def test_empty_string_alias_sets_root_logger(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "", "log-level": "DEBUG"},
        )
        assert logging.getLogger().level == logging.DEBUG
        logging.getLogger().setLevel(logging.INFO)

    def test_trace_mapped_to_debug(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "TRACE"},
        )
        assert logging.getLogger().level == logging.DEBUG
        logging.getLogger().setLevel(logging.INFO)

    def test_warn_mapped_to_warning(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "WARN"},
        )
        assert logging.getLogger().level == logging.WARNING
        logging.getLogger().setLevel(logging.INFO)

    def test_sets_named_module_level(self, client):
        client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "common.diagnostics.test_module", "log-level": "DEBUG"},
        )
        assert logging.getLogger("common.diagnostics.test_module").level == logging.DEBUG

    def test_invalid_level_returns_400(self, client):
        r = client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "NONSENSE"},
        )
        assert r.status_code == 400
        assert "error" in r.json()

    def test_missing_body_returns_422(self, client):
        r = client.put("/api/internal/diagnostics/loggers")
        assert r.status_code == 422

    def test_case_insensitive_level(self, client):
        r = client.put(
            "/api/internal/diagnostics/loggers",
            json={"module-name": "root", "log-level": "debug"},
        )
        assert r.status_code == 204




class TestOpenAPISchema:
    def test_diagnostics_not_in_openapi_schema(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        diag_paths = [p for p in paths if "/api/internal/diagnostics" in p]
        assert diag_paths == [], f"Diagnostics paths should be hidden but found: {diag_paths}"
