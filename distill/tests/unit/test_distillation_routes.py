# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi.testclient import TestClient

from distill.app.config import settings as settings_mod
from distill.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_distillation_run_rejects_non_https_callback(client, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")
    r = client.post(
        "/api/knowledge-mgmt/distillation/run",
        json={
            "header": {"workspace_id": "w", "mas_id": "m"},
            "request_id": "rid-1",
            "payload": {"callback_url": "http://evil.test/cb"},
        },
    )
    assert r.status_code == 400
    assert r.json()["status"] == "error"


def test_distillation_run_409_when_lock_held(client, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")
    import distill.app.api.routes as routes_mod

    async def fake_try_begin(*_a, **_k):
        return False

    monkeypatch.setattr(routes_mod.distillation_lock, "try_begin_run", fake_try_begin)
    r = client.post(
        "/api/knowledge-mgmt/distillation/run",
        json={
            "header": {"workspace_id": "w", "mas_id": "m"},
            "request_id": "rid-2",
            "payload": {"callback_url": "https://cfn.test/callback"},
        },
    )
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "conflict"
    assert body["workspaceId"] == "w"


def test_distillation_run_202_shape(client, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")

    async def fake_try_begin(*_a, **_k):
        return True

    import distill.app.api.routes as routes_mod
    from distill.app.services import distillation_lock as dl

    async def fake_fetch(*_a, **_k):
        return {"concepts": [], "relations": []}

    async def fake_execute(**kwargs):
        assert "prefetched_graph_read" in kwargs
        assert kwargs["prefetched_graph_read"] == {"concepts": [], "relations": []}
        h = kwargs["header"]
        await dl.end_run(h.workspace_id, h.mas_id)

    monkeypatch.setattr(routes_mod.distillation_lock, "try_begin_run", fake_try_begin)
    monkeypatch.setattr(routes_mod, "fetch_distillation_graph_read", fake_fetch)
    monkeypatch.setattr(routes_mod, "execute_distillation_run", fake_execute)

    r = client.post(
        "/api/knowledge-mgmt/distillation/run",
        json={
            "header": {"workspace_id": "w", "mas_id": "m", "agent_id": "a1"},
            "request_id": "rid-3",
            "payload": {"callback_url": "https://cfn.test/callback"},
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted"
    assert "operationId" in body
    assert body["message"] == "distillation started"


def test_distillation_run_502_when_graph_read_fails(client, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")

    async def fake_try_begin(*_a, **_k):
        return True

    import distill.app.api.routes as routes_mod
    from distill.app.services import distillation_lock as dl

    ended: list[tuple[str, str]] = []

    async def fake_fetch(*_a, **_k):
        raise RuntimeError("upstream down")

    orig_end = dl.end_run

    async def track_end(w, m):
        ended.append((w, m))
        await orig_end(w, m)

    monkeypatch.setattr(routes_mod.distillation_lock, "try_begin_run", fake_try_begin)
    monkeypatch.setattr(routes_mod, "fetch_distillation_graph_read", fake_fetch)
    monkeypatch.setattr(dl, "end_run", track_end)

    r = client.post(
        "/api/knowledge-mgmt/distillation/run",
        json={
            "header": {"workspace_id": "w", "mas_id": "m"},
            "request_id": "rid-502",
            "payload": {"callback_url": "https://cfn.test/callback"},
        },
    )
    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "error"
    assert "graph read failed" in body["message"]
    assert ended == [("w", "m")]
