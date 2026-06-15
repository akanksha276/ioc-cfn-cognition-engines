# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from evidence.app.api.schemas import Header

from distill.app.config import settings as settings_mod
from distill.app.services import distillation_job as job_mod


@pytest.mark.asyncio
async def test_execute_posts_mutation_then_callback_success(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")
    monkeypatch.setattr(settings_mod.settings, "CFN_MUTATION_APPLY_URL", None)
    monkeypatch.setattr(settings_mod.settings, "CODI_MIN_EDGES", 2)
    monkeypatch.setattr(settings_mod.settings, "CODI_MAX_RELATIONS_PER_BATCH", 8)
    monkeypatch.setattr(settings_mod.settings, "CODI_RAG_TOP_K", 2)
    monkeypatch.setattr(settings_mod.settings, "DISTILLATION_MODE", "Summary")

    graph_instance = MagicMock()
    graph_instance.distillation_graph_read = AsyncMock(
        return_value={
            "concepts": [{"id": "a1", "name": "Anchor One", "type": "X"}],
            "relations": [
                {
                    "id": "rel1",
                    "node_ids": ["a1", "n2"],
                    "relationship": "R",
                    "attributes": {"source_name": "S", "target_name": "T"},
                }
            ],
        }
    )
    graph_instance.aclose = AsyncMock()

    class FakeGraphCls:
        def __init__(self, *a, **k):
            self._inner = graph_instance

        async def distillation_graph_read(self, body):
            return await graph_instance.distillation_graph_read(body)

        async def aclose(self):
            await graph_instance.aclose()

    monkeypatch.setattr(job_mod, "CoDiGraphClient", FakeGraphCls)

    async def fake_distill_batch(**kwargs):
        return (
            {
                "id": "codin-1",
                "name": "CoDiN-Anchor One-codin-1",
                "description": "distilled summary text",
                "type": "CoDiN",
                "attributes": {"embedding": [], "distill_mode": "Summary"},
            },
            [
                {
                    "id": "rel1",
                    "node_ids": ["a1", "n2"],
                    "relationship": "R",
                    "attributes": {"source_name": "S", "target_name": "T"},
                    "internal_attributes": [
                        {
                            "owner": "m",
                            "attributes": {"distill_status": "updated"},
                        },
                    ],
                },
                {
                    "id": "newrel",
                    "node_ids": ["a1", "codin-1"],
                    "relationship": "summary",
                    "attributes": {"summarized_context": "x"},
                    "internal_attributes": [
                        {
                            "owner": "m",
                            "attributes": {"distill_status": "CoDi"},
                        },
                    ],
                },
            ],
            {},
        )

    monkeypatch.setattr(job_mod, "_distill_one_batch_payload", fake_distill_batch)

    post_urls = []

    mutation_bodies: list = []

    class RecordingAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, json=None, timeout=None):
            post_urls.append(str(url))
            return httpx.Response(200, json={"ok": True})

        async def put(self, url, json=None, timeout=None):
            post_urls.append(str(url))
            if "graph/update" in str(url) or "/apply" in str(url):
                mutation_bodies.append(json)
            return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(job_mod.httpx, "AsyncClient", RecordingAsyncClient)

    hdr = Header(workspace_id="w", mas_id="m", agent_id="ag")
    await job_mod.execute_distillation_run(
        header=hdr,
        request_id="req-x",
        callback_url="https://cb.test/done",
        operation_id="op-1",
        distill_run_at="2026-05-01T00:00:00Z",
        rag_layer=None,
    )

    assert post_urls[0] == "http://graph.test/api/internal/workspaces/w/multi-agentic-systems/m/graph/update"
    assert post_urls[1] == "https://cb.test/done"
    assert mutation_bodies[0]["request_id"] == "req-x"
    assert mutation_bodies[0]["descriptor"] == "Cognition Distillation"
    assert "metadata" in mutation_bodies[0] and "meta" not in mutation_bodies[0]
    assert mutation_bodies[0]["metadata"]["distill_mode"] == "Summary"
    assert mutation_bodies[0]["header"] == {"agent_id": "ag"}


@pytest.mark.asyncio
async def test_execute_skips_callback_when_mutation_non_2xx(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "DATA_LAYER_BASE_URL", "http://graph.test")
    monkeypatch.setattr(settings_mod.settings, "CFN_MUTATION_APPLY_URL", None)
    monkeypatch.setattr(settings_mod.settings, "CODI_MIN_EDGES", 2)
    monkeypatch.setattr(settings_mod.settings, "CODI_MAX_RELATIONS_PER_BATCH", 8)
    monkeypatch.setattr(settings_mod.settings, "CODI_RAG_TOP_K", 2)
    monkeypatch.setattr(settings_mod.settings, "DISTILLATION_MODE", "Summary")

    graph_instance = MagicMock()
    graph_instance.distillation_graph_read = AsyncMock(
        return_value={
            "concepts": [{"id": "a1", "name": "Anchor One", "type": "X"}],
            "relations": [
                {
                    "id": "rel1",
                    "node_ids": ["a1", "n2"],
                    "relationship": "R",
                    "attributes": {"source_name": "S", "target_name": "T"},
                }
            ],
        }
    )
    graph_instance.aclose = AsyncMock()

    class FakeGraphCls:
        def __init__(self, *a, **k):
            pass

        async def distillation_graph_read(self, body):
            return await graph_instance.distillation_graph_read(body)

        async def aclose(self):
            await graph_instance.aclose()

    monkeypatch.setattr(job_mod, "CoDiGraphClient", FakeGraphCls)

    async def fake_distill_batch(**kwargs):
        return (
            {
                "id": "codin-1",
                "name": "CoDiN-Anchor One-codin-1",
                "description": "distilled summary text",
                "type": "CoDiN",
                "attributes": {"distill_mode": "Summary"},
            },
            [],
            {},
        )

    monkeypatch.setattr(job_mod, "_distill_one_batch_payload", fake_distill_batch)

    http_calls = []

    class RecordingAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, json=None, timeout=None):
            http_calls.append(("post", url, json))
            return httpx.Response(200, json={})

        async def put(self, url, json=None, timeout=None):
            http_calls.append(("put", url, json))
            if "graph/update" in str(url):
                return httpx.Response(500, text="no")
            return httpx.Response(200, json={})

    monkeypatch.setattr(job_mod.httpx, "AsyncClient", RecordingAsyncClient)

    hdr = Header(workspace_id="w2", mas_id="m2")
    await job_mod.execute_distillation_run(
        header=hdr,
        request_id="req-y",
        callback_url="https://cb.test/done",
        operation_id="op-2",
        distill_run_at="2026-05-01T00:00:00Z",
        rag_layer=None,
    )

    assert len(http_calls) == 2
    assert http_calls[0][0] == "put"
    assert "graph/update" in str(http_calls[0][1])
    assert http_calls[1][0] == "post"
    assert http_calls[1][1] == "https://cb.test/done"
    assert http_calls[1][2]["status"] == "failed"
