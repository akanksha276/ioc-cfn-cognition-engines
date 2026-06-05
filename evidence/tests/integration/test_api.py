# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0


from fastapi.testclient import TestClient

from evidence.app.main import app

client = TestClient(app)


def test_evidence_endpoint_smoke():
    request = {
        "header": {
            "workspace_id": "test-workspace",
            "mas_id": "test-mas",
            "agent_id": "test-agent",
        },
        "request_id": "demo",
        "payload": {
            "intent": "what does the orchestrator do?",
            "metadata": {},
            "additional_context": [],
        },
    }

    r = client.post("/api/knowledge-mgmt/reasoning/evidence", json=request)
    assert r.status_code == 200
    body = r.json()
    assert "header" in body and "workspace_id" in body["header"] and "agent_id" in body["header"] and "mas_id" in body["header"]
    assert body["header"]["workspace_id"] == "test-workspace"
    assert body["header"]["mas_id"] == "test-mas"
    assert body["header"]["agent_id"] == "test-agent"

    assert "response_id" in body and body["response_id"] == "demo"
    assert "records" in body

def test_evidence_endpoint_no_request_id():
    request = {
        "header": {
            "workspace_id": "test-workspace",
            "mas_id": "test-mas",
            "agent_id": "test-agent",
        },
        "payload": {
            "intent": "what does the orchestrator do?",
            "metadata": {},
            "additional_context": [],
        },
    }

    r = client.post("/api/knowledge-mgmt/reasoning/evidence", json=request)
    assert r.status_code == 422


def test_evidence_endpoint_no_agent_id():
    request = {
        "header": {
            "workspace_id": "test-workspace",
            "mas_id": "test-mas"
        },
        "request_id": "demo",
        "payload": {
            "intent": "what does the orchestrator do?",
            "metadata": {},
            "additional_context": [],
        },
    }

    r = client.post("/api/knowledge-mgmt/reasoning/evidence", json=request)
    assert r.status_code == 200
    body = r.json()
    assert "header" in body and "agent_id" not in body["header"]
