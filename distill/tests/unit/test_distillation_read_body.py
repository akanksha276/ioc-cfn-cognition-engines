# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from evidence.app.api.schemas import Header

from distill.app.config import settings as settings_mod
from distill.app.services.distillation_job import _distillation_read_request_body


def test_distillation_read_request_body_uses_cfn_filters(monkeypatch):
    monkeypatch.setenv("CODI_MIN_RELATIONS", "10")
    monkeypatch.setattr(settings_mod.settings, "CODI_MIN_RELATIONS", 10)
    monkeypatch.setattr(settings_mod.settings, "CODI_DISTILL_STATUS_FILTER", "")
    monkeypatch.setattr(settings_mod.settings, "CODI_RETURN_MISSING_DISTILL_STATUS", True)

    hdr = Header(
        workspace_id="d3fc3341-ee16-43dc-956c-6c506015dc23",
        mas_id="3eaa5d42-f5c4-4d87-9663-a333343c6e6e",
        agent_id="agent-1",
    )
    body = _distillation_read_request_body(hdr, "req-1")

    assert body["request_id"] == "req-1"
    assert "min_edges" not in body
    assert body["filters"] == {
        "relations_cnt_gte": 10,
        "distill_status": "",
        "owner": "3eaa5d42-f5c4-4d87-9663-a333343c6e6e",
        "return_missing_distill_status": True,
    }
    assert body["header"]["workspace_id"] == hdr.workspace_id
