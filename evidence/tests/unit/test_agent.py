# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the evidence-gathering agent."""
import pytest
from unittest.mock import AsyncMock, patch

from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest


@pytest.mark.asyncio
async def test_process_evidence_empty_entities_returns_ok():
    """When no entities are extracted, process_evidence returns OK with empty records."""
    with patch("app.agent.evidence.LLMEntityExtractor") as MockExtractor:
        MockExtractor.return_value.extract_entities_from_request = AsyncMock(return_value=[])
        req = ReasonerCognitionRequest(
            mas_id="mas-1",
            workspace_id="ws-1",
            intent="What does the orchestrator do?",
        )
        resp = await process_evidence(req)
    assert resp.status == "OK"
    assert resp.records == []
    assert resp.mas_id == "mas-1"
    assert resp.workspace_id == "ws-1"
    assert resp.meta.get("note") == "no_entities"
