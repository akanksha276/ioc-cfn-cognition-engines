# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NegotiationCognitionEngine (semantic_neg_ce.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from semantic_negotiation.app.agent.semantic_neg_ce import (
    NegotiationAction,
    NegotiationCognitionEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(pipeline=None) -> NegotiationCognitionEngine:
    if pipeline is None:
        pipeline = MagicMock()
        pipeline.async_execute = AsyncMock(return_value={})
    return NegotiationCognitionEngine(pipeline=pipeline)


_INIT_PAYLOAD = {
    "session_id": "sess-1",
    "content_text": "Negotiate email archival policy",
    "agents": [{"id": "agent-a", "name": "Agent A"}, {"id": "agent-b", "name": "Agent B"}],
    "n_steps": 10,
    "workspace_id": "ws-1",
    "mas_id": "mas-1",
}

_DECIDE_PAYLOAD = {
    "session_id": "sess-1",
    "agent_replies": [
        {"agent_id": "agent-a", "action": "counter_offer", "offer": {"issue1": "opt1"}},
        {"agent_id": "agent-b", "action": "accept"},
    ],
    "commit_message_id": "msg-42",
}

_INITIATE_RESULT = {
    "session_id": "sess-1",
    "issues": ["archive_age", "deletion_policy"],
    "options_per_issue": {
        "archive_age": ["30d", "60d", "1y"],
        "deletion_policy": ["never", "after_archive", "on_request"],
    },
    "n_steps": 10,
    "messages": [],
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
    "final_result": {"outcome": {"archive_age": "30d"}},
}


# ---------------------------------------------------------------------------
# NegotiationAction enum
# ---------------------------------------------------------------------------


def test_action_enum_values():
    assert NegotiationAction.INITIATE == "initiate"
    assert NegotiationAction.DECIDE == "decide"


# ---------------------------------------------------------------------------
# INITIATE action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initiate_calls_pipeline_async_execute():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_INITIATE_RESULT)
    engine = _make_engine(pipeline)

    result = await engine.run(NegotiationAction.INITIATE, _INIT_PAYLOAD)

    pipeline.async_execute.assert_awaited_once()
    call_kwargs = pipeline.async_execute.call_args
    assert call_kwargs[0][0] == "sess-1"                          # session_id positional
    assert call_kwargs[1]["content_text"] == "Negotiate email archival policy"
    assert call_kwargs[1]["agents_raw"] == _INIT_PAYLOAD["agents"]
    assert call_kwargs[1]["n_steps"] == 10
    assert call_kwargs[1]["workspace_id"] == "ws-1"
    assert call_kwargs[1]["mas_id"] == "mas-1"
    assert result["issues"] == ["archive_age", "deletion_policy"]


@pytest.mark.asyncio
async def test_initiate_passes_optional_fabric_url():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_INITIATE_RESULT)
    engine = _make_engine(pipeline)

    payload = {**_INIT_PAYLOAD, "fabric_node_base_url": "http://fabric:8080", "agent_names": ["A", "B"]}
    await engine.run(NegotiationAction.INITIATE, payload)

    kwargs = pipeline.async_execute.call_args[1]
    assert kwargs["fabric_node_base_url"] == "http://fabric:8080"
    assert kwargs["agent_names"] == ["A", "B"]


@pytest.mark.asyncio
async def test_initiate_defaults_missing_optional_fields():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_INITIATE_RESULT)
    engine = _make_engine(pipeline)

    minimal_payload = {
        "session_id": "sess-2",
        "content_text": "simple mission",
        "agents": [{"id": "a1", "name": "A"}],
    }
    await engine.run(NegotiationAction.INITIATE, minimal_payload)

    kwargs = pipeline.async_execute.call_args[1]
    assert kwargs["n_steps"] is None
    assert kwargs["workspace_id"] is None
    assert kwargs["fabric_node_base_url"] is None
    assert kwargs["agent_names"] is None


# ---------------------------------------------------------------------------
# DECIDE action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_ongoing_returns_messages():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_DECIDE_ONGOING_RESULT)
    engine = _make_engine(pipeline)

    result = await engine.run(NegotiationAction.DECIDE, _DECIDE_PAYLOAD)

    pipeline.async_execute.assert_awaited_once()
    call_kwargs = pipeline.async_execute.call_args
    assert call_kwargs[0][0] == "sess-1"
    assert call_kwargs[1]["agent_replies"] == _DECIDE_PAYLOAD["agent_replies"]
    assert call_kwargs[1]["commit_message_id"] == "msg-42"
    assert result["status"] == "ongoing"
    assert result["round"] == 2


@pytest.mark.asyncio
async def test_decide_terminal_returns_final_result():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_DECIDE_TERMINAL_RESULT)
    engine = _make_engine(pipeline)

    result = await engine.run(NegotiationAction.DECIDE, _DECIDE_PAYLOAD)

    assert result["status"] == "agreed"
    assert "final_result" in result


@pytest.mark.asyncio
async def test_decide_empty_replies_is_valid():
    pipeline = MagicMock()
    pipeline.async_execute = AsyncMock(return_value=_DECIDE_ONGOING_RESULT)
    engine = _make_engine(pipeline)

    await engine.run(NegotiationAction.DECIDE, {"session_id": "sess-1"})

    kwargs = pipeline.async_execute.call_args[1]
    assert kwargs["agent_replies"] == []


# ---------------------------------------------------------------------------
# Invalid action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_action_raises_value_error():
    engine = _make_engine()
    with pytest.raises(ValueError):
        await engine.run("not_a_real_action", {})
