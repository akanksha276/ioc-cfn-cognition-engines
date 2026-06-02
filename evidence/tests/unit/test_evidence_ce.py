# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EvidenceCognitionEngine (evidence_ce.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evidence.app.agent.evidence_ce import EvidenceAction, EvidenceCognitionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(**kwargs) -> EvidenceCognitionEngine:
    return EvidenceCognitionEngine(
        repo_adapter=kwargs.get("repo_adapter", None),
        cache_layer=kwargs.get("cache_layer", None),
        rag_cache_layer=kwargs.get("rag_cache_layer", None),
    )


_SAMPLE_HEADER = {"workspace_id": "ws-1", "mas_id": "mas-1", "agent_id": "ag-1"}
_SAMPLE_REQUEST_PAYLOAD = {"intent": "what does the orchestrator do?", "metadata": {}, "additional_context": []}


# ---------------------------------------------------------------------------
# EvidenceAction enum
# ---------------------------------------------------------------------------


def test_action_enum_values():
    assert EvidenceAction.REASON == "reason"
    assert EvidenceAction.GRAPH_PATHS == "graph_paths"
    assert EvidenceAction.NEIGHBORS == "neighbors"
    assert EvidenceAction.CONCEPTS_BY_IDS == "concepts_by_ids"


# ---------------------------------------------------------------------------
# REASON action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_calls_process_evidence():
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "status": "OK",
        "records": [],
        "header": _SAMPLE_HEADER,
        "response_id": "r1",
    }

    with patch("evidence.app.agent.embeddings.EmbeddingManager"), \
         patch("evidence.app.agent.evidence.process_evidence", new=AsyncMock(return_value=mock_response)):
        engine = _make_engine()
        result = await engine.run(
            EvidenceAction.REASON,
            {
                "header": _SAMPLE_HEADER,
                "request_id": "r1",
                "payload": _SAMPLE_REQUEST_PAYLOAD,
            },
        )

    assert result["status"] == "OK"


@pytest.mark.asyncio
async def test_reason_passes_repo_to_process_evidence():
    repo = MagicMock()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"status": "OK", "records": []}

    with patch("evidence.app.agent.embeddings.EmbeddingManager"), \
         patch("evidence.app.agent.evidence.process_evidence", new=AsyncMock(return_value=mock_response)) as mock_pe:
        engine = _make_engine(repo_adapter=repo)
        await engine.run(
            EvidenceAction.REASON,
            {"header": _SAMPLE_HEADER, "request_id": "r1", "payload": _SAMPLE_REQUEST_PAYLOAD},
        )
        call_kwargs = mock_pe.call_args[1]
        assert call_kwargs["repo_adapter"] is repo


# ---------------------------------------------------------------------------
# GRAPH_PATHS action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_paths_calls_repo():
    repo = AsyncMock()
    repo.find_paths.return_value = {"status": "success", "paths": [["c1", "c2"]]}
    engine = _make_engine(repo_adapter=repo)

    result = await engine.run(
        EvidenceAction.GRAPH_PATHS,
        {"source_id": "c1", "target_id": "c2", "max_depth": 3, "limit": 10, "relations": ["calls"]},
    )

    repo.find_paths.assert_awaited_once_with(
        source_id="c1", target_id="c2", max_depth=3, limit=10, relations=["calls"]
    )
    assert result["paths"] == [["c1", "c2"]]


@pytest.mark.asyncio
async def test_graph_paths_raises_without_repo():
    engine = _make_engine()
    with pytest.raises(RuntimeError, match="repo_adapter"):
        await engine.run(EvidenceAction.GRAPH_PATHS, {"source_id": "a", "target_id": "b"})


# ---------------------------------------------------------------------------
# NEIGHBORS action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neighbors_calls_repo():
    repo = AsyncMock()
    repo.get_neighbors.return_value = {"neighbors": [{"id": "c2"}]}
    engine = _make_engine(repo_adapter=repo)

    result = await engine.run(
        EvidenceAction.NEIGHBORS,
        {"node_id": "c1", "max_depth": 2, "relations": None},
    )

    repo.get_neighbors.assert_awaited_once_with(node_id="c1", max_depth=2, relations=None)
    assert result["neighbors"] == [{"id": "c2"}]


@pytest.mark.asyncio
async def test_neighbors_raises_without_repo():
    engine = _make_engine()
    with pytest.raises(RuntimeError, match="repo_adapter"):
        await engine.run(EvidenceAction.NEIGHBORS, {"node_id": "c1"})


# ---------------------------------------------------------------------------
# CONCEPTS_BY_IDS action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concepts_by_ids_calls_repo():
    repo = AsyncMock()
    repo.get_concepts_by_ids.return_value = {"concepts": [{"id": "c1", "name": "agent_a"}]}
    engine = _make_engine(repo_adapter=repo)

    result = await engine.run(EvidenceAction.CONCEPTS_BY_IDS, {"ids": ["c1", "c2"]})

    repo.get_concepts_by_ids.assert_awaited_once_with(ids=["c1", "c2"])
    assert result["concepts"][0]["id"] == "c1"


@pytest.mark.asyncio
async def test_concepts_by_ids_raises_without_repo():
    engine = _make_engine()
    with pytest.raises(RuntimeError, match="repo_adapter"):
        await engine.run(EvidenceAction.CONCEPTS_BY_IDS, {"ids": ["c1"]})


# ---------------------------------------------------------------------------
# Invalid action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_action_raises_value_error():
    engine = _make_engine()
    with pytest.raises(ValueError):
        await engine.run("not_a_real_action", {})
