# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for IngestionCognitionEngine (ingestion_ce.py)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ingestion.app.agent.ingestion_ce import IngestionAction, IngestionCognitionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(**kwargs) -> IngestionCognitionEngine:
    """Build an IngestionCognitionEngine with fully stubbed collaborators."""
    ingest_service = kwargs.get("ingest_service", MagicMock())
    extraction_service = kwargs.get("extraction_service", MagicMock())
    knowledge_processor = kwargs.get("knowledge_processor", None)
    vector_store = kwargs.get("vector_store", None)
    data_repository = kwargs.get("data_repository", None)
    return IngestionCognitionEngine(
        ingest_service=ingest_service,
        extraction_service=extraction_service,
        knowledge_processor=knowledge_processor,
        vector_store=vector_store,
        data_repository=data_repository,
    )


_SAMPLE_RECORDS = [{"SpanId": "s1", "SpanName": "agent.call"}]
_SAMPLE_CONCEPTS = [{"id": "c1", "name": "agent_a", "type": "concept"}]
_SAMPLE_RELATIONS = [{"source": "c1", "target": "c2", "type": "calls"}]


# ---------------------------------------------------------------------------
# IngestionAction enum
# ---------------------------------------------------------------------------


def test_action_enum_values():
    assert IngestionAction.EXTRACT == "extract"
    assert IngestionAction.INGEST == "ingest"
    assert IngestionAction.EXTRACT_AND_INGEST == "extract_and_ingest"
    assert IngestionAction.METRICS == "metrics"
    assert IngestionAction.EXTRACT_FROM_FILE == "extract_from_file"
    assert IngestionAction.INGEST_FROM_FILE == "ingest_from_file"


# ---------------------------------------------------------------------------
# EXTRACT action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_calls_extraction_service():
    extraction_service = MagicMock()
    extraction_service.extract.return_value = {
        "concepts": _SAMPLE_CONCEPTS,
        "relations": _SAMPLE_RELATIONS,
    }
    engine = _make_engine(extraction_service=extraction_service)

    result = await engine.run(IngestionAction.EXTRACT, {"records": _SAMPLE_RECORDS, "format": "observe-sdk-otel"})

    extraction_service.extract.assert_called_once_with(_SAMPLE_RECORDS, "observe-sdk-otel")
    assert result["concepts"] == _SAMPLE_CONCEPTS
    assert result["relations"] == _SAMPLE_RELATIONS


@pytest.mark.asyncio
async def test_extract_uses_default_format():
    extraction_service = MagicMock()
    extraction_service.extract.return_value = {"concepts": [], "relations": []}
    engine = _make_engine(extraction_service=extraction_service)

    await engine.run(IngestionAction.EXTRACT, {"records": _SAMPLE_RECORDS})

    extraction_service.extract.assert_called_once_with(_SAMPLE_RECORDS, "observe-sdk-otel")


# ---------------------------------------------------------------------------
# INGEST / EXTRACT_AND_INGEST action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_calls_ingest_service():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS, "relations": []}
    engine = _make_engine(ingest_service=ingest_service)

    result = await engine.run(
        IngestionAction.INGEST,
        {"records": _SAMPLE_RECORDS, "format": "observe-sdk-otel", "request_id": "req-1"},
    )

    ingest_service.ingest.assert_called_once_with(_SAMPLE_RECORDS, "req-1", "observe-sdk-otel")
    assert result["concepts"] == _SAMPLE_CONCEPTS


@pytest.mark.asyncio
async def test_ingest_runs_knowledge_processor():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS, "relations": []}
    processor = MagicMock()
    processor.process.return_value = {"concepts": _SAMPLE_CONCEPTS, "relations": [], "extra": "processed"}
    engine = _make_engine(ingest_service=ingest_service, knowledge_processor=processor)

    result = await engine.run(IngestionAction.INGEST, {"records": _SAMPLE_RECORDS})

    processor.process.assert_called_once()
    assert result.get("extra") == "processed"


@pytest.mark.asyncio
async def test_ingest_awaits_async_ingest_service_before_processing():
    ingest_service = MagicMock()
    ingest_service.ingest = AsyncMock(
        return_value={
            "concepts": _SAMPLE_CONCEPTS,
            "relations": [],
            "meta": {"records_processed": 1},
        }
    )
    processor = MagicMock()
    processor.process.side_effect = lambda result: result
    engine = _make_engine(ingest_service=ingest_service, knowledge_processor=processor)

    result = await engine.run(
        IngestionAction.INGEST,
        {"records": _SAMPLE_RECORDS, "format": "otel-trace", "request_id": "req-async"},
    )

    ingest_service.ingest.assert_awaited_once_with(_SAMPLE_RECORDS, "req-async", "otel-trace")
    processor.process.assert_called_once()
    processed_arg = processor.process.call_args.args[0]
    assert isinstance(processed_arg, dict)
    assert processed_arg["meta"]["records_processed"] == 1
    assert result["concepts"] == _SAMPLE_CONCEPTS


@pytest.mark.asyncio
async def test_ingest_stores_in_faiss_when_vector_store_present():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS, "relations": []}
    vector_store = MagicMock()
    engine = _make_engine(ingest_service=ingest_service, vector_store=vector_store)

    await engine.run(IngestionAction.INGEST, {"records": _SAMPLE_RECORDS, "save_to_faiss": True})

    vector_store.store_concepts.assert_called_once_with(_SAMPLE_CONCEPTS)


@pytest.mark.asyncio
async def test_ingest_skips_faiss_when_save_to_faiss_false():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS}
    vector_store = MagicMock()
    engine = _make_engine(ingest_service=ingest_service, vector_store=vector_store)

    await engine.run(IngestionAction.INGEST, {"records": _SAMPLE_RECORDS, "save_to_faiss": False})

    vector_store.store_concepts.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_ingest_is_alias_for_ingest():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS}
    engine = _make_engine(ingest_service=ingest_service)

    result = await engine.run(IngestionAction.EXTRACT_AND_INGEST, {"records": _SAMPLE_RECORDS})

    ingest_service.ingest.assert_called_once()
    assert "concepts" in result


@pytest.mark.asyncio
async def test_faiss_failure_does_not_raise():
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS}
    vector_store = MagicMock()
    vector_store.store_concepts.side_effect = RuntimeError("FAISS down")
    engine = _make_engine(ingest_service=ingest_service, vector_store=vector_store)

    # Should not raise
    result = await engine.run(IngestionAction.INGEST, {"records": _SAMPLE_RECORDS})
    assert "concepts" in result


# ---------------------------------------------------------------------------
# METRICS action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_returns_operational_data():
    metrics_obj = MagicMock()
    metrics_obj.records_processed = 10
    metrics_obj.records_sent = 8
    metrics_obj.records_failed = 2
    metrics_obj.last_run_timestamp = None
    metrics_obj.last_run_duration_seconds = 1.5
    metrics_obj.errors = ["e1", "e2"]

    extraction_service = MagicMock()
    extraction_service.get_operational_metrics.return_value = metrics_obj
    engine = _make_engine(extraction_service=extraction_service)

    result = await engine.run(IngestionAction.METRICS, {})

    assert result["records_processed"] == 10
    assert result["records_sent"] == 8
    assert result["records_failed"] == 2
    assert result["last_run_timestamp"] is None
    assert result["last_run_duration_seconds"] == 1.5
    assert result["recent_errors"] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_metrics_formats_timestamp():
    from datetime import datetime, timezone

    ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
    metrics_obj = MagicMock()
    metrics_obj.records_processed = 0
    metrics_obj.records_sent = 0
    metrics_obj.records_failed = 0
    metrics_obj.last_run_timestamp = ts
    metrics_obj.last_run_duration_seconds = 0.0
    metrics_obj.errors = []

    extraction_service = MagicMock()
    extraction_service.get_operational_metrics.return_value = metrics_obj
    engine = _make_engine(extraction_service=extraction_service)

    result = await engine.run(IngestionAction.METRICS, {})
    assert result["last_run_timestamp"] == ts.isoformat()


# ---------------------------------------------------------------------------
# EXTRACT_FROM_FILE action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_from_file_calls_repo_and_service():
    repo = MagicMock()
    repo.load_from_file.return_value = _SAMPLE_RECORDS
    extraction_service = MagicMock()
    extraction_service.extract_entities_and_relations.return_value = {
        "concepts": _SAMPLE_CONCEPTS,
        "relations": [],
    }
    engine = _make_engine(extraction_service=extraction_service, data_repository=repo)

    result = await engine.run(
        IngestionAction.EXTRACT_FROM_FILE,
        {"file_path": "/tmp/test.json", "save_output": False},
    )

    repo.load_from_file.assert_called_once_with(Path("/tmp/test.json"))
    extraction_service.extract_entities_and_relations.assert_called_once_with(_SAMPLE_RECORDS)
    assert result["concepts"] == _SAMPLE_CONCEPTS


@pytest.mark.asyncio
async def test_extract_from_file_saves_output_when_requested():
    repo = MagicMock()
    repo.load_from_file.return_value = _SAMPLE_RECORDS
    extraction_service = MagicMock()
    extraction_service.extract_entities_and_relations.return_value = {
        "concepts": _SAMPLE_CONCEPTS,
        "knowledge_cognition_request_id": "req-42",
    }
    engine = _make_engine(extraction_service=extraction_service, data_repository=repo)

    await engine.run(
        IngestionAction.EXTRACT_FROM_FILE,
        {"file_path": "/tmp/test.json", "save_output": True},
    )

    repo.save_output.assert_called_once()
    saved_filename = repo.save_output.call_args[0][1]
    assert "extracted_entities_req-42" in saved_filename


@pytest.mark.asyncio
async def test_extract_from_file_raises_without_repo():
    engine = _make_engine()
    with pytest.raises(RuntimeError, match="data_repository"):
        await engine.run(IngestionAction.EXTRACT_FROM_FILE, {"file_path": "/tmp/x.json"})


@pytest.mark.asyncio
async def test_extract_from_file_raises_without_file_path():
    engine = _make_engine(data_repository=MagicMock())
    with pytest.raises(ValueError, match="file_path"):
        await engine.run(IngestionAction.EXTRACT_FROM_FILE, {})


# ---------------------------------------------------------------------------
# INGEST_FROM_FILE action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_from_file_calls_repo_and_service():
    repo = MagicMock()
    repo.load_from_file.return_value = _SAMPLE_RECORDS
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {"concepts": _SAMPLE_CONCEPTS}
    engine = _make_engine(ingest_service=ingest_service, data_repository=repo)

    result = await engine.run(
        IngestionAction.INGEST_FROM_FILE,
        {"file_path": "/tmp/test.json", "save_output": False},
    )

    repo.load_from_file.assert_called_once_with(Path("/tmp/test.json"))
    ingest_service.ingest.assert_called_once_with(_SAMPLE_RECORDS, None, "observe-sdk-otel")
    assert result["concepts"] == _SAMPLE_CONCEPTS


@pytest.mark.asyncio
async def test_ingest_from_file_saves_output_when_requested():
    repo = MagicMock()
    repo.load_from_file.return_value = _SAMPLE_RECORDS
    ingest_service = MagicMock()
    ingest_service.ingest.return_value = {
        "concepts": _SAMPLE_CONCEPTS,
        "knowledge_cognition_request_id": "req-99",
    }
    engine = _make_engine(ingest_service=ingest_service, data_repository=repo)

    await engine.run(
        IngestionAction.INGEST_FROM_FILE,
        {"file_path": "/tmp/test.json", "save_output": True},
    )

    repo.save_output.assert_called_once()
    saved_filename = repo.save_output.call_args[0][1]
    assert "concept_relationships_req-99" in saved_filename


@pytest.mark.asyncio
async def test_ingest_from_file_raises_without_repo():
    engine = _make_engine()
    with pytest.raises(RuntimeError, match="data_repository"):
        await engine.run(IngestionAction.INGEST_FROM_FILE, {"file_path": "/tmp/x.json"})


@pytest.mark.asyncio
async def test_ingest_from_file_raises_without_file_path():
    engine = _make_engine(data_repository=MagicMock())
    with pytest.raises(ValueError, match="file_path"):
        await engine.run(IngestionAction.INGEST_FROM_FILE, {})


# ---------------------------------------------------------------------------
# Invalid action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_action_raises_value_error():
    engine = _make_engine()
    with pytest.raises(ValueError):
        await engine.run("not_a_real_action", {})
