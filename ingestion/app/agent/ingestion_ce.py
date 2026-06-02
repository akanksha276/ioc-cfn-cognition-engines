# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Ingestion Cognition Engine.

Wraps :class:`~app.agent.ingest_data.IngestDataService` and
:class:`~app.agent.service.TelemetryExtractionService` behind the
:class:`~common.cognition_engine.CognitionEngine` interface.

Supported actions
-----------------
``extract``
    Run concept + relationship extraction on raw telemetry records without
    persisting to any store.  Payload keys: ``records`` (list), ``format``
    (str, optional).

``ingest``
    Full ingestion pipeline: extract → embed → process → store in vector/graph store.
    Payload keys: ``records`` (list), ``format`` (str, optional),
    ``request_id`` (str, optional), ``header`` (dict, optional),
    ``save_to_faiss`` (bool, optional, default True).

``extract_and_ingest``
    Alias for ``ingest`` — runs the complete extract-then-store pipeline.
    Same payload keys as ``ingest``.

``metrics``
    Return operational metrics from the extraction service.
    Payload keys: none.

``extract_from_file``
    Load OTEL data from a file path and extract entities and relations.
    Payload keys: ``file_path`` (str), ``save_output`` (bool, optional).

``ingest_from_file``
    Load OTEL data from a file path and run the full ingestion pipeline.
    Payload keys: ``file_path`` (str), ``save_output`` (bool, optional).
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type

# Make sure the workspace root is importable regardless of launch directory.
_workspace_root = str(Path(__file__).resolve().parents[4])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from common.cognition_engine import CognitionEngine, ModelConfig, handles  # noqa: E402

logger = logging.getLogger(__name__)


class IngestionAction(str, Enum):
    """Actions supported by :class:`IngestionCognitionEngine`."""

    EXTRACT = "extract"
    """Extract concepts and relationships from raw records (no persistence)."""

    INGEST = "ingest"
    """Full pipeline: extract, embed, process, and store in the knowledge graph/vector store."""

    EXTRACT_AND_INGEST = "extract_and_ingest"
    """Alias for :attr:`INGEST` — runs the complete extract-then-store pipeline."""

    METRICS = "metrics"
    """Return operational metrics from the extraction service."""

    EXTRACT_FROM_FILE = "extract_from_file"
    """Load OTEL data from a file path and extract entities and relations."""

    INGEST_FROM_FILE = "ingest_from_file"
    """Load OTEL data from a file path and run the full ingestion pipeline."""


class IngestionCognitionEngine(CognitionEngine):
    """Cognition engine for knowledge ingestion.

    Args:
        ingest_service: An initialised :class:`~app.agent.ingest_data.IngestDataService`.
        extraction_service: An initialised
            :class:`~app.agent.service.TelemetryExtractionService`.
    """

    def __init__(
        self,
        ingest_service: Any,
        extraction_service: Any,
        model_config: Optional[ModelConfig] = None,
        knowledge_processor: Any = None,
        vector_store: Any = None,
        data_repository: Any = None,
    ) -> None:
        super().__init__(model_config)
        self._ingest_service = ingest_service
        self._extraction_service = extraction_service
        self._knowledge_processor = knowledge_processor
        self._vector_store = vector_store
        self._data_repository = data_repository

    @property
    def action_enum(self) -> Type[Enum]:
        return IngestionAction

    # ── private helpers ────────────────────────────────────────────────────

    @handles(IngestionAction.EXTRACT)
    async def _extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract concepts and relationships without persisting."""
        import asyncio

        records: list = payload.get("records", [])
        fmt: str = payload.get("format", "observe-sdk-otel")
        logger.info(
            "IngestionCognitionEngine._extract records=%d fmt=%s", len(records), fmt
        )
        result = await asyncio.to_thread(self._extraction_service.extract, records, fmt)
        return {
            "concepts": result.get("concepts", []),
            "relations": result.get("relations", []),
        }

    @handles(IngestionAction.INGEST, IngestionAction.EXTRACT_AND_INGEST)
    async def _ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full ingestion pipeline: extract → process → store."""
        import asyncio

        records: list = payload.get("records", [])
        fmt: str = payload.get("format", "observe-sdk-otel")
        request_id: str | None = payload.get("request_id")
        logger.info(
            "IngestionCognitionEngine._ingest records=%d fmt=%s", len(records), fmt
        )
        result = await asyncio.to_thread(
            self._ingest_service.ingest, records, request_id, fmt
        )
        if not isinstance(result, dict):
            result = {"result": result}

        if self._knowledge_processor is not None:
            result = self._knowledge_processor.process(result)

        if self._vector_store is not None and payload.get("save_to_faiss", True):
            self._store_concepts_in_faiss(result.get("concepts", []))

        return result

    def _store_concepts_in_faiss(self, concepts: list) -> None:
        """Persist concepts in the in-process FAISS index (fire-and-log)."""
        try:
            self._vector_store.store_concepts(concepts)
        except Exception:
            logger.exception("FAISS storage failed; ingestion result is still valid")

    @handles(IngestionAction.METRICS)
    async def _metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG002
        """Return operational metrics from the extraction service."""
        logger.info("IngestionCognitionEngine._metrics")
        metrics = self._extraction_service.get_operational_metrics()
        return {
            "records_processed": metrics.records_processed,
            "records_sent": metrics.records_sent,
            "records_failed": metrics.records_failed,
            "last_run_timestamp": (
                metrics.last_run_timestamp.isoformat()
                if metrics.last_run_timestamp
                else None
            ),
            "last_run_duration_seconds": metrics.last_run_duration_seconds,
            "recent_errors": metrics.errors[-10:],
        }

    @handles(IngestionAction.EXTRACT_FROM_FILE)
    async def _extract_from_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Load OTEL data from a file path and extract entities and relations."""
        import asyncio

        if self._data_repository is None:
            raise RuntimeError(
                "data_repository is required for extract_from_file action"
            )
        file_path = payload.get("file_path")
        if not file_path:
            raise ValueError("payload.file_path is required")
        save_output: bool = payload.get("save_output", False)
        logger.info("IngestionCognitionEngine._extract_from_file path=%s", file_path)
        otel_data = self._data_repository.load_from_file(Path(file_path))
        result = await asyncio.to_thread(
            self._extraction_service.extract_entities_and_relations, otel_data
        )
        if self._knowledge_processor is not None:
            result = self._knowledge_processor.process(result)
        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            self._data_repository.save_output(result, output_filename)
        return result

    @handles(IngestionAction.INGEST_FROM_FILE)
    async def _ingest_from_file(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Load OTEL data from a file path and run the full ingestion pipeline."""
        import asyncio

        if self._data_repository is None:
            raise RuntimeError(
                "data_repository is required for ingest_from_file action"
            )
        file_path = payload.get("file_path")
        if not file_path:
            raise ValueError("payload.file_path is required")
        fmt: str = payload.get("format", "observe-sdk-otel")
        request_id: str | None = payload.get("request_id")
        save_output: bool = payload.get("save_output", False)
        logger.info("IngestionCognitionEngine._ingest_from_file path=%s", file_path)
        otel_data = self._data_repository.load_from_file(Path(file_path))
        result = await asyncio.to_thread(self._ingest_service.ingest, otel_data, request_id, fmt)
        if not isinstance(result, dict):
            result = {"result": result}
        if self._knowledge_processor is not None:
            result = self._knowledge_processor.process(result)
        if save_output:
            output_filename = f"concept_relationships_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            self._data_repository.save_output(result, output_filename)
        return result
