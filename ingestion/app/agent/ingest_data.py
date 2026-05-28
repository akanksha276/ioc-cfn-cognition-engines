# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unified ingestion orchestration for compact, RAG, and graph extraction."""

from __future__ import annotations

import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .adapters import ExtractionAdapter, ExtractionAdapterRAG
from .prompts import SUPPORTED_FORMATS

if TYPE_CHECKING:
    from .rag import RagPipeline

logger = logging.getLogger(__name__)


class IngestDataService:
    """One entrypoint that orchestrates compact payload, RAG, and graph extraction."""

    def __init__(
        self,
        concept_service: Any,
        *,
        enable_rag_ingest: bool = True,
        rag_pipeline: Optional[Any] = None,
    ) -> None:
        self._concept_service = concept_service
        self._enable_rag_ingest = enable_rag_ingest
        self._rag_pipeline = rag_pipeline
        self._extraction_adapter = ExtractionAdapter()

    @staticmethod
    def _normalize_data_format(format_descriptor: Optional[str]) -> str:
        data_format = (format_descriptor or "observe-sdk-otel").strip().lower()
        if data_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported data format: {data_format!r}. Supported: {SUPPORTED_FORMATS}"
            )
        return data_format

    @staticmethod
    def _fallback_request_id(records_processed: int) -> str:
        return hashlib.md5(
            f"{datetime.now().isoformat()}_{records_processed}".encode()
        ).hexdigest()

    def _build_rag_chunks(
        self, compact_payload: List[Dict[str, Any]], data_format: str
    ) -> List[Dict[str, Any]]:
        if not self._enable_rag_ingest or not compact_payload:
            return []
        if self._rag_pipeline is None:
            from .rag import RagPipeline
            self._rag_pipeline = RagPipeline()

        rag_docs = ExtractionAdapterRAG.nested_dict_to_text_document(
            compact_payload,
            data_format=data_format,
        )
        if not rag_docs:
            return []
        return self._rag_pipeline.run(rag_docs)

    async def ingest(
        self,
        records: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        format_descriptor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the unified ingestion pipeline and return graph response plus rag_chunks.

        Callers must ``await`` this coroutine; it is not a synchronous dict return.
        """
        data_format = self._normalize_data_format(format_descriptor)

        filtered = self._extraction_adapter.filter_records(records, data_format)
        logger.info(
            "Filtered to %d records from %d total (format=%s)",
            len(filtered),
            len(records),
            data_format,
        )
        if not filtered:
            rid = request_id or self._fallback_request_id(0)
            return {
                "knowledge_cognition_request_id": rid,
                "concepts": [],
                "relations": [],
                "descriptor": data_format,
                "meta": {
                    "records_processed": 0,
                    "concepts_extracted": 0,
                    "relations_extracted": 0,
                },
                "rag_chunks": [],
            }

        compact_payload = self._extraction_adapter.build_compact_payload(filtered, data_format)
        logger.info(
            "Built compact payload with %d entries (format=%s)",
            len(compact_payload),
            data_format,
        )

        rag_chunks: List[Dict[str, Any]] = []
        if self._enable_rag_ingest:
            try:
                rag_chunks = self._build_rag_chunks(compact_payload, data_format)
                logger.info("Generated %d rag chunks (format=%s)", len(rag_chunks), data_format)
            except Exception:
                # Graph extraction must remain available even when RAG fails.
                logger.exception("RAG stage failed; continuing with graph extraction only")

        graph_result = await self._concept_service.extract_concepts_and_relationships(
            compact_payload=compact_payload,
            request_id=request_id,
            format_descriptor=data_format,
        )
        graph_result["rag_chunks"] = rag_chunks
        return graph_result
