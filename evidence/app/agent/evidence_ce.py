# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Evidence Cognition Engine.

Wraps the evidence reasoning functions behind the
:class:`~common.cognition_engine.CognitionEngine` interface.

Supported actions
-----------------
``reason``
    RAG-based semantic graph traversal for a natural-language query.
    Payload keys: ``header`` (dict with ``workspace_id``, ``mas_id``,
    ``agent_id``), ``request_id`` (str), ``payload`` (dict with ``intent``,
    optional ``metadata`` and ``additional_context``).

``graph_paths``
    Find paths between two nodes in the knowledge graph.
    Payload keys: ``source_id`` (str), ``target_id`` (str),
    ``max_depth`` (int, optional), ``limit`` (int, optional),
    ``relations`` (list[str], optional).

``neighbors``
    Retrieve immediate neighbours of a node.
    Payload keys: ``node_id`` (str), ``max_depth`` (int, optional),
    ``relations`` (list[str], optional).

``concepts_by_ids``
    Bulk concept lookup by IDs.
    Payload keys: ``ids`` (list[str]).
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type

_workspace_root = str(Path(__file__).resolve().parents[4])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from common.cognition_engine import CognitionEngine, ModelConfig, handles  # noqa: E402

logger = logging.getLogger(__name__)


class EvidenceAction(str, Enum):
    """Actions supported by :class:`EvidenceCognitionEngine`."""

    REASON = "reason"
    """RAG-based semantic graph traversal to answer a natural-language query."""

    GRAPH_PATHS = "graph_paths"
    """Find paths between two nodes in the knowledge graph."""

    NEIGHBORS = "neighbors"
    """Retrieve immediate neighbours of a node."""

    CONCEPTS_BY_IDS = "concepts_by_ids"
    """Bulk concept lookup by a list of IDs."""


class EvidenceCognitionEngine(CognitionEngine):
    """Cognition engine for evidence reasoning.

    Args:
        repo_adapter: Async-compatible repository adapter (graph store).
        cache_layer: Optional caching layer for reasoning results.
        rag_cache_layer: Optional RAG caching layer.
    """

    def __init__(
        self,
        repo_adapter: Any = None,
        cache_layer: Any = None,
        rag_cache_layer: Any = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(model_config)
        self._repo = repo_adapter
        self._cache = cache_layer
        self._rag_cache = rag_cache_layer

    @property
    def action_enum(self) -> Type[Enum]:
        return EvidenceAction

    # ── private helpers ────────────────────────────────────────────────────

    @handles(EvidenceAction.REASON)
    async def _reason(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """RAG-based semantic graph traversal to answer a natural-language query."""
        from ..api.schemas import (
            Header,
            ReasonerCognitionRequest,
            RequestPayload,
        )
        from .evidence import process_evidence

        header_raw = payload.get("header", {})
        request_payload_raw = payload.get("payload", {})

        request = ReasonerCognitionRequest(
            header=Header(**header_raw),
            request_id=payload.get("request_id", ""),
            payload=RequestPayload(**request_payload_raw),
        )
        logger.info("EvidenceCognitionEngine._reason request_id=%s", request.request_id)
        response = await process_evidence(
            request,
            repo_adapter=self._repo,
        )
        return response.model_dump(mode="json")

    @handles(EvidenceAction.GRAPH_PATHS)
    async def _graph_paths(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Find paths between two nodes in the knowledge graph."""
        if self._repo is None:
            raise RuntimeError("repo_adapter is required for graph_paths action")
        logger.info(
            "EvidenceCognitionEngine._graph_paths source=%s target=%s",
            payload.get("source_id"),
            payload.get("target_id"),
        )
        result = await self._repo.find_paths(
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            max_depth=payload.get("max_depth"),
            limit=payload.get("limit"),
            relations=payload.get("relations"),
        )
        return result if isinstance(result, dict) else {"paths": result}

    @handles(EvidenceAction.NEIGHBORS)
    async def _neighbors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve immediate neighbours of a node in the knowledge graph."""
        if self._repo is None:
            raise RuntimeError("repo_adapter is required for neighbors action")
        logger.info(
            "EvidenceCognitionEngine._neighbors node_id=%s", payload.get("node_id")
        )
        result = await self._repo.get_neighbors(
            node_id=payload["node_id"],
            max_depth=payload.get("max_depth"),
            relations=payload.get("relations"),
        )
        return result if isinstance(result, dict) else {"neighbors": result}

    @handles(EvidenceAction.CONCEPTS_BY_IDS)
    async def _concepts_by_ids(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk concept lookup by a list of IDs."""
        if self._repo is None:
            raise RuntimeError("repo_adapter is required for concepts_by_ids action")
        logger.info(
            "EvidenceCognitionEngine._concepts_by_ids count=%d",
            len(payload.get("ids", [])),
        )
        result = await self._repo.get_concepts_by_ids(ids=payload["ids"])
        return result if isinstance(result, dict) else {"concepts": result}
