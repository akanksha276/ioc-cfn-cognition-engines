# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic Negotiation Cognition Engine.

Wraps :class:`~app.agent.semantic_negotiation.SemanticNegotiationPipeline`
behind the :class:`~common.cognition_engine.CognitionEngine` interface.

Supported actions
-----------------
``initiate``
    Start a new multi-party SAO negotiation session.
    Payload keys: ``session_id`` (str), ``content_text`` (str),
    ``agents`` (list[dict] with ``id`` and ``name``),
    ``n_steps`` (int, optional), ``workspace_id`` (str, optional),
    ``mas_id`` (str, optional), ``fabric_node_base_url`` (str, optional),
    ``agent_names`` (list[str], optional).

``decide``
    Submit one round of agent decisions to an existing session.
    Payload keys: ``session_id`` (str),
    ``agent_replies`` (list[dict] — one per agent),
    ``commit_message_id`` (str, optional).
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

_workspace_root = str(Path(__file__).resolve().parents[4])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from common.cognition_engine import CognitionEngine, ModelConfig, handles  # noqa: E402

logger = logging.getLogger(__name__)


class NegotiationAction(str, Enum):
    """Actions supported by :class:`NegotiationCognitionEngine`."""

    INITIATE = "initiate"
    """Start a new negotiation session (Components 1 + 2 + SAO seed)."""

    DECIDE = "decide"
    """Advance an existing session by one batch of agent decisions."""


class NegotiationCognitionEngine(CognitionEngine):
    """Cognition engine for semantic negotiation.

    Args:
        pipeline: An initialised
            :class:`~app.agent.semantic_negotiation.SemanticNegotiationPipeline`.
    """

    def __init__(
        self,
        pipeline: Any,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(model_config)
        self._pipeline = pipeline

    @property
    def action_enum(self) -> Type[Enum]:
        return NegotiationAction

    # ── private helpers ────────────────────────────────────────────────────

    @handles(NegotiationAction.INITIATE)
    async def _initiate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new multi-party SAO negotiation session (Components 1 + 2 + seed round)."""
        session_id: str = payload.get("session_id", "")
        agents_raw: List[Dict[str, Any]] = payload.get("agents", [])
        logger.info(
            "NegotiationCognitionEngine._initiate session_id=%s agents=%d",
            session_id,
            len(agents_raw),
        )
        return await self._pipeline.async_execute(
            session_id,
            content_text=payload.get("content_text", ""),
            agents_raw=agents_raw,
            n_steps=payload.get("n_steps"),
            workspace_id=payload.get("workspace_id"),
            mas_id=payload.get("mas_id"),
            fabric_node_base_url=payload.get("fabric_node_base_url"),
            agent_names=payload.get("agent_names"),
            initiate_message=payload.get("initiate_message"),
        )

    @handles(NegotiationAction.DECIDE)
    async def _decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Advance an existing negotiation session by one batch of agent decisions."""
        session_id: str = payload.get("session_id", "")
        agent_replies: List[Dict[str, Any]] = payload.get("agent_replies", [])
        logger.info(
            "NegotiationCognitionEngine._decide session_id=%s replies=%d",
            session_id,
            len(agent_replies),
        )
        return await self._pipeline.async_execute(
            session_id,
            agent_replies=agent_replies,
            commit_message_id=payload.get("commit_message_id", ""),
            workspace_id=payload.get("workspace_id"),
            mas_id=payload.get("mas_id"),
        )
