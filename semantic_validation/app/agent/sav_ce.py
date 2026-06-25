# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic Validation Cognition Engine.

Wraps :class:`~app.agent.evaluator.SAVEvaluator`
behind the :class:`~common.cognition_engine.CognitionEngine` interface.

Supported actions
-----------------
``evaluate``
    Evaluate the alignment between a final decision and the mission.
    Payload keys: ``mission`` (str), ``participants`` (list[dict]),
    ``context`` (str, optional), ``final_decision`` (str, optional),
    ``latest_message`` (str, optional), ``interaction_history`` (list[str], optional).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

_workspace_root = str(Path(__file__).resolve().parents[4])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from common.cognition_engine import CognitionEngine, ModelConfig, handles  # noqa: E402

from .evaluator import SAVEvaluator
from .models import Participant, SAVInput

logger = logging.getLogger(__name__)


class ValidationAction(str, Enum):
    EVALUATE = "evaluate"


class ValidationCognitionEngine(CognitionEngine):
    def __init__(
        self,
        evaluator: SAVEvaluator,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(model_config)
        self._evaluator = evaluator

    @property
    def action_enum(self) -> Type[Enum]:
        return ValidationAction

    @handles(ValidationAction.EVALUATE)
    async def _evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("ValidationCognitionEngine._evaluate mission=%s", payload.get("mission", "")[:60])

        participants = [
            Participant(
                id=p.get("id", ""),
                name=p.get("name"),
                role=p.get("role"),
            )
            for p in (payload.get("participants") or [])
        ]

        sav_input = SAVInput(
            mission=payload.get("mission", ""),
            participants=participants,
            context=payload.get("context"),
            latest_message=payload.get("latest_message"),
            final_decision=payload.get("final_decision"),
            interaction_history=payload.get("interaction_history"),
        )

        sav_output = self._evaluator.evaluate(sav_input)

        return {
            "failure_modes": [
                dataclasses.asdict(fm) for fm in sav_output.failure_modes
            ]
        }
