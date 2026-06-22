# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from functools import lru_cache

from common.cognition_engine import ModelConfig

from .agent.evaluator import SAVEvaluator
from .agent.sav_ce import ValidationCognitionEngine
from .config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_validation_cognition_engine() -> ValidationCognitionEngine:
    from .config.utils import get_llm_provider
    llm = get_llm_provider()
    evaluator = SAVEvaluator(llm_provider=llm)
    cfg = ModelConfig(
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
    )
    return ValidationCognitionEngine(evaluator=evaluator, model_config=cfg)
