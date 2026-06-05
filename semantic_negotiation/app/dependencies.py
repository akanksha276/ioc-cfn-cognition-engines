# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Dependency injection configuration for the Semantic Negotiation Agent.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from common.cognition_engine import ModelConfig

from .agent.semantic_neg_ce import NegotiationCognitionEngine
from .agent.semantic_negotiation import SemanticNegotiationPipeline
from .config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_pipeline() -> SemanticNegotiationPipeline:
    """Return a singleton :class:`SemanticNegotiationPipeline`.

    The pipeline is the single entry point for all negotiation requests.  It
    wires together the three components in order:

    1. :class:`~app.agent.intent_discovery.IntentDiscovery`
    2. :class:`~app.agent.options_generation.OptionsGeneration`
    3. :class:`~app.agent.negotiation_model.NegotiationModel`

    When issues and options are pre-supplied by the caller (via the SSTP
    payload) components 1 and 2 are skipped automatically.
    """
    logger.info(
        "Initialising SemanticNegotiationPipeline (n_steps=%d, enable_local_trace=%s)",
        settings.negotiation_n_steps,
        settings.enable_local_trace,
    )
    return SemanticNegotiationPipeline(
        n_steps=settings.negotiation_n_steps,
        enable_local_trace=settings.enable_local_trace,
    )


@lru_cache()
def get_negotiation_cognition_engine() -> NegotiationCognitionEngine:
    """Singleton :class:`NegotiationCognitionEngine` wired to the live pipeline."""
    cfg = ModelConfig(
        llm_model=settings.llm_model,
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
    )
    return NegotiationCognitionEngine(
        pipeline=get_pipeline(),
        model_config=cfg,
    )
