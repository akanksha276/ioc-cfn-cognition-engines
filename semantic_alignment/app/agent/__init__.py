# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Domain services for the semantic negotiation agent."""


import logging

from .http_repo import (
    SharedMemoryNotFoundError,
    SharedMemoryQueryError,
    issue_labels_from_negotiable_entities,
    post_shared_memories_query,
    shared_memories_query_path,
)
from .intent_discovery import IntentDiscovery
from .negotiation_model import NegotiationModel, NegotiationOutcome, NegotiationParticipant, NegotiationResult
from .options_generation import OptionsGeneration
from .semantic_alignment import SemanticNegotiationPipeline

logger = logging.getLogger(__name__)
logger.info("semantic_alignment app.agent package loaded")

__all__ = [
    "SemanticNegotiationPipeline",
    "IntentDiscovery",
    "OptionsGeneration",
    "SharedMemoryNotFoundError",
    "SharedMemoryQueryError",
    "issue_labels_from_negotiable_entities",
    "post_shared_memories_query",
    "shared_memories_query_path",
    "shared_memories_upsert_path",
    "NegotiationModel",
    "NegotiationParticipant",
    "NegotiationOutcome",
    "NegotiationResult",
]
