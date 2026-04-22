# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Dependency injection configuration.

This module provides factory functions for creating service instances
with the appropriate dependencies injected.
"""

from __future__ import annotations

from functools import lru_cache

from .config.settings import settings
from .agent.ingest_data import IngestDataService
from .agent.service import TelemetryExtractionService, ConceptRelationshipExtractionService
from .agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from .data.mock_repo import MockDataRepository

@lru_cache()
def get_data_repository() -> MockDataRepository:
    """
    Get the data repository instance.
    
    Returns:
        MockDataRepository instance (can be swapped for other implementations)
    """
    return MockDataRepository()


@lru_cache()
def get_extraction_service() -> TelemetryExtractionService:
    """Get the telemetry extraction service instance."""
    return TelemetryExtractionService()


@lru_cache()
def get_concept_relationship_service() -> ConceptRelationshipExtractionService:
    """Get the concept-relationship extraction service instance."""
    return ConceptRelationshipExtractionService()


@lru_cache()
def get_ingest_data_service() -> IngestDataService:
    """Get the unified ingest orchestration service instance."""
    return IngestDataService(
        concept_service=get_concept_relationship_service(),
        enable_rag_ingest=settings.enable_rag_ingest,
    )


@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    """Singleton ``EmbeddingManager`` shared by extraction components."""
    return EmbeddingManager(model_path=settings.embedding_model_path)


def get_knowledge_processor() -> KnowledgeProcessor:
    """
    Get a knowledge processor instance.

    The heavy ``EmbeddingManager`` is a singleton; only the lightweight
    processor wrapper is re-created per call so config can change at runtime.
    """
    return KnowledgeProcessor(
        enable_embeddings=settings.enable_embeddings,
        enable_dedup=settings.enable_dedup,
        similarity_threshold=settings.similarity_threshold,
        embedding_manager=get_embedding_manager(),
    )

