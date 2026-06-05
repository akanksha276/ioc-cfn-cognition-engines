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

from common.cognition_engine import ModelConfig

from .agent.ingest_data import IngestDataService
from .agent.ingestion_ce import IngestionCognitionEngine
from .agent.knowledge_processor import EmbeddingManager, KnowledgeProcessor
from .agent.service import ConceptRelationshipExtractionService, TelemetryExtractionService
from .config.settings import settings
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


@lru_cache()
def get_ingestion_cognition_engine() -> IngestionCognitionEngine:
    """Singleton :class:`IngestionCognitionEngine` wired to all ingestion services."""
    cfg = ModelConfig(
        llm_model=getattr(settings, "llm_model", None),
        llm_api_key=getattr(settings, "llm_api_key", None),
        llm_base_url=getattr(settings, "llm_base_url", None),
    )
    return IngestionCognitionEngine(
        ingest_service=get_ingest_data_service(),
        extraction_service=get_extraction_service(),
        model_config=cfg,
        knowledge_processor=get_knowledge_processor(),
        data_repository=get_data_repository(),
    )

