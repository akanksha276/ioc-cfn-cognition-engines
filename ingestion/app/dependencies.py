# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Dependency injection configuration.

This module provides factory functions for creating service instances
with the appropriate dependencies injected.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Request

from .config.settings import settings
from .agent.ingest_data import IngestDataService
from .agent.service import TelemetryExtractionService, ConceptRelationshipExtractionService
from .agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from .data.mock_repo import MockDataRepository

logger = logging.getLogger(__name__)


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
    """Singleton ``EmbeddingManager`` shared by the knowledge processor and FAISS store."""
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

def get_vector_store(request: Request):
    """
    In-process FAISS vector store. When running under the unified app, uses
    the shared CachingLayer from request.app.state.cache_layer (Option A).
    Otherwise creates a local ConceptVectorStore (standalone).
    Returns ``None`` when FAISS storage is disabled via settings.
    """
    if not settings.enable_faiss_storage:
        logger.info("FAISS storage is disabled via settings.")
        return None

    cache_layer = getattr(request.app.state, "cache_layer", None)
    rag_cache_layer = getattr(request.app.state, "rag_cache_layer", None)
    if cache_layer is not None or rag_cache_layer is not None:
        from .agent.concept_vector_store import VectorStore
        return VectorStore(cache_layer=cache_layer, rag_cache_layer=rag_cache_layer)
    try:
        from .agent.concept_vector_store import VectorStore
        store = VectorStore(
            embed_fn=get_embedding_manager().generate_embedding,
            vector_dimension=settings.faiss_vector_dimension,
            metric=settings.faiss_metric,
        )
        logger.info(
            "VectorStore initialised (dim=%d, metric=%s)",
            settings.faiss_vector_dimension,
            settings.faiss_metric,
        )
        return store
    except Exception:
        logger.exception("Failed to initialise ConceptVectorStore")
        return None

