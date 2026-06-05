# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Processor - Handles embeddings generation and deduplication for extracted knowledge.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _register_granite_in_fastembed(model_name: str) -> None:
    """Inject a custom model entry into fastembed's runtime ONNX registry.

    fastembed validates model_name against its internal list even when
    specific_model_path is provided.  This inserts a minimal DenseModelDescription
    so the real IBM Granite model name is accepted without using a proxy entry.
    The list mutation is process-scoped and does not touch any installed files.
    """
    from fastembed.common.model_description import DenseModelDescription, ModelSource
    from fastembed.text.onnx_embedding import supported_onnx_models

    if any(m.model == model_name for m in supported_onnx_models):
        return  # already registered

    entry = DenseModelDescription(
        model=model_name,
        dim=384,
        description="IBM Granite 30M English embedding model (local ONNX int8)",
        license="apache-2.0",
        size_in_GB=0.03,
        sources=ModelSource(hf=model_name),
        model_file="model_optimized.onnx",
    )
    supported_onnx_models.append(entry)
    logger.debug("Registered %s in fastembed ONNX registry", model_name)


# Try to import fastembed, fall back gracefully
try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    logger.warning("fastembed not available. Embeddings will be skipped.")


class EmbeddingManager:
    """Manages embedding generation using fastembed (ONNX-based, no PyTorch required)."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-embedding-30m-english",
        model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model = None

        if FASTEMBED_AVAILABLE:
            # Prefer local model path (e.g. EMBEDDING_MODEL_PATH from repo / Docker)
            path = (model_path or os.getenv("EMBEDDING_MODEL_PATH", "")).strip()
            if path and os.path.isdir(path):
                logger.info("Loading embedding model from local path: %s", path)
                # Register the granite model in fastembed's runtime registry so we can
                # pass the real model name instead of a proxy entry.
                _register_granite_in_fastembed(model_name)
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.model = TextEmbedding(
                        model_name=model_name,
                        specific_model_path=path,
                    )
            else:
                cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
                # Granite is not in fastembed's built-in registry; register it before use.
                _register_granite_in_fastembed(model_name)
                self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
                logger.info(
                    "Loaded embedding model: %s from cache: %s", model_name, cache_dir
                )

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text string."""
        if not self.model or not text:
            return None

        return next(self.model.embed([text]))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class KnowledgeProcessor:
    """
    Processes extracted knowledge by:
    1. Generating embeddings for concepts
    2. Semantic deduplication using cosine similarity (optional)
    3. Deduplicating relations (optional)
    """

    def __init__(
        self,
        enable_embeddings: bool = True,
        enable_dedup: bool = True,
        similarity_threshold: float = 0.95,
        embedding_manager: Optional["EmbeddingManager"] = None,
    ):
        """
        Args:
            enable_embeddings: Enable embedding generation
            enable_dedup: Enable deduplication of concepts and relations
            similarity_threshold: Cosine similarity threshold for semantic dedup (0.0-1.0)
            embedding_manager: Pre-built EmbeddingManager instance (avoids re-loading the model)
        """
        self.enable_embeddings = enable_embeddings and FASTEMBED_AVAILABLE
        self.embedding_manager = embedding_manager if self.enable_embeddings else None
        if self.enable_embeddings and self.embedding_manager is None:
            self.embedding_manager = EmbeddingManager()
        self.enable_dedup = enable_dedup
        self.similarity_threshold = similarity_threshold

    def generate_embeddings_for_concepts(
        self, concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for each concept using its name.
        Stores raw numpy array in _embedding for dedup, then converts to list for output.
        """
        if not self.enable_embeddings:
            return concepts

        for concept in concepts:
            name = (concept.get("name") or "").strip()

            if name:
                embedding = self.embedding_manager.generate_embedding(name)
                if embedding is not None:
                    concept["_embedding"] = embedding  # numpy array for dedup

        return concepts

    def semantic_deduplicate_concepts(
        self, concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate concepts using semantic similarity.
        Concepts with cosine similarity >= threshold are considered duplicates.
        Falls back to name-based dedup if embeddings not available.
        """
        if not self.enable_embeddings:
            # Fallback to name-based dedup
            seen_names = set()
            deduped = []
            for concept in concepts:
                name = concept.get("name")
                if name and name not in seen_names:
                    seen_names.add(name)
                    deduped.append(concept)
            return deduped

        deduped = []
        deduped_embeddings = []

        for concept in concepts:
            embedding = concept.get("_embedding")

            if embedding is None:
                # No embedding, check by name
                name = concept.get("name")
                if not any(c.get("name") == name for c in deduped):
                    deduped.append(concept)
                continue

            # Check semantic similarity against already kept concepts
            is_duplicate = False
            for kept_embedding in deduped_embeddings:
                similarity = cosine_similarity(embedding, kept_embedding)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Duplicate found: {concept.get('name')} (similarity: {similarity:.3f})"
                    )
                    break

            if not is_duplicate:
                deduped.append(concept)
                deduped_embeddings.append(embedding)

        return deduped

    def finalize_embeddings(
        self, concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert numpy embeddings to list format for JSON serialization."""
        for concept in concepts:
            embedding = concept.pop("_embedding", None)
            if embedding is not None:
                if "attributes" not in concept:
                    concept["attributes"] = {}
                concept["attributes"]["embedding"] = [embedding.tolist()]

        return concepts

    def deduplicate_relations(
        self, relations: List[Dict[str, Any]], valid_concept_ids: set
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate relations and filter out relations with invalid concept references.
        """
        seen_keys = set()
        deduped = []

        for relation in relations:
            node_ids = relation.get("node_ids", [])
            relationship = relation.get("relationship", "")

            # Skip relations with missing nodes
            if not all(nid in valid_concept_ids for nid in node_ids):
                continue

            # Create dedup key from sorted node_ids + relationship
            rel_key = f"{tuple(sorted(node_ids))}_{relationship}"
            if rel_key not in seen_keys:
                seen_keys.add(rel_key)
                deduped.append(relation)

        return deduped

    def process(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the extraction result:
        1. Generate embeddings for concepts
        2. Semantic deduplication of concepts (if enabled)
        3. Deduplicate relations (if enabled)
        4. Finalize embeddings for output

        Args:
            extraction_result: Output from extract_entities_and_relations

        Returns:
            Processed result with embeddings and optionally deduped concepts/relations
        """
        concepts = extraction_result.get("concepts", [])
        relations = extraction_result.get("relations", [])

        original_concept_count = len(concepts)
        original_relation_count = len(relations)

        # Step 1: Generate embeddings
        concepts = self.generate_embeddings_for_concepts(concepts)

        # Step 2: Semantic deduplication of concepts (if enabled)
        if self.enable_dedup:
            concepts = self.semantic_deduplicate_concepts(concepts)

        # Step 3: Get valid concept IDs
        valid_concept_ids = {c.get("id") for c in concepts if c.get("id")}

        # Step 4: Deduplicate relations (if enabled)
        if self.enable_dedup:
            relations = self.deduplicate_relations(relations, valid_concept_ids)

        # Step 5: Finalize embeddings (convert to list for JSON)
        concepts = self.finalize_embeddings(concepts)

        # Update the result
        extraction_result["concepts"] = concepts
        extraction_result["relations"] = relations
        extraction_result["meta"]["concepts_extracted"] = len(concepts)
        extraction_result["meta"]["relations_extracted"] = len(relations)
        extraction_result["meta"]["dedup_enabled"] = self.enable_dedup
        extraction_result["meta"]["concepts_deduped"] = original_concept_count - len(
            concepts
        )
        extraction_result["meta"]["relations_deduped"] = original_relation_count - len(
            relations
        )

        logger.info(
            f"Processed: {len(concepts)} concepts, {len(relations)} relations "
            f"(dedup={'enabled' if self.enable_dedup else 'disabled'}, "
            f"removed {original_concept_count - len(concepts)} concepts, "
            f"{original_relation_count - len(relations)} relations)"
        )

        return extraction_result
