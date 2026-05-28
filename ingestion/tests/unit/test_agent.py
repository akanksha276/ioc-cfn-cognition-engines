# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for extraction services and KnowledgeProcessor.
"""

import asyncio

import pytest

from ingestion.app.agent.ingest_data import IngestDataService
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.knowledge_processor import (
    KnowledgeProcessor,
    EmbeddingManager,
    cosine_similarity,
    FASTEMBED_AVAILABLE,
)


# ---------------------------------------------------------------------------
# ConceptRelationshipExtractionService
# ---------------------------------------------------------------------------


class TestConceptRelationshipExtractionService:
    """Tests for ConceptRelationshipExtractionService compact-payload flow."""

    def test_generate_id_deterministic(self, concept_relationship_service):
        id1 = concept_relationship_service._generate_id("test_string")
        id2 = concept_relationship_service._generate_id("test_string")
        assert id1 == id2

    def test_empty_compact_payload_returns_expected_structure(
        self, concept_relationship_service
    ):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[]
            )
        )
        assert "knowledge_cognition_request_id" in result
        assert "concepts" in result
        assert "relations" in result
        assert "descriptor" in result
        assert "meta" in result
        assert result["meta"]["records_processed"] == 0

    def test_custom_request_id_is_echoed(
        self, concept_relationship_service
    ):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[], request_id="cr-custom-id"
            )
        )
        assert result["knowledge_cognition_request_id"] == "cr-custom-id"

    def test_format_descriptor_is_used(
        self, concept_relationship_service
    ):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[], format_descriptor="observe-sdk-otel"
            )
        )
        assert result["descriptor"] == "observe-sdk-otel"

    def test_default_descriptor(self, concept_relationship_service):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[]
            )
        )
        assert result["descriptor"] == "observe-sdk-otel"

    def test_empty_records(self, concept_relationship_service):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[]
            )
        )
        assert result["concepts"] == []
        assert result["relations"] == []
        assert result["meta"]["records_processed"] == 0

    def test_empty_records_with_custom_id(self, concept_relationship_service):
        result = asyncio.run(
            concept_relationship_service.extract_concepts_and_relationships(
                compact_payload=[], request_id="empty-cr"
            )
        )
        assert result["knowledge_cognition_request_id"] == "empty-cr"

    def test_unsupported_format_raises(self, concept_relationship_service):
        with pytest.raises(ValueError, match="Unsupported data format"):
            asyncio.run(
                concept_relationship_service.extract_concepts_and_relationships(
                    compact_payload=[],
                    format_descriptor="unknown-format",
                )
            )

    def test_non_empty_compact_payload_requires_llm(self):
        svc = ConceptRelationshipExtractionService(mock_mode=False)
        with pytest.raises(RuntimeError, match="LLM is not configured"):
            asyncio.run(
                svc.extract_concepts_and_relationships(
                    compact_payload=[{"ServiceName": "svc"}],
                    format_descriptor="observe-sdk-otel",
                )
            )

    def test_semneg_relations_include_domain_and_session_time(self):
        class StubConceptService(ConceptRelationshipExtractionService):
            def _has_llm(self):
                return True

            async def _llm_extract_concepts(self, compact_payload, system_prompt):
                return [
                    {"name": "agent_a", "type": "agent", "description": "Agent A"},
                    {"name": "agent_b", "type": "agent", "description": "Agent B"},
                ]

            async def _llm_extract_relationships(self, concepts, compact_payload, system_prompt):
                return [
                    {
                        "source": "agent_a",
                        "target": "agent_b",
                        "relationship": "NEGOTIATES_WITH",
                        "description": "Agents negotiated an outcome.",
                    }
                ]

        svc = StubConceptService()
        payload = [
            {"dt_created": "2026-01-01T01:00:00Z"},
            {"dt_created": "2026-01-02T12:30:00Z"},
        ]
        result = asyncio.run(
            svc.extract_concepts_and_relationships(
                compact_payload=payload,
                request_id="semneg-id",
                format_descriptor="semneg",
            )
        )

        assert result["knowledge_cognition_request_id"] == "semneg-id"
        assert result["descriptor"] == "semneg"
        assert len(result["relations"]) == 1
        rel_attrs = result["relations"][0]["attributes"]
        assert rel_attrs["session_time"] == "2026-01-02T12:30:00Z"
        assert rel_attrs["domain"] == "semneg"


class TestIngestDataService:
    """Tests for unified ingest orchestration."""

    def test_ingest_graph_only_mode_returns_rag_chunks_empty(self, sample_otel_records):
        class StubConceptService:
            def __init__(self):
                self.last_payload = None

            async def extract_concepts_and_relationships(
                self, compact_payload, request_id=None, format_descriptor=None
            ):
                self.last_payload = compact_payload
                return {
                    "knowledge_cognition_request_id": request_id or "stub-id",
                    "concepts": [],
                    "relations": [],
                    "descriptor": format_descriptor or "observe-sdk-otel",
                    "meta": {
                        "records_processed": len(compact_payload),
                        "concepts_extracted": 0,
                        "relations_extracted": 0,
                    },
                }

        stub = StubConceptService()
        ingest = IngestDataService(stub, enable_rag_ingest=False)
        result = asyncio.run(
            ingest.ingest(
                sample_otel_records,
                request_id="req-1",
                format_descriptor="observe-sdk-otel",
            )
        )
        assert result["knowledge_cognition_request_id"] == "req-1"
        assert "rag_chunks" in result
        assert result["rag_chunks"] == []
        assert isinstance(stub.last_payload, list)
        assert len(stub.last_payload) > 0

    def test_ingest_empty_records_returns_empty_shape(self):
        class StubConceptService:
            async def extract_concepts_and_relationships(
                self, compact_payload, request_id=None, format_descriptor=None
            ):
                raise AssertionError("Graph stage should not run for empty filtered input")

        ingest = IngestDataService(StubConceptService(), enable_rag_ingest=False)
        result = asyncio.run(
            ingest.ingest([], request_id="empty-1", format_descriptor="observe-sdk-otel")
        )
        assert result["knowledge_cognition_request_id"] == "empty-1"
        assert result["concepts"] == []
        assert result["relations"] == []
        assert result["rag_chunks"] == []
        assert result["meta"]["records_processed"] == 0

    def test_ingest_unsupported_format_raises(self):
        class StubConceptService:
            async def extract_concepts_and_relationships(
                self, compact_payload, request_id=None, format_descriptor=None
            ):
                return {}

        ingest = IngestDataService(StubConceptService(), enable_rag_ingest=False)
        with pytest.raises(ValueError, match="Unsupported data format"):
            asyncio.run(
                ingest.ingest([{}], request_id="bad-format", format_descriptor="unknown")
            )

    def test_ingest_rag_enabled_attaches_rag_chunks(self, sample_otel_records):
        class StubConceptService:
            async def extract_concepts_and_relationships(
                self, compact_payload, request_id=None, format_descriptor=None
            ):
                return {
                    "knowledge_cognition_request_id": request_id or "stub-id",
                    "concepts": [],
                    "relations": [],
                    "descriptor": format_descriptor or "observe-sdk-otel",
                    "meta": {
                        "records_processed": len(compact_payload),
                        "concepts_extracted": 0,
                        "relations_extracted": 0,
                    },
                }

        class StubRagPipeline:
            def run(self, rag_docs):
                assert isinstance(rag_docs, list)
                return [
                    {
                        "text": "chunk text",
                        "embedding": [[0.1, 0.2]],
                        "metadata": {"doc_index": 0, "chunk_index": 0},
                    }
                ]

        ingest = IngestDataService(
            StubConceptService(),
            enable_rag_ingest=True,
            rag_pipeline=StubRagPipeline(),
        )
        result = asyncio.run(
            ingest.ingest(
                sample_otel_records,
                request_id="req-rag",
                format_descriptor="observe-sdk-otel",
            )
        )

        assert result["knowledge_cognition_request_id"] == "req-rag"
        assert len(result["rag_chunks"]) == 1
        assert result["rag_chunks"][0]["metadata"]["chunk_index"] == 0

    def test_ingest_rag_failure_does_not_fail_graph(self, sample_otel_records):
        class StubConceptService:
            async def extract_concepts_and_relationships(
                self, compact_payload, request_id=None, format_descriptor=None
            ):
                return {
                    "knowledge_cognition_request_id": request_id or "stub-id",
                    "concepts": [],
                    "relations": [],
                    "descriptor": format_descriptor or "observe-sdk-otel",
                    "meta": {
                        "records_processed": len(compact_payload),
                        "concepts_extracted": 0,
                        "relations_extracted": 0,
                    },
                }

        class FailingRagPipeline:
            def run(self, rag_docs):
                raise RuntimeError("rag failed")

        ingest = IngestDataService(
            StubConceptService(),
            enable_rag_ingest=True,
            rag_pipeline=FailingRagPipeline(),
        )
        result = asyncio.run(
            ingest.ingest(
                sample_otel_records,
                request_id="req-rag-fail",
                format_descriptor="observe-sdk-otel",
            )
        )

        assert result["knowledge_cognition_request_id"] == "req-rag-fail"
        assert result["rag_chunks"] == []


# ---------------------------------------------------------------------------
# KnowledgeProcessor
# ---------------------------------------------------------------------------


class TestKnowledgeProcessor:
    """Tests for KnowledgeProcessor."""

    def test_process_adds_dedup_meta(
        self, knowledge_processor_with_dedup, sample_extraction_result
    ):
        result = knowledge_processor_with_dedup.process(sample_extraction_result)
        assert "dedup_enabled" in result["meta"]
        assert result["meta"]["dedup_enabled"] is True
        assert "concepts_deduped" in result["meta"]
        assert "relations_deduped" in result["meta"]

    def test_name_based_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "agent_a",
                    "description": "Agent A",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "agent_a",
                    "description": "Agent A copy",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "3",
                    "name": "agent_b",
                    "description": "Agent B",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 3,
                "relations_extracted": 0,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["concepts"]) == 2
        assert result["meta"]["concepts_deduped"] == 1

    def test_no_dedup_preserves_all(self, knowledge_processor_no_dedup):
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "agent_a",
                    "description": "Agent A",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "agent_a",
                    "description": "Agent A copy",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 2,
                "relations_extracted": 0,
            },
        }
        result = knowledge_processor_no_dedup.process(extraction_result)
        assert len(result["concepts"]) == 2
        assert result["meta"]["dedup_enabled"] is False

    def test_relation_dedup_removes_duplicates(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
                {"id": "2", "name": "llm_a", "attributes": {"concept_type": "llm"}},
            ],
            "relations": [
                {
                    "id": "r1",
                    "node_ids": ["1", "2"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
                {
                    "id": "r2",
                    "node_ids": ["1", "2"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
                {
                    "id": "r3",
                    "node_ids": ["1", "2"],
                    "relationship": "INVOKES_TOOL",
                    "attributes": {},
                },
            ],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 2,
                "relations_extracted": 3,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["relations"]) == 2
        assert result["meta"]["relations_deduped"] == 1

    def test_relation_dedup_filters_invalid_nodes(self, knowledge_processor_with_dedup):
        extraction_result = {
            "concepts": [
                {"id": "1", "name": "agent_a", "attributes": {"concept_type": "agent"}},
            ],
            "relations": [
                {
                    "id": "r1",
                    "node_ids": ["1", "999"],
                    "relationship": "SENDS_PROMPT_TO",
                    "attributes": {},
                },
            ],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 1,
                "relations_extracted": 1,
            },
        }
        result = knowledge_processor_with_dedup.process(extraction_result)
        assert len(result["relations"]) == 0


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""

    def test_embedding_manager_init(self):
        manager = EmbeddingManager()
        if FASTEMBED_AVAILABLE:
            assert manager.model is not None
        else:
            assert manager.model is None

    def test_generate_embedding_empty_text(self):
        manager = EmbeddingManager()
        result = manager.generate_embedding("")
        assert result is None

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_generate_embedding_returns_array(self):
        import numpy as np

        manager = EmbeddingManager()
        result = manager.generate_embedding("test text")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_opposite_vectors(self):
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity + 1.0) < 0.0001

    def test_zero_vector(self):
        import numpy as np

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


# ---------------------------------------------------------------------------
# Semantic Deduplication
# ---------------------------------------------------------------------------


class TestSemanticDeduplication:
    """Tests for semantic deduplication with embeddings."""

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_semantic_dedup_similar_concepts(self):
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=True,
            similarity_threshold=0.9,
        )
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "weather_agent",
                    "description": "Agent for weather lookup",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "2",
                    "name": "weather_agent_v2",
                    "description": "Agent for weather lookup version 2",
                    "attributes": {"concept_type": "agent"},
                },
                {
                    "id": "3",
                    "name": "payment_agent",
                    "description": "Agent for payment processing",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 3,
                "relations_extracted": 0,
            },
        }
        result = processor.process(extraction_result)
        for concept in result["concepts"]:
            assert "embedding" in concept.get("attributes", {})

    @pytest.mark.skipif(
        not FASTEMBED_AVAILABLE,
        reason="fastembed not available",
    )
    def test_embedding_format(self):
        processor = KnowledgeProcessor(
            enable_embeddings=True,
            enable_dedup=False,
            similarity_threshold=0.95,
        )
        extraction_result = {
            "concepts": [
                {
                    "id": "1",
                    "name": "test_agent",
                    "description": "Test agent",
                    "attributes": {"concept_type": "agent"},
                },
            ],
            "relations": [],
            "meta": {
                "records_processed": 0,
                "concepts_extracted": 1,
                "relations_extracted": 0,
            },
        }
        result = processor.process(extraction_result)
        embedding = result["concepts"][0]["attributes"].get("embedding")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 1
        assert isinstance(embedding[0], list)
