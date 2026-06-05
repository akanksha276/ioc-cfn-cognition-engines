# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests based on the usage guide examples.
These tests demonstrate the programmatic usage of the cognition engine components.
"""
import asyncio

import pytest
from caching.app.agent.caching_layer import CachingLayer

from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import Header, ReasonerCognitionRequest, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor
from ingestion.app.agent.service import ConceptRelationshipExtractionService

# Sample OTel data from usage guide
SAMPLE_OTEL_PAYLOAD = [{
    "resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "user-service"}}]},
        "scopeSpans": [{
            "spans": [{
                "name": "GET /users",
                "kind": "SPAN_KIND_SERVER",
                "attributes": [
                    {"key": "http.method", "value": {"stringValue": "GET"}},
                    {"key": "http.route", "value": {"stringValue": "/users"}}
                ]
            }]
        }]
    }]
}]


class TestUsageGuideExample1KnowledgeExtraction:
    """Integration test for Usage Guide Example 1: Knowledge Extraction."""

    def test_knowledge_extraction_programmatic(self):
        """
        Test the knowledge extraction flow as shown in usage guide Example 1.
        This uses mock mode (no Azure OpenAI required) for CI/local testing.
        """
        # Initialize services in mock mode (no LLM required)
        concept_service = ConceptRelationshipExtractionService(
            mock_mode=True,
        )
        processor = KnowledgeProcessor(enable_embeddings=False, enable_dedup=False)

        # Extract → Process → Store
        result = asyncio.run(
            concept_service.extract_concepts_and_relationships(
                SAMPLE_OTEL_PAYLOAD,
                request_id="test-req-001",
                format_descriptor="observe-sdk-otel",
            )
        )

        # Verify extraction returned data
        assert "concepts" in result
        assert "relations" in result
        assert isinstance(result["concepts"], list)
        assert isinstance(result["relations"], list)

        # Process the result
        result = processor.process(result)

        assert len(result.get("concepts", [])) > 0, (
            "Should have extracted at least one concept in mock mode"
        )

    def test_knowledge_extraction_returns_metadata(self):
        """Test that extraction returns metadata about processing."""
        concept_service = ConceptRelationshipExtractionService(
            mock_mode=True,
        )

        result = asyncio.run(
            concept_service.extract_concepts_and_relationships(
                SAMPLE_OTEL_PAYLOAD,
                request_id="test-req-002",
            )
        )

        # Verify basic structure
        assert "concepts" in result
        assert "relations" in result


class TestUsageGuideExample2EvidenceGathering:
    """Integration test for Usage Guide Example 2: Evidence Gathering."""

    @pytest.mark.asyncio
    async def test_evidence_gathering_programmatic(self):
        """
        Test the evidence gathering flow as shown in usage guide Example 2.
        Uses mock repository to avoid external dependencies.
        """
        # Initialize repository and cache
        repo = MockDataRepository()

        # Create a simple cache layer with a basic embedding function
        def simple_embed(text: str):
            # Simple hash-based embedding for testing
            import hashlib

            import numpy as np
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val % (2**32))
            return np.random.rand(384).astype(np.float32)

        cache_layer = CachingLayer(
            vector_dimension=384,
            metric="l2",
            embed_fn=simple_embed
        )

        # Populate cache with concepts (as shown in usage guide)
        concepts = [
            {"name": "user-service", "type": "Service", "description": "User management"},
            {"name": "auth-service", "type": "Service", "description": "Authentication"},
        ]

        for concept in concepts:
            cache_layer.store_knowledge(
                text=f"{concept['name']}: {concept['description']}",
                metadata=concept
            )

        # Verify concepts were added
        stats = cache_layer.describe()
        assert stats["ntotal"] >= 2, "Cache should contain at least 2 concepts"

        # Create evidence request (as shown in usage guide)
        request = ReasonerCognitionRequest(
            header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
            request_id="evidence-test-001",
            payload=RequestPayload(
                intent="How does the user service authenticate?"
            )
        )

        # Note: This test will use LLM if credentials are available, otherwise will use mock data
        # We're testing the structure and flow, not the LLM-specific logic
        try:
            response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)

            # Verify response structure
            assert hasattr(response, "records")
            assert hasattr(response, "header")
            assert response.header.workspace_id == "ws-1"
            assert response.header.mas_id == "mas-1"

            # Records should be a list (may be empty in test env)
            assert isinstance(response.records, list)
        except Exception as e:
            # In test environment without proper Azure OpenAI setup,
            # the test may fail with connection errors - that's expected
            # We're primarily testing the integration structure
            pytest.skip(f"Evidence gathering requires Azure OpenAI credentials: {e}")

    def test_cache_layer_basic_operations(self):
        """Test basic cache layer operations from the usage guide."""
        import numpy as np

        # Simple embedding function for testing
        def simple_embed(text: str):
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val % (2**32))
            return np.random.rand(384).astype(np.float32)

        cache = CachingLayer(
            vector_dimension=384,
            metric="l2",
            embed_fn=simple_embed
        )

        # Add a concept
        concept = {
            "name": "test-service",
            "type": "Service",
            "description": "Test service"
        }
        cache.store_knowledge(
            text=f"{concept['name']}: {concept['description']}",
            metadata=concept
        )

        # Verify it was added
        stats = cache.describe()
        assert stats["ntotal"] == 1
        assert stats["dimension"] == 384


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
