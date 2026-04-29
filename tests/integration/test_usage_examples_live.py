# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests using real Azure OpenAI credentials from .env file.
These tests verify the full flow with actual LLM calls.
"""
import asyncio
import pytest

from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor
from ingestion.app.config.settings import Settings  # used in skipif condition
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer


# Sample OTel data in the format expected by ConceptRelationshipExtractionService
SAMPLE_OTEL_PAYLOAD = [
    {
        "Timestamp": "2026-03-17 18:30:00.000000000",
        "TraceId": "test_trace_001",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "user.service",
        "SpanKind": "Server",
        "ServiceName": "user-service",
        "SpanAttributes": {
            "agent_id": "user_agent",
            "http.method": "GET",
            "http.route": "/users",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "Get user list"
        },
        "Duration": 1500000
    },
    {
        "Timestamp": "2026-03-17 18:30:01.000000000",
        "TraceId": "test_trace_001",
        "SpanId": "span_002",
        "ParentSpanId": "span_001",
        "SpanName": "auth.service",
        "SpanKind": "Client",
        "ServiceName": "auth-service",
        "SpanAttributes": {
            "agent_id": "auth_agent",
            "auth.method": "JWT",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 800000
    },
    {
        "Timestamp": "2026-03-17 18:30:02.000000000",
        "TraceId": "test_trace_001",
        "SpanId": "span_003",
        "ParentSpanId": "span_002",
        "SpanName": "database.query",
        "SpanKind": "Client",
        "ServiceName": "postgres-db",
        "SpanAttributes": {
            "db.system": "postgresql",
            "db.statement": "SELECT * FROM users"
        },
        "Duration": 500000
    }
]


@pytest.mark.requires_credentials
@pytest.mark.skipif(
    not Settings().llm_api_key and not Settings().llm_base_url,
    reason="LLM credentials not configured (set LLM_API_KEY or LLM_BASE_URL in .env)"
)
class TestUsageExamplesLive:
    """Integration tests using actual LLM credentials.

    These tests are marked with @pytest.mark.requires_credentials and will be
    skipped in CI/CD environments. To run them locally:

        PYTHONPATH=. uv run pytest tests/integration/test_usage_examples_live.py -v -s

    Or to run all tests including live tests:

        PYTHONPATH=. uv run pytest tests/integration/ -v --run-requires-credentials
    """

    def test_knowledge_extraction_with_llm(self):
        """
        Test knowledge extraction with real LLM calls.
        This test requires LLM credentials in .env (LLM_API_KEY or LLM_BASE_URL)
        """
        # Initialize services with litellm (reads LLM_MODEL/LLM_API_KEY/LLM_BASE_URL from env)
        concept_service = ConceptRelationshipExtractionService(
            mock_mode=False,  # Use real LLM
        )
        processor = KnowledgeProcessor(enable_embeddings=False, enable_dedup=False)

        # Extract → Process → Store
        result = concept_service.extract_concepts_and_relationships(
            SAMPLE_OTEL_PAYLOAD,
            request_id="test-req-llm-001",
            format_descriptor="observe-sdk-otel"
        )

        # Verify extraction returned data
        assert "concepts" in result
        assert "relations" in result
        assert isinstance(result["concepts"], list)
        assert isinstance(result["relations"], list)

        # With LLM, we expect meaningful extraction
        assert len(result["concepts"]) > 0, "LLM should extract at least one concept"

        print(f"\n✅ LLM extracted {len(result['concepts'])} concepts and {len(result['relations'])} relations")

        # Print extracted concepts for verification
        for concept in result["concepts"][:3]:  # Show first 3
            print(f"  - Concept: {concept.get('name')} (type: {concept.get('attributes', {}).get('concept_type')})")

        # Process the result
        result = processor.process(result)

        concepts = result.get("concepts", [])
        if concepts:
            print(f"✅ Processed {len(concepts)} concepts after knowledge pipeline")

    @pytest.mark.asyncio
    async def test_evidence_gathering_with_llm(self):
        """
        Test evidence gathering with real Azure OpenAI LLM calls.
        This test requires valid Azure OpenAI credentials in .env
        """
        # Initialize repository and cache
        repo = MockDataRepository()

        # Create a simple cache layer with a basic embedding function
        def simple_embed(text: str):
            import numpy as np
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val % (2**32))
            return np.random.rand(384).astype(np.float32)

        cache_layer = CachingLayer(
            vector_dimension=384,
            metric="l2",
            embed_fn=simple_embed
        )

        # Populate cache with concepts
        concepts = [
            {"name": "user-service", "type": "Service", "description": "User management service handling authentication"},
            {"name": "auth-service", "type": "Service", "description": "Authentication service providing JWT tokens"},
            {"name": "database", "type": "Service", "description": "PostgreSQL database storing user credentials"},
        ]

        for concept in concepts:
            cache_layer.store_knowledge(
                text=f"{concept['name']}: {concept['description']}",
                metadata=concept
            )

        print(f"\n✅ Populated cache with {len(concepts)} concepts")

        # Create evidence request
        request = ReasonerCognitionRequest(
            header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
            request_id="evidence-test-llm-001",
            payload=RequestPayload(
                intent="How does the user service authenticate?"
            )
        )

        # Gather evidence using real LLM
        response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)

        # Verify response structure
        assert hasattr(response, "records")
        assert hasattr(response, "header")
        assert response.header.workspace_id == "ws-1"
        assert response.header.mas_id == "mas-1"

        # Records should be a list
        assert isinstance(response.records, list)

        print(f"✅ Evidence gathering returned {len(response.records)} records")

        # Print first few records
        for i, record in enumerate(response.records[:3]):
            print(f"  - Record {i+1}: {str(record)[:100]}...")

    def test_extraction_metadata_with_llm(self):
        """Test that LLM extraction returns detailed metadata."""
        concept_service = ConceptRelationshipExtractionService(
            mock_mode=False,
        )

        result = concept_service.extract_concepts_and_relationships(
            SAMPLE_OTEL_PAYLOAD,
            request_id="test-req-llm-002"
        )

        # Verify structure
        assert "concepts" in result
        assert "relations" in result
        assert "meta" in result

        # Verify metadata fields
        meta = result["meta"]
        assert "records_processed" in meta
        assert "concepts_extracted" in meta
        assert "relations_extracted" in meta

        print(f"\n✅ Metadata: {meta}")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
