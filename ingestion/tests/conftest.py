# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures and test utilities.
"""
from typing import Any, Dict, List

import pytest

from ingestion.app.agent.knowledge_processor import KnowledgeProcessor
from ingestion.app.agent.service import ConceptRelationshipExtractionService, TelemetryExtractionService
from ingestion.app.data.mock_repo import MockDataRepository

SAMPLE_OTEL_RECORDS: List[Dict[str, Any]] = [
    {
        "Timestamp": "2025-12-22 18:37:22.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "orchestrator.agent",
        "SpanKind": "Server",
        "ServiceName": "corto.orchestrator",
        "SpanAttributes": {
            "agent_id": "orchestrator_agent",
            "execution.success": "true",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "What is the weather in NYC?"
        },
        "Duration": 21346166
    },
    {
        "Timestamp": "2025-12-22 18:37:23.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_002",
        "ParentSpanId": "span_001",
        "SpanName": "worker.agent",
        "SpanKind": "Client",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "agent_id": "worker_agent",
            "execution.success": "true",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 10000000
    },
    {
        "Timestamp": "2025-12-22 18:37:24.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_003",
        "ParentSpanId": "span_002",
        "SpanName": "llm.call",
        "SpanKind": "Client",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.response.model": "gpt-4o",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "What is the weather?",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "The weather is sunny."
        },
        "Duration": 5000000
    },
    {
        "Timestamp": "2025-12-22 18:37:25.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_004",
        "ParentSpanId": "span_003",
        "SpanName": "tool.execution",
        "SpanKind": "Client",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.completion.0.tool_calls.0.name": "weather_lookup",
            "gen_ai.completion.0.tool_calls.0.arguments": "{\"location\": \"NYC\"}"
        },
        "Duration": 2000000
    }
]


def build_extraction_request(
    otel_records: List[Dict[str, Any]],
    workspace_id: str = "test-ws",
    mas_id: str = "test-mas",
    agent_id: str | None = None,
    request_id: str | None = None,
    data_format: str = "observe-sdk-otel",
) -> Dict[str, Any]:
    """Build a valid ExtractionRequest envelope for testing."""
    header: Dict[str, Any] = {"workspace_id": workspace_id, "mas_id": mas_id}
    if agent_id is not None:
        header["agent_id"] = agent_id
    body: Dict[str, Any] = {
        "header": header,
        "payload": {
            "metadata": {"format": data_format},
            "data": otel_records,
        },
    }
    if request_id is not None:
        body["request_id"] = request_id
    return body


@pytest.fixture
def sample_otel_records() -> List[Dict[str, Any]]:
    """Provide sample OTel records for testing."""
    return [r.copy() for r in SAMPLE_OTEL_RECORDS]


@pytest.fixture
def extraction_service() -> TelemetryExtractionService:
    """Create a TelemetryExtractionService without LLM (basic mode)."""
    return TelemetryExtractionService()


@pytest.fixture
def concept_relationship_service() -> ConceptRelationshipExtractionService:
    """Create a ConceptRelationshipExtractionService without LLM (mock mode)."""
    return ConceptRelationshipExtractionService(mock_mode=True)


@pytest.fixture
def knowledge_processor_with_dedup() -> KnowledgeProcessor:
    """Create a KnowledgeProcessor with dedup enabled."""
    return KnowledgeProcessor(
        enable_embeddings=False,
        enable_dedup=True,
        similarity_threshold=0.95
    )


@pytest.fixture
def knowledge_processor_no_dedup() -> KnowledgeProcessor:
    """Create a KnowledgeProcessor with dedup disabled."""
    return KnowledgeProcessor(
        enable_embeddings=False,
        enable_dedup=False,
        similarity_threshold=0.95
    )


@pytest.fixture
def data_repository() -> MockDataRepository:
    """Create a MockDataRepository instance."""
    return MockDataRepository()


@pytest.fixture
def sample_extraction_result(extraction_service, sample_otel_records) -> Dict[str, Any]:
    """Provide a sample extraction result from TelemetryExtractionService."""
    return extraction_service.extract_entities_and_relations(sample_otel_records)


@pytest.fixture
def sample_concept_result(concept_relationship_service, sample_otel_records) -> Dict[str, Any]:
    """Provide a sample concept extraction result shape from compact-payload API."""
    import asyncio

    return asyncio.run(
        concept_relationship_service.extract_concepts_and_relationships(
            compact_payload=[],
            format_descriptor="observe-sdk-otel",
        )
    )
