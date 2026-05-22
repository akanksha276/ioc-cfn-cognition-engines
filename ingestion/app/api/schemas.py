# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic request/response models for API endpoints.
"""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ============== Extraction API Request Models ==============

class ExtractionHeader(BaseModel):
    """Header for knowledge-mgmt extraction requests."""
    workspace_id: str = Field(..., description="Workspace identifier (mandatory)")
    mas_id: str = Field(..., description="MAS identifier (mandatory)")
    agent_id: Optional[str] = Field(None, description="Agent identifier (optional)")


class PayloadFormat(str, Enum):
    observe_sdk_otel = "observe-sdk-otel"
    openclaw = "openclaw"
    locomo = "locomo"
    semneg = "semneg"


class PayloadMetadata(BaseModel):
    """Metadata describing the payload format and additional labels."""
    format: PayloadFormat = Field(
        ...,
        description="Data format: 'observe-sdk-otel', 'openclaw', 'locomo', or 'semneg'",
    )

    class Config:
        extra = "allow"


class ExtractionPayload(BaseModel):
    """Payload containing metadata and the raw data records."""
    metadata: PayloadMetadata
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Array of trace/data records matching the declared format",
    )


class ExtractionRequest(BaseModel):
    """Top-level request body for /api/knowledge-mgmt/extraction."""
    header: ExtractionHeader
    request_id: str = Field(
        description="Client-supplied request ID; service echoes it back as response_id",
    )
    payload: ExtractionPayload


# ============== Extraction API Response Models ==============

class ExtractionErrorDetail(BaseModel):
    """Detailed debugging information attached to an error response."""
    class Config:
        extra = "allow"


class ExtractionError(BaseModel):
    """Error block returned when processing fails."""
    message: str = Field(..., description="User-meaningful error message")
    detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Stack trace or debugging context",
    )

class TokenUsage(BaseModel):
    """LLM token consumption metrics."""
    prompt: int = Field(..., description="Input tokens consumed")
    completion: int = Field(..., description="Output tokens generated")
    total: int = Field(..., description="Total tokens (prompt + completion)")
    model: str = Field(..., description="Model identifier")


class TokenUsageMeta(BaseModel):
    """LLM call metadata including token usage."""
    tokens: TokenUsage
    latency_ms: float = Field(..., description="Time taken for LLM call in milliseconds")
    cost_usd: Optional[float] = Field(None, description="Calculated cost in USD")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class ExtractionResponseModel(BaseModel):
    """
    Unified response for /api/knowledge-mgmt/extraction.

    Either ``error`` is present **or** the concepts/relations/descriptor/metadata
    fields are present, but never both.
    """
    header: ExtractionHeader
    response_id: str = Field(
        ...,
        description="Populated from request_id",
    )
    error: Optional[ExtractionError] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    relations: Optional[List[Dict[str, Any]]] = None
    descriptor: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    rag_chunks: Optional[List[Dict[str, Any]]] = None
    meta: Optional[TokenUsageMeta] = Field(None, description="LLM token usage metadata")


# ============== Legacy OTel Record Models ==============

class OTelSpanAttributes(BaseModel):
    """OpenTelemetry span attributes."""
    agent_id: Optional[str] = None
    execution_success: Optional[str] = Field(None, alias="execution.success")
    gen_ai_request_model: Optional[str] = Field(None, alias="gen_ai.request.model")
    gen_ai_response_model: Optional[str] = Field(None, alias="gen_ai.response.model")
    
    class Config:
        extra = "allow"


class OTelRecord(BaseModel):
    """Single OpenTelemetry trace record."""
    Timestamp: Optional[str] = None
    TraceId: Optional[str] = None
    SpanId: Optional[str] = None
    ParentSpanId: Optional[str] = None
    SpanName: Optional[str] = None
    SpanKind: Optional[str] = None
    ServiceName: Optional[str] = None
    ResourceAttributes: Optional[Dict[str, Any]] = None
    SpanAttributes: Optional[Dict[str, Any]] = None
    Duration: Optional[int] = None
    StatusCode: Optional[str] = None
    Events_Name: Optional[List[str]] = Field(None, alias="Events.Name")
    Events_Attributes: Optional[List[Dict[str, Any]]] = Field(None, alias="Events.Attributes")
    
    class Config:
        extra = "allow"
        populate_by_name = True


# ============== LLM Structured Output Models ==============

class LLMConcept(BaseModel):
    """A single concept extracted by the LLM."""
    name: str = Field(..., description="Unique name for this concept")
    type: str = Field(
        ...,
        description=(
            "Concept type label. Common values: query, agent, service, llm, tool, "
            "function, output, other_concept. Format-specific types also accepted "
            "(e.g. document, entity, speaker, topic, intent, fact)."
        ),
    )
    description: str = Field(..., description="2-3 sentence description of this concept")


class LLMRelationship(BaseModel):
    """A single relationship between two concepts extracted by the LLM."""
    source: str = Field(..., description="Name of the source concept")
    target: str = Field(..., description="Name of the target concept")
    relationship: str = Field(
        ...,
        description="UPPER_SNAKE_CASE label describing the relationship",
    )
    description: str = Field(..., description="One sentence context for the relationship")


class LLMExtractionResult(BaseModel):
    """Structured output schema for LLM-based concept and relationship extraction."""
    concepts: List[LLMConcept] = Field(..., description="All extracted concepts")
    relationships: List[LLMRelationship] = Field(..., description="All extracted relationships")


class LLMConceptsResult(BaseModel):
    """Structured output for the concepts-only extraction stage."""
    concepts: List[LLMConcept] = Field(..., description="All extracted concepts")


class LLMRelationshipsResult(BaseModel):
    """Structured output for the relationships-only extraction stage."""
    relationships: List[LLMRelationship] = Field(..., description="All extracted relationships")


# ============== Response Models ==============

class ConceptAttributes(BaseModel):
    """Attributes for a concept."""
    concept_type: str
    embedding: Optional[List[List[float]]] = None
    
    class Config:
        extra = "allow"


class Concept(BaseModel):
    """Extracted concept/entity."""
    id: str
    name: str
    description: str
    type: str = "concept"
    attributes: ConceptAttributes


class RelationAttributes(BaseModel):
    """Attributes for a relation."""
    source_name: str
    target_name: str
    summarized_context: str
    
    class Config:
        extra = "allow"


class Relation(BaseModel):
    """Extracted relationship between concepts."""
    id: str
    node_ids: List[str]
    relationship: str
    attributes: RelationAttributes


class ExtractionMeta(BaseModel):
    """Metadata about the extraction process."""
    records_processed: int
    concepts_extracted: int
    relations_extracted: int
    dedup_enabled: Optional[bool] = None
    concepts_deduped: Optional[int] = None
    relations_deduped: Optional[int] = None


class ExtractionResponse(BaseModel):
    """Response from entity and relation extraction."""
    knowledge_cognition_request_id: str
    concepts: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    descriptor: str
    meta: ExtractionMeta
    rag_chunks: Optional[List[Dict[str, Any]]] = None


class MetricsResponse(BaseModel):
    """Operational metrics response."""
    records_processed: int
    records_sent: int
    records_failed: int
    last_run_timestamp: Optional[str] = None
    last_run_duration_seconds: float
    recent_errors: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    initialized: bool
    metrics: Dict[str, Any]
    recent_errors: List[str]

