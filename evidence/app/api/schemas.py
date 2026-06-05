# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ============== Request Models ==============

# To-be-added: "Vector Search", "Regular Graph Query", etc
QueryType = Literal["Semantic Graph Traversal"]

class Header(BaseModel):
    workspace_id: str = Field(..., description="Mandatory workspace identifier")
    mas_id: str = Field(..., description="Mandatory MAS identifier")
    agent_id: Optional[str] = Field(None, description="Optional agent identifier")

class RagRetrievalParams(BaseModel):
    """Optional per-request overrides; defaults come from evidence service settings (EVIDENCE_RAG_*)."""

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Max similar chunks to retrieve from /rag/search.",
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Seconds to wait for RAG retrieval before treating as empty.",
    )


class QueryMetadata(BaseModel):
    query_type: Optional[QueryType] = Field(
        default="Semantic Graph Traversal",
        description="For now this metadata element is optional since we only support Semantic Graph traversal"
    )
    rag: Optional[RagRetrievalParams] = Field(
        default=None,
        description="Overrides EVIDENCE_RAG_TOP_K / EVIDENCE_RAG_TIMEOUT_SEC for RAG similarity lookup.",
    )

class RequestPayload(BaseModel):
    intent: str
    metadata: Optional[QueryMetadata] = Field(default_factory=QueryMetadata)
    additional_context: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    records: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class ReasonerCognitionRequest(BaseModel):
    header: Header
    request_id: str = Field(
        description="ID for the incoming request",
    )
    payload: RequestPayload

# ============== Response Models ==============

class TokenUsage(BaseModel):
    """Token usage information from LLM calls."""
    prompt: int = Field(..., description="Number of prompt tokens")
    completion: int = Field(..., description="Number of completion tokens")
    total: int = Field(..., description="Total tokens (prompt + completion)")
    model: str = Field(..., description="Model used for the LLM call")


class TokenUsageMeta(BaseModel):
    """Metadata including token usage, latency, cost, and CE attribution."""
    tokens: TokenUsage = Field(..., description="Token usage information")
    latency_ms: float = Field(..., description="LLM call latency in milliseconds")
    cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the LLM call")
    ce_id: Optional[str] = Field(None, description="CE that performed the operation (for CFN metrics attribution)")


class KnowledgeRecord(BaseModel):
    id: str = Field(default="auto")
    type: Literal["json"] = "json"
    content: Dict[str, Any]


# Alias for backward compatibility
TKFKnowledgeRecord = KnowledgeRecord

class ErrorDetail(BaseModel):
    message: str
    detail: Optional[Dict[str, Any]] = None

class ReasonerCognitionResponse(BaseModel):
    header: Header
    response_id: str = Field(..., description="This will be returned populated from the request_id.")
    # Either error is present OR records/metadata are present
    error: Optional[ErrorDetail] = None
    records: List[KnowledgeRecord] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Token usage metadata (optional, present when LLM calls are made)
    meta: Optional[TokenUsageMeta] = Field(None, description="LLM token usage and performance metrics")


# ---- Placeholder DB-facing schemas (to be finalized later) ----

class GraphPathsRequest(BaseModel):
    source_id: str
    target_id: str
    max_depth: int = 3
    limit: int = 10
    relations: Optional[List[str]] = None


class PathEdge(BaseModel):
    from_id: str
    relation: str
    to_id: str
    from_name: Optional[str] = None
    to_name: Optional[str] = None


class Path(BaseModel):
    node_ids: Optional[List[str]] = None
    edges: List[PathEdge]
    path_length: Optional[int] = None
    symbolic: str


class GraphPathsResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    paths: List[Path] = Field(default_factory=list)


class NeighborsResponse(BaseModel):
    # Keep flexible for now; can be specialized later
    records: List[Dict[str, Any]] = Field(default_factory=list)


class ConceptsByIdsRequest(BaseModel):
    ids: List[str]


class Concept(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""


class ConceptsByIdsResponse(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)
