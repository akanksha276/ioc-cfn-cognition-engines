# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
CFN Cognitive Agents Data Model Library

A Python library for CFN Cognitive Agents data models with configuration management
and health monitoring capabilities.
"""

__version__ = "0.1.0"
__author__ = "Cisco ETI"
__email__ = "eti-team@cisco.com"

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .config import Config


class RecordType(str, Enum):
    string = "string"
    json = "json"
    ## Future supported types
    binary = "binary"
    image = "image"
    audio = "audio"
    video = "video"
    timeseries = "timeseries"


class Concept(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    node_ids: List[str] = Field(
        default_factory=list, description="List of node IDs connected by this edge"
    )
    relationship: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class Path(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    path_sequence: List[str] = Field(
        description="Alternating sequence of node_id, edge_id, node_id, etc."
    )
    attributes: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_path_structure(self):
        """Validate that path has correct structure: alternating node-edge-node sequence"""
        if len(self.path_sequence) < 3:
            raise ValueError("Path must contain at least 3 elements (node-edge-node)")

        if len(self.path_sequence) % 2 == 0:
            raise ValueError(
                "Path sequence must have odd length (start and end with nodes)"
            )

        # Could add additional validation here to check that odd indices are node IDs
        # and even indices are edge IDs if needed

        return self


class EmbeddingRecord(BaseModel):
    config: Optional[Dict[str, Any]] = None
    data: Optional[List[float]] = None


from .cfn_cognitive_agents import (
    CAKnowledgeRecord,
    KnowledgeCognitionRequest,
    KnowledgeCognitionResponse,
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
)
from .cfn_memory import (
    CFNEvidenceRecord,
    CFNKnowledgeRecord,
    CFNQueryRequest,
    CFNQueryResponse,
    EmbeddingRecord,
    EntityRecord,
    MemoryType,
    ReasoningRequest,
    ReasoningResponse,
)

__all__ = [
    "Config",
    "Concept",
    "Relation",
    "Path",
    "RecordType",
    "EmbeddingRecord",
    "CAKnowledgeRecord",
    "KnowledgeCognitionRequest",
    "KnowledgeCognitionResponse",
    "ReasonerCognitionRequest",
    "ReasonerCognitionResponse",
    "MemoryType",
    "CFNKnowledgeRecord",
    "CFNEvidenceRecord",
    "ReasoningRequest",
    "ReasoningResponse",
    "EntityRecord",
    "CFNQueryRequest",
    "CFNQueryResponse",
]
