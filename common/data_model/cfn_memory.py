# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from . import EmbeddingRecord, RecordType


class MemoryType(Enum):
    ## TODO I an not sure if we have to do this but for now its okay
    unknown = "unknown"
    auto = "auto"
    semantic = "semantic"
    procedural = "procedural"
    episodic = "episodic"
    gmemory = "gmemory"

    @classmethod
    def get_values(cls):
        return [member.value for member in cls if member != cls.unknown]


class CFNKnowledgeRecord(BaseModel):
    id: str
    type: Optional[RecordType] = None
    content: Optional[Union[str, Dict[str, Any]]] = None
    # reference_query: Optional[str] = None

    @model_validator(mode="after")
    def validate_content(self):
        if self.content is not None:
            if not isinstance(self.content, (str, dict)):
                raise ValueError(
                    "content must be either a string or a dictionary (JSON)"
                )
        return self


class CFNEvidenceRecord(BaseModel):
    id: str
    type: Optional[RecordType] = None
    content: Optional[Union[str, Dict[str, Any]]] = None
    score: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_content(self):
        if self.content is not None:
            if not isinstance(self.content, (str, dict)):
                raise ValueError(
                    "content must be either a string or a dictionary (JSON)"
                )
        return self


class ReasoningRequest(BaseModel):
    request_id: str
    records: List[CFNKnowledgeRecord]
    intent: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ReasoningResponse(BaseModel):
    response_id: str  ## Must be same as request id
    evidence: List[CFNEvidenceRecord]
    meta: Optional[Dict[str, Any]] = None


class QueryCriteria(BaseModel):
    depth: Optional[int] = None
    threshold: Optional[float] = None
    additional_params: Optional[Dict[str, Any]] = None


class EntityRecord(BaseModel):
    entity_name: str
    embeddings: Optional[EmbeddingRecord] = None
    query_criteria: Optional[QueryCriteria] = None


## CFN/retrieve_knowledge
class CFNQueryRequest(BaseModel):
    memory_type: MemoryType = Field(default=MemoryType.auto)
    mas_id: Optional[str] = None
    entities: Optional[List[EntityRecord]] = None
    meta: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_entities(self):
        if self.entities is None:
            raise ValueError("Must provide entities")
        return self


class CFNQueryResponse(BaseModel):
    queried_entities: List[EntityRecord]
    retrieved_concepts: List[CFNKnowledgeRecord]
    relations: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]] = None
