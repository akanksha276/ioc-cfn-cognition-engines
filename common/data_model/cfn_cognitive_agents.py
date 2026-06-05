# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

from . import Concept, RecordType, Relation

if TYPE_CHECKING:
    from .cfn_memory import CFNKnowledgeRecord


class CAKnowledgeRecord(BaseModel):
    record_id: str
    record_type: Optional[RecordType] = None
    content: Optional[Any] = None  ##
    meta: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_object_content(self):
        if self.record_type and self.content is not None:
            if self.record_type == RecordType.string and not isinstance(
                self.content, str
            ):
                raise ValueError(
                    "content must be a string when record_type is 'string'"
                )
            elif self.record_type == RecordType.json and not isinstance(
                self.content, (dict, list)
            ):
                raise ValueError(
                    "content must be dict or list when record_type is 'json'"
                )
            elif self.record_type in [
                RecordType.binary,
                RecordType.image,
                RecordType.audio,
                RecordType.video,
            ] and not isinstance(self.content, bytes):
                raise ValueError(
                    f"content must be bytes when record_type is '{self.record_type.value}'"
                )
            elif self.record_type == RecordType.timeseries and not isinstance(
                self.content, list
            ):
                raise ValueError(
                    "content must be a list when record_type is 'timeseries'"
                )
        return self

class KnowledgeCognitionRequest(BaseModel):
    knowledge_cognition_request_id: str
    concepts: List[Concept] = []
    relations: List[Relation] = []
    descriptor: Optional[str] = None
    ## Additional metadata including request time
    meta: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_relation_concept_references(self):
        """Validate that all relation references exist in the concepts"""
        if not self.relations:
            return self

        # Collect all concept IDs
        all_concept_ids = {concept.id for concept in self.concepts}

        # Validate all relation references
        invalid_relations = []
        for relation_idx, relation in enumerate(self.relations):
            # Check that all node_ids in the relation exist in the concepts list
            for node_id in relation.node_ids:
                if node_id not in all_concept_ids:
                    invalid_relations.append(
                        f"Relation {relation_idx}: node_id '{node_id}' not found in concepts"
                    )

        if invalid_relations:
            raise ValueError(
                f"Invalid relation concept references: {'; '.join(invalid_relations)}"
            )

        return self

    def get_all_concept_ids(self) -> set:
        """Return a set of all concept IDs from the concepts list"""
        return {concept.id for concept in self.concepts}

    def find_concept_by_id(self, concept_id: str) -> Optional[Concept]:
        """Find a concept by its ID in the concepts list"""
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        return None

    def validate_relation_references(self) -> List[str]:
        """Manually validate relation references and return list of validation errors"""
        all_concept_ids = self.get_all_concept_ids()
        invalid_relations = []

        for relation_idx, relation in enumerate(self.relations):
            # Check that all node_ids in the relation exist in the concepts list
            for node_id in relation.node_ids:
                if node_id not in all_concept_ids:
                    invalid_relations.append(
                        f"Relation {relation_idx}: node_id '{node_id}' not found"
                    )
            # Validate that relation has at least 2 nodes
            if len(relation.node_ids) < 2:
                invalid_relations.append(
                    f"Relation {relation_idx}: must have at least 2 node_ids"
                )

        return invalid_relations


class KnowledgeCognitionResponse(BaseModel):
    knowledge_cognition_response_id: str
    status: str
    knowledge_cognition_request_id: str
    meta: Optional[Dict[str, Any]] = None


class ReasonerCognitionRequest(BaseModel):
    reasoner_cognition_request_id: str
    records: List[CAKnowledgeRecord] = []
    intent: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ReasonerCognitionResponse(BaseModel):
    reasoner_cognition_response_id: str
    status: str
    reasoner_cognition_request_id: str
    records: List["CFNKnowledgeRecord"] = []
    meta: Optional[Dict[str, Any]] = None


# Rebuild the model after CFNKnowledgeRecord is defined
def _rebuild_models():

    ReasonerCognitionResponse.model_rebuild()


_rebuild_models()
