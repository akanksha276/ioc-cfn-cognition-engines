# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.data_model import (
    CAKnowledgeRecord,
    CFNEvidenceRecord,
    CFNKnowledgeRecord,
    CFNQueryRequest,
    Concept,
    EntityRecord,
    KnowledgeCognitionRequest,
    RecordType,
    Relation,
)


def test_caknowledge_record_accepts_matching_types():
    record = CAKnowledgeRecord(
        record_id="rec-string",
        record_type=RecordType.string,
        content="router description",
    )
    assert record.content == "router description"

    json_record = CAKnowledgeRecord(
        record_id="rec-json",
        record_type=RecordType.json,
        content={"vendor": "cisco"},
    )
    assert json_record.content == {"vendor": "cisco"}


def test_caknowledge_record_rejects_mismatched_content():
    with pytest.raises(ValueError):
        CAKnowledgeRecord(
            record_id="rec-bad-string",
            record_type=RecordType.string,
            content=123,
        )

    with pytest.raises(ValueError):
        CAKnowledgeRecord(
            record_id="rec-bad-json",
            record_type=RecordType.json,
            content="not json",
        )

    with pytest.raises(ValueError):
        CAKnowledgeRecord(
            record_id="rec-bad-image",
            record_type=RecordType.image,
            content="pixels",
        )

    with pytest.raises(ValueError):
        CAKnowledgeRecord(
            record_id="rec-bad-timeseries",
            record_type=RecordType.timeseries,
            content={"bad": "shape"},
        )


def test_knowledge_cognition_request_validates_relation_references():
    concepts = [
        Concept(id="c1", name="router"),
        Concept(id="c2", name="switch"),
    ]
    relations = [Relation(id="r1", node_ids=["c1", "c3"])]

    with pytest.raises(ValueError) as exc:
        KnowledgeCognitionRequest(
            knowledge_cognition_request_id="req-1",
            concepts=concepts,
            relations=relations,
        )

    assert "node_id 'c3'" in str(exc.value)


def test_knowledge_cognition_request_helpers_return_expected_values():
    concepts = [
        Concept(id="c1", name="router"),
        Concept(id="c2", name="switch"),
    ]
    relations = [Relation(id="r1", node_ids=["c1", "c2"], relationship="links")]

    request = KnowledgeCognitionRequest(
        knowledge_cognition_request_id="req-2",
        concepts=concepts,
        relations=relations,
    )

    assert request.get_all_concept_ids() == {"c1", "c2"}
    assert request.find_concept_by_id("c2").name == "switch"
    assert request.validate_relation_references() == []


def test_cfn_knowledge_record_validates_content_type():
    CFNKnowledgeRecord(id="kn-1", content="details")
    CFNKnowledgeRecord(id="kn-2", content={"field": "value"})

    with pytest.raises(ValueError):
        CFNKnowledgeRecord(id="kn-3", content=[1, 2, 3])


def test_cfn_evidence_record_validates_content_type():
    CFNEvidenceRecord(id="ev-1", content="text evidence")
    CFNEvidenceRecord(id="ev-2", content={"score": 0.9})

    with pytest.raises(ValueError):
        CFNEvidenceRecord(id="ev-3", content=42)


def test_cfn_query_request_requires_entities():
    with pytest.raises(ValueError, match="Must provide entities"):
        CFNQueryRequest()


def test_cfn_query_request_accepts_entities():
    entity = EntityRecord(entity_name="router_a")

    request = CFNQueryRequest(entities=[entity])

    assert request.entities[0].entity_name == "router_a"
