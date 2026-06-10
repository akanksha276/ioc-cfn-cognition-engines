# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from distill.app.services.distillation_job import _coerce_distillation_read_lists


def test_coerce_flat_legacy_shape():
    raw = {
        "concepts": [{"id": "c1", "name": "N1"}],
        "relations": [{"id": "r1", "node_ids": ["c1", "c2"], "relation": "USES"}],
    }
    concepts, rels = _coerce_distillation_read_lists(raw)
    assert len(concepts) == 1
    assert rels[0]["relationship"] == "USES"


def test_coerce_memory_records_shape():
    raw = {
        "request_id": "abc",
        "status": "success",
        "message": "ok",
        "records": [
            {
                "concepts": [{"id": "a1", "name": "Anchor"}],
                "relationships": [
                    {"id": "rel1", "relation": "R1", "node_ids": ["a1", "x2"], "attributes": {}},
                ],
            }
        ],
    }
    concepts, rels = _coerce_distillation_read_lists(raw)
    assert concepts == [{"id": "a1", "name": "Anchor"}]
    assert rels[0]["id"] == "rel1"
    assert rels[0]["relationship"] == "R1"
    assert rels[0]["node_ids"] == ["a1", "x2"]


def test_coerce_cfn_distillation_read_shape_with_relation_field():
    """CFN returns relationships[].relation (not relationship); concepts in same record."""
    raw = {
        "records": [
            {
                "relationships": [
                    {
                        "id": "rel1",
                        "relation": "DELEGATES_TASK_TO",
                        "node_ids": ["anchor-a", "anchor-b"],
                        "attributes": {"mas_id": "mas-1"},
                    }
                ],
                "concepts": [
                    {"id": "anchor-a", "name": "Agent A", "attributes": {}},
                    {"id": "anchor-b", "name": "Agent B", "attributes": {}},
                ],
            }
        ]
    }
    concepts, rels = _coerce_distillation_read_lists(raw)
    assert len(concepts) == 2
    assert len(rels) == 1
    assert rels[0]["relationship"] == "DELEGATES_TASK_TO"


def test_records_nonempty_takes_precedence_over_top_level():
    raw = {
        "concepts": [{"id": "ignored", "name": "I"}],
        "records": [{"concepts": [{"id": "used", "name": "U"}], "relationships": []}],
    }
    concepts, rels = _coerce_distillation_read_lists(raw)
    assert [c["id"] for c in concepts] == ["used"]
    assert rels == []
