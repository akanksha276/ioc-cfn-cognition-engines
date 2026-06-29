# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from distill.app.agent.distill_core import (
    bucket_relations_incident_to_anchors,
    build_concept_map,
    symbolic_lines_distillation_batch,
)


def test_bucket_incident_includes_relation_for_both_anchor_endpoints():
    anchors = ["a", "b"]
    rels = [{"id": "r1", "node_ids": ["a", "b"], "relationship": "LINK"}]
    b = bucket_relations_incident_to_anchors(anchors, rels)
    assert len(b["a"]) == 1 and len(b["b"]) == 1


def test_symbolic_uses_source_target_names_when_present():
    batch = [
        {
            "id": "e1",
            "node_ids": ["x", "y"],
            "relationship": "KNOWS",
            "attributes": {"source_name": "Alice", "target_name": "Bob"},
        }
    ]
    lines = symbolic_lines_distillation_batch("x", "AnchorX", batch, {"x": "WrongX", "y": "WrongY"})
    assert len(lines) == 1
    assert "Alice" in lines[0] and "Bob" in lines[0]
    assert "[anchor=AnchorX]" in lines[0]


def test_build_concept_map_full_records():
    concepts = [{"id": "c1", "name": "N1", "type": "t", "attributes": {"k": 1}}]
    m = build_concept_map(concepts)
    assert m["c1"]["name"] == "N1"
