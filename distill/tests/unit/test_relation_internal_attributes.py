# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from distill.app.services.distillation_job import _relation_internal_attributes


def test_relation_internal_attributes_sets_owner_and_status():
    out = _relation_internal_attributes(None, owner="mas-1", status="distilled")
    assert out == [
        {"owner": "mas-1", "attributes": {"status": "distilled"}},
    ]


def test_relation_internal_attributes_replaces_same_owner_keeps_others():
    existing = [
        {"owner": "other-mas", "attributes": {"distill_status": "pending"}},
        {"owner": "mas-1", "attributes": {"distill_status": "old"}},
    ]
    out = _relation_internal_attributes(existing, owner="mas-1", status="synthesized")
    assert len(out) == 2
    assert out[0] == {
        "owner": "other-mas",
        "attributes": {"distill_status": "pending"},
    }
    assert out[1] == {
        "owner": "mas-1",
        "attributes": {"status": "synthesized"},
    }


def test_relation_internal_attributes_accepts_flat_existing_entry():
    existing = [{"owner": "other-mas", "distill_status": "pending"}]
    out = _relation_internal_attributes(existing, owner="mas-1", status="distilled")
    assert out[0] == {
        "owner": "other-mas",
        "attributes": {"distill_status": "pending"},
    }
    assert out[1]["attributes"]["status"] == "distilled"


def test_relation_internal_attributes_merges_status_into_existing_owner_attrs():
    existing = [
        {
            "owner": "mas-1",
            "attributes": {
                "rate": 19.5,
                "category": "Technology1",
                "session_time": 1672531207,
                "distill_status": "",
            },
        },
    ]
    out = _relation_internal_attributes(existing, owner="mas-1", status="distilled")
    assert len(out) == 1
    assert out[0] == {
        "owner": "mas-1",
        "attributes": {
            "rate": 19.5,
            "category": "Technology1",
            "session_time": 1672531207,
            "status": "distilled",
        },
    }
