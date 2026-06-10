# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared distillation helpers (graph bucketing, chunking, symbolic batch text)."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def bucket_relations_incident_to_anchors(
    anchor_ids: List[str], relations: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """§8.3.2: each anchor gets every undistilled relation where anchor_id is an endpoint."""
    anchor_set = set(anchor_ids)
    buckets: Dict[str, List[Dict[str, Any]]] = {a: [] for a in anchor_ids}
    for r in relations or []:
        ids = r.get("node_ids") or []
        if len(ids) < 2:
            continue
        u, v = str(ids[0]), str(ids[1])
        if u in anchor_set:
            buckets[u].append(r)
        if v in anchor_set and u != v:
            buckets[v].append(r)
    return buckets


def chunk_lists(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        return [list(items)]
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_concept_map(concepts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """§8.3.1: concept_id -> full concept dict."""
    out: Dict[str, Dict[str, Any]] = {}
    for c in concepts or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", "")).strip()
        if not cid:
            continue
        out[cid] = c
    return out


def concept_display_name(concept: Dict[str, Any], concept_id: str) -> str:
    n = str(concept.get("name", "") or "").strip()
    return n or concept_id


def relation_attributes(rel: Dict[str, Any]) -> Dict[str, Any]:
    raw = rel.get("attributes")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _relation_type_segment(rel: Dict[str, Any]) -> str:
    rtype = str(rel.get("relationship") or "RELATED_TO").strip() or "RELATED_TO"
    attrs = relation_attributes(rel)
    st = attrs.get("session_time")
    if st is not None and str(st).strip():
        return f"{rtype} (mentioned at: {st})"
    return rtype


def symbolic_lines_distillation_batch(
    anchor_id: str,
    anchor_name: str,
    batch: List[Dict[str, Any]],
    id_to_display: Dict[str, str],
) -> List[str]:
    """
    §8.3.4: one line per relation; anchor name from map; endpoints from source_name/target_name
    when present, else fall back to id_to_display for graph-true wording.
    """
    lines: List[str] = []
    for rel in batch:
        ids = rel.get("node_ids") or []
        if len(ids) < 2:
            continue
        u, v = str(ids[0]), str(ids[1])
        if anchor_id not in (u, v):
            continue
        attrs = relation_attributes(rel)
        src = str(attrs.get("source_name") or "").strip() or id_to_display.get(u, u)
        tgt = str(attrs.get("target_name") or "").strip() or id_to_display.get(v, v)
        seg = _relation_type_segment(rel)
        lines.append(f"[anchor={anchor_name}] {src} -{seg}-> {tgt}")
    return lines


def id_to_name_from_concept_map(concept_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {cid: concept_display_name(c, cid) for cid, c in concept_map.items()}
