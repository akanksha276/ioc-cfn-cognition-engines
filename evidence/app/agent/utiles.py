# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def coerce_graph_node_ids(value: Any) -> List[str]:
    """
    Normalize graph edge endpoints to a list of string ids.
    Handles JSON arrays and double-encoded JSON strings from some graph adapters.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [s]
    return [str(value).strip()] if str(value).strip() else []


def select_by_relative_top(
    index_to_score: Dict[int, float],
    relative_gap: float = 0.25,
    max_k: Optional[int] = None,
) -> List[int]:
    """
    Select indices based on a relative gap to the top score.
    Keeps the highest-scoring index and any other indices whose score is at least
    (1 - relative_gap) * top_score. Example: relative_gap=0.25 keeps scores >= 0.75 * top.
    """
    if not index_to_score:
        return []
    sorted_items: List[Tuple[int, float]] = sorted(index_to_score.items(), key=lambda kv: (-kv[1], kv[0]))
    top_idx, top_score = sorted_items[0]
    threshold = (1.0 - float(relative_gap)) * float(top_score)
    selected: List[int] = [top_idx]
    for idx, score in sorted_items[1:]:
        if score >= threshold:
            selected.append(idx)
        else:
            break
    if max_k is not None and max_k > 0:
        return selected[:max_k]
    return selected


class PathFormatter:
    @staticmethod
    def _concept_label(meta: Dict[str, Any]) -> str:
        cid = meta.get("concept_id") or meta.get("id") or ""
        name = (meta.get("name") or "").strip()
        description = (meta.get("description") or "").strip()
        label = name or (description[:60] + ("..." if len(description) > 60 else "")) or cid
        return f"{{{cid}: {label}}}" if cid else f"{{{label}}}"

    @staticmethod
    def _relation_label(rel: Dict[str, Any]) -> str:
        r = rel.get("relationship") or rel.get("relation") or ""
        r = str(r).strip()
        return "{" + f"-> {r} ->" + "}"

    def to_natural_language(self, paths: List[List[Dict[str, Any]]]) -> List[str]:
        out: List[str] = []
        for path in paths:
            parts: List[str] = []
            for seg in path:
                if seg.get("kind") == "concept":
                    parts.append(self._concept_label(seg["value"]))
                elif seg.get("kind") == "relation":
                    parts.append(self._relation_label(seg["value"]))
            out.append(" - ".join(parts))
        return out

    def to_symbolic_paths(self, paths: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Render each path as compact symbolic hops while preserving path identity.
        If a relation has attributes.session_time (from graph DB), that edge is shown as
        " -REL (mentioned at: <session_time>)-> "; otherwise " -REL-> ".
        Example: source -DELEGATES_TO (mentioned at: 2025-03-07T10:00:00Z)-> target
        """

        def _c(meta: Dict[str, Any]) -> str:
            name_raw = str((meta or {}).get("name") or "").strip()
            if name_raw:
                return name_raw.replace(" ", "_")[:64]
            desc_raw = str((meta or {}).get("description") or "").strip()
            if desc_raw:
                # Use a short, readable fallback from description; no IDs included
                return (desc_raw[:64] + ("..." if len(desc_raw) > 64 else "")).replace(" ", "_")
            return "unknown"

        def _r(rel: Dict[str, Any]) -> str:
            r = str((rel or {}).get("relationship") or (rel or {}).get("relation") or "").strip()
            r = r.upper().replace(" ", "_")
            return r or "RELATED_TO"

        def _rel_segment(rel: Dict[str, Any]) -> str:
            """Relation label for a hop; appends temporal and summarized context from attributes when present."""
            label = _r(rel)
            attrs = (rel or {}).get("attributes")
            if not isinstance(attrs, dict):
                return label
            parts: List[str] = []
            st = attrs.get("session_time")
            if st:
                parts.append(f"mentioned at: {st}")
            sc = (attrs.get("summarized_context") or "").strip()
            if sc:
                parts.append(f"context: {sc}")
            if parts:
                return f"{label} ({', '.join(parts)})"
            return label

        out: List[str] = []
        for p_idx, path in enumerate(paths or []):
            hops: List[str] = []
            for i in range(0, max(0, len(path) - 2), 2):
                a = path[i]
                b = path[i + 1]
                c = path[i + 2]
                if a.get("kind") == "concept" and b.get("kind") == "relation" and c.get("kind") == "concept":
                    rel_val = b.get("value") or {}
                    hops.append(f"{_c(a.get('value') or {})} -{_rel_segment(rel_val)}-> {_c(c.get('value') or {})}")
            out.append(f"[path_{p_idx}] " + (" ; ".join(hops) if hops else "<empty>"))
        return out


class GraphSession:
    """
    Maintains a lightweight graph built from concepts/relations returned by the repository.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, Any]] = []
        self.adjacency: Dict[str, List[int]] = {}

    def _ensure_node(self, meta: Dict[str, Any]):
        cid = meta.get("concept_id") or meta.get("id")
        if not cid:
            return
        if cid not in self.nodes:
            self.nodes[cid] = meta
        if cid not in self.adjacency:
            self.adjacency[cid] = []

    def _add_relation(self, relation: Dict[str, Any]) -> int:
        # Normalize minimal shape
        rel = {
            "id": relation.get("id"),
            "node_ids": coerce_graph_node_ids(relation.get("node_ids")),
            "relationship": relation.get("relationship") or relation.get("relation"),
            "attributes": relation.get("attributes"),
        }
        self.relations.append(rel)
        return len(self.relations) - 1

    def ingest_enriched_results(self, enriched: List[Dict[str, Any]]):
        """
        enriched item:
        {
            "distance": float,
            "concept": {concept_id, name, description, ...},
            "relations": [ { id, node_ids, relationship, attributes }, ... ],
            "neighbor_concepts": [ {concept_id, name, ...}, ... ]
        }
        """
        added_nodes = 0
        added_rels = 0
        for item in enriched:
            meta = item.get("concept") or {}
            self._ensure_node(meta)
            added_nodes += 1
            for nmeta in item.get("neighbor_concepts") or []:
                self._ensure_node(nmeta)
                added_nodes += 1
            # Register relations and adjacency
            for rel in item.get("relations") or []:
                idx = self._add_relation(rel)
                added_rels += 1
                for nid in coerce_graph_node_ids(rel.get("node_ids")):
                    if nid not in self.adjacency:
                        self.adjacency[nid] = []
                    self.adjacency[nid].append(idx)

    def _neighbors_for(self, concept_id: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        For a concept_id, return list of (relation, neighbor_meta) pairs.
        """
        out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for rel_idx in self.adjacency.get(concept_id, []):
            if 0 <= rel_idx < len(self.relations):
                rel = self.relations[rel_idx]
                for nid in rel.get("node_ids", []):
                    if nid and nid != concept_id:
                        nmeta = self.nodes.get(nid)
                        if nmeta:
                            out.append((rel, nmeta))
        return out

    def build_paths_from(self, start_id: Optional[str], hop: int) -> List[List[Dict[str, Any]]]:
        """
        Build simple paths starting at start_id with exactly `hop` edges.
        Structured path format:
        [
            {"kind":"concept","value":meta},
            {"kind":"relation","value":rel},
            {"kind":"concept","value":meta},
            ...
        ]
        """
        if not start_id or start_id not in self.nodes:
            return []

        paths: List[List[Dict[str, Any]]] = []
        start_meta = self.nodes[start_id]

        def dfs(curr_id: str, depth: int, visited: Set[str], acc: List[Dict[str, Any]]):
            if depth == hop:
                paths.append(list(acc))
                return
            for rel, nei in self._neighbors_for(curr_id):
                nid = nei.get("concept_id")
                if not nid:
                    nid = nei.get("id")
                if not nid or nid in visited:
                    continue
                # extend path with relation and neighbor concept
                acc.append({"kind": "relation", "value": rel})
                acc.append({"kind": "concept", "value": nei})
                visited.add(nid)
                dfs(nid, depth + 1, visited, acc)
                visited.remove(nid)
                acc.pop()
                acc.pop()

        dfs(
            start_id,
            0,
            {start_id},
            [{"kind": "concept", "value": start_meta}],
        )
        return paths

    def add_relations_and_nodes(self, relations: List[Dict[str, Any]], neighbor_concepts: List[Dict[str, Any]]):
        """
        Add new relations and any provided neighbor concepts into the session.
        Ensures adjacency is updated so subsequent hops can traverse newly added edges.
        """
        for meta in neighbor_concepts or []:
            self._ensure_node(meta or {})
        for rel in relations or []:
            idx = self._add_relation(rel or {})
            for nid in coerce_graph_node_ids((rel or {}).get("node_ids")):
                if nid not in self.adjacency:
                    self.adjacency[nid] = []
                self.adjacency[nid].append(idx)


# -------- Embedding helpers and MMR selection (reusable) --------
def generate_text_embedding(embedding_manager, text: str) -> np.ndarray:
    """
    Generate a single embedding vector for the given text using the provided embedding_manager.
    Handles chunking/averaging consistent with other code paths.
    """
    chunks = embedding_manager.preprocess_text(text or "")
    vecs = embedding_manager.generate_embeddings(chunks)
    if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
        return np.mean(np.array(vecs, dtype=np.float32), axis=0)
    return np.array(vecs, dtype=np.float32)


def normalize_l2(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized vector (no-op for zero vector)."""
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else v


def mmr_select_indices(
    scores: Dict[int, float],
    candidate_texts: List[str],
    query_text: str,
    embedding_manager,
    k: int,
    alpha: float = 0.7,
    lam: float = 0.7,
) -> List[int]:
    """
    Select k indices using Maximal Marginal Relevance.
    - scores: index->LLM score (arbitrary scale)
    - candidate_texts: textual representations to embed
    - query_text: question/intent
    - embedding_manager: to compute embeddings
    - alpha: blend LLM score vs query cosine in relevance
    - lam: relevance vs diversity tradeoff
    """
    k = max(1, int(k))
    n = len(candidate_texts)
    if n == 0:
        return []
    idx_list = list(range(n))

    # Embed query and candidates
    q_vec = normalize_l2(generate_text_embedding(embedding_manager, query_text))
    d_vecs = []
    for s in candidate_texts:
        try:
            d_vecs.append(normalize_l2(generate_text_embedding(embedding_manager, s)))
        except Exception:
            d_vecs.append(np.zeros_like(q_vec, dtype=np.float32))

    # Normalize scores to [0,1]
    if not scores:
        scores = {i: 0.5 for i in idx_list}
    sc_vals = np.array([float(scores.get(i, 0.0)) for i in idx_list], dtype=np.float32)
    sc_min, sc_max = float(sc_vals.min()), float(sc_vals.max())
    if sc_max > sc_min:
        sc_norm = (sc_vals - sc_min) / (sc_max - sc_min)
    else:
        sc_norm = np.ones_like(sc_vals, dtype=np.float32) * 0.5

    # Relevance = alpha*LLM + (1-alpha)*cos(query, doc)
    rel = alpha * sc_norm + (1.0 - alpha) * np.array([float(np.dot(q_vec, dv)) for dv in d_vecs], dtype=np.float32)

    # Greedy MMR
    selected: List[int] = []
    pool: List[int] = idx_list.copy()
    while len(selected) < k and pool:
        best_i = None
        best_score = -1e9
        for i in pool:
            if not selected:
                mmr_i = float(rel[i])
            else:
                max_sim = max(float(np.dot(d_vecs[i], d_vecs[j])) for j in selected)
                mmr_i = lam * float(rel[i]) - (1.0 - lam) * max_sim
            if mmr_i > best_score:
                best_score = mmr_i
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        pool.remove(best_i)
    return selected
