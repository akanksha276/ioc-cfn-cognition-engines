# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

from ..api.schemas import KnowledgeRecord, ReasonerCognitionRequest
from .embeddings import EmbeddingManager
from .llm_clients import EvidenceJudge, EvidenceRanker, ResponseGenerator, get_llm_call_count
from .utiles import (
    GraphSession,
    PathFormatter,
    coerce_graph_node_ids,
    mmr_select_indices,
    select_by_relative_top,
)

load_dotenv(find_dotenv())


class SingleEntityConfig:
    def __init__(
        self,
        top_k_similar: int = 2,
        select_k_per_hop: int = 3,
        max_depth: int = 3,
        llm_temperature: float = 0.1,
    ):
        self.top_k_similar = top_k_similar
        self.select_k_per_hop = select_k_per_hop
        self.max_depth = max_depth
        self.llm_temperature = llm_temperature


class ConceptRepository:
    """
    Adapter around the graph repository.

    Similarity search is resolved through the concepts similarity-search API
    exposed by the same HTTP repository used for graph traversal.
    """

    def __init__(self, repo, request_id: str):
        self.repo = repo
        self.request_id = request_id

    async def similar_with_neighbors_async(
        self, query_vec: List[List[float]], k: int, entity_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Top-k similar concepts with one-hop neighbours from API similarity + repo.neighbors(concept_id)."""
        if not hasattr(self.repo, "search_similar_concepts"):
            logger.info(
                "[SingleEntity][Similar] Similarity API client unavailable; returning empty anchors."
            )
            return []

        embedded_text = str(entity_text or "").strip()
        if not embedded_text:
            embedded_text = ""
        cache_results = await self.repo.search_similar_concepts(
            embedded_text=embedded_text,
            embedding_vector=query_vec[0] if len(query_vec) == 1 else query_vec,
            request_id=self.request_id,
            top_k=k,
            search_metrics="l2",
        )

        if not cache_results:
            logger.info("[SingleEntity][SimilarityAPI] similarity-search returned 0 results.")
            return []
        logger.info(
            "[SingleEntity][SimilarityAPI] similarity-search returned %d results (entity_text=%r).",
            len(cache_results),
            entity_text,
        )
        # Require graph concept_id on each hit.
        concept_ids_ordered: List[str] = []
        score_by_id: Dict[str, float] = {}
        description_by_id: Dict[str, str] = {}
        name_by_id: Dict[str, str] = {}
        for r in cache_results:
            concept_id = str((r or {}).get("concept_id", "")).strip()
            if not concept_id:
                logger.info("[SingleEntity][SimilarityAPI] Skip row without concept_id.")
                continue
            if concept_id in score_by_id:
                continue
            name = str((r or {}).get("concept_name", "")).strip()
            display_name = name or concept_id
            concept_ids_ordered.append(concept_id)
            score_by_id[concept_id] = float((r or {}).get("score", 0.0))
            description_by_id[concept_id] = ""
            name_by_id[concept_id] = display_name
        if not concept_ids_ordered:
            logger.info("[SingleEntity][SimilarityAPI] No rows with a non-empty concept_id.")
            return []
        out: List[Dict[str, Any]] = []
        for concept_id in concept_ids_ordered[:k]:
            neighbors_result = await self.repo.neighbors(concept_id)
            records = (neighbors_result or {}).get("records") or []
            if not records:
                logger.info(
                    "[SingleEntity][SimilarityAPI] Graph returned no records for concept_id=%r.",
                    concept_id,
                )
                continue
            rec = records[0]
            concept = rec.get("node") or {}
            resolved_id = str(concept_id or concept.get("id", "") or "").strip()

            anchor_row: Optional[Dict[str, Any]] = None
            for c in rec.get("concepts") or []:
                if isinstance(c, dict) and str(c.get("id", "")).strip() == resolved_id:
                    anchor_row = c
                    break

            def _concept_type_from(m: Optional[Dict[str, Any]]) -> str:
                if not m:
                    return ""
                t = m.get("type")
                if isinstance(t, str) and t.strip():
                    return t.strip()
                attrs = m.get("attributes")
                if isinstance(attrs, dict):
                    ct = attrs.get("concept_type")
                    if ct is not None and str(ct).strip():
                        return str(ct).strip()
                return ""

            anchor_name = ""
            if anchor_row:
                anchor_name = str(anchor_row.get("name") or "").strip()
            if not anchor_name:
                anchor_name = str(concept.get("name") or "").strip()
            if not anchor_name:
                anchor_name = name_by_id.get(concept_id, concept_id)

            anchor_desc = ""
            if anchor_row:
                anchor_desc = str(anchor_row.get("description") or "").strip()
            if not anchor_desc:
                anchor_desc = str(concept.get("description") or "").strip()
            if not anchor_desc:
                anchor_desc = description_by_id.get(concept_id, "")

            anchor_type = _concept_type_from(anchor_row) or _concept_type_from(concept) or "concept"

            relations = []
            for rel in rec.get("relationships") or []:
                if not rel:
                    continue
                label = rel.get("relationship") or rel.get("relation")
                relations.append(
                    {
                        "id": rel.get("id"),
                        "node_ids": coerce_graph_node_ids(rel.get("node_ids")),
                        "relationship": label,
                        "attributes": rel.get("attributes"),
                    }
                )
            # Neighbor payloads: legacy key "neighbors" or KG-style "concepts" on the ego record
            _neighbor_rows: List[Any] = list(rec.get("neighbors") or []) + list(rec.get("concepts") or [])
            _seen_neighbor_ids: Set[str] = set()
            neighbor_concepts: List[Dict[str, Any]] = []
            for n in _neighbor_rows:
                if not isinstance(n, dict) or not n.get("id"):
                    continue
                nid = str(n.get("id")).strip()
                if not nid or nid == resolved_id or nid in _seen_neighbor_ids:
                    continue
                _seen_neighbor_ids.add(nid)
                neighbor_concepts.append(n)
            out.append({
                "distance": score_by_id.get(concept_id, 0.0),
                "concept": {
                    "id": resolved_id,
                    "name": anchor_name,
                    "description": anchor_desc,
                    "type": anchor_type,
                },
                "relations": relations,
                "neighbor_concepts": neighbor_concepts,
            })
        return out

    async def relations_for_async(self, concept_id: str):
        # Normalize neighbors result into relation entries with node_ids/relationship
        result = await self.repo.neighbors(concept_id)
        rels: List[Dict[str, Any]] = []
        for rec in (result or {}).get("records", []) or []:
            for rel in rec.get("relationships", []) or []:
                rels.append(
                    {
                        "id": rel.get("id"),
                        "node_ids": coerce_graph_node_ids(rel.get("node_ids")),
                        "relationship": rel.get("relationship") or rel.get("relation"),
                        "attributes": rel.get("attributes"),
                    }
                )
        return rels

    async def concepts_by_ids_async(self, concept_ids: List[str]):
        return await self.repo.get_concepts_by_ids(concept_ids)


@dataclass
class LaneState:
    anchor_id: str
    anchor_name: str
    graph: GraphSession
    selected_structured: List[List[Dict[str, Any]]] = field(default_factory=list)
    frontier_paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    seen_path_keys: Set[Tuple[Any, ...]] = field(default_factory=set)
    last_candidates_structured: Optional[List[List[Dict[str, Any]]]] = None
    last_candidates_symbolic: Optional[List[str]] = None
    last_reason: Optional[str] = None
    sufficient: bool = False


def path_key(path: List[Dict[str, Any]]) -> Tuple:
    """
    Stable identity for a path based on segment kinds and ids/relationships.
    """
    out: List[Any] = []
    for seg in path:
        if seg.get("kind") == "concept":
            out.append((seg["kind"], (seg.get("value") or {}).get("id")))
        elif seg.get("kind") == "relation":
            out.append((seg["kind"], (seg.get("value") or {}).get("relationship")))
    return tuple(out)


def last_edge(
    path: List[Dict[str, Any]],
    name_fn,
    rel_fn,
) -> Optional[Dict[str, str]]:
    """
    Return the last edge (from, relation, to) in a structured path, or None.
    """
    for i in range(len(path) - 2, -1, -1):
        if path[i].get("kind") == "relation" and i - 1 >= 0 and i + 1 < len(path):
            prev_c = path[i - 1].get("value") or {}
            next_c = path[i + 1].get("value") or {}
            return {
                "from": name_fn(prev_c),
                "relation": rel_fn(path[i].get("value") or {}),
                "to": name_fn(next_c),
            }
    return None


def _expand_paths_one_hop(paths: List[List[Dict[str, Any]]], graph: GraphSession) -> List[List[Dict[str, Any]]]:
    """
    Expand each structured path by one hop from its tail concept (simple BFS frontier expansion).
    """
    next_paths: List[List[Dict[str, Any]]] = []
    for path in paths or []:
        if not path or path[-1].get("kind") != "concept":
            continue
        tail_meta = path[-1]["value"] or {}
        tail_id = tail_meta.get("id")
        if not tail_id:
            continue
        for rel, nei in graph._neighbors_for(tail_id):
            nid = (nei or {}).get("id")
            if not nid:
                continue
            visited_ids = {seg["value"].get("id") for seg in path if seg.get("kind") == "concept"}
            if nid in visited_ids:
                continue
            extended = list(path)
            extended.append({"kind": "relation", "value": rel})
            extended.append({"kind": "concept", "value": nei})
            next_paths.append(extended)
    return next_paths


class SingleEntityEvidenceEngine:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        repo: ConceptRepository,
        path_formatter: PathFormatter,
        judge: EvidenceJudge,
        ranker: EvidenceRanker,
        config: Optional[SingleEntityConfig] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.embedding_manager = embedding_manager
        self.repo = repo
        self.path_formatter = path_formatter
        self.judge = judge
        self.ranker = ranker
        self.config = config or SingleEntityConfig()
        self.response_generator = response_generator or ResponseGenerator(temperature=0.2)

    def _entity_to_query_vec(self, entity: Dict[str, Any]) -> List[float]:
        text = f"{entity.get('description') or ''}{entity.get('name') or ''}"
        chunks = self.embedding_manager.preprocess_text(text)
        vectors = self.embedding_manager.generate_embeddings(chunks)
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 2:
            vec = np.mean(arr, axis=0).tolist()
        else:
            vec = arr.flatten().tolist()
        return vec

    async def gather(
        self,
        request: ReasonerCognitionRequest,
        entity: Dict[str, Any],
        extra_context: Optional[str] = None,
        skip_final_response: bool = False,
    ) -> KnowledgeRecord:
        # Snapshot LLM call count to compute per-gather delta
        llm_calls_before = get_llm_call_count()
        query_vec = self._entity_to_query_vec(entity)
        logger.info("[SingleEntity] Starting similar concept retrieval for entity: %s", entity.get("name") or "(unnamed)")
        entity_name = (entity.get("name") or "").strip()
        enriched = await self.repo.similar_with_neighbors_async(
            query_vec, k=self.config.top_k_similar, entity_text=entity_name or None
        )
        logger.info("[SingleEntity] Retrieved %d similar concepts.", len(enriched))

        # Initialize trace structure
        def _name_for(meta: Dict[str, Any]) -> str:
            n = (meta or {}).get("name")
            return (n.strip() if isinstance(n, str) else "") or (meta or {}).get("id") or "unknown"

        def _rel_label(rel: Dict[str, Any]) -> str:
            return str((rel or {}).get("relationship") or (rel or {}).get("relation") or "").strip() or "related_to"

        trace: Dict[str, Any] = {
            "extracted_entity": _name_for(entity),
            "tope_similar_concepts": [],
            "iterations": [],
            "lanes_count": 0,
            "sufficient": False,
            "winning": None,
            "request_decomposition": ((getattr(request, "meta", {}) or {}) or {}).get("request_decomposition"),
            "pass_on_context": (extra_context or ""),
        }
        # Populate top_similar using enriched relations and neighbor concepts
        try:
            for item in enriched or []:
                concept = (item or {}).get("concept") or {}
                anchor_id = concept.get("id")
                anchor_name = _name_for(concept)
                neighbor_list: List[Dict[str, Any]] = []
                # Map id->name from neighbor_concepts
                id_to_name: Dict[str, str] = {
                    (nc or {}).get("id"): _name_for(nc) for nc in ((item or {}).get("neighbor_concepts") or [])
                }
                # Also include anchor itself
                if anchor_id and anchor_name:
                    id_to_name[anchor_id] = anchor_name
                seen_edge: Set[Tuple[str, str, str]] = set()
                for rel in (item or {}).get("relations") or []:
                    for nid in coerce_graph_node_ids((rel or {}).get("node_ids")):
                        if not anchor_id or not nid or nid == anchor_id:
                            continue
                        tup = (anchor_id, nid, _rel_label(rel))
                        if tup in seen_edge:
                            continue
                        seen_edge.add(tup)
                        neighbor_list.append({"to": id_to_name.get(nid) or nid, "relation": _rel_label(rel)})
                trace["tope_similar_concepts"].append({"anchor_concept": anchor_name, "firs_neighbour": neighbor_list})
        except Exception:
            pass
        # Build a separate GraphSession per similar concept (anchor)
        lanes: List[LaneState] = []
        for idx, item in enumerate(enriched or []):
            concept = (item or {}).get("concept") or {}
            anchor_id = concept.get("id")
            if not anchor_id:
                continue
            g = GraphSession()
            g.ingest_enriched_results([item])
            lanes.append(LaneState(anchor_id=anchor_id, anchor_name=_name_for(concept), graph=g))
        if not lanes:
            logger.info("[SingleEntity] No anchors available; returning empty evidence.")
            content = {
                "evidence": {
                    "entity": entity,
                    "status": "insufficient",
                    "summary": {"supporting_paths": 0, "unique_concepts": 0},
                    "paths": [],
                    "context_paths_for_next": [],
                    "metadata": {
                        "retrieval_mode": "single_entity",
                        "pruning_applied": True,
                        "llm_assisted": True,
                    },
                },
                "trace": trace,
            }
            if not skip_final_response:
                content["evidence"]["final_response"] = "Insufficient Evidence"
            try:
                trace["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)
            except Exception:
                trace["llm_calls"] = 0
            return KnowledgeRecord(type="json", content=content)

        trace["lanes_count"] = len(lanes)
        winning_lane_index: Optional[int] = None

        # Hop-wise parallel exploration across lanes
        for hop in range(1, self.config.max_depth + 1):

            # First, try sufficiency selection on each lane (async per-lane), restricting candidates
            # to only the paths that start from the previously selected outer nodes (frontier).
            async def select_lane(idx: int, lane: LaneState):
                anchor_id = lane.anchor_id
                graph = lane.graph
                if hop == 1:
                    candidates_structured = graph.build_paths_from(anchor_id, hop=1)
                else:
                    frontier_paths = lane.frontier_paths or []
                    candidates_structured = _expand_paths_one_hop(frontier_paths, graph) if frontier_paths else []
                # Use compact symbolic representation for LLM selection (preserve path identity, low tokens)
                candidates_symbolic = self.path_formatter.to_symbolic_paths(candidates_structured)
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                logger.info(
                    "[SingleEntity] Invoking judge | hop=%s | lane=%s | anchor=%r | candidates=%d",
                    hop,
                    idx,
                    lane.anchor_name,
                    len(candidates_symbolic),
                )
                chosen_idx, sufficient, reason = await self.judge.async_select_paths_and_check_sufficiency(
                    question=question_text,
                    candidate_paths=candidates_symbolic,
                    select_k=self.config.select_k_per_hop,
                )
                logger.info(
                    "[SingleEntity] Judge returned | hop=%s | lane=%s | selected=%s | sufficient=%s | reason=%r",
                    hop,
                    idx,
                    chosen_idx,
                    sufficient,
                    reason,
                )
                return idx, candidates_structured, candidates_symbolic, chosen_idx, sufficient, reason

            # Create tasks explicitly so we can cancel the rest if any lane is sufficient
            selection_tasks = [asyncio.create_task(select_lane(i, lane)) for i, lane in enumerate(lanes)]
            for fut in asyncio.as_completed(selection_tasks):
                try:
                    lane_idx, candidates_structured, candidates_symbolic, chosen_idx, sufficient, reason = await fut
                except asyncio.CancelledError:
                    continue
                lane = lanes[lane_idx]
                # Persist the candidates for ranking stage to avoid recomputation
                lane.last_candidates_structured = candidates_structured
                lane.last_candidates_symbolic = candidates_symbolic
                lane.last_reason = reason
                try:
                    _ = chosen_idx, candidates_symbolic  # for trace/debug if needed
                except Exception as e:
                    logger.warning("[SingleEntity][select-log] %s", e)
                # Append to trace for this hop (record the outermost edge of each selected path)
                try:
                    # Prepare judge reason only (do not record judge-selected paths)
                    reason_text = reason
                    try:
                        # keep original reason text as-is
                        _ = reason_text  # no-op to satisfy linter in try
                    except Exception:
                        pass
                    # Do not add a judge-only iteration entry; ranker entry will include the judge reason
                except Exception as e:
                    logger.warning("[SingleEntity][trace-select] %s", e)
                # Initialize per-lane seen set for deduplication
                if lane.seen_path_keys is None:
                    lane.seen_path_keys = set()
                # When judge says sufficient: use judge's selected paths as final; no rank/expand after.
                if sufficient and winning_lane_index is None:
                    lane.sufficient = True
                    winning_lane_index = lane_idx
                    trace["sufficient"] = True
                    trace["winning"] = {"anchor_concept": lane.anchor_name, "reason_for_sufficiency": reason_text}
                    struct = lane.last_candidates_structured or []
                    lane.selected_structured = [struct[i] for i in chosen_idx if 0 <= i < len(struct)]
                    for t in selection_tasks:
                        if not t.done():
                            t.cancel()
                    for t in selection_tasks:
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                    break

            # If none sufficient, rank candidates per-lane and apply relative top-25% filtering
            async def rank_and_expand_lane(idx: int, lane: LaneState):
                anchor_id = lane.anchor_id
                graph = lane.graph
                # Use the same candidate set as selection stage; if missing, compute with the same restriction
                candidates_structured = lane.last_candidates_structured
                candidates_symbolic = lane.last_candidates_symbolic
                if candidates_structured is None or candidates_symbolic is None:
                    if hop == 1:
                        candidates_structured = graph.build_paths_from(anchor_id, hop=1)
                    else:
                        frontier_paths = lane.frontier_paths or []
                        candidates_structured = _expand_paths_one_hop(frontier_paths, graph) if frontier_paths else []
                    candidates_symbolic = self.path_formatter.to_symbolic_paths(candidates_structured)
                if not candidates_structured:
                    return
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                scores = await self.ranker.async_rank_paths(question=question_text, candidate_paths_repr=candidates_symbolic)
                # MMR-based selection to promote diversity among similarly scored paths
                try:
                    chosen_idx = mmr_select_indices(
                        scores=scores,
                        candidate_texts=candidates_symbolic,
                        query_text=request.payload.intent or "",
                        embedding_manager=self.embedding_manager,
                        k=self.config.select_k_per_hop,
                        alpha=0.7,
                        lam=0.7,
                    )
                except Exception:
                    chosen_idx = select_by_relative_top(
                        scores, relative_gap=0.25, max_k=self.config.select_k_per_hop
                    )
                # Append selected (ranking stage) to trace as well (outermost edge)
                try:
                    # Carry forward the judge reason from the selection stage for this lane if available.
                    carried_reason = (lane.last_reason or "").strip() or "No judge reason provided"
                    # Build detailed scored candidates: index, score, and path string
                    scored_detail = []
                    try:
                        for k_i, v_sc in (scores or {}).items():
                            try:
                                ii = int(k_i)
                                if 0 <= ii < len(candidates_symbolic):
                                    scored_detail.append(
                                        {"index": ii, "score": float(v_sc), "path": candidates_symbolic[ii]}
                                    )
                            except Exception:
                                continue
                    except Exception:
                        scored_detail = []
                    hop_entry = {
                        "iteration": hop,
                        "anchor_concept": lane.anchor_name,
                        "selected": [],
                        # For transparency: raw LLM rank scores and final MMR-chosen indices
                        "ranker_scored_paths": scored_detail,
                        "mmr_selected_indices": list(chosen_idx),
                        "selected_paths_symbolic": [
                            candidates_symbolic[i] for i in chosen_idx if 0 <= i < len(candidates_symbolic)
                        ],
                        "judge_reason_for_sufficiency": carried_reason,
                    }
                    seen_hop_edges: Set[Tuple[str, str, str]] = set()
                    for i in chosen_idx:
                        if 0 <= i < len(candidates_structured):
                            path = candidates_structured[i]
                            edge = last_edge(path, _name_for, _rel_label)
                            if edge:
                                key = (edge["from"], edge["to"], edge["relation"])
                                if key not in seen_hop_edges:
                                    seen_hop_edges.add(key)
                                    hop_entry["selected"].append(edge)
                    if hop_entry["selected"]:
                        trace["iterations"].append(hop_entry)
                except Exception as e:
                    logger.warning("[SingleEntity][trace-rank] %s", e)
                # Accumulate selections and prepare expansion frontier
                outer_node_ids: Set[str] = set()
                for i in chosen_idx:
                    if 0 <= i < len(candidates_structured):
                        path = candidates_structured[i]
                        k = path_key(path)
                        if k not in lane.seen_path_keys:
                            lane.seen_path_keys.add(k)
                            lane.selected_structured.append(path)
                        if path and path[-1].get("kind") == "concept":
                            last_meta = path[-1].get("value") or {}
                            oid = last_meta.get("id")
                            if oid:
                                outer_node_ids.add(oid)
                # Update lane frontier for next hop: selected paths from this hop
                lane.frontier_paths = [
                    candidates_structured[i] for i in chosen_idx if 0 <= i < len(candidates_structured)
                ]
                if not outer_node_ids:
                    return
                # Fetch relations for each outer node concurrently
                rel_tasks = [self.repo.relations_for_async(oid) for oid in outer_node_ids]
                rel_results = await asyncio.gather(*rel_tasks, return_exceptions=True)
                all_relations: List[Dict[str, Any]] = []
                needed_concept_ids: Set[str] = set()
                for rels in rel_results:
                    if isinstance(rels, Exception):
                        continue
                    for rel in rels or []:
                        all_relations.append(rel)
                        for nid in coerce_graph_node_ids(rel.get("node_ids")):
                            if nid and nid not in graph.nodes:
                                needed_concept_ids.add(nid)
                neighbor_metas: List[Dict[str, Any]] = []
                if needed_concept_ids:
                    try:
                        neighbor_metas = await self.repo.concepts_by_ids_async(list(needed_concept_ids))
                    except Exception:
                        neighbor_metas = []
                graph.add_relations_and_nodes(all_relations, neighbor_metas)

            if winning_lane_index is not None:
                break
            await asyncio.gather(*(rank_and_expand_lane(i, lane) for i, lane in enumerate(lanes)))

        if winning_lane_index is not None:
            wl = lanes[winning_lane_index]
            selected_structured = wl.selected_structured
            sufficient = True
        else:
            selected_structured = []
            sufficient = False

        evidence_paths_str: List[str] = []
        global_concept_ids: Set[str] = set()
        details_concepts: Dict[str, Dict[str, Any]] = {}
        details_relations: List[Dict[str, Any]] = []
        for path in selected_structured or []:
            try:
                symbolic = self.path_formatter.to_symbolic_paths([path])[0]
            except Exception:
                symbolic = ""
            evidence_paths_str.append(symbolic)
            for seg in path:
                if seg.get("kind") == "concept":
                    c = seg.get("value") or {}
                    cid = c.get("id")
                    if cid:
                        global_concept_ids.add(cid)
                        if cid not in details_concepts:
                            details_concepts[cid] = {
                                "concept_id": cid,
                                "name": _name_for(c),
                                "description": c.get("description"),
                                "type": c.get("type"),
                                "attributes": c.get("attributes"),
                            }
                elif seg.get("kind") == "relation":
                    r = seg.get("value") or {}
                    details_relations.append(
                        {
                            "id": r.get("id"),
                            "relationship": r.get("relationship") or r.get("relation"),
                            "node_ids": list(r.get("node_ids") or []),
                            "attributes": r.get("attributes"),
                        }
                    )

        context_paths_for_next: List[str] = list(evidence_paths_str)
        if not context_paths_for_next and not sufficient:
            for lane in lanes:
                for path in lane.selected_structured or []:
                    try:
                        s = self.path_formatter.to_symbolic_paths([path])[0]
                        if s:
                            context_paths_for_next.append(s)
                    except Exception:
                        pass

        # When insufficient and no paths selected, include top similar concept with its relations and first neighbors
        if not evidence_paths_str:
            try:
                top = (enriched or [None])[0] or {}
                top_concept = top.get("concept") or {}
                neighbor_concepts = top.get("neighbor_concepts") or []
                relations = top.get("relations") or []
                # Add concept + neighbors to details
                for c in [top_concept, *neighbor_concepts]:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id")
                    if not cid:
                        continue
                    global_concept_ids.add(cid)
                    if cid not in details_concepts:
                        details_concepts[cid] = {
                            "concept_id": cid,
                            "name": _name_for(c),
                            "description": c.get("description"),
                            "type": c.get("type"),
                            "attributes": c.get("attributes"),
                        }
                # Add relations as-is (normalized)
                for r in relations:
                    if not isinstance(r, dict):
                        continue
                    details_relations.append(
                        {
                            "id": r.get("id"),
                            "relationship": r.get("relationship") or r.get("relation"),
                            "node_ids": coerce_graph_node_ids(r.get("node_ids")),
                            "attributes": r.get("attributes"),
                        }
                    )
            except Exception:
                pass

        final_response: Optional[str] = None
        if not skip_final_response:
            if sufficient:
                verdict = (trace.get("winning") or {}).get("reason_for_sufficiency") or ""
            else:
                verdict = ""
            intent = (request.payload.intent or "").strip()
            try:
                final_response = await self.response_generator.async_generate_final_response(
                    intent, evidence_paths_str, verdict
                )
            except Exception:
                final_response = verdict or "Insufficient Evidence"

        content = {
            "evidence": {
                "entity": entity,
                "status": "sufficient" if sufficient else "insufficient",
                "summary": {
                    "supporting_paths": len(evidence_paths_str),
                    "unique_concepts": len(global_concept_ids),
                },
                "paths": evidence_paths_str,
                "context_paths_for_next": context_paths_for_next,
                "details": {
                    "concepts": list(details_concepts.values()),
                    "relations": details_relations,
                },
                "metadata": {
                    "retrieval_mode": "single_entity",
                    "pruning_applied": True,
                    "llm_assisted": True,
                },
            },
            "trace": trace,
        }
        if final_response is not None:
            content["evidence"]["final_response"] = final_response
        # Add per-gather LLM call count to the trace
        try:
            trace["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)
        except Exception:
            trace["llm_calls"] = 0
        # Print the trace to terminal for visibility
        try:
            pass

        except Exception:
            pass
        return KnowledgeRecord(type="json", content=content)
