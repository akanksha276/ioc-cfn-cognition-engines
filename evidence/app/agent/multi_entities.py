# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from ..api.schemas import KnowledgeRecord, ReasonerCognitionRequest
from .embeddings import EmbeddingManager
from .llm_clients import EvidenceJudge, EvidenceRanker, ResponseGenerator, get_llm_call_count
from .utiles import coerce_graph_node_ids, mmr_select_indices


def _name_for(meta: Dict[str, Any]) -> str:
    n = (meta or {}).get("name")
    return (n.strip() if isinstance(n, str) else "") or (meta or {}).get("id") or "unknown"


def _rel_label(rel: Dict[str, Any]) -> str:
    return str((rel or {}).get("relationship") or (rel or {}).get("relation") or "").strip() or "related_to"


def _hop_str(frn: str, ton: str, rel: str, attrs: Optional[Dict[str, Any]] = None) -> str:
    """Build one hop string; append temporal and summarized context from attrs when present."""
    if not attrs or not isinstance(attrs, dict):
        return f"{frn} -{rel}-> {ton}"
    parts: List[str] = []
    st = attrs.get("session_time")
    if st:
        parts.append(f"mentioned at: {st}")
    sc = (attrs.get("summarized_context") or "").strip()
    if sc:
        parts.append(f"context: {sc}")
    if parts:
        return f"{frn} -{rel} ({', '.join(parts)})-> {ton}"
    return f"{frn} -{rel}-> {ton}"


def _target_id_to_one_hop_neighbors(tgt_top: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, str]]]:
    """Build target_id -> [(relation_label, neighbor_name)] from enriched target candidates."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for item in tgt_top or []:
        top = item or {}
        concept = (top or {}).get("concept") or {}
        cid = concept.get("id")
        if not cid:
            continue
        id_to_name: Dict[str, str] = {cid: _name_for(concept)}
        for nc in (top or {}).get("neighbor_concepts") or []:
            nid = (nc or {}).get("id")
            if nid:
                id_to_name[nid] = _name_for(nc)
        seen_edge: Set[Tuple[str, str, str]] = set()
        neighbors: List[Tuple[str, str]] = []
        for rel in (top or {}).get("relations") or []:
            for nid in coerce_graph_node_ids((rel or {}).get("node_ids")):
                if not nid or nid == cid:
                    continue
                edge = (cid, nid, _rel_label(rel))
                if edge in seen_edge:
                    continue
                seen_edge.add(edge)
                neighbors.append((_rel_label(rel), id_to_name.get(nid) or nid))
        if neighbors:
            out[cid] = neighbors
    return out


class MultiEntityConfig:
    def __init__(
        self,
        top_k_candidates: int = 2,
        max_depth: int = 4,
        pre_rank_limit: int = 20,
        mmr_top_k: int = 5,
        concurrency_limit: int = 3,
        mmr_alpha: float = 0.7,
        mmr_lam: float = 0.7,
    ):
        self.top_k_candidates = top_k_candidates
        self.max_depth = max_depth
        self.pre_rank_limit = pre_rank_limit
        self.mmr_top_k = mmr_top_k
        self.concurrency_limit = concurrency_limit
        self.mmr_alpha = mmr_alpha
        self.mmr_lam = mmr_lam


@dataclass(frozen=True)
class Pair:
    source_id: str
    target_id: str
    source_name: str
    target_name: str


class MultiEntityEvidenceEngine:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        data_layer,
        judge: EvidenceJudge,
        ranker: EvidenceRanker,
        config: Optional[MultiEntityConfig] = None,
        concept_repo=None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.embedding_manager = embedding_manager
        self.data_layer = data_layer
        self.judge = judge
        self.ranker = ranker
        self.config = config or MultiEntityConfig()
        # When set, used for similar-concept retrieval (cache_layer + neighbors by id only); no repo.search_similar_with_neighbors
        self.concept_repo = concept_repo
        self.response_generator = response_generator or ResponseGenerator(temperature=0.2)

    def _entity_to_query_vec(self, entity: Dict[str, Any]) -> List[float]:
        text = f"{entity.get('description') or ''}{entity.get('name') or ''}"
        chunks = self.embedding_manager.preprocess_text(text)
        vectors = self.embedding_manager.generate_embeddings(chunks)
        if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
            vec = np.mean(np.array(vectors, dtype=np.float32), axis=0).tolist()
        else:
            vec = np.array(vectors, dtype=np.float32).tolist()
        return vec

    async def _top_k_candidates(self, entity: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        query_vec = self._entity_to_query_vec(entity)
        entity_name = (entity.get("name") or "").strip() or None
        if self.concept_repo:
            enriched = await self.concept_repo.similar_with_neighbors_async(
                query_vec, k, entity_text=entity_name
            )
        else:
            enriched = []
        out: List[Dict[str, Any]] = []
        for it in enriched or []:
            if (it or {}).get("concept", {}).get("id"):
                out.append(it)
            if len(out) >= k:
                break
        return out

    def _build_pairs(self, src: List[Dict[str, Any]], tgt: List[Dict[str, Any]]) -> List[Pair]:
        """Match multi_entities.py: top-1 source × all top-k targets, then top-1 target × all top-k sources."""
        pairs: List[Pair] = []
        seen: Set[Tuple[str, str]] = set()
        k = self.config.top_k_candidates
        src_slice = src[:k]
        tgt_slice = tgt[:k]

        def add_pair(s_id: str, t_id: str, s_name: str, t_name: str) -> None:
            key = (s_id, t_id)
            if key in seen:
                return
            seen.add(key)
            pairs.append(Pair(source_id=s_id, target_id=t_id, source_name=s_name, target_name=t_name))

        if src_slice:
            s = src_slice[0]
            sc = (s or {}).get("concept") or {}
            s_id = sc.get("id")
            s_name = _name_for(sc)
            if s_id:
                for t in tgt_slice:
                    tc = (t or {}).get("concept") or {}
                    t_id = tc.get("id")
                    t_name = _name_for(tc)
                    if t_id:
                        add_pair(s_id, t_id, s_name, t_name)

        if tgt_slice:
            t = tgt_slice[0]
            tc = (t or {}).get("concept") or {}
            t_id = tc.get("id")
            t_name = _name_for(tc)
            if t_id:
                for s in src_slice:
                    sc = (s or {}).get("concept") or {}
                    s_id = sc.get("id")
                    s_name = _name_for(sc)
                    if s_id:
                        add_pair(s_id, t_id, s_name, t_name)

        return pairs

    async def _fetch_paths_between(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """POST graph/paths on the data layer (async find_paths); returns paths list or []."""
        try:
            j = await self.data_layer.find_paths(
                source_id,
                target_id,
                self.config.max_depth,
                self.config.pre_rank_limit,
                relations=None,
            )
            paths = (j or {}).get("paths") or []
            if not paths:
                logger.info(
                    "[MultiEntity] Paths API returned no paths: source_id=%r, target_id=%r, status=%s",
                    source_id,
                    target_id,
                    (j or {}).get("status"),
                )
            return paths
        except Exception as e:
            logger.error("[MultiEntity] Paths API error: source_id=%r, target_id=%r, error=%s", source_id, target_id, e)
            return []

    async def gather(
        self,
        request: ReasonerCognitionRequest,
        entities: Dict[str, Any],
        extra_context: Optional[str] = None,
        skip_final_response: bool = False,
    ) -> KnowledgeRecord:
        # Extract names from entities mapping
        e1 = {"name": entities.get("source") or ""}
        e2 = {"name": entities.get("target") or ""}
        llm_calls_before = get_llm_call_count()

        # Top-k for both entities
        k = self.config.top_k_candidates
        src_top, tgt_top = await asyncio.gather(self._top_k_candidates(e1, k), self._top_k_candidates(e2, k))

        # Trace: top similar concepts for each entity
        def _similar_trace(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            out_neighbors: List[Dict[str, str]] = []
            try:
                top = (items or [None])[0] or {}
                concept = (top or {}).get("concept") or {}
                anchor_name = _name_for(concept)
                seen_edge: Set[Tuple[str, str, str]] = set()
                id_to_name: Dict[str, str] = { (concept or {}).get("id"): anchor_name }
                for nc in (top or {}).get("neighbor_concepts") or []:
                    cid = (nc or {}).get("id")
                    if cid:
                        id_to_name[cid] = _name_for(nc)
                for rel in (top or {}).get("relations") or []:
                    for nid in coerce_graph_node_ids((rel or {}).get("node_ids")):
                        if not nid or nid == (concept or {}).get("id"):
                            continue
                        edge = ( (concept or {}).get("id"), nid, _rel_label(rel) )
                        if edge in seen_edge:
                            continue
                        seen_edge.add(edge)
                        out_neighbors.append({"to": id_to_name.get(nid) or nid, "relation": _rel_label(rel)})
                return {"anchor_concept": anchor_name, "firs_neighbour": out_neighbors}
            except Exception:
                return {"anchor_concept": _name_for((items or [{}])[0].get("concept") if items else {}), "firs_neighbour": []}

        trace: Dict[str, Any] = {
            "extracted_entities": [e1.get("name"), e2.get("name")],
            "tope_similar_concepts": [_similar_trace(src_top), _similar_trace(tgt_top)],
            "iterations": [],
            "lanes_count": 0,
            "sufficient": False,
            "winning": None,
            "request_decomposition": ((getattr(request, "meta", {}) or {}) or {}).get("request_decomposition"),
            "pass_on_context": (extra_context or ""),
        }

        pairs = self._build_pairs(src_top, tgt_top)
        trace["lanes_count"] = len(pairs)
        target_id_to_neighbors = _target_id_to_one_hop_neighbors(tgt_top)

        # Per-pair pipeline
        sem = asyncio.Semaphore(self.config.concurrency_limit)

        async def process_pair(pair: Pair):
            async with sem:
                # Step 1: fetch paths from data-logic (by id — copy API)
                paths = await self._fetch_paths_between(pair.source_id, pair.target_id)
                # Build name mapping for all involved ids
                concept_ids_local: Set[str] = set()
                for p in paths or []:
                    for ed in (p or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id")
                        tid = (ed or {}).get("to_id")
                        if fid:
                            concept_ids_local.add(fid)
                        if tid:
                            concept_ids_local.add(tid)
                id_to_name_local: Dict[str, str] = {}
                try:
                    metas_local = await self.data_layer.get_concepts_by_ids(list(concept_ids_local))
                    for c in metas_local or []:
                        cid = (c or {}).get("id")
                        if cid:
                            id_to_name_local[cid] = _name_for(c)
                except Exception:
                    pass
                target_neighbors: List[Tuple[str, str]] = target_id_to_neighbors.get(pair.target_id) or []
                candidates_symbolic: List[str] = []
                for p in paths or []:
                    hops: List[str] = []
                    for ed in (p or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id")
                        tid = (ed or {}).get("to_id")
                        rel = (ed or {}).get("relationship") or (ed or {}).get("relation")
                        frn = id_to_name_local.get(fid) or fid
                        ton = id_to_name_local.get(tid) or tid
                        if frn and ton and rel:
                            hops.append(_hop_str(frn, ton, rel, (ed or {}).get("attributes")))
                    base_str = " ; ".join(hops)
                    if not base_str:
                        continue
                    candidates_symbolic.append(base_str)
                    for rel_label, nei_name in target_neighbors:
                        candidates_symbolic.append(
                            base_str + " ; " + _hop_str(pair.target_name, nei_name, rel_label, None)
                        )
                candidates_symbolic = [s for s in candidates_symbolic if s]
                # Step 2: rank with LLM, then MMR
                if not candidates_symbolic:
                    return {
                        "pair": pair,
                        "paths": [],
                        "selected_indices": [],
                        "scores": {},
                        "candidates_symbolic": [],
                        "sufficient": False,
                        "reason": "no_candidate_paths",
                    }
                question_text = (request.payload.intent or "")
                if extra_context:
                    question_text = f"{question_text}\n\nPrior evidence:\n{extra_context}"
                scores = await self.ranker.async_rank_paths(question=question_text, candidate_paths_repr=candidates_symbolic)
                try:
                    chosen_idx = mmr_select_indices(
                        scores=scores,
                        candidate_texts=candidates_symbolic,
                        query_text=request.payload.intent or "",
                        embedding_manager=self.embedding_manager,
                        k=self.config.mmr_top_k,
                        alpha=self.config.mmr_alpha,
                        lam=self.config.mmr_lam,
                    )
                except Exception:
                    # simple fallback: top by score
                    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                    chosen_idx = [int(i) for i, _ in ordered[: self.config.mmr_top_k]]
                # Step 3: judge sufficiency on selected paths
                chosen_symbolic = [candidates_symbolic[i] for i in chosen_idx if 0 <= i < len(candidates_symbolic)]
                _, sufficient, reason = await self.judge.async_select_paths_and_check_sufficiency(
                    question=question_text, candidate_paths=chosen_symbolic, select_k=len(chosen_symbolic) or 1
                )
                return {
                    "pair": pair,
                    "paths": paths,
                    "selected_indices": chosen_idx,
                    "scores": scores,
                    "candidates_symbolic": candidates_symbolic,
                    "sufficient": bool(sufficient),
                    "reason": reason,
                }

        results = await asyncio.gather(*(process_pair(p) for p in pairs))

        # Decide winning lane
        winning = next((r for r in results if r.get("sufficient")), None)
        if winning:
            trace["sufficient"] = True
            trace["winning"] = {
                "source": winning["pair"].source_name,
                "target": winning["pair"].target_name,
                "reason_for_sufficiency": winning.get("reason"),
            }

        # Add iterations trace per pair
        for hop_idx, r in enumerate(results, start=1):
            pair: Pair = r["pair"]
            # Use the name-based candidates that were ranked/judged
            candidates_symbolic = list(r.get("candidates_symbolic") or [])
            # detail scored paths
            scored_detail = []
            try:
                for k_i, v_sc in (r.get("scores") or {}).items():
                    i = int(k_i)
                    if 0 <= i < len(candidates_symbolic):
                        scored_detail.append({"index": i, "score": float(v_sc), "path": candidates_symbolic[i]})
            except Exception:
                scored_detail = []
            trace["iterations"].append(
                {
                    "iteration": hop_idx,
                    "anchor_concept": f"{pair.source_name} -> {pair.target_name}",
                    "selected": [],
                    "ranker_scored_paths": scored_detail,
                    "mmr_selected_indices": list(r.get("selected_indices") or []),
                    "selected_paths_symbolic": [
                        candidates_symbolic[i] for i in (r.get("selected_indices") or []) if 0 <= i < len(candidates_symbolic)
                    ],
                    "judge_reason_for_sufficiency": r.get("reason"),
                    "sufficient": bool(r.get("sufficient")),
                }
            )

        # Build evidence based on winning or best-scored (fallback). Aligns with multi_entities.py shape.
        def build_evidence_from(
            r: Dict[str, Any],
            resolved_concepts: Optional[List[Dict[str, Any]]] = None,
        ) -> Tuple[List[str], Set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
            """Returns (evidence_paths_as_strings, concept_ids, details_relations, details_concepts)."""
            selected_indices: List[int] = list(r.get("selected_indices") or [])
            paths: List[Dict[str, Any]] = list(r.get("paths") or [])
            selected_paths: List[Dict[str, Any]] = [paths[i] for i in selected_indices if 0 <= i < len(paths)]
            evidence_paths_str: List[str] = []
            concept_ids: Set[str] = set()
            for p in selected_paths or []:
                for ed in p.get("edges") or []:
                    fid = (ed or {}).get("from_id")
                    tid = (ed or {}).get("to_id")
                    if fid:
                        concept_ids.add(fid)
                    if tid:
                        concept_ids.add(tid)
            details_concepts: List[Dict[str, Any]] = []
            id_to_name: Dict[str, str] = {}
            for c in resolved_concepts or []:
                cid = (c or {}).get("id")
                if not cid:
                    continue
                id_to_name[cid] = _name_for(c)
                details_concepts.append(
                    {
                        "concept_id": cid,
                        "name": _name_for(c),
                        "type": c.get("type"),
                        "description": c.get("description"),
                    }
                )
            details_relations: List[Dict[str, Any]] = []
            for p in selected_paths or []:
                hops: List[str] = []
                for ed in p.get("edges") or []:
                    fid = (ed or {}).get("from_id")
                    tid = (ed or {}).get("to_id")
                    rel = (ed or {}).get("relationship") or (ed or {}).get("relation")
                    frn = id_to_name.get(fid) or fid
                    ton = id_to_name.get(tid) or tid
                    if frn and ton and rel:
                        hops.append(_hop_str(frn, ton, rel, (ed or {}).get("attributes")))
                    details_relations.append(
                        {
                            "id": None,
                            "relationship": (ed or {}).get("relationship") or (ed or {}).get("relation"),
                            "node_ids": [fid, tid],
                            "attributes": (ed or {}).get("attributes"),
                        }
                    )
                evidence_paths_str.append(" ; ".join(hops))
            return evidence_paths_str, concept_ids, details_relations, details_concepts

        def _collect_concept_ids(r: Dict[str, Any]) -> Set[str]:
            ids: Set[str] = set()
            paths = list(r.get("paths") or [])
            selected_indices = list(r.get("selected_indices") or [])
            for i in selected_indices:
                if 0 <= i < len(paths):
                    for ed in (paths[i] or {}).get("edges") or []:
                        fid = (ed or {}).get("from_id")
                        tid = (ed or {}).get("to_id")
                        if fid:
                            ids.add(fid)
                        if tid:
                            ids.add(tid)
            return ids

        if winning:
            concept_ids_resolved = _collect_concept_ids(winning)
            metas_resolved = await self.data_layer.get_concepts_by_ids(list(concept_ids_resolved))
            evidence_paths_str, global_ids, details_relations, details_concepts = build_evidence_from(
                winning, resolved_concepts=metas_resolved
            )
            status = "sufficient"
            context_paths_for_next: List[str] = list(evidence_paths_str)
        else:
            candidate = max(results, key=lambda r: len(r.get("selected_indices") or []), default=None)
            if candidate and candidate.get("selected_indices"):
                concept_ids_resolved = _collect_concept_ids(candidate)
                metas_resolved = await self.data_layer.get_concepts_by_ids(list(concept_ids_resolved))
                evidence_paths_str, global_ids, details_relations, details_concepts = build_evidence_from(
                    candidate, resolved_concepts=metas_resolved
                )
                cand_sym = candidate.get("candidates_symbolic") or []
                sel_idx = candidate.get("selected_indices") or []
                context_paths_for_next = [cand_sym[i] for i in sel_idx if 0 <= i < len(cand_sym)]
            else:
                evidence_paths_str, global_ids, details_relations, details_concepts = [], set(), [], []
                context_paths_for_next = []
            status = "insufficient"
            trace["insufficient_verdict"] = (candidate.get("reason") or "").strip() if candidate else ""

        final_response: Optional[str] = None
        if not skip_final_response:
            if status == "sufficient":
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
                "entity": {
                    "source": {"name": e1.get("name")},
                    "target": {"name": e2.get("name")},
                },
                "status": status,
                "summary": {
                    "supporting_paths": len(evidence_paths_str),
                    "unique_concepts": len(global_ids),
                },
                "paths": evidence_paths_str,
                "context_paths_for_next": context_paths_for_next,
                "details": {
                    "concepts": details_concepts,
                    "relations": details_relations,
                },
                "metadata": {
                    "retrieval_mode": "multi_entity",
                    "pruning_applied": True,
                    "llm_assisted": True,
                },
            },
            "trace": trace,
        }
        if final_response is not None:
            content["evidence"]["final_response"] = final_response

        try:
            content["trace"]["llm_calls"] = max(0, get_llm_call_count() - llm_calls_before)  # type: ignore[index]
        except Exception:
            content["trace"]["llm_calls"] = 0  # type: ignore[index]

        return KnowledgeRecord(type="json", content=content)

