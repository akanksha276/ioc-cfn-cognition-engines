# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Async distillation run: graph read → batches → CFN mutation POST → completion callback."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from evidence.app.api.schemas import Header

from distill.app.agent.distill_core import (
    bucket_relations_incident_to_anchors,
    build_concept_map,
    chunk_lists,
    concept_display_name,
    id_to_name_from_concept_map,
    relation_attributes,
    symbolic_lines_distillation_batch,
)
from distill.app.agent.embeddings_bge import embed_text_bge_small
from distill.app.agent.llm_distill import run_distillation_llm
from distill.app.agent.rag_retrieval import retrieve_rag_top_k
from distill.app.config.settings import settings
from distill.app.data.graph_client import CoDiGraphClient, graph_update_post_url
from distill.app.services import distillation_lock

logger = logging.getLogger(__name__)
_POLICY_PATH = Path(__file__).resolve().parent.parent / "agent" / "distillation_policy.md"


def _load_policy() -> str:
    try:
        return _POLICY_PATH.read_text(encoding="utf-8")
    except Exception:
        return "(distillation_policy.md missing)"


def _graph_safe_single_line(text: str, *, max_len: int = 512) -> str:
    """AgensGraph JSON properties reject raw newlines (0x0a); collapse whitespace for graph writes."""
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return ""
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def _sanitize_graph_string_attributes(attrs: Dict[str, Any], *, max_len: int = 4096) -> Dict[str, Any]:
    """Ensure string attribute values are safe for AgensGraph JSON property encoding."""
    out: Dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if isinstance(value, str):
            out[key] = _graph_safe_single_line(value, max_len=max_len)
        else:
            out[key] = value
    return out


def _codin_display_name(anchor_name: str, codin_id: str) -> str:
    base = (anchor_name or "anchor").strip() or "anchor"
    short_id = (codin_id or "")[:8]
    return _graph_safe_single_line(f"CoDiN-{base}-{short_id}", max_len=256)


def _relation_internal_attributes(
    existing: Any,
    *,
    owner: str,
    distill_status: str,
) -> List[Dict[str, Any]]:
    """
    Build ``internal_attributes`` for graph/update (CFN/KM nested shape).

    Each entry: ``{"owner": "<uuid>", "attributes": {..., "distill_status": "..."}}``.
    Preserves other owners unchanged; merges ``distill_status`` into this owner's
    existing nested ``attributes`` when present.
    """
    owner_key = (owner or "").strip()
    out: List[Dict[str, Any]] = []
    owner_attrs: Dict[str, Any] = {}
    if isinstance(existing, list):
        for item in existing:
            if not isinstance(item, dict):
                continue
            item_owner = str(item.get("owner") or "").strip()
            if not item_owner:
                continue
            nested = item.get("attributes")
            if isinstance(nested, dict):
                item_attrs = dict(nested)
            else:
                item_attrs = {k: v for k, v in item.items() if k != "owner"}
            if item_owner == owner_key:
                owner_attrs = item_attrs
                continue
            if item_attrs:
                out.append({"owner": item_owner, "attributes": item_attrs})
    owner_attrs["distill_status"] = distill_status
    out.append({"owner": owner_key, "attributes": owner_attrs})
    return out


async def _retrieve_rag(
    rag_layer: Any,
    batch_text: str,
    top_k: int,
    *,
    request_id: str,
    embedding_vector: Optional[List[float]] = None,
) -> str:
    try:
        hits = await retrieve_rag_top_k(
            rag_layer,
            batch_text,
            int(top_k),
            request_id,
            embedding_vector=embedding_vector or [],
            timeout_seconds=float(settings.CODI_RAG_TIMEOUT_SEC),
        )
    except Exception as rag_exc:
        logger.warning("[CoDi distill] RAG retrieval failed: %s", rag_exc)
        return ""
    return "\n".join(str(h.get("display_line") or h.get("text") or "") for h in (hits or []))


async def _distill_one_batch_payload(
    *,
    anchor_id: str,
    anchor_name: str,
    mas_id: str,
    batch: List[Dict[str, Any]],
    batch_index: int,
    id_to_display: Dict[str, str],
    concept_map: Dict[str, Dict[str, Any]],
    policy: str,
    rag_layer: Any,
    dist_mode: str,
    request_id: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (new_concept_dict, list of relation payloads [edge patches..., new anchor→CoDiN], batch_debug).
    Relations match CFN ``/graph/update``: ``id``, ``node_ids``, ``relationship``, ``attributes``,
    and ``internal_attributes`` (``distill_status`` + ``owner`` for distillation/read filters).
    """
    symbolic = symbolic_lines_distillation_batch(anchor_id, anchor_name, batch, id_to_display)
    batch_text = "\n".join(symbolic) if symbolic else f"{anchor_name} (no symbolic paths in batch)"
    logger.info(
        "[CoDi distill] batch | anchor=%s idx=%d lines=%d",
        anchor_id,
        batch_index,
        len(symbolic),
    )

    emb_vec: List[float] = []
    try:
        emb_vec = await embed_text_bge_small(batch_text)
    except Exception as emb_exc:
        logger.warning("[CoDi distill] embedding failed anchor=%s batch=%s: %s", anchor_id, batch_index, emb_exc)

    rag_block = await _retrieve_rag(
        rag_layer,
        batch_text,
        settings.CODI_RAG_TOP_K,
        request_id=request_id,
        embedding_vector=emb_vec or [],
    )
    user_prompt = (
        f"Distillation mode: {dist_mode}\n"
        f"Anchor name: {anchor_name}\n"
        f"Symbolic one-hop paths (batch):\n{batch_text}\n\n"
        f"Retrieved RAG lines:\n{rag_block}\n"
    )
    resolved_policy = policy.replace("{entity_name}", anchor_name)
    system_prompt = (
        "You distill graph context for downstream memory. Follow the policy below.\n\n"
        f"{resolved_policy}"
    )
    distill = await asyncio.to_thread(
        run_distillation_llm,
        system_prompt,
        user_prompt,
    )

    codin_id = str(uuid.uuid4())
    new_rel_id = str(uuid.uuid4())
    distilled = (distill.distilled_description or "").strip() or "(empty distillation)"
    summarized = (distill.summarized_context or "").strip() or "distilled_batch"
    # AgensGraph rejects unescaped newlines in JSON property values (knowledge-memory-svc).
    description_text = _graph_safe_single_line(distilled, max_len=8192)
    summarized_safe = _graph_safe_single_line(summarized, max_len=4096)

    new_concept: Dict[str, Any] = {
        "id": codin_id,
        "name": _codin_display_name(anchor_name, codin_id),
        "description": description_text,
        "type": "CoDiN",
        "attributes": {
            "embedding": list(emb_vec) if emb_vec else [],
            "distill_mode": _graph_safe_single_line(dist_mode, max_len=64),
        },
    }

    relation_mutations: List[Dict[str, Any]] = []
    for rel in batch:
        rid = str(rel.get("id") or "").strip()
        ids = rel.get("node_ids") or []
        if len(ids) < 2 or not rid:
            continue
        u, v = str(ids[0]), str(ids[1])
        attrs = _sanitize_graph_string_attributes(relation_attributes(rel))
        attrs.pop("distill_status", None)
        relation_mutations.append(
            {
                "id": rid,
                "node_ids": [u, v],
                "relationship": str(rel.get("relationship") or "RELATED_TO").strip() or "RELATED_TO",
                "attributes": attrs,
                "internal_attributes": _relation_internal_attributes(
                    rel.get("internal_attributes"),
                    owner=mas_id,
                    distill_status="updated",
                ),
            }
        )

    relation_mutations.append(
        {
            "id": new_rel_id,
            "node_ids": [anchor_id, codin_id],
            "relationship": "summary",
            "attributes": _sanitize_graph_string_attributes(
                {
                    "source_name": anchor_name,
                    "target_name": "CoDiN",
                    "summarized_context": summarized_safe,
                }
            ),
            "internal_attributes": _relation_internal_attributes(
                None,
                owner=mas_id,
                distill_status="CoDi",
            ),
        }
    )

    debug = {"anchor_id": anchor_id, "batch_index": batch_index, "batch_size": len(batch)}
    return new_concept, relation_mutations, debug


async def _post_json(client: httpx.AsyncClient, url: str, payload: Dict[str, Any], timeout: float) -> httpx.Response:
    return await client.post(url, json=payload, timeout=timeout)


async def _put_json(client: httpx.AsyncClient, url: str, payload: Dict[str, Any], timeout: float) -> httpx.Response:
    return await client.put(url, json=payload, timeout=timeout)


def _normalize_relation_dict(rel: Dict[str, Any]) -> Dict[str, Any]:
    """Align memory-service ``relation`` with CoDi/CFN ``relationship`` on edge dicts."""
    out = dict(rel)
    rel_label = out.get("relationship")
    if rel_label is None or (isinstance(rel_label, str) and not rel_label.strip()):
        alt = out.get("relation")
        if alt is not None and str(alt).strip():
            out["relationship"] = str(alt).strip()
    return out


def _coerce_distillation_read_lists(raw: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build ``concepts`` / ``relations`` lists for distillation.

    - If ``records`` is a non-empty list (knowledge-memory ``KnowledgeGraphQueryResponse`` shape),
      aggregate ``concepts`` and ``relationships`` from each record (dedupe by id).
    - Otherwise use top-level ``concepts`` / ``relations`` (legacy distillation/read shape).
    """
    records = raw.get("records")
    if isinstance(records, list) and len(records) > 0:
        concepts_acc: List[Dict[str, Any]] = []
        relations_acc: List[Dict[str, Any]] = []
        seen_cids: set[str] = set()
        seen_rids: set[str] = set()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for c in rec.get("concepts") or []:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id", "")).strip()
                if not cid or cid in seen_cids:
                    continue
                seen_cids.add(cid)
                concepts_acc.append(c)
            rel_items = list(rec.get("relationships") or [])
            if not rel_items:
                rel_items = list(rec.get("relations") or [])
            for rel in rel_items:
                if not isinstance(rel, dict):
                    continue
                rid = str(rel.get("id", "")).strip()
                if rid and rid in seen_rids:
                    continue
                if rid:
                    seen_rids.add(rid)
                relations_acc.append(_normalize_relation_dict(rel))
        return concepts_acc, relations_acc

    concepts_raw = raw.get("concepts") or []
    relations_raw = raw.get("relations") or []
    if not isinstance(concepts_raw, list):
        concepts_raw = []
    if not isinstance(relations_raw, list):
        relations_raw = []
    concepts_out = [c for c in concepts_raw if isinstance(c, dict)]
    relations_out = [_normalize_relation_dict(r) for r in relations_raw if isinstance(r, dict)]
    return concepts_out, relations_out


def _relations_cnt_gte_filter() -> int:
    """Read at call time so .env changes apply without re-importing Settings."""
    raw = os.getenv("CODI_MIN_EDGES")
    if raw is not None and str(raw).strip():
        return int(raw)
    return int(settings.CODI_MIN_EDGES)


def _distillation_read_request_body(header: Header, request_id: str) -> Dict[str, Any]:
    """Build CFN ``POST .../graph/distillation/read`` body (filters contract)."""
    mas_id = (header.mas_id or "").strip()
    relations_cnt = _relations_cnt_gte_filter()
    body: Dict[str, Any] = {
        "request_id": request_id,
        "filters": {
            "relations_cnt_gte": relations_cnt,
            "distill_status": settings.CODI_DISTILL_STATUS_FILTER,
            "owner": mas_id,
            "return_missing_distill_status": settings.CODI_RETURN_MISSING_DISTILL_STATUS,
        },
    }
    header_payload = header.model_dump(exclude_none=True)
    if header_payload:
        body["header"] = header_payload
    return body


async def fetch_distillation_graph_read(header: Header, request_id: str) -> Dict[str, Any]:
    """
    POST distillation read to the data layer (same contract as the former in-job fetch).

    Called from the HTTP route so the graph payload is obtained before ``202 Accepted`` and
    passed into :func:`execute_distillation_run` as ``prefetched_graph_read``.
    """
    base = (settings.DATA_LAYER_BASE_URL or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("DATA_LAYER_BASE_URL / MOCKED_DB_BASE_URL is not configured.")
    graph = CoDiGraphClient(
        base,
        workspace_id=header.workspace_id,
        mas_id=header.mas_id,
        agent_id=header.agent_id,
        timeout=120.0,
        distillation_read_path=settings.CODI_GRAPH_DISTILLATION_READ_SEGMENT,
    )
    try:
        return await graph.distillation_graph_read(_distillation_read_request_body(header, request_id))
    finally:
        await graph.aclose()


async def execute_distillation_run(
    *,
    header: Header,
    request_id: str,
    callback_url: str,
    operation_id: str,
    distill_run_at: str,
    rag_layer: Any = None,
    prefetched_graph_read: Optional[Dict[str, Any]] = None,
) -> None:
    """
    §6–§15: full async job. Always releases the (workspace_id, mas_id) lock in ``finally``.

    When ``prefetched_graph_read`` is set (normal path from the route), the data-layer graph
    read is skipped. Otherwise the job performs the read itself (tests / direct callers).
    """
    wid = header.workspace_id
    mid = header.mas_id
    agent_id = (header.agent_id or "").strip() or None
    base = (settings.DATA_LAYER_BASE_URL or "").strip().rstrip("/")
    mutation_url = (settings.CFN_MUTATION_APPLY_URL or "").strip() or graph_update_post_url(
        base,
        wid,
        mid,
        settings.CODI_GRAPH_UPDATE_SEGMENT,
    )
    try:
        if not base:
            raise RuntimeError("DATA_LAYER_BASE_URL / MOCKED_DB_BASE_URL is not configured.")
        if not mutation_url:
            raise RuntimeError("Graph mutation URL could not be resolved (check workspace_id / mas_id).")

        if prefetched_graph_read is not None:
            raw = prefetched_graph_read if isinstance(prefetched_graph_read, dict) else {}
        else:
            graph = CoDiGraphClient(
                base,
                workspace_id=wid,
                mas_id=mid,
                agent_id=header.agent_id,
                timeout=120.0,
                distillation_read_path=settings.CODI_GRAPH_DISTILLATION_READ_SEGMENT,
            )
            try:
                raw = await graph.distillation_graph_read(
                    _distillation_read_request_body(header, request_id)
                )
            finally:
                await graph.aclose()

        concepts_raw, relations_raw = _coerce_distillation_read_lists(raw if isinstance(raw, dict) else {})

        concept_map = build_concept_map(concepts_raw)
        anchor_ids = list(concept_map.keys())
        id_to_display = id_to_name_from_concept_map(concept_map)

        mutation_concepts: List[Dict[str, Any]] = []
        mutation_relations: List[Dict[str, Any]] = []
        added_nodes = 0
        added_anchor_links = 0
        updated_edges = 0

        if not anchor_ids:
            records_n = len(raw.get("records") or []) if isinstance(raw, dict) else 0
            logger.info(
                "[CoDi distill] no anchors from graph read | request_id=%s "
                "parsed_concepts=%d parsed_relations=%d records=%d "
                "filters.relations_cnt_gte=%d distill_status=%r owner=%s "
                "(if manual curl used relations_cnt_gte=1, set CODI_MIN_EDGES=1 in .env)",
                request_id,
                len(concepts_raw),
                len(relations_raw),
                records_n,
                _relations_cnt_gte_filter(),
                settings.CODI_DISTILL_STATUS_FILTER,
                mid,
            )
        else:
            buckets = bucket_relations_incident_to_anchors(anchor_ids, relations_raw)
            policy = _load_policy()
            dist_mode = settings.CODI_DIST_MODE
            processed_relation_ids: set[str] = set()

            for anchor_id in sorted(buckets.keys()):
                rels = [r for r in (buckets.get(anchor_id) or []) if isinstance(r, dict)]
                rels = [r for r in rels if str(r.get("id", "")).strip() and str(r.get("id")) not in processed_relation_ids]
                if not rels:
                    continue
                random.shuffle(rels)
                anchor_name = concept_display_name(concept_map.get(anchor_id, {}), anchor_id)
                batches = chunk_lists(rels, int(settings.CODI_MAX_RELATIONS_PER_BATCH))
                for bi, batch in enumerate(batches):
                    if not batch:
                        continue
                    new_c, rel_ops, _ = await _distill_one_batch_payload(
                        anchor_id=anchor_id,
                        anchor_name=anchor_name,
                        mas_id=mid,
                        batch=batch,
                        batch_index=bi,
                        id_to_display=id_to_display,
                        concept_map=concept_map,
                        policy=policy,
                        rag_layer=rag_layer,
                        dist_mode=dist_mode,
                        request_id=request_id,
                    )
                    mutation_concepts.append(new_c)
                    mutation_relations.extend(rel_ops)
                    added_nodes += 1
                    added_anchor_links += 1
                    updated_edges += len(batch)
                    for r in batch:
                        rid = str(r.get("id", "")).strip()
                        if rid:
                            processed_relation_ids.add(rid)

        metadata = {
            "added_distilled_nodes": added_nodes,
            "added_distilled_relations": added_anchor_links,
            "updated_relations": updated_edges,
            "operation_id": operation_id,
            "distill_run_at": distill_run_at,
            "distill_mode": settings.CODI_DIST_MODE,
        }

        mutation_body: Dict[str, Any] = {
            "request_id": request_id,
            "concepts": mutation_concepts,
            "relations": mutation_relations,
            "descriptor": settings.CODI_GRAPH_UPDATE_DESCRIPTOR,
            "metadata": metadata,
        }
        if agent_id:
            mutation_body["header"] = {"agent_id": agent_id}

        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(timeout=timeout) as http:
            if not mutation_concepts and not mutation_relations:
                ok_body = {
                    "status": "successful",
                    "operation_id": operation_id,
                    "response_id": request_id,
                    "workspace_id": wid,
                    "mas_id": mid,
                    "distill_run_at": distill_run_at,
                    "distill_mode": settings.CODI_DIST_MODE,
                    "meta": metadata,
                    "message": "no anchors or relations to distill",
                }
                if agent_id:
                    ok_body["agent_id"] = agent_id
                await _send_callback(http, callback_url, ok_body)
                return

            try:
                mut_resp = await _put_json(http, mutation_url, mutation_body, 120.0)
            except Exception as post_exc:
                logger.exception("[CoDi distill] mutation POST failed | op=%s", operation_id)
                await _send_callback(
                    http,
                    callback_url,
                    {
                        "status": "unsuccessful",
                        "operation_id": operation_id,
                        "response_id": request_id,
                        "workspace_id": wid,
                        "mas_id": mid,
                        "reason_code": "MUTATION_POST_FAILED",
                        "message": str(post_exc),
                        **({"agent_id": agent_id} if agent_id else {}),
                    },
                )
                return

            if mut_resp.status_code >= 300:
                logger.error(
                    "[CoDi distill] mutation non-2xx | op=%s status=%s body=%s",
                    operation_id,
                    mut_resp.status_code,
                    (mut_resp.text or "")[:2000],
                )
                await _send_callback(
                    http,
                    callback_url,
                    {
                        "status": "unsuccessful",
                        "operation_id": operation_id,
                        "response_id": request_id,
                        "workspace_id": wid,
                        "mas_id": mid,
                        "reason_code": "MUTATION_REJECTED",
                        "message": f"HTTP {mut_resp.status_code}",
                        **({"agent_id": agent_id} if agent_id else {}),
                    },
                )
                return

            ok_body: Dict[str, Any] = {
                "status": "successful",
                "operation_id": operation_id,
                "response_id": request_id,
                "workspace_id": wid,
                "mas_id": mid,
                "distill_run_at": distill_run_at,
                "distill_mode": settings.CODI_DIST_MODE,
                "meta": {
                    "added_distilled_nodes": added_nodes,
                    "added_distilled_relations": added_anchor_links,
                    "updated_relations": updated_edges,
                    "distill_mode": settings.CODI_DIST_MODE,
                },
            }
            if agent_id:
                ok_body["agent_id"] = agent_id
            await _send_callback(http, callback_url, ok_body)
            logger.info(
                "[CoDi distill] complete | op=%s workspace=%s mas=%s "
                "codin_nodes=%d anchor_links=%d updated_edges=%d",
                operation_id, wid, mid, added_nodes, added_anchor_links, updated_edges,
            )
    except Exception as run_exc:
        logger.exception("[CoDi distill] run failed | op=%s", operation_id)
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                await _send_callback(
                    http,
                    callback_url,
                    {
                        "status": "unsuccessful",
                        "operation_id": operation_id,
                        "response_id": request_id,
                        "workspace_id": wid,
                        "mas_id": mid,
                        "reason_code": "RUN_FAILED",
                        "message": str(run_exc),
                        **({"agent_id": agent_id} if agent_id else {}),
                    },
                )
        except Exception:
            logger.exception("[CoDi distill] failure callback also failed | op=%s", operation_id)
    finally:
        await distillation_lock.end_run(wid, mid)


async def _send_callback(http: httpx.AsyncClient, callback_url: str, payload: Dict[str, Any]) -> None:
    try:
        r = await http.post(callback_url, json=payload, timeout=60.0)
        if r.status_code >= 300:
            logger.warning("[CoDi distill] callback non-2xx | status=%s", r.status_code)
    except Exception:
        logger.exception("[CoDi distill] callback POST failed")
