# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

# app/evidence/evidence.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from ..config.settings import settings
from .embeddings import EmbeddingManager
from .rag_retrieval import retrieve_rag_top_k
from ..api.schemas import (
    ReasonerCognitionRequest,
    ReasonerCognitionResponse,
    KnowledgeRecord,
    Header,
)
from .single_entity import (
    SingleEntityEvidenceEngine,
    SingleEntityConfig,
    ConceptRepository,
)
from .multi_entities import MultiEntityEvidenceEngine, MultiEntityConfig
from .utiles import PathFormatter
from .llm_clients import (
    EvidenceJudge,
    EvidenceRanker,
    ResponseGenerator,
    QueryDecomposer,
    EntityExtractor as LLMEntityExtractor,
)

embedding_manager = EmbeddingManager()

# Max prior path strings to pass as extra_context to the next sub-query (aligned with old_code/evidence.py).
_PRIOR_PATHS_CONTEXT_LIMIT = 8


async def extract_entities(request: ReasonerCognitionRequest) -> List[Dict]:
    """Thin async wrapper around the LLM entity extractor."""
    client = LLMEntityExtractor(temperature=0.2)
    return await client.extract_entities_from_request(request)


def _get_context_paths_for_next(rec: KnowledgeRecord) -> List[str]:
    """Paths to pass as context to the next sub-query (sufficient -> judge-selected; insufficient -> last ranked)."""
    evidence = (rec.content or {}).get("evidence") or {}
    paths = evidence.get("context_paths_for_next")
    if isinstance(paths, list):
        return [str(p) for p in paths if (p or "").strip()]
    return []


def _get_paths_strings(rec: KnowledgeRecord) -> List[str]:
    """Evidence paths as list of strings (may be stored as list of strings or list of dicts)."""
    evidence = (rec.content or {}).get("evidence") or {}
    paths = evidence.get("paths") or []
    out: List[str] = []
    for p in paths:
        if isinstance(p, str) and (p or "").strip():
            out.append(p.strip())
        elif isinstance(p, dict) and (p.get("symbolic") or "").strip():
            out.append(str(p["symbolic"]).strip())
    return out


def _verdict_from_record(rec: KnowledgeRecord) -> str:
    """Judge / ranker verdict text for unified final generation."""
    content = rec.content or {}
    evidence = content.get("evidence") or {}
    trace = content.get("trace") or {}
    if evidence.get("status") == "sufficient":
        w = trace.get("winning") or {}
        return (w.get("reason_for_sufficiency") or "").strip()
    return (trace.get("insufficient_verdict") or "").strip()


def _resolve_rag_params(request: ReasonerCognitionRequest) -> Tuple[int, float]:
    """Defaults from evidence settings; optional overrides on payload.metadata.rag."""
    top_k = settings.EVIDENCE_RAG_TOP_K
    timeout_sec = settings.EVIDENCE_RAG_TIMEOUT_SEC
    md = request.payload.metadata
    if md and getattr(md, "rag", None) is not None:
        rag = md.rag
        if rag is not None:
            if rag.top_k is not None:
                top_k = rag.top_k
            if rag.timeout_seconds is not None:
                timeout_sec = rag.timeout_seconds
    return top_k, timeout_sec


def _intent_query_vec(intent: str) -> List[float]:
    text = (intent or "").strip()
    if not text:
        return []
    try:
        chunks = embedding_manager.preprocess_text(text)
        vectors = embedding_manager.generate_embeddings(chunks)
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim == 2:
            return np.mean(arr, axis=0).tolist()
        return arr.flatten().tolist()
    except Exception:
        logger.warning("[Evidence] Failed to embed intent for RAG similarity search.")
        return []


async def process_evidence(
    request: ReasonerCognitionRequest,
    repo_adapter=None,
) -> ReasonerCognitionResponse:
    response_id = request.request_id

    response_header = Header(
        workspace_id=request.header.workspace_id,
        mas_id=request.header.mas_id,
        agent_id=request.header.agent_id,
    )

    logger.info("[Evidence] Starting entity extraction via LLM.")
    entities = await LLMEntityExtractor(temperature=0).extract_entities_from_request(request)
    logger.info("[Evidence] Extracted %d entities.", len(entities))
    if not entities:
        return ReasonerCognitionResponse(
            header=response_header,
            response_id=response_id,
            records=[],
            metadata={"source": "evidence.process_evidence", "note": "no_entities"},
        )

    try:
        ent_names = [
            str(e.get("name")).strip()
            for e in (entities or [])
            if isinstance(e, dict) and e.get("name")
        ]
    except Exception:
        ent_names = []
    n_entities = len(ent_names)
    intent = (request.payload.intent or "").strip()

    # Decomposition only when 3+ entities; 1 → single-entity, 2 → multi-entity directly
    decomposition: List[Dict[str, Any]] = []
    if n_entities >= 3:
        try:
            decomposer = QueryDecomposer()
            decomposition = await decomposer.async_decompose(intent, ent_names)
        except Exception:
            decomposition = []

    if n_entities == 1:
        items = [{"index": 1, "sentence": intent, "entities": ent_names}]
        mode = "single_entity"
    elif n_entities == 2:
        items = [{"index": 1, "sentence": intent, "entities": ent_names}]
        mode = "multi_entity"
    else:
        # 3+ entities: use decomposition or fallback to first two
        items = (
            decomposition
            if decomposition
            else [{"index": 1, "sentence": intent, "entities": ent_names[:2]}]
        )
        mode = "decomposed"

    try:
        request.payload.metadata = (request.payload.metadata or {})  # type: ignore[attr-defined]
        request.payload.metadata["request_decomposition"] = decomposition  # type: ignore[index]
    except Exception:
        pass

    judge = EvidenceJudge()
    ranker = EvidenceRanker()
    response_generator = ResponseGenerator(temperature=0.2)

    path_formatter = PathFormatter()
    repo = ConceptRepository(repo_adapter, request_id=response_id)
    config = SingleEntityConfig(top_k_similar=1, select_k_per_hop=3, max_depth=4)

    records_out: List[KnowledgeRecord] = []
    subquery_results: List[Dict[str, Any]] = []
    prior_paths: List[str] = []
    is_decomposed = mode == "decomposed"
    use_unified_rag_final = hasattr(repo_adapter, "search_similar_rag")
    rag_top_k, rag_timeout_sec = (
        _resolve_rag_params(request) if use_unified_rag_final else (settings.EVIDENCE_RAG_TOP_K, settings.EVIDENCE_RAG_TIMEOUT_SEC)
    )

    rag_task: Optional[asyncio.Task] = None
    if use_unified_rag_final:
        rag_query_vec = _intent_query_vec(intent)
        rag_task = asyncio.create_task(
            retrieve_rag_top_k(
                repo_adapter,
                intent,
                rag_top_k,
                response_id,
                rag_query_vec,
                rag_timeout_sec,
            )
        )
        logger.info(
            "[Evidence] RAG retrieval started in parallel (top_k=%s, timeout=%ss).",
            rag_top_k,
            rag_timeout_sec,
        )

    # With RAG similarity API available: defer per-lane final LLM to one unified call after graph + RAG.
    skip_final_response = is_decomposed or use_unified_rag_final

    for item in items:
        sent = str(item.get("sentence") or "").strip()
        ents = item.get("entities") or []
        extra_context = (
            "\n".join(prior_paths[-_PRIOR_PATHS_CONTEXT_LIMIT:]) if prior_paths else ""
        )

        if len(ents) == 1:
            engine = SingleEntityEvidenceEngine(
                embedding_manager=embedding_manager,
                repo=repo,
                path_formatter=path_formatter,
                judge=judge,
                ranker=ranker,
                config=config,
                response_generator=response_generator,
            )
            ent_dict = {"name": ents[0]}
            rec = await engine.gather(
                request,
                ent_dict,
                extra_context=extra_context,
                skip_final_response=skip_final_response,
            )
            records_out.append(rec)
            prior_paths = _get_context_paths_for_next(rec)
            path_strs = _get_paths_strings(rec)
            subquery_results.append(
                {
                    "sentence": sent,
                    "entities": ents,
                    "paths_symbolic": path_strs,
                    "status": (rec.content or {}).get("evidence", {}).get("status"),
                }
            )
        elif len(ents) >= 2:
            me_engine = MultiEntityEvidenceEngine(
                embedding_manager=embedding_manager,
                data_layer=repo_adapter,
                judge=judge,
                ranker=ranker,
                config=MultiEntityConfig(
                    top_k_candidates=1,
                    max_depth=4,
                    pre_rank_limit=20,
                    mmr_top_k=5,
                    concurrency_limit=3,
                ),
                concept_repo=repo,
                response_generator=response_generator,
            )
            pair_entities = {"source": ents[0], "target": ents[1]}
            rec = await me_engine.gather(
                request,
                pair_entities,
                extra_context=extra_context,
                skip_final_response=skip_final_response,
            )
            records_out.append(rec)
            prior_paths = _get_context_paths_for_next(rec)
            path_strs = _get_paths_strings(rec)
            subquery_results.append(
                {
                    "sentence": sent,
                    "entities": ents[:2],
                    "paths_symbolic": path_strs,
                    "status": (rec.content or {}).get("evidence", {}).get("status"),
                }
            )
        else:
            prior_paths = []
            subquery_results.append(
                {
                    "sentence": sent,
                    "entities": [],
                    "paths_symbolic": [],
                    "status": "no_entities",
                }
            )

    rag_snippets: List[Dict[str, Any]] = []
    if rag_task is not None:
        try:
            rag_snippets = await rag_task
        except Exception as e:
            logger.warning("[Evidence] RAG task failed: %s", e)
            rag_snippets = []

    if is_decomposed and records_out:
        cumulated_paths: List[str] = []
        sufficient_reasons: List[str] = []
        for rec in records_out:
            cumulated_paths.extend(_get_context_paths_for_next(rec))
            evidence = (rec.content or {}).get("evidence") or {}
            if evidence.get("status") == "sufficient":
                trace = (rec.content or {}).get("trace") or {}
                winning = trace.get("winning") or {}
                reason = (winning.get("reason_for_sufficiency") or "").strip()
                if reason:
                    sufficient_reasons.append(reason)
        verdict = " ".join(sufficient_reasons) if sufficient_reasons else ""
        token_meta = None
        try:
            final_response, token_meta = await response_generator.async_generate_final_response(
                intent,
                cumulated_paths,
                verdict,
                rag_snippets if use_unified_rag_final else None,
                return_tokens=True,
            )
        except Exception:
            final_response = verdict or "Insufficient Evidence"
        evidence_metadata: Dict[str, Any] = {
            "retrieval_mode": "decomposed",
            "pruning_applied": True,
            "llm_assisted": True,
        }
        if use_unified_rag_final:
            evidence_metadata["rag"] = {
                "chunks_returned": len(rag_snippets),
                "unified_final": True,
                "top_k": rag_top_k,
            }
        combined_content = {
            "evidence": {
                "entity": ent_names,
                "status": "sufficient" if sufficient_reasons else "insufficient",
                "summary": {
                    "supporting_paths": len(cumulated_paths),
                    "decomposed_steps": len(records_out),
                },
                "paths": cumulated_paths,
                "final_response": final_response,
                "details": {},
                "metadata": evidence_metadata,
            },
            "trace": {
                "request_decomposition": decomposition,
                "subquery_results": subquery_results,
            },
        }
        if use_unified_rag_final:
            combined_content["trace"]["rag_snippets"] = rag_snippets
        combined_record = KnowledgeRecord(type="json", content=combined_content)

        # Convert token metadata to response format
        response_meta = None
        if token_meta:
            from ..api.schemas import TokenUsage, TokenUsageMeta
            response_meta = TokenUsageMeta(
                tokens=TokenUsage(
                    prompt=token_meta.prompt_tokens,
                    completion=token_meta.completion_tokens,
                    total=token_meta.total_tokens,
                    model=token_meta.model,
                ),
                latency_ms=token_meta.latency_ms,
                cost_usd=token_meta.cost_usd,
                timestamp=token_meta.timestamp,
            )

        return ReasonerCognitionResponse(
            header=response_header,
            response_id=response_id,
            records=[combined_record],
            metadata={
                "source": "evidence.process_evidence",
                "mode": mode,
                "lanes": len(items),
                "returned": 1,
                "request_decomposition": decomposition,
                "subquery_results": subquery_results,
                "decomposed_step_records_count": len(records_out),
                "rag_unified_final": use_unified_rag_final,
                "rag_chunks_returned": len(rag_snippets) if use_unified_rag_final else 0,
            },
            meta=response_meta,
        )

    if use_unified_rag_final and records_out:
        last_token_meta = None
        for rec in records_out:
            graph_paths = _get_paths_strings(rec)
            verdict_one = _verdict_from_record(rec)
            try:
                final_text, last_token_meta = await response_generator.async_generate_final_response(
                    intent, graph_paths, verdict_one, rag_snippets, return_tokens=True
                )
            except Exception:
                final_text = verdict_one or "Insufficient Evidence"
            ev = (rec.content or {}).setdefault("evidence", {})
            ev["final_response"] = final_text
            md = ev.setdefault("metadata", {})
            md["rag"] = {
                "chunks_returned": len(rag_snippets),
                "unified_final": True,
                "top_k": rag_top_k,
            }
            tr = (rec.content or {}).setdefault("trace", {})
            tr["rag_snippets"] = rag_snippets

        # Convert last token metadata to response format (if available)
        response_meta_final = None
        if use_unified_rag_final and last_token_meta:
            from ..api.schemas import TokenUsage, TokenUsageMeta
            response_meta_final = TokenUsageMeta(
                tokens=TokenUsage(
                    prompt=last_token_meta.prompt_tokens,
                    completion=last_token_meta.completion_tokens,
                    total=last_token_meta.total_tokens,
                    model=last_token_meta.model,
                ),
                latency_ms=last_token_meta.latency_ms,
                cost_usd=last_token_meta.cost_usd,
                timestamp=last_token_meta.timestamp,
            )

    return ReasonerCognitionResponse(
        header=response_header,
        response_id=response_id,
        records=records_out,
        metadata={
            "source": "evidence.process_evidence",
            "mode": mode,
            "lanes": len(items),
            "returned": len(records_out),
            "request_decomposition": decomposition,
            "subquery_results": subquery_results,
            "rag_unified_final": use_unified_rag_final,
            "rag_chunks_returned": len(rag_snippets) if use_unified_rag_final else 0,
        },
        meta=response_meta_final if use_unified_rag_final else None,
    )
