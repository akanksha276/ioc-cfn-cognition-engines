# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
API routes for the Knowledge Extraction Service.
"""

import asyncio
import logging
import os
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..dependencies import (
    get_extraction_service,
    get_ingest_data_service,
    get_knowledge_processor,
    get_data_repository,
)
from ..agent.ingest_data import IngestDataService
from ..agent.service import TelemetryExtractionService
from ..agent.prompts import SUPPORTED_FORMATS
from ..config.settings import settings
from ..data.mock_repo import MockDataRepository
from .schemas import ExtractionRequest, ExtractionResponseModel, ExtractionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["extraction"])
extraction_router = APIRouter(prefix="/api/knowledge-mgmt", tags=["knowledge-mgmt"])


# ============== Extraction Endpoint ==============


@extraction_router.post(
    "/extraction",
    response_model=ExtractionResponseModel,
    response_model_exclude_none=True,
)
async def knowledge_extraction(
    body: ExtractionRequest,
    ingest_service: IngestDataService = Depends(get_ingest_data_service),
):
    """
    Unified knowledge extraction endpoint.

    Accepts a structured request with header, request_id, and a payload
    whose ``metadata.format`` declares the data format (e.g. 'observe-sdk-otel',
    'openclaw').  All formats are processed through the ConceptRelationship
    extraction pipeline (LLM-based concept and relationship identification).

    Pipeline:
    1. Validate header and format
    2. Extract concepts and relationships via ConceptRelationshipExtractionService
    3. Generate embeddings and optionally deduplicate
    4. Resolve similar existing concepts via similarity-search API
    5. Return response with header echo and response_id
    """
    response_id = body.request_id
    data_format = body.payload.metadata.format.strip().lower()

    if data_format not in SUPPORTED_FORMATS:
        error_resp = ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            error=ExtractionError(
                message="BAD_REQUEST",
                detail={
                    "Validation error": (
                        f"Unsupported format: {data_format!r}. "
                        f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
                    )
                },
            ),
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump())

    payload_data = body.payload.data
    if not payload_data:
        error_resp = ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            error=ExtractionError(
                message="BAD_REQUEST",
                detail={
                    "Validation error": "payload.data must be a non-empty array of records."
                },
            ),
        )
        return JSONResponse(status_code=400, content=error_resp.model_dump())

    try:
        result = ingest_service.ingest(
            payload_data,
            request_id=response_id,
            format_descriptor=data_format,
        )

        processor = get_knowledge_processor()
        result = processor.process(result)

        try:
            similarity_hits = await _fetch_similar_concepts(
                concepts=result.get("concepts", []),
                body=body,
            )
        except Exception:
            logger.exception("Concept similarity lookup failed; continuing without remote dedupe context")
            similarity_hits = []
        if similarity_hits:
            result.setdefault("meta", {})
            result["meta"]["concept_similarity_hits"] = len(similarity_hits)
            result["meta"]["concept_similarity"] = similarity_hits

        return ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            concepts=result.get("concepts", []),
            relations=result.get("relations", []),
            descriptor=result.get("descriptor", data_format),
            metadata=result.get("meta", {}),
            rag_chunks=result.get("rag_chunks", []),
        )

    except Exception as e:
        logger.error(f"Error in batch extraction {response_id}: {e}")
        error_resp = ExtractionResponseModel(
            header=body.header,
            response_id=response_id,
            error=ExtractionError(
                message="INTERNAL_ERROR",
                detail={"traceback": traceback.format_exc()},
            ),
            concepts=[],
        )
        return JSONResponse(status_code=500, content=error_resp.model_dump())


# ============== Similarity API Helper ==============


def _similarity_base_url() -> str:
    return (
        (settings.cfn_url or "").strip()
        or (os.getenv("MOCKED_DB_BASE_URL") or "").strip()
    )


def _concept_embedding(concept: Dict[str, Any]) -> List[float]:
    emb = ((concept.get("attributes") or {}).get("embedding") or [])
    if not emb:
        return []
    row = emb[0] if isinstance(emb, list) else []
    if not isinstance(row, list):
        return []
    out: List[float] = []
    for v in row:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return []
    return out


async def _fetch_similar_concepts(
    concepts: list,
    body: ExtractionRequest,
) -> List[Dict[str, Any]]:
    """
    Resolve top-k concept matches from /concepts/similarity-search.
    All concepts are queried concurrently via asyncio.gather.
    Errors per concept are logged and never fail extraction.
    """
    base_url = _similarity_base_url()
    if not base_url:
        return []

    workspace_id = quote(body.header.workspace_id, safe="")
    mas_id = quote(body.header.mas_id, safe="")
    endpoint = (
        f"{base_url.rstrip('/')}/api/internal/workspaces/{workspace_id}"
        f"/multi-agentic-systems/{mas_id}/concepts/similarity-search"
    )
    top_k = max(1, int(settings.similarity_top_k))
    metric = (settings.similarity_metric or "l2").strip() or "l2"
    agent_id = (body.header.agent_id or "ingestion-agent").strip()

    async def _search_one(
        client: httpx.AsyncClient,
        concept: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        vector = _concept_embedding(concept if isinstance(concept, dict) else {})
        if not vector:
            return None
        name = str((concept or {}).get("name") or "").strip()
        req_id = f"{body.request_id}-concept-search-{uuid.uuid4().hex[:8]}"
        payload = {
            "header": {"agent_id": agent_id},
            "request_id": req_id,
            "payload": {
                "embedded_text": name,
                "embedding_vector": vector,
                "top_k": top_k,
                "search_metrics": metric,
            },
        }
        try:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception(
                "Concept similarity-search failed for concept=%s",
                (concept or {}).get("id"),
            )
            return None
        results = data.get("results") or []
        if not isinstance(results, list):
            return None
        return {
            "concept_id": (concept or {}).get("id"),
            "concept_name": name,
            "matches": results,
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *[_search_one(client, c) for c in (concepts or [])],
            return_exceptions=False,
        )

    return [r for r in results if r is not None]


# ============== Operational Endpoints ==============


@router.get("/metrics")
async def get_metrics(
    service: TelemetryExtractionService = Depends(get_extraction_service),
):
    """Get operational metrics."""
    metrics = service.get_operational_metrics()
    return {
        "records_processed": metrics.records_processed,
        "records_sent": metrics.records_sent,
        "records_failed": metrics.records_failed,
        "last_run_timestamp": (
            metrics.last_run_timestamp.isoformat()
            if metrics.last_run_timestamp
            else None
        ),
        "last_run_duration_seconds": metrics.last_run_duration_seconds,
        "recent_errors": metrics.errors[-10:],
    }


# ============== File-based Endpoints (kept for dev/testing) ==============


@router.get("/extract/entities_and_relations/from_file")
async def extract_entities_and_relations_from_file(
    file_path: str,
    save_output: bool = False,
    service: TelemetryExtractionService = Depends(get_extraction_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Load OTEL data from a JSON file, extract entities and relations,
    generate embeddings, and optionally perform semantic deduplication.
    """
    try:
        path = Path(file_path)
        otel_data = repository.load_from_file(path)

        result = service.extract_entities_and_relations(otel_data)

        processor = get_knowledge_processor()
        result = processor.process(result)

        if save_output:
            output_filename = f"extracted_entities_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            repository.save_output(result, output_filename)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error extracting entities from file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/concepts_and_relationships/from_file")
async def extract_concepts_and_relationships_from_file(
    file_path: str,
    save_output: bool = False,
    ingest_service: IngestDataService = Depends(get_ingest_data_service),
    repository: MockDataRepository = Depends(get_data_repository),
):
    """
    Load OTEL data from a JSON file and extract high-level concepts and relationships.
    """
    try:
        path = Path(file_path)
        otel_data = repository.load_from_file(path)

        result = ingest_service.ingest(otel_data)

        processor = get_knowledge_processor()
        result = processor.process(result)

        if save_output:
            output_filename = f"concept_relationships_{result.get('knowledge_cognition_request_id', 'no_id')}.json"
            repository.save_output(result, output_filename)

        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error extracting concepts from file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
