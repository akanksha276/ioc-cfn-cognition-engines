# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
API routes for the Telemetry Extraction Service.
"""

import logging
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..dependencies import (
    get_extraction_service,
    get_ingest_data_service,
    get_knowledge_processor,
    get_data_repository,
    get_vector_store,
)
from ..agent.ingest_data import IngestDataService
from ..agent.service import TelemetryExtractionService
from ..agent.prompts import SUPPORTED_FORMATS
from ..data.mock_repo import MockDataRepository
from .schemas import ExtractionRequest, ExtractionResponseModel, ExtractionError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extraction"])
extraction_router = APIRouter(tags=["knowledge-mgmt"])


# ============== Extraction Endpoint ==============


@extraction_router.post(
    "/extraction",
    response_model=ExtractionResponseModel,
    response_model_exclude_none=True,
)
async def knowledge_extraction(
    body: ExtractionRequest,
    ingest_service: IngestDataService = Depends(get_ingest_data_service),
    vector_store=Depends(get_vector_store),
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
    4. Store concepts in FAISS vector DB
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

        _store_concepts_in_faiss(result.get("concepts", []), vector_store)
        _store_rag_chunks_in_faiss(result.get("rag_chunks", []), vector_store)

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


# ============== FAISS Helper ==============


def _store_concepts_in_faiss(concepts: list, vector_store=None) -> None:
    """
    Persist concepts in the in-process FAISS index (fire-and-log).

    vector_store is injected from Depends(get_concept_vector_store); when None, skip.
    Failures are logged but never bubble up to the caller so the
    extraction response is always returned.
    """
    if vector_store is None:
        return
    try:
        vector_store.store_concepts(concepts)
    except Exception:
        logger.exception("FAISS storage failed; extraction result is still valid")


def _store_rag_chunks_in_faiss(rag_chunks: list, vector_store=None) -> None:
    """
    Persist concepts in the in-process FAISS index (fire-and-log).

    vector_store is injected from Depends(get_concept_vector_store); when None, skip.
    Failures are logged but never bubble up to the caller so the
    extraction response is always returned.
    """
    if vector_store is None:
        return
    try:
        vector_store.store_rag_chunks(rag_chunks)
    except Exception:
        logger.exception(
            "FAISS storage failed for RAG chunks; extraction result is still valid"
        )


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
