# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from distill.app.api.schemas import (
    DistillationAcceptedResponse,
    DistillationConflictResponse,
    DistillationErrorResponse,
    DistillationRunRequest,
)
from distill.app.config.settings import settings
from distill.app.services import distillation_lock
from distill.app.services.distillation_job import execute_distillation_run, fetch_distillation_graph_read

router = APIRouter()
logger = logging.getLogger(__name__)


def _distill_run_at_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_distillation_start(req: DistillationRunRequest) -> str | None:
    """Return error message or None if ok."""
    if not (req.header.workspace_id or "").strip():
        return "workspace_id is required"
    if not (req.header.mas_id or "").strip():
        return "mas_id is required"
    cb = (req.payload.callback_url or "").strip()
    if not cb.startswith(("https://", "http://")):
        return "callback_url must be an HTTP or HTTPS URL"
    if not (settings.DATA_LAYER_BASE_URL or "").strip():
        return "DATA_LAYER_BASE_URL is not configured"
    return None


@router.post("/distillation/run")
async def start_distillation_run(
    req: DistillationRunRequest,
    background_tasks: BackgroundTasks,
):
    """§6: accept async distillation; returns 202 with operationId after acquiring per-(workspace, mas) lock."""
    err = _validate_distillation_start(req)
    if err:
        return JSONResponse(
            status_code=400,
            content=DistillationErrorResponse(message=err).model_dump(mode="json"),
        )

    wid = req.header.workspace_id.strip()
    mid = req.header.mas_id.strip()
    if not await distillation_lock.try_begin_run(wid, mid):
        body = DistillationConflictResponse(
            workspace_id=wid,
            mas_id=mid,
        ).model_dump(mode="json", by_alias=True)
        return JSONResponse(status_code=409, content=body)

    operation_id = str(uuid.uuid4())
    distill_run_at = _distill_run_at_rfc3339()

    try:
        graph_payload = await fetch_distillation_graph_read(req.header, req.request_id)
        from distill.app.services.distillation_job import _coerce_distillation_read_lists

        concepts_preview, relations_preview = _coerce_distillation_read_lists(
            graph_payload if isinstance(graph_payload, dict) else {}
        )
        logger.info(
            "[CoDi distill] graph read ok | request_id=%s concepts=%d relations=%d records=%d",
            req.request_id,
            len(concepts_preview),
            len(relations_preview),
            len((graph_payload or {}).get("records") or [])
            if isinstance(graph_payload, dict)
            else 0,
        )
    except Exception as exc:
        logger.exception(
            "[CoDi distill] graph read failed before run | workspace_id=%s mas_id=%s",
            wid,
            mid,
        )
        await distillation_lock.end_run(wid, mid)
        return JSONResponse(
            status_code=502,
            content=DistillationErrorResponse(
                message=f"graph read failed: {exc}",
            ).model_dump(mode="json"),
        )

    background_tasks.add_task(
        execute_distillation_run,
        header=req.header,
        request_id=req.request_id,
        callback_url=req.payload.callback_url.strip(),
        operation_id=operation_id,
        distill_run_at=distill_run_at,
        rag_layer=None,
        prefetched_graph_read=graph_payload,
    )

    accepted = DistillationAcceptedResponse(
        operation_id=operation_id,
    )
    return JSONResponse(
        status_code=202,
        content=accepted.model_dump(mode="json", by_alias=True),
    )
