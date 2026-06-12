# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""CFN-compatible distillation task endpoint.

Receives task dispatch requests from cfn-svc and adapts them to the
internal distillation job pipeline. Mirrors the pattern used by
semantic_negotiation/app/api/cfn_compat.py.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from distill.app.api.routes import _probe_callback_url, _validate_distillation_start
from distill.app.api.schemas import DistillationErrorResponse, DistillationRunRequest, DistillationStartPayload
from distill.app.services import distillation_lock
from distill.app.services.distillation_job import (
    _coerce_distillation_read_lists,
    execute_distillation_run,
    fetch_distillation_graph_read,
)
from evidence.app.api.schemas import Header

router = APIRouter(tags=["cfn-compat"])
logger = logging.getLogger(__name__)


class TaskExecutionRequest(BaseModel):
    """Flat payload cfn-svc sends when dispatching a task to a CE."""

    workspace_id: str
    mas_id: str
    ce_id: str
    callback_url: str


class TaskExecutionResponse(BaseModel):
    """202 Accepted response returned to cfn-svc."""

    execution_id: str


@router.post("/knowledge-mgmt/distillation")
async def run_distillation_task(
    req: TaskExecutionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Entry point for cfn-svc task framework to trigger an async distillation run.

    Adapts the flat TaskExecutionRequest from cfn-svc into a DistillationRunRequest
    and delegates to the distillation service. Returns 202 with execution_id on
    acceptance; cfn-svc receives the completion signal via callback_url.
    """
    distill_req = DistillationRunRequest(
        header=Header(workspace_id=req.workspace_id, mas_id=req.mas_id),
        request_id=str(uuid.uuid4()),
        payload=DistillationStartPayload(callback_url=req.callback_url),
    )

    err = _validate_distillation_start(distill_req)
    if err:
        return JSONResponse(
            status_code=400,
            content=DistillationErrorResponse(message=err).model_dump(mode="json"),
        )

    probe_err = await _probe_callback_url(req.callback_url.strip())
    if probe_err:
        return JSONResponse(
            status_code=400,
            content=DistillationErrorResponse(message=probe_err).model_dump(mode="json"),
        )

    wid = req.workspace_id.strip()
    mid = req.mas_id.strip()
    if not await distillation_lock.try_begin_run(wid, mid):
        return JSONResponse(
            status_code=409,
            content={"message": "distillation already in progress", "workspace_id": wid, "mas_id": mid},
        )

    operation_id = str(uuid.uuid4())
    distill_run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        graph_payload = await fetch_distillation_graph_read(distill_req.header, distill_req.request_id)
        concepts_preview, relations_preview = _coerce_distillation_read_lists(
            graph_payload if isinstance(graph_payload, dict) else {}
        )
        logger.info(
            "[CoDi runDistillation] graph read ok | request_id=%s concepts=%d relations=%d",
            distill_req.request_id,
            len(concepts_preview),
            len(relations_preview),
        )
    except Exception as exc:
        logger.exception(
            "[CoDi runDistillation] graph read failed | workspace_id=%s mas_id=%s", wid, mid
        )
        await distillation_lock.end_run(wid, mid)
        return JSONResponse(
            status_code=502,
            content=DistillationErrorResponse(message=f"graph read failed: {exc}").model_dump(mode="json"),
        )

    background_tasks.add_task(
        execute_distillation_run,
        header=distill_req.header,
        request_id=distill_req.request_id,
        callback_url=req.callback_url.strip(),
        operation_id=operation_id,
        distill_run_at=distill_run_at,
        rag_layer=None,
        prefetched_graph_read=graph_payload,
    )

    return JSONResponse(
        status_code=202,
        content=TaskExecutionResponse(execution_id=operation_id).model_dump(),
    )
