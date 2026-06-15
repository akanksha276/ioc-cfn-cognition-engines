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
from typing import Any, Dict

import httpx
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from distill.app.api.routes import _validate_distillation_start
from distill.app.api.schemas import DistillationRunRequest, DistillationStartPayload
from distill.app.services import distillation_lock
from distill.app.services.distillation_job import execute_distillation_run
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

    Always returns 202 immediately. All validation, lock acquisition, and
    execution happen in the background; any failure is reported back to cfn-svc
    via callback_url so it is recorded in task execution history.
    """
    operation_id = str(uuid.uuid4())
    distill_run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    background_tasks.add_task(
        _run_with_callback,
        req=req,
        operation_id=operation_id,
        distill_run_at=distill_run_at,
    )

    return JSONResponse(
        status_code=202,
        content=TaskExecutionResponse(execution_id=operation_id).model_dump(),
    )


async def _run_with_callback(
    *,
    req: TaskExecutionRequest,
    operation_id: str,
    distill_run_at: str,
) -> None:
    """Background task: validate, acquire lock, then delegate to the distillation job."""
    callback_url = req.callback_url.strip()
    wid = req.workspace_id.strip()
    mid = req.mas_id.strip()
    request_id = str(uuid.uuid4())

    distill_req = DistillationRunRequest(
        header=Header(workspace_id=wid, mas_id=mid),
        request_id=request_id,
        payload=DistillationStartPayload(callback_url=callback_url),
    )

    err = _validate_distillation_start(distill_req)
    if err:
        logger.warning("[CoDi task] validation failed | op=%s reason=%s", operation_id, err)
        await _post_failure_callback(callback_url, wid, mid, req.ce_id, operation_id, err)
        return

    if not await distillation_lock.try_begin_run(wid, mid):
        msg = "distillation already in progress"
        logger.warning("[CoDi task] lock conflict | op=%s ws=%s mas=%s", operation_id, wid, mid)
        await _post_failure_callback(callback_url, wid, mid, req.ce_id, operation_id, msg)
        return

    # Lock is held — execute_distillation_run releases it in its finally block.
    await execute_distillation_run(
        header=distill_req.header,
        request_id=request_id,
        callback_url=callback_url,
        operation_id=operation_id,
        distill_run_at=distill_run_at,
        ce_id=req.ce_id,
        rag_layer=None,
        prefetched_graph_read=None,
    )


async def _post_failure_callback(
    callback_url: str,
    workspace_id: str,
    mas_id: str,
    ce_id: str,
    operation_id: str,
    reason: str,
) -> None:
    payload: Dict[str, Any] = {
        "status": "failed",
        "workspace_id": workspace_id,
        "mas_id": mas_id,
        "ce_id": ce_id,
        "error": reason,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(callback_url, json=payload)
            if r.status_code >= 300:
                logger.warning("[CoDi task] failure callback non-2xx | status=%s op=%s", r.status_code, operation_id)
    except Exception:
        logger.exception("[CoDi task] failure callback POST failed | op=%s", operation_id)
