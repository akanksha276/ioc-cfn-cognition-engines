# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evidence.app.api.schemas import Header


# --- CFN-orchestrated async distillation ---


class DistillationStartPayload(BaseModel):
    callback_url: str = Field(..., description="HTTPS URL for completion callback after mutation apply succeeds.")


class DistillationRunRequest(BaseModel):
    """POST /distillation/run body (§6.1)."""

    header: Header
    request_id: str = Field(..., description="Echoed as response_id on the CFN mutation payload.")
    payload: DistillationStartPayload

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_rid(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    @model_validator(mode="after")
    def _require_request_id(self):
        if not self.request_id:
            raise ValueError("request_id is required")
        return self


class DistillationAcceptedResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    status: Literal["accepted"] = "accepted"
    operation_id: str = Field(serialization_alias="operationId")
    message: str = "distillation started"


class DistillationConflictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    status: Literal["conflict"] = "conflict"
    message: str = "distillation already in progress"
    workspace_id: str = Field(serialization_alias="workspaceId")
    mas_id: str = Field(serialization_alias="masId")


class DistillationErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
