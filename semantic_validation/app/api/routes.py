# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..agent.sav_ce import ValidationAction, ValidationCognitionEngine
from ..dependencies import get_validation_cognition_engine
from .schemas import ValidateRequest, ValidateResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["semantic-validation"])


@router.post("/validate", response_model=ValidateResponse)
async def validate(
    body: ValidateRequest,
    engine: ValidationCognitionEngine = Depends(get_validation_cognition_engine),
) -> ValidateResponse:
    payload = body.model_dump()
    result = await engine.run(ValidationAction.EVALUATE, payload)
    return ValidateResponse(**result)
