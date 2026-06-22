# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ParticipantSchema(BaseModel):
    id: str
    name: Optional[str] = None
    role: Optional[str] = None


class ValidateRequest(BaseModel):
    mission: str
    participants: List[ParticipantSchema]
    context: Optional[str] = None
    latest_message: Optional[str] = None
    final_decision: Optional[str] = None
    interaction_history: Optional[List[str]] = None


class FailureModeSchema(BaseModel):
    type: str
    score: float
    severity: dict
    description: str
    reasoning: str


class ValidateResponse(BaseModel):
    failure_modes: List[FailureModeSchema]
