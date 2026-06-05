# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Data models for CE lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CERegistrationRequest:
    """
    CE registration payload matching CFN API.

    This is what the CE sends to CFN (which forwards to Management Plane).
    The CE does NOT include cfn_id or ce_id - those are managed by the system.

    Example:
        request = CERegistrationRequest(
            name="Knowledge Management CE",
            url="http://ce-host:9004",
            version="1.2.3",
            kind="knowledge",
            subkind="query",
            capabilities=["ingestion", "retrieval"],
            metrics=["llm.token.input", "llm.latency_ms"],
            config={"model": "gpt-4o"},
        )
    """

    name: str
    url: str
    version: str
    kind: str
    subkind: str
    capabilities: List[str]
    metrics: List[str]
    config: Dict[str, Any]
    mas_config: Optional[Dict[str, Any]] = None
    mas_auto_associate: bool = False


@dataclass
class CERegistrationResponse:
    """
    CE registration response from Management Plane (via CFN).

    Contains the generated ce_id that the CE must store for heartbeats.
    The Management Plane generates ce_id on first registration, or returns
    the existing ce_id if the CE (cfn_id, name, version) already exists.

    Example response:
        {
            "ce_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "cfn_id": "cfn-abc-123",
            "name": "Knowledge Management CE",
            "version": "1.2.3",
            "kind": "knowledge",
            "subkind": "query",
            "enabled": true,
            "status": "offline",  # transitions to "online" on first heartbeat
            "created": true       # true if new, false if updated
        }
    """

    ce_id: str
    cfn_id: str
    name: str
    version: str
    kind: str
    subkind: str
    enabled: bool
    status: str  # "offline" initially, "online" after first heartbeat
    created: bool  # true if newly created, false if updated


@dataclass
class CEHeartbeatResponse:
    """
    CE heartbeat response from Management Plane (via CFN).

    Returned after each successful heartbeat to indicate CE status.

    Example response:
        {
            "status": "online",
            "last_seen": "2026-05-21T10:30:00Z"
        }
    """

    status: str  # "online", "offline"
    last_seen: str  # ISO 8601 timestamp
