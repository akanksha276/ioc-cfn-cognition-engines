# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for CoDi graph reads (distillation) scoped by workspace / MAS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

_LEGACY_GRAPH_PREFIX = "/api/graph"


def _graph_prefix(workspace_id: str, mas_id: str) -> str:
    wid = (workspace_id or "").strip()
    mid = (mas_id or "").strip()
    if wid and mid:
        return (
            f"/api/internal/workspaces/{quote(wid, safe='')}/multi-agentic-systems/{quote(mid, safe='')}/graph"
        )
    return _LEGACY_GRAPH_PREFIX


def graph_update_post_url(base_url: str, workspace_id: str, mas_id: str, update_segment: str = "update") -> str:
    """
    Full URL for CFN graph batch update:
    ``PUT /api/internal/workspaces/{workspaceId}/multi-agentic-systems/{masId}/graph/update``.
    """
    b = (base_url or "").strip().rstrip("/")
    seg = (update_segment or "update").strip().strip("/") or "update"
    return f"{b}{_graph_prefix(workspace_id, mas_id)}/{seg}"


class CoDiGraphClient:
    """Async graph HTTP access scoped by workspace / MAS (mirrors evidence HttpDataRepository routing)."""

    def __init__(
        self,
        base_url: str,
        *,
        workspace_id: str,
        mas_id: str,
        agent_id: Optional[str] = None,
        timeout: float = 120.0,
        distillation_read_path: str = "distillation/read",
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._prefix = _graph_prefix(workspace_id, mas_id)
        self._distill_read_path = distillation_read_path.strip().strip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._agent_id = (agent_id or "").strip()

    async def _client_async(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def distillation_graph_read(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST normalized Phase A/B distillation read (concepts=anchors, relations=undistilled)."""
        client = await self._client_async()
        r = await client.post(f"{self._prefix}/{self._distill_read_path}", json=body)
        r.raise_for_status()
        return r.json()
