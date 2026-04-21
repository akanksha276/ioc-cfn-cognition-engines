# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
HTTP client for the graph / mocked-db service.

Configuration: set MOCKED_DB_BASE_URL or CFN_URL to the service origin (no trailing slash).
Evidence wiring: dependencies.builds this when that URL is set; graph prefix depends on workspace/MAS:

  • Both workspace_id and mas_id non-empty (POST /reasoning/evidence):
      GET/POST {base}/api/workspaces/{wid}/multi-agentic-systems/{mas_id}/graph/...
  • Otherwise (e.g. standalone /graph/* routes):
      GET/POST {base}/api/graph/...

Call sites (summary):
  • neighbors, get_concepts_by_ids — ConceptRepository / single-entity hop expansion.
  • find_paths, get_concepts_by_ids — evidence /graph/* proxy, multi-entity path enumeration.
"""
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

_LEGACY_GRAPH_PREFIX = "/api/graph"


class HttpDataRepository:
    """Async HTTP graph access."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        workspace_id: Optional[str] = None,
        mas_id: Optional[str] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        wid = (workspace_id or "").strip()
        mid = (mas_id or "").strip()
        if wid and mid:
            self._graph_prefix = (
                f"/api/internal/workspaces/{quote(wid, safe='')}/multi-agentic-systems/{quote(mid, safe='')}/graph"
            )
        else:
            self._graph_prefix = _LEGACY_GRAPH_PREFIX

    async def _client_async(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base, timeout=self._timeout)
        return self._client

    async def neighbors(self, concept_id: str) -> Dict[str, Any]:
        """One-hop ego network: GET {graph_prefix}/neighbors/{concept_id} → JSON with records[]."""
        client = await self._client_async()
        r = await client.get(f"{self._graph_prefix}/neighbors/{concept_id}")
        r.raise_for_status()
        return r.json()

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int,
        limit: int,
        relations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """POST {graph_prefix}/paths — used by evidence /graph/paths proxy and multi-entity."""
        client = await self._client_async()
        r = await client.post(
            f"{self._graph_prefix}/paths",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "max_depth": max_depth,
                "limit": limit,
                "relations": relations,
            },
        )
        r.raise_for_status()
        return r.json()

    async def get_concepts_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Batch concept metadata: POST .../concepts/by_ids → response.concepts list."""
        if not ids:
            return []
        client = await self._client_async()
        r = await client.post(f"{self._graph_prefix}/concepts/by_ids", json={"ids": ids})
        r.raise_for_status()
        data = r.json()
        return data.get("concepts", [])
