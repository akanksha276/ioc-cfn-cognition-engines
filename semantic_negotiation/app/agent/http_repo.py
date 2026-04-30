# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
HTTP helpers for the cognition-fabric **shared-memories query** route.

Callers build the ``intent`` string (e.g. in :mod:`app.agent.options_generation` or
:mod:`app.agent.intent_discovery`) and invoke :func:`post_shared_memories_query` once
per request. No multi-issue loop lives here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class SharedMemoryQueryError(Exception):
    """Raised when the fabric shared-memories query fails (transport or non-success HTTP)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class SharedMemoryNotFoundError(SharedMemoryQueryError):
    """Raised when the fabric shared-memories query returns HTTP 404 (no cache for MAS)."""


def issue_labels_from_negotiable_entities(
    negotiable_entities: Optional[list[Any]],
) -> list[str]:
    """Normalize negotiable entities to issue strings (``term`` attribute when present)."""
    if not negotiable_entities:
        return []
    if isinstance(negotiable_entities, (list, tuple, set)):
        out: list[str] = []
        for x in negotiable_entities:
            label = getattr(x, "term", x)
            s = str(label).strip()
            if s:
                out.append(s)
        return out
    s = str(negotiable_entities).strip()
    return [s] if s else []


def shared_memories_query_path(workspace_id: str, mas_id: str) -> str:
    """URL path (no origin) for POST shared-memories query."""
    wid = quote(workspace_id.strip(), safe="")
    mid = quote(mas_id.strip(), safe="")
    return f"/api/workspaces/{wid}/multi-agentic-systems/{mid}/shared-memories/query"


def post_shared_memories_query(
    client: httpx.Client,
    path: str,
    intent: str,
) -> dict[str, Any]:
    """
    Perform exactly one POST to *path* with the given ``intent``.

    Returns the parsed JSON body (typically ``message``, ``response_id``).

    Raises:
        SharedMemoryNotFoundError: HTTP 404 (no shared memory for workspace/mas).
        SharedMemoryQueryError: Other HTTP errors, or transport failures (no response).
    """
    payload: dict[str, Any] = {
        "intent": intent,
        "request_id": str(uuid.uuid4()),
    }
    rid = payload.get("request_id", "")
    logger.info(
        "shared-memories query POST path=%s request_id=%s intent_len=%d",
        path,
        rid,
        len(intent),
    )
    try:
        r = client.post(path, json=payload)
    except httpx.RequestError as exc:
        raise SharedMemoryQueryError(
            f"Shared-memories query request failed: {exc}",
            status_code=None,
        ) from exc
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        code = resp.status_code if resp is not None else None
        if code == 404:
            # Extract workspace/mas from the request path for a clearer message
            _parts = path.split("/")
            _wid = (
                _parts[_parts.index("workspaces") + 1]
                if "workspaces" in _parts
                else "?"
            )
            _mid = (
                _parts[_parts.index("multi-agentic-systems") + 1]
                if "multi-agentic-systems" in _parts
                else "?"
            )
            raise SharedMemoryNotFoundError(
                f"No shared memory for workspace_id={_wid} mas_id={_mid} (HTTP 404)",
                status_code=404,
            ) from exc
        body_preview = ""
        if resp is not None:
            try:
                body_preview = (resp.text or "")[:300]
            except Exception:
                body_preview = ""
        suffix = f" — body: {body_preview!r}" if body_preview else ""
        raise SharedMemoryQueryError(
            f"Shared-memories query failed (HTTP {code}){suffix}",
            status_code=code,
        ) from exc
    logger.info(
        "shared-memories query OK path=%s status=%s request_id=%s",
        path,
        r.status_code,
        rid,
    )
    return r.json()
