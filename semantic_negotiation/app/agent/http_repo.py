# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
HTTP helpers for the cognition-fabric **shared-memories query** route.

Callers build the ``intent`` string (e.g. in :mod:`app.agent.options_generation` or
:mod:`app.agent.intent_discovery`) and invoke :func:`post_shared_memories_query` or
:func:`gather_shared_memories_queries` (parallel async POSTs per issue).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Coroutine, Optional, TypeVar
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
    """URL path (no origin) for POST shared-memories/query."""
    wid = quote(workspace_id.strip(), safe="")
    mid = quote(mas_id.strip(), safe="")
    return f"/api/workspaces/{wid}/multi-agentic-systems/{mid}/shared-memories/query"


def _finalize_shared_memories_response(r: httpx.Response, path: str, rid: str) -> dict[str, Any]:
    """Map HTTP outcome to JSON or raise; shared by sync and async POST helpers."""
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        code = resp.status_code if resp is not None else None
        if code == 404:
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


_T = TypeVar("_T")


def run_coro_in_own_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run *coro* to completion from synchronous code (no running event loop).

    Do not call from ``async def`` handlers or other async code: ``await`` the coroutine
    directly. Using this helper while a loop is running blocks the caller and can
    deadlock loopback HTTP to the same process (see cognition-engines issue #23).
    """
    _coro_name = getattr(coro, "__qualname__", type(coro).__name__)
    logger.info("run_coro_in_own_loop: scheduling coroutine=%s", _coro_name)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.info(
            "run_coro_in_own_loop: no running loop; asyncio.run in current thread coroutine=%s",
            _coro_name,
        )
        return asyncio.run(coro)
    raise RuntimeError(
        f"run_coro_in_own_loop cannot be used from async code (coroutine={_coro_name!r}); "
        "await the coroutine on the running event loop instead."
    )


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
    return _finalize_shared_memories_response(r, path, rid)


async def post_shared_memories_query_async(
    client: httpx.AsyncClient,
    path: str,
    intent: str,
) -> dict[str, Any]:
    """Async variant of :func:`post_shared_memories_query` (same errors and return shape).

    Intended for use under :func:`gather_shared_memories_queries` with one shared
    :class:`httpx.AsyncClient` so multiple issues await I/O concurrently.
    """
    payload: dict[str, Any] = {
        "intent": intent,
        "request_id": str(uuid.uuid4()),
    }
    rid = str(payload.get("request_id", ""))
    logger.info(
        "shared-memories query POST path=%s request_id=%s intent_len=%d",
        path,
        rid,
        len(intent),
    )
    try:
        r = await client.post(path, json=payload)
    except httpx.RequestError as exc:
        raise SharedMemoryQueryError(
            f"Shared-memories query request failed: {exc}",
            status_code=None,
        ) from exc
    return _finalize_shared_memories_response(r, path, rid)


async def _post_shared_memories_query_async_with_deadline(
    client: httpx.AsyncClient,
    path: str,
    intent: str,
    *,
    deadline_s: float,
) -> dict[str, Any] | None:
    """Run :func:`post_shared_memories_query_async` with a per-request ``asyncio`` deadline.

    On :class:`asyncio.TimeoutError`, logs and returns ``None`` so callers can treat
    that slot as timed out while other parallel POSTs may still succeed.
    """
    try:
        return await asyncio.wait_for(
            post_shared_memories_query_async(client, path, intent),
            timeout=deadline_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "shared-memories query asyncio timeout path=%s intent_len=%d deadline_s=%s",
            path,
            len(intent),
            deadline_s,
        )
        return None


async def gather_shared_memories_queries(
    base_url: str,
    path: str,
    intents: list[str],
    *,
    timeout: float = 120.0,
) -> list[dict[str, Any] | None]:
    """POST all *intents* to *path* in parallel (one ``AsyncClient``, order matches *intents*).

    Each POST is bounded by *timeout* seconds via :func:`asyncio.wait_for` (in addition
    to the httpx client timeout). Entries are ``None`` when that deadline elapses for
    the corresponding intent; otherwise the dict is the same shape as
    :func:`post_shared_memories_query_async` on success.
    """
    if not intents:
        return []
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        return await asyncio.gather(
            *[
                _post_shared_memories_query_async_with_deadline(
                    client, path, intent, deadline_s=timeout
                )
                for intent in intents
            ],
        )
