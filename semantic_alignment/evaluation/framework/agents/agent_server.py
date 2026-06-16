# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared FastAPI ``/decide`` server for all agents.

``make_decide_app`` builds a FastAPI app whose ``POST /decide`` endpoint
dispatches each inbound ``SSTPNegotiateMessage`` to the correct
:class:`~evaluation.framework.agents.base_agent.BaseAgent` by
``payload.participant_id``, then returns the full list of SSTP reply dicts.

This contract is satisfied by both runtime paths:

* **direct_env** — ``BatchCallbackRunner._post_batch`` sends
  ``List[SSTPNegotiateMessage]`` and reads ``List[SSTPNegotiateMessage]``
  from the HTTP response body.
* **callback_env** — the external negotiation server calls ``/decide``
  with the same list-in / list-out shape on every SAO round.

``start_agent_server`` / ``wait_for_server`` are the lifecycle helpers
used by both mechanism files.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── sys.path safety ───────────────────────────────────────────────────────────
_repo_root = str(Path(__file__).resolve().parents[4])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from .base_agent import BaseAgent  # noqa: E402


def make_decide_app(registry: Dict[str, BaseAgent]) -> FastAPI:
    """Return a FastAPI app whose ``POST /decide`` routes to registered agents.

    The endpoint:

    1. Receives ``List[SSTPNegotiateMessage]`` (one per participant).
    2. Resolves each message's ``payload.participant_id`` to a
       :class:`~evaluation.framework.agents.base_agent.BaseAgent` in
       *registry*.
    3. Calls :meth:`~evaluation.framework.agents.base_agent.BaseAgent.handle_message`
       for each (runs in parallel threads so LLM round-trips don't block).
    4. Returns ``List[SSTPNegotiateMessage]`` in the same order.

    *registry* maps ``participant_id → BaseAgent``.  The dict may be
    mutated in-place between missions without restarting the server — the
    endpoint always reads the current mapping.

    A fuzzy fallback resolves partial-match participant IDs (e.g.
    ``"agent-a-session-xyz"`` → ``registry["agent-a"]``).  If no match is
    found the first registered agent is used as a last resort.

    Args:
        registry: Mutable ``{participant_id: BaseAgent}`` mapping.

    Returns:
        A :class:`fastapi.FastAPI` instance ready to be served with uvicorn.
    """
    app = FastAPI(title="Evaluation Agent Server")

    @app.post("/decide")
    async def decide(request: Request) -> JSONResponse:
        messages: List[Dict[str, Any]] = await request.json()

        def _process(body: Dict[str, Any]) -> Dict[str, Any]:
            payload = body.get("payload") or {}
            pid: str = payload.get("participant_id", "")

            # Exact match first
            agent: BaseAgent | None = registry.get(pid)

            # Fuzzy: try substring match in both directions
            if agent is None:
                for key, a in registry.items():
                    if pid in key or key in pid:
                        agent = a
                        break

            # Last resort: first registered agent
            if agent is None:
                agent = next(iter(registry.values()))

            return agent.handle_message(body)

        # All decisions run in parallel threads (handles LLM latency)
        replies = await asyncio.gather(
            *[asyncio.to_thread(_process, msg) for msg in messages]
        )
        return JSONResponse(list(replies))

    return app


def start_agent_server(
    registry: Dict[str, BaseAgent],
    port: int,
    host: str = "0.0.0.0",
) -> uvicorn.Server:
    """Start the shared agent server in a daemon thread.

    Args:
        registry: Agent registry passed to :func:`make_decide_app`.
        port: TCP port to listen on.
        host: Bind address (default ``"0.0.0.0"``).

    Returns:
        The running :class:`uvicorn.Server` instance.  Call
        ``server.should_exit = True`` to stop it gracefully.
    """
    app = make_decide_app(registry)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server


def wait_for_server(
    port: int,
    host: str = "localhost",
    retries: int = 40,
    delay: float = 0.2,
) -> None:
    """Block until the agent server on *port* is accepting connections.

    Args:
        port: TCP port to poll.
        host: Hostname to connect to (default ``"localhost"``).
        retries: Maximum number of connection attempts.
        delay: Seconds to wait between attempts.

    Raises:
        RuntimeError: When the server does not become ready within
            ``retries * delay`` seconds.
    """
    for _ in range(retries):
        try:
            httpx.get(f"http://{host}:{port}/openapi.json", timeout=1.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError(
        f"Agent server on {host}:{port} did not start within "
        f"{retries * delay:.1f}s."
    )
