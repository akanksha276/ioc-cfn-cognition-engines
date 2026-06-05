# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared diagnostics router factory.

Produces a FastAPI APIRouter exposing:
  GET  /health   – liveness probe (with dependency health checks)
  GET  /info     – service metadata
  GET  /metrics  – process-level metrics (uptime, threads, memory)
  GET  /loggers  – list all Python loggers and their levels
  PUT  /loggers – set a logger's level at runtime (body: module-name + log-level)

Usage (in each service's main.py):
    from common.diagnostics.router import make_diagnostics_router, HealthCheck
    app.include_router(
        make_diagnostics_router(
            service_name="my-service",
            version="1.0.0",
            health_checks=[
                HealthCheck("db", check_db, critical=True),
                HealthCheck("cache", check_cache, critical=False),
            ],
        ),
        prefix="/api/internal/diagnostics",
    )
"""
from __future__ import annotations

import datetime
import logging
import os
import platform
import resource
import sys
import threading
import time
from enum import Enum
from typing import Callable, NamedTuple

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# Captured once at module import — represents the process start time.
_STARTUP_TIME: float = time.monotonic()


class HealthState(Enum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class HealthCheck(NamedTuple):
    """A single dependency health check.

    Args:
        name:     Identifier shown in the response (e.g. ``"embedding_model"``).
        check:    Zero-argument callable returning ``True`` when the dependency
                  is healthy, ``False`` (or raising) when it is not.
        critical: When ``True`` a failed check drives the overall state to DOWN
                  (HTTP 500).  When ``False`` it drives it to DEGRADED (HTTP 200).
        external: When ``True`` the check makes a network call to a downstream
                  service and is skipped unless ``?dependencies=true`` is passed.
    """
    name: str
    check: Callable[[], bool]
    critical: bool = True
    external: bool = False


# Log levels accepted by this API.  TRACE and WARN are aliases used by
# callers with a Java/Spring background; they are normalised before being
# passed to Python's logging module.
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "TRACE"}
_LOG_LEVEL_ALIASES = {
    "TRACE": "DEBUG",    # Python has no TRACE; map to DEBUG
    "WARN": "WARNING",   # common shorthand
}


class _SetLevelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    module_name: str = Field(alias="module-name")
    log_level: str = Field(alias="log-level")


def make_diagnostics_router(
    service_name: str,
    version: str,
    description: str = "",
    health_checks: list[HealthCheck] | None = None,
    include_health: bool = True,
) -> APIRouter:
    """Return a configured APIRouter with all four diagnostic endpoints.

    Set ``include_health=False`` when the caller registers its own ``/health``
    route.
    """

    router = APIRouter(tags=["diagnostics"])

    if include_health:
        @router.get("/health")
        async def health(dependencies: bool = False):
            checks = health_checks or []
            state = HealthState.UP
            check_results: dict[str, bool] = {}

            for hc in checks:
                if hc.external and not dependencies:
                    continue
                try:
                    ok = hc.check()
                except Exception:
                    ok = False
                check_results[hc.name] = ok
                if not ok:
                    if hc.critical:
                        state = HealthState.DOWN
                    elif state != HealthState.DOWN:
                        state = HealthState.DEGRADED

            response_body: dict = {
                "status": state.value,
                "service_name": service_name,
                "service_state": state.name,
                "last_updated": datetime.datetime.now().isoformat(),
            }
            if check_results:
                response_body["checks"] = check_results

            http_status = 500 if state == HealthState.DOWN else 200
            return JSONResponse(content=response_body, status_code=http_status)

    @router.get("/info")
    async def info():
        return {
            "service": service_name,
            "version": version,
            "description": description,
            "python_version": sys.version,
            "platform": platform.platform(),
            "environment": os.getenv("ENV", "development"),
            "git": {
                "commit": {
                    "id": os.getenv("GIT_COMMIT_SHA", "unknown"),
                    "time": os.getenv("GIT_COMMIT_TIME", "unknown"),
                },
                "branch": os.getenv("GIT_BRANCH", "unknown"),
            },
        }


    @router.get("/metrics")
    async def metrics():
        result: dict = {
            "uptime_seconds": round(time.monotonic() - _STARTUP_TIME, 2),
            "threads": threading.active_count(),
        }
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is bytes on macOS, kilobytes on Linux
            divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
            result["memory_rss_mb"] = round(usage.ru_maxrss / divisor, 2)
        except Exception:
            pass
        return result


    @router.get("/loggers")
    async def get_loggers():
        root = logging.getLogger()
        loggers: dict = {}
        for name, obj in logging.Logger.manager.loggerDict.items():
            if not isinstance(obj, logging.Logger):
                # PlaceHolder entries (not yet instantiated) – skip
                continue
            loggers[name] = {
                "configured_level": (
                    logging.getLevelName(obj.level)
                    if obj.level != logging.NOTSET
                    else "NOT_SET"
                ),
                "effective_level": logging.getLevelName(obj.getEffectiveLevel()),
            }
        loggers["root"] = {
            "configured_level": logging.getLevelName(root.level),
            "effective_level": logging.getLevelName(root.getEffectiveLevel()),
        }
        return {
            "log-level": logging.getLevelName(root.level),  # root level at top for quick scanning
            "loggers": loggers,
        }


    @router.put("/loggers", status_code=status.HTTP_204_NO_CONTENT)
    async def set_logger_level(body: _SetLevelRequest):
        level_upper = body.log_level.upper()
        if level_upper not in _VALID_LOG_LEVELS:
            return JSONResponse(
                content={"error": f"Invalid log level {body.log_level!r}. "
                                  f"Valid values: {sorted(_VALID_LOG_LEVELS)}"},
                status_code=400,
            )
        # Normalise aliases (TRACE → DEBUG, WARN → WARNING)
        normalised = _LOG_LEVEL_ALIASES.get(level_upper, level_upper)
        numeric = getattr(logging, normalised)

        # "ROOT" and "" are accepted aliases for the root logger (alongside "root")
        module = body.module_name
        target = (
            logging.getLogger()
            if module in ("ROOT", "root", "")
            else logging.getLogger(module)
        )
        target.setLevel(numeric)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
