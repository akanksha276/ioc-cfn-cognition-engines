# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified app: single process, one uvicorn. Mounts ingestion, evidence, distill, and
semantic-alignment as sub-apps.
Run: uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004
With PYTHONPATH set to the repo root containing gateway, ingestion, evidence, distill, etc. (e.g. /app in Docker).
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

_gateway_root = Path(__file__).resolve().parent.parent.parent

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Ensure parent of gateway is on path so we can import ingestion, evidence, etc.
if str(_gateway_root) not in sys.path:
    sys.path.insert(0, str(_gateway_root))


import httpx
from fastapi.responses import JSONResponse

from common.diagnostics.router import make_diagnostics_router
from distill.app.api.routes import router as distill_api_router
from distill.app.main import app as _distill_app
from evidence.app.api.routes import router as evidence_api_router
from evidence.app.main import app as _evidence_app

# Routers for Confluence paths (no /ingestion or /evidence prefix)
from ingestion.app.api.routes import extraction_router as ingestion_extraction_router
from ingestion.app.main import app as _ingestion_app
from semantic_alignment.app.api.cfn_compat import router as cfn_compat_router
from semantic_alignment.app.api.routes import router as semantic_alignment_api_router
from semantic_alignment.app.main import app as _semantic_alignment_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified app lifespan: register cognition engines on startup."""
    logger.info("Unified app startup")

    from .registration import register_cognition_engines, shutdown_lifecycle_clients
    await register_cognition_engines()

    yield

    # Shutdown: stop heartbeats and close HTTP clients
    await shutdown_lifecycle_clients()
    logger.info("Unified app shutdown")


app = FastAPI(
    title="IoC CFN Cognitive Agents (Unified)",
    description="Single process: ingestion, evidence, distill, and semantic-alignment sub-apps",
    version="0.2.0",
    lifespan=lifespan,
)

app.mount("/ingestion", _ingestion_app)
app.mount("/evidence", _evidence_app)
app.mount("/distill", _distill_app)

app.mount("/semantic-alignment", _semantic_alignment_app)

# Confluence paths: /api/knowledge-mgmt/... (no /ingestion or /evidence prefix)
app.include_router(ingestion_extraction_router)
app.include_router(evidence_api_router, prefix="/api/knowledge-mgmt")
app.include_router(distill_api_router, prefix="/api/knowledge-mgmt")


@app.get("/api/internal/diagnostics/health", include_in_schema=False)
async def aggregate_health(dependencies: bool = False):
    """Aggregate health across all sub-services."""
    overall = "UP"
    services = {}

    services["gateway"] = {"status": "UP", "checks": {}}

    # Sub-app checks via in-process ASGI transport (no network hop)
    health_path = "/api/internal/diagnostics/health"
    if dependencies:
        health_path += "?dependencies=true"
    for name, sub_app in [
        ("ingestion", _ingestion_app),
        ("evidence", _evidence_app),
        ("distill", _distill_app),
        ("semantic_alignment", _semantic_alignment_app),
    ]:
        try:
            transport = httpx.ASGITransport(app=sub_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(health_path)
            data = resp.json()
            svc_status = data.get("status", "UNKNOWN")
            services[name] = {"status": svc_status, "checks": data.get("checks", {})}
            if resp.status_code >= 500:
                overall = "DOWN"
            elif svc_status == "DEGRADED" and overall == "UP":
                overall = "DEGRADED"
        except Exception as e:
            services[name] = {"status": "UNKNOWN", "error": str(e)}
            overall = "DOWN"

    http_status = 500 if overall == "DOWN" else 200
    return JSONResponse(content={"status": overall, "services": services}, status_code=http_status)


app.include_router(
    make_diagnostics_router(
        service_name="IoC CFN Cognitive Agents (Unified)",
        version="0.2.0",
        description="Single process: ingestion, evidence, distill, and semantic-alignment sub-apps",
        include_health=False,
    ),
    prefix="/api/internal/diagnostics",
    include_in_schema=False,
)

app.include_router(semantic_alignment_api_router, prefix="/api/semantic-alignment")
# CFN-compatible routes: mirrors the Go cfn-svc binary's API so evaluation scripts
# can point to this gateway instead of the Go binary.
app.include_router(cfn_compat_router, prefix="/api")


@app.get("/health")
async def unified_health():
    """Unified app health; does not check sub-apps."""
    return {"status": "healthy", "service": "unified"}


@app.get("/")
async def root():
    return {
        "message": "IoC CFN Cognitive Agents (Unified)",
        "routes": {
            "confluence": "Confluence paths (no prefix): /api/knowledge-mgmt/extraction, /api/knowledge-mgmt/reasoning/evidence, /api/knowledge-mgmt/distillation/run",
            "prefixed": "/ingestion/, /evidence/, /distill/ (e.g. /distill/api/knowledge-mgmt/distillation/run)",
        },
        "note": "Single process; no proxy.",
    }
