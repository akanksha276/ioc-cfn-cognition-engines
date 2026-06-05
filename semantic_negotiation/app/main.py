# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Main entry point for the Semantic Negotiation Agent.

A cognitive agent that runs multi-issue bilateral and multilateral negotiations
using the NegMAS Stacked Alternating Offers (SAO) mechanism.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from common.diagnostics.router import HealthCheck, make_diagnostics_router
from common.metrics import get_metrics_client, init_metrics_client

from .api.routes import router as api_router
from .api.schemas import HealthResponse
from .config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    logger.info("Starting %s...", settings.service_name)

    # Initialize metrics client if CFN URL is configured
    if settings.cfn_url:
        init_metrics_client(settings.cfn_url, enabled=True)
        logger.info("✅ Metrics client initialized: %s", settings.cfn_url)
    else:
        logger.warning("⚠️  Metrics client disabled (CFN_URL not set)")

    yield

    # Cleanup
    client = get_metrics_client()
    if client:
        await client.close()

    logger.info("Shutting down %s...", settings.service_name)


# Initialise FastAPI app
app = FastAPI(
    title="Semantic Negotiation Agent",
    description=(
        "Runs multi-issue semantic negotiations via the NegMAS SAO mechanism. "
        "Exposes component 3 of the three-part semantic negotiation pipeline "
        "(intent discovery → options generation → negotiation model)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routes
app.include_router(api_router, prefix="/api")


def _check_cfn() -> bool:
    if not settings.cfn_url:
        return False
    try:
        import httpx

        r = httpx.get(
            f"{settings.cfn_url}/api/internal/diagnostics/health", timeout=3.0
        )
        return r.status_code < 500
    except Exception:
        return False


app.include_router(
    make_diagnostics_router(
        service_name=settings.service_name,
        version="1.0.0",
        description="Multi-issue semantic negotiation agent via NegMAS SAO mechanism",
        health_checks=[
            HealthCheck(
                name="cognition_fabric_node",
                check=_check_cfn,
                critical=True,
                external=True,
            ),
        ],
    ),
    prefix="/api/internal/diagnostics",
    include_in_schema=False,
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Semantic Negotiation Agent",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version="1.0.0",
    )


def run_server():
    """Run the uvicorn server."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
