# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from .api.routes import router as api_router
from .config.settings import settings
from common.diagnostics.router import make_diagnostics_router, HealthCheck
from common.metrics import init_metrics_client, get_metrics_client


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s...", settings.service_name)

    # Initialize metrics client if CFN URL is configured
    if settings.CFN_URL:
        init_metrics_client(settings.CFN_URL, enabled=True)
        logger.info("✅ Metrics client initialized: %s", settings.CFN_URL)
    else:
        logger.warning("⚠️  Metrics client disabled (CFN_URL not set)")

    yield

    # Cleanup
    client = get_metrics_client()
    if client:
        await client.close()

    logger.info("Shutting down %s...", settings.service_name)


def get_app() -> FastAPI:
    app = FastAPI(
        title="Evidence Gathering Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/knowledge-mgmt")
    def _check_cfn() -> bool:
        if not settings.CFN_URL:
            return False
        try:
            import httpx
            r = httpx.get(f"{settings.CFN_URL}/api/internal/diagnostics/health", timeout=3.0)
            return r.status_code < 500
        except Exception:
            return False

    app.include_router(
        make_diagnostics_router(
            service_name=settings.service_name,
            version="0.1.0",
            description="Evidence gathering agent for CFN cognitive pipeline",
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

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = get_app()
