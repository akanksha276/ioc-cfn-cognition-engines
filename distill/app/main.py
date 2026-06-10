# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from common.diagnostics.router import HealthCheck, make_diagnostics_router

from distill.app.api.routes import router as api_router
from distill.app.config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s...", settings.service_name)
    yield
    logger.info("Shutting down %s...", settings.service_name)


def _data_layer_configured() -> bool:
    return bool((settings.DATA_LAYER_BASE_URL or "").strip())


def get_app() -> FastAPI:
    app = FastAPI(
        title="Cognition Distillation (CoDi)",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/knowledge-mgmt")
    app.include_router(
        make_diagnostics_router(
            service_name=settings.service_name,
            version="0.1.0",
            description="Cognition Distillation (CoDi) for CFN async graph distillation",
            health_checks=[
                HealthCheck(
                    name="data_layer_configured",
                    check=_data_layer_configured,
                    critical=False,
                    external=False,
                ),
            ],
        ),
        prefix="/api/internal/diagnostics",
        include_in_schema=False,
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "distill"}

    return app


app = get_app()
