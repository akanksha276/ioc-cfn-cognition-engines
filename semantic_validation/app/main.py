# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from common.diagnostics.router import HealthCheck, make_diagnostics_router

from .api.routes import router as api_router
from .config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Semantic Validation...")
    yield
    logger.info("Shutting down Semantic Validation...")


app = FastAPI(
    title="Semantic Validation",
    description="Evaluates alignment between agent decisions and mission goals.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")

app.include_router(
    make_diagnostics_router(
        service_name="semantic-validation",
        version="1.0.0",
        description="Semantic alignment validation for multi-agent systems",
        health_checks=[],
    ),
    prefix="/api/internal/diagnostics",
    include_in_schema=False,
)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "semantic-validation", "version": "1.0.0"}


def run_server():
    uvicorn.run(
        "semantic_validation.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
