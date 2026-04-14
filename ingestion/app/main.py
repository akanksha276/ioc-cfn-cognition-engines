# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Main entry point for the Telemetry Extraction Service.

A cognitive agent that ingests OpenTelemetry (OTel) trace data and extracts 
entities, relationships, and knowledge graphs.
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from .config.settings import settings
from .api.routes import router as api_router, extraction_router
from .dependencies import get_extraction_service
from common.diagnostics.router import make_diagnostics_router, HealthCheck
from .agent.knowledge_processor import FASTEMBED_AVAILABLE


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    logger.info(f"Starting {settings.service_name}...")
    yield
    logger.info(f"Shutting down {settings.service_name}...")


# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Extraction Service",
    description="Extracts knowledge from OpenTelemetry data",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router)
app.include_router(extraction_router)
app.include_router(
    make_diagnostics_router(
        service_name=settings.service_name,
        version="1.0.0",
        description="Extracts knowledge from OpenTelemetry data",
        health_checks=[
            HealthCheck(
                name="embedding_model",
                check=lambda: FASTEMBED_AVAILABLE,
                critical=True,
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
        "message": "Knowledge Extraction Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    service = get_extraction_service()
    health_info = service.report_health_and_diagnostics()
    return JSONResponse(content=health_info)


def run_server():
    """Run the uvicorn server."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )


if __name__ == "__main__":
    run_server()

