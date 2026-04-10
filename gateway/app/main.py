# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified app: single process, one uvicorn. Mounts ingestion and evidence as sub-apps.
Creates one shared in-memory CachingLayer at startup and passes it to both via app state.
No HTTP proxy; no separate cache server. Run: uvicorn gateway.app.main:app --host 0.0.0.0 --port 8000
With PYTHONPATH set to the directory containing gateway, ingestion, evidence, caching (e.g. /app in Docker).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Ensure parent of gateway is on path so we can import ingestion, evidence, caching
# In Docker: /app/gateway/app/main.py -> parent.parent.parent = /app
_gateway_root = Path(__file__).resolve().parent.parent.parent
if str(_gateway_root) not in sys.path:
    sys.path.insert(0, str(_gateway_root))


def _create_shared_caching_layer():
    """Build one CachingLayer with shared embed_fn and dimension for ingestion and evidence."""
    import os
    from ingestion.app.agent.knowledge_processor import EmbeddingManager
    from caching.app.agent.caching_layer import CachingLayer

    model_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None
    embedding_manager = EmbeddingManager(model_path=model_path)
    vector_dimension = 384
    metric = "l2"

    def embed_fn(text: str):
        out = embedding_manager.generate_embedding(text)
        if out is None:
            raise ValueError("Embedding returned None")
        return out

    return CachingLayer(
        vector_dimension=vector_dimension,
        metric=metric,
        embed_fn=embed_fn,
    )


# Import sub-apps once (used in lifespan and for mount)
from ingestion.app.main import app as _ingestion_app
from evidence.app.main import app as _evidence_app
from semantic_negotiation.app.main import app as _semantic_negotiation_app
# Routers for Confluence paths (no /ingestion or /evidence prefix)
from ingestion.app.api.routes import extraction_router as ingestion_extraction_router
from evidence.app.api.routes import router as evidence_api_router
from semantic_negotiation.app.api.routes import router as semantic_negotiation_api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one CachingLayer and attach it to ingestion and evidence sub-app state."""
    logger.info("Unified app startup: creating shared CachingLayer")

    app.state.cache_layer = _create_shared_caching_layer()
    app.state.rag_cache_layer = _create_shared_caching_layer()

    logger.info("Unified app: cache_layer and rag_cache_layer attached to unified app")

    # Auto-register cognition engines with management plane
    from .registration import register_cognition_engines
    await register_cognition_engines()

    yield

    logger.info("Unified app shutdown")


app = FastAPI(
    title="IoC CFN Cognitive Agents (Unified)",
    description="Single process: ingestion and evidence sub-apps with shared in-memory cache",
    version="0.2.0",
    lifespan=lifespan,
)

app.mount("/ingestion", _ingestion_app)
app.mount("/evidence", _evidence_app)

app.mount("/semantic-negotiation", _semantic_negotiation_app)

# Confluence paths: /api/knowledge-mgmt/... (no /ingestion or /evidence prefix)
app.include_router(ingestion_extraction_router)
app.include_router(evidence_api_router, prefix="/api/knowledge-mgmt")
app.include_router(semantic_negotiation_api_router, prefix="/api/semantic-negotiation")


@app.get("/health")
async def unified_health():
    """Unified app health; does not check sub-apps."""
    return {"status": "healthy", "service": "unified"}


@app.get("/")
async def root():
    return {
        "message": "IoC CFN Cognitive Agents (Unified)",
        "routes": {
            "confluence": "Confluence paths (no prefix): /api/knowledge-mgmt/extraction, /api/knowledge-mgmt/reasoning/evidence",
            "prefixed": "/ingestion/ and /evidence/ (e.g. /ingestion/api/knowledge-mgmt/extraction, /evidence/api/knowledge-mgmt/reasoning/evidence)",
        },
        "note": "Single process; shared in-memory CachingLayer; no proxy.",
    }
