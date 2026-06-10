# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Distillation batch embeddings via fastembed (IBM Granite 30M English; Cisco-allowlisted path)."""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding

    _FASTEMBED = True
except ImportError:
    _FASTEMBED = False

_model: Optional["TextEmbedding"] = None

_DEFAULT_MODEL_NAME = "ibm-granite/granite-embedding-30m-english"


def _register_granite_in_fastembed(model_name: str, local_model_path: str) -> None:
    """Inject Granite ONNX into fastembed's registry (same pattern as Evidence ``embeddings.py``)."""
    from fastembed.text.onnx_embedding import supported_onnx_models
    from fastembed.common.model_description import DenseModelDescription, ModelSource

    if any(m.model == model_name for m in supported_onnx_models):
        return

    entry = DenseModelDescription(
        model=model_name,
        dim=384,
        description="IBM Granite 30M English embedding model (local ONNX int8)",
        license="apache-2.0",
        size_in_GB=0.03,
        sources=ModelSource(hf=model_name),
        model_file="model_optimized.onnx",
    )
    supported_onnx_models.append(entry)
    logger.debug("Registered %s in fastembed ONNX registry", model_name)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _default_local_granite_dir() -> str:
    p = _repo_root() / "granite-embedding-30m-english"
    return str(p) if p.is_dir() else ""


def _model_name() -> str:
    return (
        (os.getenv("CODI_EMBEDDING_MODEL_NAME") or os.getenv("EMBEDDING_MODEL_NAME") or "").strip()
        or _DEFAULT_MODEL_NAME
    )


def _ensure_model() -> "TextEmbedding":
    global _model
    if _model is not None:
        return _model
    if not _FASTEMBED:
        raise RuntimeError("fastembed is required for CoDi embeddings (pip install fastembed)")

    model_name = _model_name()
    local_model_path = (os.getenv("EMBEDDING_MODEL_PATH", "").strip() or _default_local_granite_dir()).strip()

    if local_model_path and os.path.isdir(local_model_path):
        logger.info("CoDi loading embedding model from local path: %s", local_model_path)
        _register_granite_in_fastembed(model_name, local_model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _model = TextEmbedding(
                model_name=model_name,
                specific_model_path=local_model_path,
            )
    else:
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
        _model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    return _model


async def embed_text_bge_small(text: str) -> List[float]:
    """Async-friendly embedding for a single concatenated batch string (Granite 384-d)."""
    if not (text or "").strip():
        return []
    loop = asyncio.get_running_loop()

    def _encode() -> List[float]:
        model = _ensure_model()
        return next(model.embed([text])).tolist()

    return await loop.run_in_executor(None, _encode)
