# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic negotiation config utilities (LiteLLM: Bedrock uses sync ``completion`` only)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import litellm

from .settings import settings

logger = logging.getLogger(__name__)


def litellm_model_uses_bedrock_sync_path(model: str | None) -> bool:
    """True when the model string indicates Bedrock (sync ``litellm.completion`` only)."""
    if not model or not str(model).strip():
        return False
    return "bedrock" in str(model).strip().lower()


def litellm_completion_compat(**kwargs: Any) -> Any:
    """Sync entry: Bedrock → ``litellm.completion``; otherwise ``litellm.acompletion`` via ``asyncio.run``."""
    model = kwargs.get("model")
    if litellm_model_uses_bedrock_sync_path(model if isinstance(model, str) else None):
        logger.debug(
            "litellm_completion_compat: sync litellm.completion for Bedrock model=%r",
            model,
        )
        return litellm.completion(**kwargs)
    return asyncio.run(litellm.acompletion(**kwargs))


async def litellm_acompletion_compat(**kwargs: Any) -> Any:
    """Async entry: Bedrock → threaded sync ``completion``; otherwise ``litellm.acompletion``."""
    model = kwargs.get("model")
    if litellm_model_uses_bedrock_sync_path(model if isinstance(model, str) else None):
        logger.debug(
            "litellm_acompletion_compat: threaded litellm.completion for Bedrock model=%r",
            model,
        )
        return await asyncio.to_thread(litellm.completion, **kwargs)
    return await litellm.acompletion(**kwargs)


def get_llm_provider(model: Optional[str] = None, token_accumulator: Any = None) -> Callable[[str], str]:
    """Return a callable(prompt) -> str backed by litellm."""
    _model = model or settings.llm_model

    def _call(prompt: str) -> str:
        import time
        kwargs: dict = {
            "model": _model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.llm_temperature,
            "max_tokens": 8000,
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        t0 = time.monotonic()
        resp = litellm_completion_compat(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        if token_accumulator is not None and hasattr(resp, "usage"):
            token_accumulator.add(resp.usage)
            token_accumulator.add_latency(latency_ms)
            token_accumulator.set_model(getattr(resp, "model", _model))
        return resp.choices[0].message.content or ""

    return _call
