# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Ingestion config utilities (LiteLLM: Bedrock uses sync ``completion`` only)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)


def litellm_model_uses_bedrock_sync_path(model: str | None) -> bool:
    """True when the model string indicates Bedrock (sync ``litellm.completion`` only).

    Case-insensitive ``bedrock`` substring covers ``bedrock/``, ``aws_bedrock/``, ARNs, etc.
    """
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
