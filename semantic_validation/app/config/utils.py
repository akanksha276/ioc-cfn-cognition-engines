# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import litellm

from .settings import settings

logger = logging.getLogger(__name__)


def litellm_completion_compat(**kwargs: Any) -> Any:
    model = kwargs.get("model")
    if model and "bedrock" in str(model).lower():
        return litellm.completion(**kwargs)
    return asyncio.run(litellm.acompletion(**kwargs))


def get_llm_provider(model: Optional[str] = None) -> Callable[[str], str]:
    _model = model or settings.llm_model

    def _call(prompt: str) -> str:
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
        resp = litellm_completion_compat(**kwargs)
        return resp.choices[0].message.content or ""

    return _call
