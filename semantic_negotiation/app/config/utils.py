# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Utils file for the semantic negotiation cognition agent.
"""

from typing import Callable, Optional

import litellm

from .settings import settings


def get_llm_provider(model: Optional[str] = None) -> Callable[[str], str]:
    """Return a callable(prompt) -> str backed by litellm.

    The model string uses litellm provider/model format, e.g.:
        openai/gpt-4o
        anthropic/claude-sonnet-4-6
        azure/gpt-4o
        ollama/llama3
        bedrock/anthropic.claude-3-sonnet-20240229-v1:0
    """
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
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""

    return _call
