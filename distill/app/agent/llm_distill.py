# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import time
from typing import Type, TypeVar

import litellm
from pydantic import BaseModel

from distill.app.config.settings import settings

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseModel)


class DistillLLMResult(BaseModel):
    distilled_description: str
    summarized_context: str


def _validate_distill_payload(parsed: DistillLLMResult) -> DistillLLMResult:
    """
    Require at least one non-empty field so caller can trust there is real model output.
    """
    desc = (parsed.distilled_description or "").strip()
    ctx = (parsed.summarized_context or "").strip()
    if not (desc or ctx):
        raise RuntimeError("LLM returned empty distillation payload.")
    return parsed


def _llm_creds() -> dict:
    out: dict = {}
    if settings.CODI_LLM_API_KEY:
        out["api_key"] = settings.CODI_LLM_API_KEY
    if settings.CODI_LLM_BASE_URL:
        out["base_url"] = settings.CODI_LLM_BASE_URL
    return out


def _model_to_tool_schema(response_model: Type[_T]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": response_model.__name__,
            "description": f"Return a structured {response_model.__name__} response.",
            "parameters": response_model.model_json_schema(),
        },
    }


def _call_chat_structured(system: str, user: str, response_model: Type[_T]) -> _T:
    tool = _model_to_tool_schema(response_model)
    kwargs: dict = {
        "model": settings.CODI_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": response_model.__name__}},
        "temperature": 0.2,
        **_llm_creds(),
    }
    resp = litellm.completion(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    finish_reason = getattr(choice, "finish_reason", None) if choice else None

    if finish_reason == "content_filter":
        raise RuntimeError(
            f"LLM response blocked by content filter (finish_reason={finish_reason!r})."
        )

    tool_calls = choice.message.tool_calls if choice and choice.message else None
    if not tool_calls:
        refusal = getattr(choice.message, "refusal", None) if choice and choice.message else None
        if refusal:
            raise RuntimeError(f"LLM refused to respond: {refusal}")
        raise RuntimeError(
            f"LLM returned no tool call (finish_reason={finish_reason!r}). "
            "Likely content filter or token limit issue."
        )

    raw = tool_calls[0].function.arguments
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON in tool arguments: {raw!r}") from exc

    return response_model(**data)


def run_distillation_llm(system: str, user: str) -> DistillLLMResult:
    """LiteLLM structured completion with exponential retry and validation."""
    max_retries = max(1, int(settings.CODI_LLM_MAX_RETRIES))
    max_backoff = max(1, int(settings.CODI_LLM_MAX_BACKOFF_SEC))
    last_error: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            parsed = _call_chat_structured(system, user, DistillLLMResult)
            out = _validate_distill_payload(parsed)
            logger.info("[CoDi] Distillation LLM ok | desc_len=%d", len(out.distilled_description or ""))
            return out
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = min(2 ** (attempt - 1), max_backoff)
                logger.warning(
                    "[CoDiDistillLLM] Attempt %d/%d failed: %s. Retrying in %ss...",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("[CoDiDistillLLM] All %d attempts failed.", max_retries)

    raise RuntimeError(
        f"[CoDiDistillLLM] run_distillation_llm failed after {max_retries} attempts"
    ) from last_error
