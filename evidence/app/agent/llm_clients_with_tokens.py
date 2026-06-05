"""
Modified LLM client that returns token usage metadata.
This is a minimal example showing the hybrid approach.
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Type, TypeVar

import litellm
from pydantic import BaseModel

from ..config.settings import settings
from .llm_clients import _inc_llm_call_count, _llm_creds, _model_to_tool_schema

_T = TypeVar("_T", bound=BaseModel)


@dataclass
class TokenUsage:
    """Token usage information."""
    prompt: int
    completion: int
    total: int
    model: str


@dataclass
class LLMCallMetadata:
    """Metadata about an LLM call including tokens and latency."""
    tokens: TokenUsage
    latency_ms: float
    cost_usd: float | None
    timestamp: str


class LLMClientWithTokens:
    """
    LLM client that returns both the parsed response AND token metadata.
    This is the hybrid approach: we return tokens to the caller immediately.
    """

    def __init__(self, temperature: float, client_label: str):
        self.temperature = temperature
        self._client_label = client_label

    def call_chat_structured_with_tokens(
        self, system: str, user: str, response_model: Type[_T]
    ) -> Tuple[_T, LLMCallMetadata]:
        """
        Invoke the LLM via litellm tool_calls with a Pydantic schema.
        Returns (parsed_response, metadata) tuple.
        """
        tool = _model_to_tool_schema(response_model)
        kwargs: dict = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": response_model.__name__}},
            "temperature": self.temperature,
            **_llm_creds(),
        }

        # Track timing
        start_time = time.time()

        # Call LLM
        resp = litellm.completion(**kwargs)
        _inc_llm_call_count()

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Extract tokens
        usage = resp.usage
        token_usage = TokenUsage(
            prompt=usage.prompt_tokens,
            completion=usage.completion_tokens,
            total=usage.total_tokens,
            model=resp.model,
        )

        # Calculate cost (optional)
        try:
            cost_usd = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost_usd = None

        # Build metadata
        metadata = LLMCallMetadata(
            tokens=token_usage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Parse response (existing logic)
        choice = resp.choices[0] if resp.choices else None
        if not choice or not choice.message.tool_calls:
            raise RuntimeError("LLM returned no tool call")

        import json
        raw = choice.message.tool_calls[0].function.arguments
        data = json.loads(raw) if isinstance(raw, str) else raw
        parsed = response_model(**data)

        return parsed, metadata
