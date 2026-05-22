# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Token tracking utility for aggregating LLM token usage across multiple calls.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List


@dataclass
class LLMTokenMetadata:
    """Token metadata from a single LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    cost_usd: Optional[float]
    timestamp: str


class TokenAccumulator:
    """Accumulates token usage across multiple LLM calls."""

    def __init__(self):
        self.total_prompt = 0
        self.total_completion = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.model = None
        self.call_count = 0
        self.first_timestamp = None

    def add(self, usage):
        """Add token usage from litellm response."""
        if hasattr(usage, 'prompt_tokens'):
            self.total_prompt += usage.prompt_tokens
            self.total_completion += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.call_count += 1
            if self.first_timestamp is None:
                self.first_timestamp = datetime.now(timezone.utc).isoformat()

    def add_latency(self, latency_ms: float):
        """Add latency from an LLM call."""
        self.total_latency_ms += latency_ms

    def set_model(self, model: str):
        """Set the model name (uses first seen)."""
        if self.model is None:
            self.model = model

    def to_metadata(self) -> Optional[LLMTokenMetadata]:
        """Convert accumulated tokens to metadata."""
        if self.call_count == 0:
            return None
        return LLMTokenMetadata(
            prompt_tokens=self.total_prompt,
            completion_tokens=self.total_completion,
            total_tokens=self.total_tokens,
            model=self.model or "unknown",
            latency_ms=self.total_latency_ms,
            cost_usd=None,
            timestamp=self.first_timestamp or datetime.now(timezone.utc).isoformat(),
        )
