"""
common/metrics/models.py

Shared response models with token usage.
"""
from pydantic import BaseModel
from typing import Optional


class TokenUsage(BaseModel):
    """Token usage information from LLM calls."""

    prompt: int
    completion: int
    total: int
    model: str


class TokenUsageMeta(BaseModel):
    """Metadata including token usage, latency, and cost."""

    tokens: TokenUsage
    latency_ms: float
    cost_usd: Optional[float] = None
    timestamp: str
