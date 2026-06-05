"""
common/metrics/models.py

Shared response models with token usage.
"""
from typing import Optional

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage information from LLM calls."""

    prompt: int
    completion: int
    total: int
    model: str


class TokenUsageMeta(BaseModel):
    """Metadata including token usage, latency, cost, and CE attribution."""

    tokens: TokenUsage
    latency_ms: float
    cost_usd: Optional[float] = None
    timestamp: str
    ce_id: Optional[str] = None  # CE that performed the operation (for CFN metrics attribution)
