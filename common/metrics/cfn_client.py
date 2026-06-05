"""
common/metrics/cfn_client.py

Fire-and-forget metrics client for CFN API.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """
    Single metric datapoint matching CFN API structure.

    Corresponds to CFN's MetricDataPoint:
    {
        "timestamp": "2026-05-27T10:00:00Z",  # optional, defaults to now
        "name": "llm.token.input",
        "value": 150.0,
        "attributes": {"model": "gpt-4o"}
    }
    """

    name: str
    value: float
    attributes: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


@dataclass
class TokenMetric:
    """Token usage metric to post to CFN (legacy wrapper for convenience)."""

    workspace_id: str
    mas_id: str
    agent_id: str
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: Optional[float] = None
    operation: str = "unknown"


class CFNMetricsClient:
    """
    Fire-and-forget metrics client for posting CE infrastructure metrics to CFN.

    IMPORTANT: This client is ONLY for CE infrastructure metrics. MAS operation
    metrics (tokens, latency, cost) are automatically stored by CFN when CE
    returns responses with token metadata - DO NOT post them via this client.

    Features:
    - Non-blocking (uses asyncio.create_task)
    - Graceful degradation (logs errors but never raises)
    - Batching (groups multiple metrics)
    - Retry with exponential backoff

    Usage:
        from common.metrics import CFNMetricsClient, MetricDataPoint
        from datetime import datetime, timezone
        import os

        # Initialize with CE ID (identifies this CE instance)
        ce_id = os.getenv("CE_ID", "550e8400-e29b-41d4-a716-446655440000")
        client = CFNMetricsClient(
            cfn_base_url="http://localhost:9002",
            ce_id=ce_id
        )

        # Create metrics with any names (CFN doesn't enforce naming)
        metrics = [
            MetricDataPoint(
                name="ce.queue.depth",
                value=12.0,
                attributes={"hostname": "ce-prod-01"},
                timestamp=datetime.now(timezone.utc),
            ),
            MetricDataPoint(
                name="ce.memory.usage_pct",
                value=67.5,
                attributes={"hostname": "ce-prod-01"},
            ),
        ]

        # Post batch (fire-and-forget, doesn't block response)
        await client.post_metrics_batch(
            metrics=metrics,
            attributes={"region": "us-west-2"},
        )

    Endpoint: POST /api/cognition-engines/{ceId}/metrics
    Payload: {"attributes": {...}, "metrics": [...]}
    """

    def __init__(
        self,
        cfn_base_url: str,
        ce_id: Optional[str] = None,
        enabled: bool = True,
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        """
        Initialize CFN metrics client.

        Args:
            cfn_base_url: Base URL of CFN service (e.g. "http://localhost:9002")
            ce_id: Cognition Engine UUID (identifies this CE instance). If None, metrics will be disabled.
            enabled: Enable/disable metrics posting (for feature flag)
            timeout: HTTP timeout in seconds
            max_retries: Number of retries on failure
        """
        self.cfn_base_url = cfn_base_url.rstrip("/")
        self.ce_id = ce_id
        self.endpoint = f"{self.cfn_base_url}/api/cognition-engines/{ce_id}/metrics" if ce_id else None
        self.enabled = enabled and ce_id is not None  # Disable if ce_id not provided
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client (asyncio-safe)."""
        if self._client is None:
            async with self._client_lock:
                # Double-check after acquiring lock
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def post_token_metric(
        self,
        workspace_id: str,
        mas_id: str,
        agent_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        cost_usd: Optional[float] = None,
        operation: str = "unknown",
    ) -> None:
        """
        DEPRECATED: This method is no longer needed.

        MAS operation metrics (tokens, latency, cost) are automatically stored by CFN
        when CE returns responses with token metadata. The CE should NOT post these
        metrics directly.

        This method is kept for backward compatibility but does nothing.
        """
        logger.warning(
            "post_token_metric() is deprecated - MAS metrics are stored automatically by CFN. "
            "Remove this call from your code."
        )
        return

    async def post_metrics_batch(
        self,
        metrics: List[MetricDataPoint],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Post batch of CE infrastructure metrics to CFN (fire-and-forget).

        This endpoint is for CE infrastructure metrics only (queue depth, memory, CPU).
        MAS operation metrics (tokens, latency) are stored automatically by CFN
        when CE returns responses with token metadata.

        Args:
            metrics: List of MetricDataPoint objects
            attributes: Optional batch-level attributes (merged with metric attributes)
        """
        if not self.enabled:
            return

        payload = {
            "attributes": attributes or {},
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "name": m.name,
                    "value": m.value,
                    "attributes": m.attributes or {},
                }
                for m in metrics
            ],
        }

        # Fire-and-forget (non-blocking)
        asyncio.create_task(self._post_payload_internal(payload))

    async def _post_payload_internal(
        self,
        payload: Dict[str, Any],
        retry_count: int = 0,
    ) -> None:
        """Internal method to post payload with retry logic."""
        try:
            # Post to CFN
            client = await self._get_client()
            response = await client.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 202:
                num_metrics = len(payload.get("metrics", []))
                logger.debug(f"Posted {num_metrics} metrics to CFN")
            elif response.status_code >= 500 and retry_count < self.max_retries:
                # Retry on server errors
                logger.warning(
                    f"CFN metrics API returned {response.status_code}, retrying... (attempt {retry_count + 1})"
                )
                await self._backoff(retry_count)
                await self._post_payload_internal(payload, retry_count + 1)
            else:
                logger.error(
                    f"CFN metrics API failed: {response.status_code} - {response.text[:200]}"
                )

        except httpx.TimeoutException:
            if retry_count < self.max_retries:
                logger.warning(
                    f"CFN metrics API timeout, retrying... (attempt {retry_count + 1})"
                )
                await self._backoff(retry_count)
                await self._post_payload_internal(payload, retry_count + 1)
            else:
                logger.error("CFN metrics API timeout after all retries")

        except Exception as e:
            logger.error(f"Failed to post metrics to CFN (non-fatal): {e}")

    async def _backoff(self, retry_count: int):
        """Exponential backoff between retries."""
        delay = min(2**retry_count, 10)  # max 10s
        await asyncio.sleep(delay)


# Global instance (optional - can be injected via dependency injection)
_metrics_client: Optional[CFNMetricsClient] = None


def init_metrics_client(
    cfn_base_url: str,
    ce_id: Optional[str] = None,
    enabled: bool = True,
) -> CFNMetricsClient:
    """
    Initialize global metrics client.
    Call once during app startup.

    Args:
        cfn_base_url: Base URL of CFN service (e.g. "http://localhost:9002")
        ce_id: Cognition Engine UUID (identifies this CE instance). If None, metrics will be disabled.
        enabled: Enable/disable metrics posting
    """
    global _metrics_client
    _metrics_client = CFNMetricsClient(cfn_base_url, ce_id=ce_id, enabled=enabled)
    if ce_id:
        logger.info(f"Metrics client initialized: {cfn_base_url}/api/cognition-engines/{ce_id}/metrics (enabled={enabled})")
    else:
        logger.warning("Metrics client initialized without ce_id - metrics disabled")
    return _metrics_client


def get_metrics_client() -> Optional[CFNMetricsClient]:
    """Get global metrics client (or None if not initialized)."""
    return _metrics_client
