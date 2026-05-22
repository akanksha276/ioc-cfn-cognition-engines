"""
common/metrics/cfn_client.py

Fire-and-forget metrics client for CFN API.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TokenMetric:
    """Token usage metric to post to CFN."""

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
    Fire-and-forget metrics client for CFN.

    Features:
    - Non-blocking (uses asyncio.create_task)
    - Graceful degradation (logs errors but never raises)
    - Batching (groups multiple metrics)
    - Retry with exponential backoff

    Usage:
        client = CFNMetricsClient("http://localhost:9002")

        # Fire-and-forget (doesn't block response)
        await client.post_token_metric(
            workspace_id="ws-123",
            mas_id="mas-456",
            agent_id="agent-789",
            model="gpt-4",
            prompt_tokens=150,
            completion_tokens=200,
            total_tokens=350,
            latency_ms=1234.5
        )
    """

    def __init__(
        self,
        cfn_base_url: str,
        enabled: bool = True,
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        """
        Initialize CFN metrics client.

        Args:
            cfn_base_url: Base URL of CFN service (e.g. "http://localhost:9002")
            enabled: Enable/disable metrics posting (for feature flag)
            timeout: HTTP timeout in seconds
            max_retries: Number of retries on failure
        """
        self.cfn_base_url = cfn_base_url.rstrip("/")
        self.endpoint = f"{self.cfn_base_url}/api/internal/cognition-engine/metrics"
        self.enabled = enabled
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
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
        Post token metric to CFN (fire-and-forget).

        This is non-blocking - it creates an async task and returns immediately.
        Errors are logged but never raised.
        """
        if not self.enabled:
            return

        metric = TokenMetric(
            workspace_id=workspace_id,
            mas_id=mas_id,
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation=operation,
        )

        # Fire-and-forget (non-blocking)
        asyncio.create_task(self._post_metric_internal(metric))

    async def _post_metric_internal(
        self,
        metric: TokenMetric,
        retry_count: int = 0,
    ) -> None:
        """Internal method to post metric with retry logic."""
        try:
            # Build payload
            payload = {
                "workspace_id": metric.workspace_id,
                "mas_id": metric.mas_id,
                "agent_id": metric.agent_id,
                "attributes": {
                    "service": "cognitive_agent",
                    "operation": metric.operation,
                },
                "metrics": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.prompt",
                        "value": metric.prompt_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.completion",
                        "value": metric.completion_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.total",
                        "value": metric.total_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.latency_ms",
                        "value": metric.latency_ms,
                        "attributes": {"model": metric.model},
                    },
                ],
            }

            # Add cost if available
            if metric.cost_usd is not None:
                payload["metrics"].append(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.cost_usd",
                        "value": metric.cost_usd,
                        "attributes": {"model": metric.model},
                    }
                )

            # Post to CFN
            client = await self._get_client()
            response = await client.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 202:
                logger.debug(f"Posted token metrics to CFN: {metric.total_tokens} tokens")
            elif response.status_code >= 500 and retry_count < self.max_retries:
                # Retry on server errors
                logger.warning(
                    f"CFN metrics API returned {response.status_code}, retrying... (attempt {retry_count + 1})"
                )
                await self._backoff(retry_count)
                await self._post_metric_internal(metric, retry_count + 1)
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
                await self._post_metric_internal(metric, retry_count + 1)
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


def init_metrics_client(cfn_base_url: str, enabled: bool = True) -> CFNMetricsClient:
    """
    Initialize global metrics client.
    Call once during app startup.
    """
    global _metrics_client
    _metrics_client = CFNMetricsClient(cfn_base_url, enabled=enabled)
    logger.info(f"Metrics client initialized: {cfn_base_url} (enabled={enabled})")
    return _metrics_client


def get_metrics_client() -> Optional[CFNMetricsClient]:
    """Get global metrics client (or None if not initialized)."""
    return _metrics_client
