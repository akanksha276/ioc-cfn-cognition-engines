# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Client for CE registration and heartbeat management."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from .models import (
    CERegistrationRequest,
    CERegistrationResponse,
)

logger = logging.getLogger(__name__)


class CELifecycleClient:
    """
    Client for CE registration and heartbeat management.

    Features:
    - Auto-registration on startup
    - Fire-and-forget heartbeats (every 30s)
    - Graceful degradation (logs errors, never raises)
    - Background task management

    Usage:
        client = CELifecycleClient("http://localhost:9002")

        # Register CE
        request = CERegistrationRequest(
            name="Knowledge Management CE",
            url="http://ce-host:9004",
            version="1.2.3",
            kind="knowledge",
            subkind="query",
            capabilities=["ingestion"],
            metrics=["llm.token.input"],
            config={},
        )

        response = await client.register(request)
        if response:
            print(f"Registered: ce_id={response.ce_id}")
            client.start_heartbeat()  # Start background heartbeat

        # Later: cleanup
        await client.close()
    """

    def __init__(
        self,
        cfn_base_url: str,
        heartbeat_interval_sec: float = 30.0,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        """
        Initialize CE lifecycle client.

        Args:
            cfn_base_url: Base URL of CFN service (e.g. "http://localhost:9002")
            heartbeat_interval_sec: Interval between heartbeats (default 30s)
            timeout: HTTP timeout in seconds
            max_retries: Max retry attempts for registration on connection failures (default 3)
        """
        self.cfn_base_url = cfn_base_url.rstrip("/")
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.timeout = timeout
        self.max_retries = max_retries

        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Stored after successful registration
        self.ce_id: Optional[str] = None
        self.cfn_id: Optional[str] = None
        self.name: Optional[str] = None
        self.version: Optional[str] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client (thread-safe)."""
        if self._client is None:
            async with self._client_lock:
                # Double-check after acquiring lock
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client and stop heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._client:
            await self._client.aclose()
            self._client = None

    async def register(
        self,
        request: CERegistrationRequest,
    ) -> Optional[CERegistrationResponse]:
        """
        Register CE with Management Plane (via CFN).

        The CE sends registration metadata to CFN, which injects cfn_id and
        forwards to Management Plane. Management Plane generates ce_id (or
        returns existing ce_id if CE already registered) and returns response.

        Args:
            request: CE registration request containing name, version, url, etc.

        Returns:
            CERegistrationResponse if successful (contains ce_id), None otherwise.
            On failure, logs error but does not raise (graceful degradation).

        Example:
            response = await client.register(request)
            if response:
                print(f"Registered: ce_id={response.ce_id}")
                client.start_heartbeat()
            else:
                print("Registration failed, will retry")
        """
        endpoint = f"{self.cfn_base_url}/api/cognition-engines"
        payload = {
            "name": request.name,
            "url": request.url,
            "version": request.version,
            "kind": request.kind,
            "subkind": request.subkind,
            "capabilities": request.capabilities,
            "metrics": request.metrics,
            "config": request.config,
            "mas_config": request.mas_config,
            "mas_auto_associate": request.mas_auto_associate,
        }

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                client = await self._get_client()

                if attempt == 0:
                    logger.info(f"Registering CE '{request.name}' at {endpoint}")
                else:
                    logger.info(f"Retrying CE registration (attempt {attempt + 1}/{self.max_retries})")

                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    result = CERegistrationResponse(
                        ce_id=data["ce_id"],
                        cfn_id=data["cfn_id"],
                        name=data["name"],
                        version=data["version"],
                        kind=data["kind"],
                        subkind=data["subkind"],
                        enabled=data["enabled"],
                        status=data["status"],
                        created=data["created"],
                    )

                    # Store for heartbeats
                    self.ce_id = result.ce_id
                    self.cfn_id = result.cfn_id
                    self.name = result.name
                    self.version = result.version

                    action = "created" if result.created else "updated"
                    logger.info(
                        f"CE '{request.name}' {action}: ce_id={result.ce_id}, "
                        f"status={result.status}, enabled={result.enabled}"
                    )
                    return result
                else:
                    # Don't retry on non-2xx responses (e.g., validation errors)
                    logger.error(
                        f"CE registration failed: {response.status_code} - {response.text[:200]}"
                    )
                    return None

            except httpx.ConnectError as e:
                # CFN service is not available (e.g., not running or network issue).
                # Retry with exponential backoff if we haven't exhausted retries.
                is_last_attempt = (attempt == self.max_retries - 1)

                if is_last_attempt:
                    # All retries exhausted - log warning and give up gracefully.
                    # This is expected in local dev when CFN isn't started yet.
                    logger.warning(
                        f"CFN service not reachable at {self.cfn_base_url} after {self.max_retries} "
                        f"attempts — skipping CE registration"
                    )
                    return None
                else:
                    # Wait with exponential backoff: 1s, 2s, 4s
                    backoff_sec = 2 ** attempt
                    logger.debug(
                        f"CFN connection failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {backoff_sec}s..."
                    )
                    await asyncio.sleep(backoff_sec)
                    # Continue to next attempt

            except Exception as e:
                # Other errors (parsing, timeout, etc.) - don't retry
                logger.error(f"Failed to register CE (non-fatal): {e}")
                return None

    async def send_heartbeat(self) -> bool:
        """
        Send heartbeat to Management Plane (via CFN).

        Heartbeats keep the CE status as "online". If heartbeats stop,
        Management Plane marks CE as "offline" after timeout (default 2 min).

        Returns:
            True if successful, False otherwise.
            Errors are logged but not raised (fire-and-forget).

        Example:
            success = await client.send_heartbeat()
            if success:
                print("Heartbeat sent successfully")
        """
        if not self.ce_id:
            return False

        endpoint = f"{self.cfn_base_url}/api/cognition-engines/{self.ce_id}/heartbeat"

        try:
            client = await self._get_client()
            response = await client.put(endpoint)

            if response.status_code == 200:
                data = response.json()
                logger.debug(
                    f"Heartbeat sent for '{self.name}': status={data.get('status')}, "
                    f"last_seen={data.get('last_seen')}"
                )
                return True
            else:
                logger.warning(
                    f"Heartbeat failed for '{self.name}': {response.status_code} - "
                    f"{response.text[:200]}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send heartbeat for '{self.name}' (non-fatal): {e}")
            return False

    def start_heartbeat(self):
        """
        Start background heartbeat task.

        Creates an asyncio task that sends heartbeats periodically.
        Only starts if CE is registered (has ce_id).

        Example:
            response = await client.register(request)
            if response:
                client.start_heartbeat()  # Start background task
        """
        if not self.ce_id:
            logger.info(f"Heartbeat not started: ce_id not set")
            return

        if self._heartbeat_task:
            logger.warning("Heartbeat task already running")
            return

        logger.info(
            f"Starting heartbeat task for '{self.name}' (interval={self.heartbeat_interval_sec}s)"
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """
        Background task that sends heartbeats periodically.

        Runs until cancelled (on shutdown). Errors are logged but do not
        stop the loop - it continues retrying.
        """
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval_sec)
                await self.send_heartbeat()
            except asyncio.CancelledError:
                logger.info(f"Heartbeat task cancelled for '{self.name}'")
                raise
            except Exception as e:
                logger.error(f"Heartbeat failed for '{self.name}': {e}")
                # Continue loop - retry on next interval
