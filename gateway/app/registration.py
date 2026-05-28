# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-registration logic for Cognition Engines with the management plane.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Constants
COGNITION_ENGINE_KNOWLEDGE_MANAGEMENT = "Knowledge Management Cognition Engine"
COGNITION_ENGINE_SEMANTIC_NEGOTIATION = "Semantic Negotiation Cognition Engine"
DEFAULT_WORKSPACE_NAME = "Default Workspace"


async def register_cognition_engines() -> None:
    """
    Register both Cognition Engines (Knowledge Management and Semantic Negotiation)
    with the management plane on startup.
    """
    # Get configuration from environment
    mgmt_url = os.getenv("MGMT_PLANE_URL", "").rstrip("/")
    ce_host = os.getenv("COGNITION_ENGINE_HOST", "localhost")
    ce_port = os.getenv("COGNITION_ENGINE_PORT", "9004")

    if not mgmt_url:
        logger.warning("MGMT_PLANE_URL not set, skipping cognition engine registration")
        return

    ce_url = f"{ce_host}:{ce_port}"
    logger.info(f"Starting cognition engine registration with mgmt_url={mgmt_url}, ce_url={ce_url}")

    try:
        # Fetch workspace ID
        workspace_id = await _get_workspace_id(mgmt_url)
        logger.info(f"Using workspace_id={workspace_id}")

        # Register both engines - let mgmt plane handle duplicates
        engine_names = [
            COGNITION_ENGINE_KNOWLEDGE_MANAGEMENT,
            COGNITION_ENGINE_SEMANTIC_NEGOTIATION,
        ]

        for engine_name in engine_names:
            await _register_cognition_engine(mgmt_url, workspace_id, engine_name, ce_url)

    except httpx.ConnectError:
        logger.warning(
            "Management plane not reachable at %s — skipping cognition engine "
            "registration (start mgmt on :9000 or unset MGMT_PLANE_URL for local dev)",
            mgmt_url,
        )
    except Exception as e:
        logger.warning(
            "Failed to register cognition engines with %s: %s — continuing without registration",
            mgmt_url,
            e,
        )


async def _get_workspace_id(mgmt_url: str) -> str:
    """
    Fetch workspace ID from the management plane.
    - If only one workspace exists, return that ID
    - If multiple workspaces exist, search for "Default Workspace"
    - Raises exception if no workspaces found or "Default Workspace" not found among multiple
    """
    workspaces_url = f"{mgmt_url}/api/workspaces"
    logger.info(f"Fetching workspaces from {workspaces_url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(workspaces_url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        result = resp.json()

    workspaces = result.get("workspaces", [])

    if not workspaces:
        raise ValueError("No workspaces found in management plane")

    # If only one workspace, use it
    if len(workspaces) == 1:
        workspace_id = workspaces[0]["id"]
        logger.info(f"Found single workspace: id={workspace_id}, name={workspaces[0].get('name')}")
        return workspace_id

    # Multiple workspaces: search for "Default Workspace"
    for ws in workspaces:
        if ws.get("name") == DEFAULT_WORKSPACE_NAME:
            logger.info(f"Found {DEFAULT_WORKSPACE_NAME}: id={ws['id']}")
            return ws["id"]

    raise ValueError(
        f"Multiple workspaces found but '{DEFAULT_WORKSPACE_NAME}' not found - "
        "cognition engine registration failed"
    )


async def _register_cognition_engine(
    mgmt_url: str, workspace_id: str, engine_name: str, engine_url: str
) -> None:
    """Register a single Cognition Engine with the management plane."""
    register_url = f"{mgmt_url}/api/workspaces/{workspace_id}/cognition-engines"
    logger.info(f"Registering cognition engine '{engine_name}' at {register_url}")

    payload = {
        "name": engine_name,
        "config": {
            "url": engine_url,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            register_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        # 201 = success, 409 = conflict (already exists)
        if resp.status_code == 201:
            logger.info(f"Successfully registered '{engine_name}' at {engine_url}")
            return

        if resp.status_code == 409:
            # 409 Conflict means engine already exists - ignore and continue
            try:
                result = resp.json()
            except Exception:
                result = resp.text
            logger.info(
                f"Cognition engine '{engine_name}' already exists (409 Conflict): {result}"
            )
            return

        # For any other error, log and move on (don't crash)
        try:
            result = resp.json()
        except Exception:
            result = resp.text

        logger.warning(
            f"Failed to register cognition engine '{engine_name}': "
            f"status={resp.status_code}, response={result}. Continuing anyway."
        )
