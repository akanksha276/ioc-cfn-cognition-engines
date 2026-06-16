# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Client helpers for registering cognition engines with the management plane.

Use these functions when deploying cognition-engine as a service to register
with the IOC management plane.
"""
import logging

import httpx

logger = logging.getLogger(__name__)


async def register_knowledge_management_engine(
    mgmt_plane_url: str,
    engine_host: str,
    engine_port: int = 9004,
    workspace_name: str = "Default Workspace",
) -> dict:
    """
    Register the Knowledge Management Cognitive Engine with the management plane.

    Args:
        mgmt_plane_url: Management plane URL (e.g., "http://localhost:9000")
        engine_host: Host where this engine is accessible (e.g., "localhost" or "cognition-engine.example.com")
        engine_port: Port where this engine is running (default: 9004)
        workspace_name: Target workspace name (default: "Default Workspace")

    Returns:
        dict with status information

    Example:
        >>> await register_knowledge_management_engine(
        ...     mgmt_plane_url="http://localhost:9000",
        ...     engine_host="localhost",
        ...     engine_port=9004
        ... )
        {'status': 'success', 'engine': 'Knowledge Management Cognitive Engine'}
    """
    return await _register_engine(
        mgmt_plane_url=mgmt_plane_url,
        engine_host=engine_host,
        engine_port=engine_port,
        engine_name="Knowledge Management Cognitive Engine",
        workspace_name=workspace_name,
    )


async def register_semantic_alignment_engine(
    mgmt_plane_url: str,
    engine_host: str,
    engine_port: int = 9004,
    workspace_name: str = "Default Workspace",
) -> dict:
    """
    Register the Semantic Alignment Cognitive Engine with the management plane.

    Args:
        mgmt_plane_url: Management plane URL (e.g., "http://localhost:9000")
        engine_host: Host where this engine is accessible (e.g., "localhost")
        engine_port: Port where this engine is running (default: 9004)
        workspace_name: Target workspace name (default: "Default Workspace")

    Returns:
        dict with status information

    Example:
        >>> await register_semantic_alignment_engine(
        ...     mgmt_plane_url="http://localhost:9000",
        ...     engine_host="cognition-engine.prod.example.com",
        ...     engine_port=443
        ... )
        {'status': 'success', 'engine': 'Semantic Alignment Cognitive Engine'}
    """
    return await _register_engine(
        mgmt_plane_url=mgmt_plane_url,
        engine_host=engine_host,
        engine_port=engine_port,
        engine_name="Semantic Alignment Cognitive Engine",
        workspace_name=workspace_name,
    )


async def register_both_engines(
    mgmt_plane_url: str,
    engine_host: str,
    engine_port: int = 9004,
    workspace_name: str = "Default Workspace",
) -> dict:
    """
    Register both Knowledge Management and Semantic Alignment engines.

    Args:
        mgmt_plane_url: Management plane URL
        engine_host: Host where engines are accessible
        engine_port: Port where engines are running (default: 9004)
        workspace_name: Target workspace name (default: "Default Workspace")

    Returns:
        dict with results for both engines

    Example:
        >>> await register_both_engines(
        ...     mgmt_plane_url="http://localhost:9000",
        ...     engine_host="localhost"
        ... )
        {
            'knowledge_management': {'status': 'success', ...},
            'semantic_alignment': {'status': 'success', ...}
        }
    """
    km_result = await register_knowledge_management_engine(
        mgmt_plane_url, engine_host, engine_port, workspace_name
    )
    sn_result = await register_semantic_alignment_engine(
        mgmt_plane_url, engine_host, engine_port, workspace_name
    )

    return {
        "knowledge_management": km_result,
        "semantic_alignment": sn_result,
    }


# ========== Internal Implementation ==========


async def _register_engine(
    mgmt_plane_url: str,
    engine_host: str,
    engine_port: int,
    engine_name: str,
    workspace_name: str,
) -> dict:
    """Internal: Register a single engine with the management plane."""
    mgmt_url = mgmt_plane_url.rstrip("/")
    engine_url = f"{engine_host}:{engine_port}"

    try:
        # Get workspace ID
        workspace_id = await _get_workspace_id(mgmt_url, workspace_name)
        logger.info(f"Using workspace_id={workspace_id} for '{engine_name}'")

        # Register engine
        register_url = f"{mgmt_url}/api/workspaces/{workspace_id}/cognition-engines"
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

            if resp.status_code == 201:
                logger.info(f"Successfully registered '{engine_name}' at {engine_url}")
                return {
                    "status": "success",
                    "engine": engine_name,
                    "url": engine_url,
                    "workspace_id": workspace_id,
                }

            if resp.status_code == 409:
                logger.info(f"Engine '{engine_name}' already exists (409 Conflict)")
                return {
                    "status": "already_exists",
                    "engine": engine_name,
                    "url": engine_url,
                    "workspace_id": workspace_id,
                }

            # Other errors
            try:
                error_detail = resp.json()
            except Exception:
                error_detail = resp.text

            logger.error(
                f"Failed to register '{engine_name}': status={resp.status_code}, detail={error_detail}"
            )
            return {
                "status": "error",
                "engine": engine_name,
                "http_status": resp.status_code,
                "detail": error_detail,
            }

    except Exception as e:
        logger.error(f"Exception registering '{engine_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "engine": engine_name,
            "error": str(e),
        }


async def _get_workspace_id(mgmt_url: str, workspace_name: str) -> str:
    """Fetch workspace ID from management plane."""
    workspaces_url = f"{mgmt_url}/api/workspaces"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(workspaces_url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        result = resp.json()

    workspaces = result.get("workspaces", [])

    if not workspaces:
        raise ValueError("No workspaces found in management plane")

    # If only one workspace, use it
    if len(workspaces) == 1:
        return workspaces[0]["id"]

    # Multiple workspaces: search by name
    for ws in workspaces:
        if ws.get("name") == workspace_name:
            return ws["id"]

    raise ValueError(
        f"Workspace '{workspace_name}' not found. Available: {[w.get('name') for w in workspaces]}"
    )
