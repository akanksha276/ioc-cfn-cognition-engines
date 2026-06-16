# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-registration logic for Cognition Engines with the management plane.

This module handles CE registration and heartbeat management using the
new CE lifecycle API (POST /api/cognition-engines via CFN).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from common.ce_lifecycle import CELifecycleClient, CERegistrationRequest
from common.metrics.names import get_llm_metric_names

logger = logging.getLogger(__name__)

# Constants
CE_VERSION = os.getenv("CE_VERSION", "1.2.3")
CE_KNOWLEDGE_NAME = "Knowledge Management CE"
CE_SEMANTIC_NEG_NAME = "Semantic Alignment CE"
CE_DISTILLATION_NAME = "Cognition Distillation CE"


# Global lifecycle clients (one per CE type)
_lifecycle_clients: List[CELifecycleClient] = []

# CE Registry: name -> ce_id mapping
_ce_registry: Dict[str, str] = {}

# Global CFN ID (shared by all CEs in this gateway)
_cfn_id: Optional[str] = None


async def register_cognition_engines() -> None:
    """
    Register Cognition Engines with Management Plane on startup.

    This function:
    1. Creates CELifecycleClient for each CE (Knowledge Management, Semantic Alignment)
    2. Calls client.register() to register with Management Plane (via CFN)
    3. Starts heartbeat background tasks for successfully registered CEs
    4. Stores clients in global list for cleanup on shutdown

    Environment variables:
        CFN_URL: Base URL of CFN service (e.g. "http://localhost:9002"). If not set, skips registration.
        COGNITION_ENGINE_HOST: Advertised host for this service (default: "localhost")
        COGNITION_ENGINE_PORT: Advertised port for this service (default: "9004")
        CE_VERSION: CE version for registration (default: "1.2.3")
        LLM_MODEL: LLM model name for config (default: "openai/gpt-4o")
    """
    # Get configuration from environment
    cfn_url = os.getenv("CFN_URL", "").rstrip("/")
    ce_host = os.getenv("COGNITION_ENGINE_HOST", "localhost")
    ce_port = os.getenv("COGNITION_ENGINE_PORT", "9004")

    if not cfn_url:
        logger.warning(
            "CFN_URL not set - CE will run without Management Plane registration. "
            "This CE will not appear in Management Plane and metrics will not be tracked. "
            "Set CFN_URL environment variable for production deployments."
        )
        return

    ce_url = f"http://{ce_host}:{ce_port}"
    logger.info(f"Starting CE registration with cfn_url={cfn_url}, ce_url={ce_url}")

    # Define CE configurations
    # NOTE: Both CEs use LLM metrics (captured automatically by CFN from response metadata)
    # TODO: Add CE-specific metrics (kb.*, negotiation.*, ce.*) - must align with CFN expectations
    ce_configs = [
        CERegistrationRequest(
            name=CE_KNOWLEDGE_NAME,
            url=ce_url,
            version=CE_VERSION,
            kind="knowledge",
            subkind="query",
            capabilities=["ingestion", "retrieval", "extraction"],
            metrics=get_llm_metric_names(),  # LLM metrics captured by CFN
            config={
                "model": os.getenv("LLM_MODEL", "openai/gpt-4o"),
                "enable_embeddings": True,
                "enable_dedup": True,
            },
            mas_config=None,
            mas_auto_associate=True,
        ),
        CERegistrationRequest(
            name=CE_SEMANTIC_NEG_NAME,
            url=ce_url,
            version=CE_VERSION,
            kind="negotiation",
            subkind="semantic",
            capabilities=["semantic_alignment", "multi_party_coordination"],
            metrics=get_llm_metric_names(),  # LLM metrics captured by CFN
            config={
                "model": os.getenv("LLM_MODEL", "openai/gpt-4o"),
            },
            mas_config={"schedule": "0 0 * * *"},  # Daily at midnight
            mas_auto_associate=True,
        ),
        CERegistrationRequest(
            name=CE_DISTILLATION_NAME,
            url=ce_url,
            version=CE_VERSION,
            kind="knowledge",
            subkind="distillation",
            capabilities=["graph_distillation", "knowledge_summarization"],
            metrics=get_llm_metric_names(),
            config={
                "model": os.getenv("LLM_MODEL", "openai/gpt-4o"),
                "distill_mode": os.getenv("DISTILLATION_MODE", "Summary"),
            },
            mas_config={"schedule": "0 * * * *"},  # Hourly
            mas_auto_associate=True,
        ),
    ]

    # Register each CE and start heartbeats
    for ce_config in ce_configs:
        client = CELifecycleClient(
            cfn_base_url=cfn_url,
            heartbeat_interval_sec=float(os.getenv("CE_HEARTBEAT_INTERVAL_SEC", "30.0")),
            timeout=10.0,
        )

        # Attempt registration
        response = await client.register(ce_config)

        if response:
            action = "created" if response.created else "updated"
            logger.info(
                f"CE '{ce_config.name}' {action}: ce_id={response.ce_id}, "
                f"status={response.status}, enabled={response.enabled}"
            )

            # Register in global CE registry
            global _ce_registry, _cfn_id
            _ce_registry[ce_config.name] = response.ce_id

            # Store CFN ID (shared by all CEs in this gateway)
            if _cfn_id is None:
                _cfn_id = response.cfn_id
                logger.info(f"CFN ID set: cfn_id={_cfn_id}")

            # Start heartbeat background task
            client.start_heartbeat()
            logger.info(f"Heartbeat task started for '{ce_config.name}'")

            # Store client for shutdown
            _lifecycle_clients.append(client)
        else:
            logger.warning(
                f"Failed to register CE '{ce_config.name}' - "
                f"CE will operate without Management Plane visibility"
            )
            # Still store client for proper cleanup
            _lifecycle_clients.append(client)


async def shutdown_lifecycle_clients() -> None:
    """
    Shutdown all lifecycle clients on gateway shutdown.

    This function:
    1. Cancels all heartbeat background tasks
    2. Closes all HTTP clients
    3. Clears the global client list

    Called from gateway lifespan shutdown phase.
    """
    logger.info("Shutting down CE lifecycle clients...")
    for client in _lifecycle_clients:
        if client.name:
            logger.info(f"Closing lifecycle client for '{client.name}'")
        await client.close()
    _lifecycle_clients.clear()
    logger.info("CE lifecycle clients shutdown complete")


def get_ce_id(ce_name: str) -> Optional[str]:
    """
    Get CE ID for a registered CE by name.

    Args:
        ce_name: Name of the CE (e.g. "Knowledge Management CE")

    Returns:
        CE ID if CE is registered, None otherwise.

    Example:
        ce_id = get_ce_id("Knowledge Management CE")
        if ce_id:
            metrics_client = CFNMetricsClient(cfn_url, ce_id=ce_id)
    """
    return _ce_registry.get(ce_name)


def get_all_ce_ids() -> Dict[str, str]:
    """
    Get all registered CEs (name -> ce_id mapping).

    Returns:
        Dictionary mapping CE names to ce_ids.

    Example:
        ce_ids = get_all_ce_ids()
        # {"Knowledge Management CE": "uuid-aaa", "Semantic Alignment CE": "uuid-bbb"}
    """
    return _ce_registry.copy()


def get_knowledge_ce_id() -> Optional[str]:
    """
    Convenience: Get Knowledge Management CE ID.

    Returns:
        CE ID for Knowledge Management CE, or None if not registered.
    """
    return _ce_registry.get(CE_KNOWLEDGE_NAME)


def get_semantic_alignment_ce_id() -> Optional[str]:
    """
    Convenience: Get Semantic Alignment CE ID.

    Returns:
        CE ID for Semantic Alignment CE, or None if not registered.
    """
    return _ce_registry.get(CE_SEMANTIC_NEG_NAME)


def get_distillation_ce_id() -> Optional[str]:
    """Convenience: Get Cognition Distillation CE ID."""
    return _ce_registry.get(CE_DISTILLATION_NAME)


def get_cfn_id() -> Optional[str]:
    """
    Get the CFN ID for this gateway instance.

    Returns:
        CFN ID if registration succeeded, None otherwise.
    """
    return _cfn_id
