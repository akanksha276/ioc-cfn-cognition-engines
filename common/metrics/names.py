# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Centralized metric name definitions.

IMPORTANT: Metric names MUST match CFN's hardcoded expectations.
See: ioc-cfn-svc/pkg/app/handlers_metrics.go:storeTokenMetricsAsync()
"""


class LLMMetrics:
    """
    LLM operation metrics - MUST match CFN's hardcoded names.

    These metrics are automatically captured by CFN from CE response metadata
    (TokenUsageMeta) and stored to TimescaleDB.

    Reference: ioc-cfn-svc/pkg/app/handlers_metrics.go:storeTokenMetricsAsync()
    """
    TOKENS_PROMPT = "llm.tokens.prompt"
    TOKENS_COMPLETION = "llm.tokens.completion"
    TOKENS_TOTAL = "llm.tokens.total"
    LATENCY_MS = "llm.latency_ms"
    COST_USD = "llm.cost_usd"


def get_llm_metric_names():
    """
    Get list of LLM metric names for CE registration.

    Returns:
        List of metric names that CFN will automatically capture from CE responses.

    Example:
        ce_config = CERegistrationRequest(
            name="My CE",
            metrics=get_llm_metric_names(),  # Register LLM metrics
        )
    """
    return [
        LLMMetrics.TOKENS_PROMPT,
        LLMMetrics.TOKENS_COMPLETION,
        LLMMetrics.TOKENS_TOTAL,
        LLMMetrics.LATENCY_MS,
        LLMMetrics.COST_USD,
    ]


# TODO: Add more metric types as needed:
# - Knowledge Base metrics (kb.documents.indexed, kb.search.latency_ms, etc.)
# - Negotiation metrics (negotiation.rounds, negotiation.agreement_reached, etc.)
# - CE infrastructure metrics (ce.queue.depth, ce.cache.hits, etc.)
#
# All new metrics must be aligned with CFN expectations and registered during CE setup.
