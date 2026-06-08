# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Environment configuration using Pydantic Settings.
"""
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (semantic_negotiation/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # allow unrecognised env vars (e.g. LLM keys)
    )

    # CFN service URL for health check
    cfn_url: str | None = Field(default=None)

    # Service configuration
    service_name: str = Field(default="semantic_negotiation")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8089)
    log_level: str = Field(default="INFO")

    # LLM configuration (litellm provider/model format, e.g. openai/gpt-4o, anthropic/claude-sonnet-4-6)
    # For Azure: set llm_model to azure/deployment-name, llm_base_url to AZURE_OPENAI_ENDPOINT, llm_api_key to AZURE_OPENAI_API_KEY
    llm_model: str = Field(default="openai/gpt-4o")
    llm_api_key: str | None = Field(default=None)
    llm_base_url: str | None = Field(default=None)
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LiteLLM completion temperature (env: LLM_TEMPERATURE). 0 = deterministic.",
    )

    # Negotiation defaults
    negotiation_n_steps: int = Field(
        default=0,
        description=(
            "Hard cap on SAO rounds per session. 0 (default) means no cap — "
            "the step budget is computed dynamically from negotiation complexity "
            "(n_agents, n_issues, options). Set to a positive integer to enforce "
            "an upper bound regardless of what compute_n_steps() returns."
        ),
    )

    enable_local_trace: bool = Field(
        default=False,
        description=(
            "When True, commit and error envelopes are written to "
            "<cwd>/neg_trace/<session_id>/sstp_message_trace.json. "
            "Disabled by default; intended for local debugging only."
        ),
    )

    negotiator_strategy: str = Field(
        default="BoulwareTBNegotiator",
        description=(
            "NegMAS SAO negotiator class name. "
            "Available: BoulwareTBNegotiator, ConcederTBNegotiator, LinearTBNegotiator, "
            "NaiveTitForTatNegotiator, SimpleTitForTatNegotiator, "
            "AspirationNegotiator, CABNegotiator, CANNegotiator, "
            "ToughNegotiator, NiceNegotiator, MiCRONegotiator."
        ),
    )

    # ── Semantic alignment validation thresholds ──────────────────────────────
    validation_score_aligned: float = Field(default=0.75)
    validation_score_intervention: float = Field(default=0.6)
    validation_cognitive_intervention: float = Field(default=0.4)
    validation_score_high_severity: float = Field(default=0.4)
    validation_score_medium_severity: float = Field(default=0.70)
    validation_agreement_coherence_medium_threshold: float = Field(default=0.85)
    validation_constraint_fit_critical: float = Field(default=0.25)
    validation_critical_cap_single: float = Field(default=0.50)
    validation_critical_cap_multiple: float = Field(default=0.40)
    validation_weight_resolution_quality: float = Field(default=0.4)
    validation_weight_constraint_fit: float = Field(default=0.3)
    validation_weight_consistency: float = Field(default=0.2)
    validation_weight_focus_retention: float = Field(default=0.1)
    validation_weight_agreement_coherence: float = Field(default=0.1)
    validation_consistency_change_penalty: float = Field(default=0.25)
    validation_consistency_reversal_penalty: float = Field(default=0.35)
    validation_focus_drift_threshold: float = Field(default=0.25)
    validation_consistency_drift_threshold: float = Field(default=0.4)

    # ── Retry ─────────────────────────────────────────────────────────────────
    retry_max_attempts: int = Field(default=3)
    retry_eligible_failure_modes: List[str] = Field(default=["SM-1", "SM-2", "SM-4"])


# Singleton settings instance
settings = Settings()
