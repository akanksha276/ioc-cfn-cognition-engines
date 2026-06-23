# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_model: str = Field(default="openai/gpt-4o")
    llm_api_key: Optional[str] = Field(default=None)
    llm_base_url: Optional[str] = Field(default=None)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8090)
    log_level: str = Field(default="INFO")

    # SAV severity thresholds (score <= high → HIGH, score <= medium → MEDIUM)
    sav_default_high_threshold: float = Field(default=0.30)
    sav_default_medium_threshold: float = Field(default=0.50)
    sav_sm1_high_threshold: float = Field(default=0.25)
    sav_sm1_medium_threshold: float = Field(default=0.40)
    sav_sm2_high_threshold: float = Field(default=0.30)
    sav_sm2_medium_threshold: float = Field(default=0.40)
    sav_sm3_high_threshold: float = Field(default=0.30)
    sav_sm3_medium_threshold: float = Field(default=0.40)
    sav_sm4_high_threshold: float = Field(default=0.30)
    sav_sm4_medium_threshold: float = Field(default=0.40)
    sav_sm5_high_threshold: float = Field(default=0.30)
    sav_sm5_medium_threshold: float = Field(default=0.40)
    # PM (process-level) thresholds
    sav_pm1_high_threshold: float = Field(default=0.25)
    sav_pm1_medium_threshold: float = Field(default=0.45)
    sav_pm2_high_threshold: float = Field(default=0.30)
    sav_pm2_medium_threshold: float = Field(default=0.50)
    sav_pm3_high_threshold: float = Field(default=0.30)
    sav_pm3_medium_threshold: float = Field(default=0.50)
    sav_pm4_high_threshold: float = Field(default=0.30)
    sav_pm4_medium_threshold: float = Field(default=0.50)
    sav_pm5_high_threshold: float = Field(default=0.30)
    sav_pm5_medium_threshold: float = Field(default=0.50)
    sav_pm6_high_threshold: float = Field(default=0.30)
    sav_pm6_medium_threshold: float = Field(default=0.50)
    sav_pm7_high_threshold: float = Field(default=0.25)
    sav_pm7_medium_threshold: float = Field(default=0.45)


settings = Settings()
