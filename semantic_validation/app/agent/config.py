# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Threshold configuration for SAVEvaluator."""

from __future__ import annotations

from dataclasses import dataclass, field


def _setting(name: str, default: float) -> float:
    try:
        from ..config.settings import settings
        return float(getattr(settings, name, default))
    except Exception:
        return default


@dataclass
class SAVConfig:
    """Per-failure-mode severity thresholds.

    Values are read from Settings (env / .env file) and can be overridden
    directly, e.g. in tests.

    score <= high_threshold  → HIGH severity
    score <= medium_threshold → MEDIUM severity
    score >  medium_threshold → LOW severity
    """

    # Default thresholds (apply to all failure modes unless overridden)
    default_high: float = field(default_factory=lambda: _setting("sav_default_high_threshold", 0.30))
    default_medium: float = field(default_factory=lambda: _setting("sav_default_medium_threshold", 0.50))

    # Per-failure-mode overrides
    sm1_high: float = field(default_factory=lambda: _setting("sav_sm1_high_threshold", 0.25))
    sm1_medium: float = field(default_factory=lambda: _setting("sav_sm1_medium_threshold", 0.40))

    sm2_high: float = field(default_factory=lambda: _setting("sav_sm2_high_threshold", 0.30))
    sm2_medium: float = field(default_factory=lambda: _setting("sav_sm2_medium_threshold", 0.40))

    sm3_high: float = field(default_factory=lambda: _setting("sav_sm3_high_threshold", 0.30))
    sm3_medium: float = field(default_factory=lambda: _setting("sav_sm3_medium_threshold", 0.40))

    sm4_high: float = field(default_factory=lambda: _setting("sav_sm4_high_threshold", 0.30))
    sm4_medium: float = field(default_factory=lambda: _setting("sav_sm4_medium_threshold", 0.40))

    sm5_high: float = field(default_factory=lambda: _setting("sav_sm5_high_threshold", 0.30))
    sm5_medium: float = field(default_factory=lambda: _setting("sav_sm5_medium_threshold", 0.40))

    # PM (process-level) thresholds
    pm1_high: float = field(default_factory=lambda: _setting("sav_pm1_high_threshold", 0.25))
    pm1_medium: float = field(default_factory=lambda: _setting("sav_pm1_medium_threshold", 0.45))

    pm2_high: float = field(default_factory=lambda: _setting("sav_pm2_high_threshold", 0.30))
    pm2_medium: float = field(default_factory=lambda: _setting("sav_pm2_medium_threshold", 0.50))

    pm3_high: float = field(default_factory=lambda: _setting("sav_pm3_high_threshold", 0.30))
    pm3_medium: float = field(default_factory=lambda: _setting("sav_pm3_medium_threshold", 0.50))

    pm4_high: float = field(default_factory=lambda: _setting("sav_pm4_high_threshold", 0.30))
    pm4_medium: float = field(default_factory=lambda: _setting("sav_pm4_medium_threshold", 0.50))

    pm5_high: float = field(default_factory=lambda: _setting("sav_pm5_high_threshold", 0.30))
    pm5_medium: float = field(default_factory=lambda: _setting("sav_pm5_medium_threshold", 0.50))

    pm6_high: float = field(default_factory=lambda: _setting("sav_pm6_high_threshold", 0.30))
    pm6_medium: float = field(default_factory=lambda: _setting("sav_pm6_medium_threshold", 0.50))

    pm7_high: float = field(default_factory=lambda: _setting("sav_pm7_high_threshold", 0.25))
    pm7_medium: float = field(default_factory=lambda: _setting("sav_pm7_medium_threshold", 0.45))
