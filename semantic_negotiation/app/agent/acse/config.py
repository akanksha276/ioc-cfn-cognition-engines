# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Algorithmic configuration for the ACSE semantic alignment evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field


def _setting(name: str, default: float) -> float:
    """Read a validation threshold from Settings, fall back to *default*."""
    try:
        from app.config.settings import settings
        return float(getattr(settings, name, default))
    except Exception:
        return default


@dataclass
class ValidationConfig:
    """Thresholds and drift definitions for the semantic alignment validation pipeline.

    When instantiated with no arguments, values are read from
    :class:`~app.config.settings.Settings` (env / .env file).
    Individual fields can be overridden directly, e.g. in tests.
    """

    # ── Alignment thresholds ──────────────────────────────────────────────────
    score_aligned: float = field(default_factory=lambda: _setting("validation_score_aligned", 0.75))
    score_intervention: float = field(default_factory=lambda: _setting("validation_score_intervention", 0.6))
    cognitive_intervention: float = field(default_factory=lambda: _setting("validation_cognitive_intervention", 0.5))
    score_high_severity: float = field(default_factory=lambda: _setting("validation_score_high_severity", 0.4))
    score_medium_severity: float = field(default_factory=lambda: _setting("validation_score_medium_severity", 0.75))

    # ── Critical-issue penalty caps ───────────────────────────────────────────
    constraint_fit_critical: float = field(default_factory=lambda: _setting("validation_constraint_fit_critical", 0.25))
    critical_cap_single: float = field(default_factory=lambda: _setting("validation_critical_cap_single", 0.50))
    critical_cap_multiple: float = field(default_factory=lambda: _setting("validation_critical_cap_multiple", 0.40))

    # ── Issue score weights (must sum to 1.0) ─────────────────────────────────
    weight_resolution_quality: float = field(default_factory=lambda: _setting("validation_weight_resolution_quality", 0.4))
    weight_constraint_fit: float = field(default_factory=lambda: _setting("validation_weight_constraint_fit", 0.3))
    weight_consistency: float = field(default_factory=lambda: _setting("validation_weight_consistency", 0.2))
    weight_focus_retention: float = field(default_factory=lambda: _setting("validation_weight_focus_retention", 0.1))

    # ── Alignment score formula ───────────────────────────────────────────────
    weight_agreement_coherence: float = field(default_factory=lambda: _setting("validation_weight_agreement_coherence", 0.1))

    # ── Drift / derailment definitions ────────────────────────────────────────
    consistency_change_penalty: float = field(default_factory=lambda: _setting("validation_consistency_change_penalty", 0.25))
    consistency_reversal_penalty: float = field(default_factory=lambda: _setting("validation_consistency_reversal_penalty", 0.35))
    focus_drift_threshold: float = field(default_factory=lambda: _setting("validation_focus_drift_threshold", 0.25))
    consistency_drift_threshold: float = field(default_factory=lambda: _setting("validation_consistency_drift_threshold", 0.4))

    # ── High-stakes detection ─────────────────────────────────────────────────
    high_stakes_keywords: frozenset = field(
        default_factory=lambda: frozenset([
            "enterprise", "critical", "infrastructure", "sla", "production",
            "multi-year", "high-stakes",
        ])
    )
