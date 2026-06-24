# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Fuzzy issue and option-value validation for semantic negotiation offers.

LLM agents occasionally return offer dicts whose keys or values differ
slightly from the canonical issue identifiers / option labels registered at
session start (e.g. ``"Price"`` instead of ``"price"``, or ``"EXPRESS delivery"``
instead of ``"express delivery"``).  The helpers here snap such near-misses to
the nearest valid value so the offer can be accepted instead of silently
downgraded to a reject.

Matching strategy (applied in order, first match wins):

Issue keys
  1. Exact match.
  2. Case-insensitive exact match.
  3. Normalised match (strip, lower, collapse whitespace / underscores / hyphens).
  4. ``rapidfuzz.fuzz.token_set_ratio`` ≥ ``issue_threshold`` (default 80).
     Token-set ratio handles word-order variation and partial containment.
  5. Granite-30M embedding cosine similarity ≥ ``issue_embed_threshold`` (default 75).
     Semantic fallback for cases where surface form diverges completely.

Option values (per issue)
  1. Exact match.
  2. Case-insensitive exact match.
  3. Normalised match (same as above).
  4. ``rapidfuzz.fuzz.ratio`` ≥ ``option_threshold`` (default 80).
     Character-level similarity; good for typos and minor formatting differences.
  5. Granite-30M embedding cosine similarity ≥ ``option_embed_threshold`` (default 75).
     Semantic fallback for paraphrased option labels (e.g. ``"next day"`` ≈ ``"overnight"``).

``rapidfuzz`` is 35-36× faster than ``difflib.SequenceMatcher`` on short
negotiation strings.  The embedding tier is only invoked when all four lighter
tiers fail, keeping inference calls rare.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz as _fuzz

from .embedding_similarity import cosine_similarity as _embed_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag — set to True to enable Granite-30M embedding (tier 5).
# When False the embedding model is never loaded and tier 5 is skipped entirely.
# ---------------------------------------------------------------------------
EMBEDDING_ENABLED: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace, underscores, and hyphens to a single space."""
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snap_issue(
    raw_key: str,
    valid_issues: list[str],
    threshold: float = 80.0,
    embed_threshold: float = 101.0,
) -> str | None:
    """Return the best-matching valid issue for *raw_key*, or ``None``.

    Tiers 1-4 use rapidfuzz (fast, CPU-only). Tier 5 uses Granite-30M
    embedding cosine similarity when all lighter tiers fail, but is
    disabled by default (embed_threshold=101.0). Set embed_threshold<=100
    to enable it.

    Args:
        raw_key:         The issue key as returned by the LLM.
        valid_issues:    Canonical issue identifiers for this session.
        threshold:       Minimum token_set_ratio score (0-100) for tier-4 match.
        embed_threshold: Minimum cosine similarity (0-100) for tier-5 match.

    Returns:
        The matched canonical issue string, or ``None`` if no match found.
    """
    # 1. Exact
    if raw_key in valid_issues:
        return raw_key
    # 2. Case-insensitive exact
    for v in valid_issues:
        if raw_key.lower() == v.lower():
            return v
    # 3. Normalised
    raw_norm = _normalise(raw_key)
    for v in valid_issues:
        if raw_norm == _normalise(v):
            return v
    # 4. rapidfuzz token_set_ratio — handles word-order and partial containment
    best_score, best_match = 0.0, None
    for v in valid_issues:
        score = _fuzz.token_set_ratio(raw_key, v)
        if score > best_score:
            best_score, best_match = score, v
    if best_score >= threshold:
        return best_match
    # 5. Granite-30M embedding cosine similarity — semantic fallback (opt-in)
    if EMBEDDING_ENABLED:
        best_embed, best_embed_match = -1.0, None
        for v in valid_issues:
            score = _embed_similarity(raw_key, v)
            if score > best_embed:
                best_embed, best_embed_match = score, v
        if best_embed >= embed_threshold:
            logger.info(
                "snap_issue: embedding tier matched %r → %r (score=%.1f)",
                raw_key,
                best_embed_match,
                best_embed,
            )
            return best_embed_match
    return None


def snap_option(
    raw_value: Any,
    valid_options: list[str],
    threshold: float = 80.0,
    embed_threshold: float = 101.0,
) -> str | None:
    """Return the best-matching valid option for *raw_value*, or ``None``.

    Tiers 1-4 use rapidfuzz (fast, CPU-only).  Tier 5 falls back to
    Granite-30M embedding cosine similarity when all lighter tiers fail.

    Args:
        raw_value:       The option value as returned by the LLM.
        valid_options:   Canonical option labels for this issue.
        threshold:       Minimum ratio score (0-100) for tier-4 match.
        embed_threshold: Minimum cosine similarity (0-100) for tier-5 match.

    Returns:
        The matched canonical option string, or ``None`` if no match found.
    """
    s = str(raw_value)
    # 1. Exact
    if s in valid_options:
        return s
    # 2. Case-insensitive exact
    for v in valid_options:
        if s.lower() == v.lower():
            return v
    # 3. Normalised
    s_norm = _normalise(s)
    for v in valid_options:
        if s_norm == _normalise(v):
            return v
    # 4. rapidfuzz ratio — character-level similarity
    best_score, best_match = 0.0, None
    for v in valid_options:
        score = _fuzz.ratio(s, v)
        if score > best_score:
            best_score, best_match = score, v
    if best_score >= threshold:
        return best_match
    # 5. Granite-30M embedding cosine similarity — semantic fallback (opt-in)
    if EMBEDDING_ENABLED:
        best_embed, best_embed_match = -1.0, None
        for v in valid_options:
            score = _embed_similarity(s, v)
            if score > best_embed:
                best_embed, best_embed_match = score, v
        if best_embed >= embed_threshold:
            logger.info(
                "snap_option: embedding tier matched %r → %r (score=%.1f)",
                s,
                best_embed_match,
                best_embed,
            )
            return best_embed_match
    return None


def validate_and_snap_offer(
    offer_raw: dict[str, Any],
    issues: list[str],
    options_per_issue: dict[str, list[str]],
    session_id: str = "",
    agent_id: str = "",
    issue_threshold: float = 80.0,
    option_threshold: float = 80.0,
    issue_embed_threshold: float = 101.0,
    option_embed_threshold: float = 101.0,
) -> tuple[dict[str, str], list[str]]:
    """Validate *offer_raw* against canonical issues/options, snapping near-misses.

    Args:
        offer_raw:             Raw offer dict from the LLM (``{key: value, ...}``).
        issues:                Ordered list of canonical issue identifiers.
        options_per_issue:     ``{issue_id: [option, ...]}`` for every issue.
        session_id:            Used for log context only.
        agent_id:              Used for log context only.
        issue_threshold:       Min ``token_set_ratio`` score (0-100) for tier-4 issue snap.
        option_threshold:      Min ``ratio`` score (0-100) for tier-4 option snap.
        issue_embed_threshold: Min cosine similarity (0-100) for tier-5 issue snap.
        option_embed_threshold:Min cosine similarity (0-100) for tier-5 option snap.

    Returns:
        A tuple ``(snapped_offer, problems)`` where:

        * ``snapped_offer`` maps every *issues* entry to a canonical option.
          May be incomplete if some issues/values could not be resolved.
        * ``problems`` is a list of human-readable strings describing snapping
          that occurred or failures encountered.  Empty means fully valid.
    """
    if not isinstance(offer_raw, dict):
        return {}, [f"offer is not a dict (got {type(offer_raw).__name__})"]

    snapped: dict[str, str] = {}
    problems: list[str] = []

    for canonical_issue in issues:
        # ── resolve the key ───────────────────────────────────────────────
        if canonical_issue in offer_raw:
            matched_key = canonical_issue
        else:
            matched_key = None
            for raw_key in offer_raw:
                if (
                    snap_issue(
                        raw_key,
                        [canonical_issue],
                        threshold=issue_threshold,
                        embed_threshold=issue_embed_threshold,
                    )
                    is not None
                ):
                    matched_key = raw_key
                    problems.append(
                        f"issue key snapped: {raw_key!r} → {canonical_issue!r}"
                    )
                    logger.info(
                        "[%s] agent=%s issue key snapped %r → %r",
                        session_id,
                        agent_id,
                        raw_key,
                        canonical_issue,
                    )
                    break

            if matched_key is None:
                problems.append(f"issue {canonical_issue!r} missing from offer")
                logger.warning(
                    "[%s] agent=%s issue %r not found in offer keys %s",
                    session_id,
                    agent_id,
                    canonical_issue,
                    list(offer_raw.keys()),
                )
                continue

        # ── resolve the value ─────────────────────────────────────────────
        raw_value = offer_raw[matched_key]
        valid_options = options_per_issue.get(canonical_issue, [])

        if str(raw_value) not in valid_options:
            logger.info(
                "[%s] XX agent=%s offer value %r for issue %r does not match "
                "options_per_issue: %s",
                session_id,
                agent_id,
                raw_value,
                canonical_issue,
                valid_options,
            )

        snapped_value = snap_option(
            raw_value,
            valid_options,
            threshold=option_threshold,
            embed_threshold=option_embed_threshold,
        )

        if snapped_value is None:
            problems.append(
                f"option value {str(raw_value)!r} for issue {canonical_issue!r} "
                f"could not be snapped to any of {valid_options}"
            )
            logger.warning(
                "[%s] agent=%s option value %r for issue %r unresolvable; valid=%s",
                session_id,
                agent_id,
                raw_value,
                canonical_issue,
                valid_options,
            )
            continue

        if str(raw_value) != snapped_value:
            problems.append(
                f"option value snapped: {str(raw_value)!r} → {snapped_value!r} "
                f"(issue {canonical_issue!r})"
            )
            logger.info(
                "[%s] agent=%s option value snapped %r → %r for issue %r",
                session_id,
                agent_id,
                raw_value,
                snapped_value,
                canonical_issue,
            )

        snapped[canonical_issue] = snapped_value

    return snapped, problems
