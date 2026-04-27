# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for offer_validation.py and the Granite-30M embedding similarity tier.

Three test categories:

1. ``TestFuzzyMatchCorrect``   — rapidfuzz tiers (1-4) snap near-misses correctly.
2. ``TestFuzzyMatchIncorrect`` — rapidfuzz correctly refuses bad/unrelated strings.
3. ``TestEmbeddingFallback``   — embedding tier (5) catches semantically similar
                                strings that rapidfuzz misses entirely.

Notes on threshold sentinel values used in tests:
- ``embed_threshold=101.0``  → disables tier 5 (max cosine score is 100)
- ``threshold=101.0``        → disables tier 4 rapidfuzz
"""

import pytest
from unittest.mock import patch

import app.agent.offer_validation as _ov
from app.agent.offer_validation import (
    snap_issue,
    snap_option,
    validate_and_snap_offer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ISSUES = ["price", "delivery speed", "warranty period"]
OPTIONS: dict[str, list[str]] = {
    "price": ["$100/unit", "$120/unit", "$150/unit"],
    "delivery speed": ["standard", "express", "overnight"],
    "warranty period": ["1 year", "2 years", "3 years"],
}


# ===========================================================================
# 1. Rapidfuzz tiers (1–4) snap near-misses correctly
# ===========================================================================

class TestFuzzyMatchCorrect:
    """Tier 1–4: exact, case-insensitive, normalised, and rapidfuzz all snap."""

    # ── Tier 1: exact ───────────────────────────────────────────────────────

    def test_snap_issue_exact(self):
        assert snap_issue("price", ISSUES) == "price"

    def test_snap_option_exact(self):
        assert snap_option("$100/unit", OPTIONS["price"]) == "$100/unit"

    # ── Tier 2: case-insensitive exact ──────────────────────────────────────

    def test_snap_issue_case_insensitive(self):
        assert snap_issue("Price", ISSUES) == "price"

    def test_snap_issue_all_caps(self):
        assert snap_issue("DELIVERY SPEED", ISSUES) == "delivery speed"

    def test_snap_option_case_insensitive(self):
        assert snap_option("EXPRESS", OPTIONS["delivery speed"]) == "express"

    def test_snap_option_mixed_case(self):
        assert snap_option("Standard", OPTIONS["delivery speed"]) == "standard"

    # ── Tier 3: normalised (underscores / hyphens / extra whitespace) ────────

    def test_snap_issue_underscores(self):
        assert snap_issue("delivery_speed", ISSUES) == "delivery speed"

    def test_snap_issue_hyphens(self):
        assert snap_issue("warranty-period", ISSUES) == "warranty period"

    def test_snap_issue_extra_whitespace(self):
        assert snap_issue("  delivery  speed  ", ISSUES) == "delivery speed"

    def test_snap_option_normalised(self):
        assert snap_option("1_year", OPTIONS["warranty period"]) == "1 year"

    # ── Tier 4: rapidfuzz ───────────────────────────────────────────────────

    def test_snap_issue_word_order_swap(self):
        # token_set_ratio handles word-order variation
        assert snap_issue("speed delivery", ISSUES) == "delivery speed"

    def test_snap_issue_typo(self):
        assert snap_issue("waranty period", ISSUES) == "warranty period"

    def test_snap_option_typo(self):
        assert snap_option("expresss", OPTIONS["delivery speed"]) == "express"

    def test_snap_option_partial_price(self):
        # "$100" is a substring of "$100/unit" — embedding tier resolves it
        # (rapidfuzz ratio ~62%, below the 80 default; requires EMBEDDING_ENABLED + embed_threshold<=100)
        _ov.EMBEDDING_ENABLED = True
        try:
            assert snap_option("$100", OPTIONS["price"], embed_threshold=75.0) == "$100/unit"
        finally:
            _ov.EMBEDDING_ENABLED = False

    # ── validate_and_snap_offer: full round-trip ─────────────────────────────

    def test_validate_full_offer_already_valid(self):
        offer = {"price": "$100/unit", "delivery speed": "express", "warranty period": "2 years"}
        snapped, problems = validate_and_snap_offer(offer, ISSUES, OPTIONS)
        assert snapped == offer
        assert problems == []

    def test_validate_offer_snaps_case_and_typo(self):
        # Issue keys have wrong case; values are typos — tier 2 and 4 should snap
        offer = {
            "Price": "$100/unit",          # tier 2: case-insensitive key snap
            "DELIVERY SPEED": "expresss",  # tier 2 key + tier 4 value typo
            "warranty period": "2 years",  # already correct
        }
        snapped, problems = validate_and_snap_offer(offer, ISSUES, OPTIONS)
        assert snapped["price"] == "$100/unit"
        assert snapped["delivery speed"] == "express"
        assert snapped["warranty period"] == "2 years"
        # At least the case-snap and typo-snap should be logged as problems
        assert len(problems) > 0

    def test_validate_offer_returns_all_issues(self):
        offer = {"price": "$120/unit", "delivery_speed": "overnight", "warranty-period": "1 year"}
        snapped, problems = validate_and_snap_offer(offer, ISSUES, OPTIONS)
        assert set(snapped.keys()) == set(ISSUES)


# ===========================================================================
# 2. Rapidfuzz correctly refuses bad / unrelated strings
# ===========================================================================

class TestFuzzyMatchIncorrect:
    """Tier 4 refuses strings that are not close enough on surface form.

    ``embed_threshold=101.0`` disables tier 5 so only tiers 1-4 are exercised.
    ``threshold=101.0`` additionally disables tier 4 for specific tests.
    """

    def test_snap_issue_completely_unrelated_returns_none(self):
        # "bandwidth" rapidfuzz scores: tsr=26 vs "delivery speed", 14 vs "price"
        # With embed disabled (101), no tier fires
        assert snap_issue("bandwidth", ISSUES, threshold=80.0, embed_threshold=101.0) is None

    def test_snap_option_completely_unrelated_returns_none(self):
        # "purple" rapidfuzz scores: ratio=46 vs "express" (best) — below 80
        assert snap_option("purple", OPTIONS["delivery speed"], threshold=80.0, embed_threshold=101.0) is None

    def test_snap_issue_empty_string_returns_none(self):
        # Empty string: token_set_ratio=0 vs all issues
        assert snap_issue("", ISSUES, threshold=80.0, embed_threshold=101.0) is None

    def test_snap_option_numeric_noise_returns_none(self):
        # "$999" rapidfuzz ratio=15 vs all price options — far below 80
        assert snap_option("$999", OPTIONS["price"], threshold=80.0, embed_threshold=101.0) is None

    def test_validate_offer_missing_issue_reported(self):
        # Offer completely omits 'warranty period' with no close key
        offer = {"price": "$100/unit", "delivery speed": "express"}
        snapped, problems = validate_and_snap_offer(
            offer, ISSUES, OPTIONS,
            issue_embed_threshold=101.0,  # disable embedding tier
        )
        assert "warranty period" not in snapped
        assert any("warranty period" in p for p in problems)

    def test_validate_offer_unresolvable_value_reported(self):
        # "xyzzy" is not close to any warranty option on surface or semantically
        offer = {
            "price": "$100/unit",
            "delivery speed": "express",
            "warranty period": "xyzzy",
        }
        snapped, problems = validate_and_snap_offer(
            offer, ISSUES, OPTIONS,
            option_embed_threshold=101.0,  # disable embedding tier
        )
        assert "warranty period" not in snapped
        assert any("warranty period" in p for p in problems)

    def test_validate_offer_non_dict_returns_empty_with_problem(self):
        # Non-dict input must return gracefully with a descriptive problem
        snapped, problems = validate_and_snap_offer(
            "not a dict",  # type: ignore[arg-type]
            ISSUES,
            OPTIONS,
        )
        assert snapped == {}
        assert any("not a dict" in p for p in problems)

    def test_snap_issue_threshold_too_high_blocks_rapidfuzz(self):
        # "waranty period" scores 97 via token_set_ratio against "warranty period"
        # Raising rapidfuzz threshold to 99 blocks tier 4;
        # embed_threshold=101 blocks tier 5 → no match
        result = snap_issue("waranty period", ISSUES, threshold=99.0, embed_threshold=101.0)
        assert result is None


# ===========================================================================
# 3. Embedding tier catches semantically similar strings rapidfuzz misses
# ===========================================================================

class TestEmbeddingFallback:
    """Tier 5 (Granite-30M) catches semantic matches when tiers 1–4 all fail.

    Pairs are chosen from measured embedding scores (see comments).
    All rapidfuzz scores for these pairs are below the 80-threshold,
    so tiers 1-4 do not fire.

    ``EMBEDDING_ENABLED`` is set to ``True`` for every test in this class via
    ``setup_method`` / ``teardown_method`` so the flag is always restored even
    on test failure.
    """

    def setup_method(self):
        _ov.EMBEDDING_ENABLED = True

    def teardown_method(self):
        _ov.EMBEDDING_ENABLED = False

    # ── snap_issue embedding matches ────────────────────────────────────────

    def test_snap_issue_synonym_cost_to_price(self):
        # "cost" rapidfuzz tsr=14 vs "price" — misses tier 4
        # embedding score = 79.2 → above 75 threshold ✓
        result = snap_issue("cost", ISSUES, threshold=80.0, embed_threshold=75.0)
        assert result == "price"

    def test_snap_issue_synonym_shipping_speed_to_delivery_speed(self):
        # "shipping speed" rapidfuzz tsr=33 vs "delivery speed" — misses tier 4
        # embedding score = 86.5 → above 75 threshold ✓
        result = snap_issue("shipping speed", ISSUES, threshold=80.0, embed_threshold=75.0)
        assert result == "delivery speed"

    def test_snap_issue_synonym_warranty_length_to_warranty_period(self):
        # "warranty length" rapidfuzz tsr=67 — misses 80-threshold
        # embedding score = 88.8 → above 75 threshold ✓
        result = snap_issue("warranty length", ISSUES, threshold=80.0, embed_threshold=75.0)
        assert result == "warranty period"

    # ── snap_option embedding matches ────────────────────────────────────────

    def test_snap_option_synonym_express_shipping_to_express(self):
        # "express shipping" rapidfuzz ratio=62 vs "express" — misses 80-threshold
        # embedding score = 85.5 → above 75 threshold ✓
        result = snap_option("express shipping", OPTIONS["delivery speed"], threshold=80.0, embed_threshold=75.0)
        assert result == "express"

    def test_snap_option_synonym_overnight_delivery_to_overnight(self):
        # "overnight delivery" rapidfuzz ratio=65 vs "overnight" — misses 80-threshold
        # embedding score = 83.5 → above 75 threshold ✓
        result = snap_option("overnight delivery", OPTIONS["delivery speed"], threshold=80.0, embed_threshold=75.0)
        assert result == "overnight"

    def test_snap_option_synonym_item_price_to_price_option(self):
        # "per-unit price" (embedding score 82.9) snaps to a valid price option
        # via embedding — we don't pin which one, just that it resolves
        result = snap_option("per-unit price", OPTIONS["price"], threshold=80.0, embed_threshold=75.0)
        assert result in OPTIONS["price"]

    # ── embedding tier is bypassed when rapidfuzz already matched ────────────

    def test_embedding_not_called_when_rapidfuzz_matches(self):
        # "expresss" (typo, score 92) should be caught by tier 4 — embedding never called
        with patch(
            "app.agent.offer_validation._embed_similarity"
        ) as mock_embed:
            result = snap_option("expresss", OPTIONS["delivery speed"], threshold=80.0)
            assert result == "express"
            mock_embed.assert_not_called()

    def test_embedding_not_called_for_tier3_normalised_match(self):
        # "delivery_speed" resolves at tier 3 (normalisation) — embedding never called
        with patch(
            "app.agent.offer_validation._embed_similarity"
        ) as mock_embed:
            result = snap_issue("delivery_speed", ISSUES)
            assert result == "delivery speed"
            mock_embed.assert_not_called()

    # ── embedding graceful degradation when model unavailable ───────────────

    def test_snap_issue_returns_none_when_model_unavailable(self):
        # Simulate model load failure: cosine_similarity returns -1.0
        with patch(
            "app.agent.offer_validation._embed_similarity", return_value=-1.0
        ):
            result = snap_issue("cost", ISSUES, threshold=80.0, embed_threshold=75.0)
            assert result is None  # gracefully unresolved, no crash

    def test_snap_option_returns_none_when_model_unavailable(self):
        with patch(
            "app.agent.offer_validation._embed_similarity", return_value=-1.0
        ):
            result = snap_option(
                "express shipping", OPTIONS["delivery speed"],
                threshold=80.0, embed_threshold=75.0,
            )
            assert result is None

    # ── full validate_and_snap_offer with embedding tier ────────────────────

    def test_validate_offer_resolves_issue_key_via_embedding(self):
        # "cost" is a semantic synonym of "price" — should snap via tier 5
        offer = {
            "cost": "$100/unit",        # "cost" → "price" via embedding (score 79.2)
            "delivery speed": "express",
            "warranty period": "2 years",
        }
        snapped, problems = validate_and_snap_offer(
            offer, ISSUES, OPTIONS,
            issue_threshold=80.0,
            issue_embed_threshold=75.0,
        )
        assert snapped.get("price") == "$100/unit"
        # The snap should be recorded as a problem entry
        assert any("cost" in p for p in problems)

    def test_validate_offer_resolves_option_value_via_embedding(self):
        # "express shipping" → "express" via embedding (score 85.5)
        offer = {
            "price": "$100/unit",
            "delivery speed": "express shipping",
            "warranty period": "2 years",
        }
        snapped, problems = validate_and_snap_offer(
            offer, ISSUES, OPTIONS,
            option_threshold=80.0,
            option_embed_threshold=75.0,
        )
        assert snapped.get("delivery speed") == "express"
        assert any("express shipping" in p for p in problems)

