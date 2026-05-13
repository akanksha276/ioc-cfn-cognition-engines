# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for the ACSE evaluator.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from an LLM response, tolerating markdown code fences."""
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[len("json"):].strip()
            if part.startswith("{"):
                text = part
                break

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


_VAGUE_PATTERNS = [
    r"\bbest effort\b",
    r"\bas needed\b",
    r"\bcase by case\b",
    r"\bappropriate\b",
    r"\breasonable\b",
    r"\bstandard\b",
    r"\bdefault\b",
    r"\bflexible\b",
    r"\bto be decided\b",
    r"\bdepends\b",
]


def is_vague_option(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in _VAGUE_PATTERNS)


def ordinal_index(value: str, options: List[str]) -> Optional[int]:
    try:
        return options.index(value)
    except ValueError:
        return None


def normalized_index_distance(a: str, b: str, options: List[str]) -> Optional[float]:
    if len(options) <= 1:
        return 0.0
    ia = ordinal_index(a, options)
    ib = ordinal_index(b, options)
    if ia is None or ib is None:
        return None
    return abs(ia - ib) / (len(options) - 1)
