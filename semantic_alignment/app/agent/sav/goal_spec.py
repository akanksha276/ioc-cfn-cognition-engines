# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
GoalSpecExtractor — extracts mission constraints and success criteria from the
mission goal text using an LLM, with a heuristic fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional

from .models import GoalSpec
from .prompts import GOAL_SPEC_PROMPT
from .utils import safe_json_parse

logger = logging.getLogger(__name__)


class GoalSpecExtractor:
    def __init__(self, llm_provider: Optional[Callable[[str], str]] = None) -> None:
        self._llm = llm_provider

    def extract(
        self,
        mission_goal: str,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
    ) -> GoalSpec:
        if self._llm:
            try:
                prompt = (
                    GOAL_SPEC_PROMPT
                    .replace("{mission_goal}", mission_goal or "(no explicit mission goal provided)")
                    .replace("{issues}", json.dumps(issues, indent=2))
                    .replace("{options_per_issue}", json.dumps(options_per_issue, indent=2))
                )
                raw = self._llm(prompt)
                parsed = safe_json_parse(str(raw))
                if parsed:
                    constraints = list(parsed.get("mission_constraints") or [])
                    criteria = list(parsed.get("success_criteria") or [])
                    logger.debug(
                        "GoalSpecExtractor: goal=%r constraints=%s criteria=%s",
                        mission_goal[:60].strip(),
                        constraints,
                        criteria,
                    )
                    return GoalSpec(
                        original_goal=mission_goal,
                        issues=issues,
                        options_per_issue=options_per_issue,
                        mission_constraints=constraints,
                        success_criteria=criteria,
                    )
            except Exception:
                logger.warning("GoalSpecExtractor LLM call failed — using heuristic fallback", exc_info=True)

        # Heuristic fallback
        generic_constraints = []
        if mission_goal:
            generic_constraints.append(
                f"Final agreement should remain appropriate for the mission: {mission_goal}"
            )

        return GoalSpec(
            original_goal=mission_goal,
            issues=issues,
            options_per_issue=options_per_issue,
            mission_constraints=generic_constraints,
            success_criteria=[
                "Final agreement should resolve the ambiguity clearly",
                "Final agreement should preserve important mission constraints",
                "Final agreement should reduce ambiguity rather than restate it vaguely",
            ],
        )
