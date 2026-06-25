# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List, Optional

import pytest

from semantic_validation.app.agent.evaluator import SAVEvaluator
from semantic_validation.app.agent.models import (
    Participant,
    PMType,
    SAVInput,
    SeverityType,
    SMType,
)


def _make_input(
    final_decision: Optional[str] = None,
    latest_message: Optional[str] = None,
    interaction_history: Optional[List[str]] = None,
) -> SAVInput:
    return SAVInput(
        mission="Negotiate a software contract.",
        participants=[Participant(id="agent-a", name="Agent A", role="buyer")],
        final_decision=final_decision,
        latest_message=latest_message,
        interaction_history=interaction_history,
    )


# ── No inputs ─────────────────────────────────────────────────────────────────

class TestSAVEvaluatorNoInput:
    def test_returns_empty_output_when_no_inputs(self):
        evaluator = SAVEvaluator(llm_provider=lambda p: "{}")
        result = evaluator.evaluate(_make_input())
        assert result.failure_modes == []


# ── SM evaluation ─────────────────────────────────────────────────────────────

class TestSAVEvaluatorSM:
    def test_parses_sm_failure_modes(self):
        llm_response = json.dumps({
            "failure_modes": [
                {
                    "type": "SM-1",
                    "score": 0.15,
                    "reasoning": "No concrete definition was provided.",
                }
            ]
        })
        evaluator = SAVEvaluator(llm_provider=lambda p: llm_response)
        result = evaluator.evaluate(_make_input(final_decision="deliver within N hours"))
        assert len(result.failure_modes) == 1
        fm = result.failure_modes[0]
        assert fm.type == SMType.SM_1
        assert fm.score == 0.15
        assert fm.severity.type == SeverityType.HIGH
        # description is injected from type (static taxonomy), not from LLM output
        assert fm.description  # non-empty
        assert "ambiguous" in fm.description.lower() or "vague" in fm.description.lower()
        assert fm.reasoning == "No concrete definition was provided."

    def test_returns_empty_on_llm_failure(self):
        def failing_llm(p):
            raise RuntimeError("LLM unavailable")

        evaluator = SAVEvaluator(llm_provider=failing_llm)
        result = evaluator.evaluate(_make_input(final_decision="some decision"))
        assert result.failure_modes == []

    def test_returns_empty_on_invalid_json(self):
        evaluator = SAVEvaluator(llm_provider=lambda p: "not valid json")
        result = evaluator.evaluate(_make_input(final_decision="some decision"))
        assert result.failure_modes == []

    def test_empty_failure_modes_when_aligned(self):
        llm_response = json.dumps({"failure_modes": []})
        evaluator = SAVEvaluator(llm_provider=lambda p: llm_response)
        result = evaluator.evaluate(_make_input(final_decision="deliver by 10am next Monday"))
        assert result.failure_modes == []


# ── PM evaluation ─────────────────────────────────────────────────────────────

class TestSAVEvaluatorPM:
    def test_parses_pm_failure_modes_from_latest_message(self):
        llm_response = json.dumps({
            "failure_modes": [
                {
                    "type": "PM-3",
                    "score": 0.20,
                    "description": "Repeated same offer across 3 rounds.",
                    "reasoning": "Agent A keeps proposing identical option.",
                }
            ]
        })
        evaluator = SAVEvaluator(llm_provider=lambda p: llm_response)
        result = evaluator.evaluate(_make_input(latest_message="I propose option X again."))
        assert len(result.failure_modes) == 1
        fm = result.failure_modes[0]
        assert fm.type == PMType.PM_3
        assert fm.score == 0.20
        assert fm.severity.type == SeverityType.HIGH

    def test_pm_skipped_when_only_history_no_latest_message(self):
        """PM evaluation requires latest_message. History alone is not enough."""
        llm_called = {"yes": False}

        def llm(p: str) -> str:
            llm_called["yes"] = True
            return json.dumps({"failure_modes": [{"type": "PM-1", "score": 0.25, "reasoning": "..."}]})

        evaluator = SAVEvaluator(llm_provider=llm)
        result = evaluator.evaluate(_make_input(
            interaction_history=["round 1: ...", "round 2: ...", "round 3: ..."]
        ))
        # Neither SM (no final_decision) nor PM (no latest_message) should run
        assert result.failure_modes == []
        assert not llm_called["yes"]

    def test_returns_empty_on_pm_llm_failure(self):
        def failing_llm(p):
            raise RuntimeError("LLM unavailable")

        evaluator = SAVEvaluator(llm_provider=failing_llm)
        result = evaluator.evaluate(_make_input(latest_message="hi"))
        assert result.failure_modes == []


# ── Concurrent SM + PM ───────────────────────────────────────────────────────

class TestSAVEvaluatorConcurrent:
    def test_runs_both_when_both_inputs_present(self):
        """When both final_decision and latest_message are present, SAV runs
        both SM and PM evaluations and merges results."""
        call_count = {"n": 0}

        def llm(prompt: str) -> str:
            call_count["n"] += 1
            # SM prompt mentions FINAL DECISION, PM prompt mentions LATEST MESSAGE
            if "FINAL DECISION" in prompt and "deliver" in prompt:
                return json.dumps({
                    "failure_modes": [{
                        "type": "SM-1", "score": 0.20,
                        "description": "vague decision",
                        "reasoning": "no specifics",
                    }]
                })
            elif "LATEST MESSAGE" in prompt:
                return json.dumps({
                    "failure_modes": [{
                        "type": "PM-3", "score": 0.25,
                        "description": "repetition",
                        "reasoning": "same offer twice",
                    }]
                })
            return "{}"

        evaluator = SAVEvaluator(llm_provider=llm)
        result = evaluator.evaluate(_make_input(
            final_decision="deliver eventually",
            latest_message="I propose X again",
            interaction_history=["round 1: ..."],
        ))

        # Both SM and PM produced failure modes
        assert call_count["n"] == 2  # SM and PM each called LLM once
        types = {fm.type for fm in result.failure_modes}
        assert SMType.SM_1 in types
        assert PMType.PM_3 in types

    def test_only_sm_when_only_final_decision(self):
        call_count = {"n": 0}

        def llm(prompt: str) -> str:
            call_count["n"] += 1
            return json.dumps({"failure_modes": []})

        evaluator = SAVEvaluator(llm_provider=llm)
        evaluator.evaluate(_make_input(final_decision="done"))
        assert call_count["n"] == 1  # only SM called

    def test_only_pm_when_only_latest_message(self):
        call_count = {"n": 0}

        def llm(prompt: str) -> str:
            call_count["n"] += 1
            return json.dumps({"failure_modes": []})

        evaluator = SAVEvaluator(llm_provider=llm)
        evaluator.evaluate(_make_input(latest_message="some message"))
        assert call_count["n"] == 1  # only PM called
