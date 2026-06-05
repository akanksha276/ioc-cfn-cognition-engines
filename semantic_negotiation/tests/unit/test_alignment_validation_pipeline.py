# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SemanticAlignmentValidationPipeline (Step 4).

All tests in this file are pure unit tests — no LLM calls are made.
The LLM provider is patched to return controlled responses or disabled entirely,
so the heuristic evaluation path is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from app.agent.semantic_alignment_validation_pipeline import (
    SemanticAlignmentValidationPipeline,
    ValidationInputError,
    ValidationResult,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ── Trace builders ────────────────────────────────────────────────────────────


def _make_initiate_msg(session_id: str, content_text: str) -> Dict[str, Any]:
    return {
        "kind": "negotiate",
        "message_id": "msg-init",
        "dt_created": "2026-01-01T00:00:00Z",
        "origin": {"actor_id": "test-runner", "tenant_id": "demo"},
        "semantic_context": {
            "session_id": session_id,
            "issues": [],
            "options_per_issue": {},
            "sao_state": None,
        },
        "payload_hash": "0" * 64,
        "payload": {"content_text": content_text, "agents": []},
    }


def _make_server_msg(
    session_id: str,
    msg_id: str,
    issues: List[str],
    options_per_issue: Dict[str, List[str]],
    current_offer: Optional[Dict[str, Any]] = None,
    round_idx: int = 1,
    proposer_id: str = "agent-a",
) -> Dict[str, Any]:
    """Build a negotiation-server "respond" message in the new SSTP payload format."""
    return {
        "kind": "negotiate",
        "message_id": msg_id,
        "dt_created": "2026-01-01T00:00:01Z",
        "origin": {"actor_id": "negotiation-server", "tenant_id": session_id},
        "semantic_context": {
            "session_id": session_id,
            "issues": issues,
            "options_per_issue": options_per_issue,
            "sao_state": None,
            "sao_response": None,
        },
        "payload_hash": "0" * 64,
        "payload": {
            "action": "respond",
            "round": round_idx,
            "current_offer": current_offer or {},
            "proposer_id": proposer_id,
        },
    }


def _make_agent_msg(
    session_id: str,
    msg_id: str,
    actor_id: str,
    response: str,
    outcome: Dict[str, str],
    round_idx: int = 1,
) -> Dict[str, Any]:
    """Build an agent reply message in the new SSTP payload format.

    ``response`` is the SAO protocol response string (e.g. "ACCEPT_OFFER",
    "REJECT_OFFER").  It is stored in ``semantic_context.sao_response`` and
    also mapped to the lowercase ``payload.action`` used by the new wire format.
    """
    _action_map = {
        "ACCEPT_OFFER": "accept",
        "REJECT_OFFER": "reject",
        "COUNTER_OFFER": "counter_offer",
    }
    payload_action = _action_map.get(response, "reject")
    payload: Dict[str, Any] = {
        "action": payload_action,
        "round": round_idx,
        "issues": [],
        "options_per_issue": {},
    }
    if payload_action == "counter_offer" and outcome:
        payload["offer"] = outcome
    return {
        "kind": "negotiate",
        "message_id": msg_id,
        "dt_created": "2026-01-01T00:00:02Z",
        "origin": {"actor_id": actor_id, "tenant_id": session_id},
        "semantic_context": {
            "session_id": session_id,
            "issues": [],
            "options_per_issue": {},
            "sao_state": None,
            "sao_response": {"response": response, "outcome": outcome, "data": None},
        },
        "payload_hash": "0" * 64,
        "payload": payload,
    }


def _minimal_agreed_trace(
    session_id: str = "sess-test",
    content_text: str = "Negotiate a software contract.",
    issues: List[str] | None = None,
    options_per_issue: Dict[str, List[str]] | None = None,
    final_offer: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Build a minimal 3-message SSTP trace that passes validate_input."""
    if issues is None:
        issues = ["budget", "timeline"]
    if options_per_issue is None:
        options_per_issue = {
            "budget": ["low", "medium", "high"],
            "timeline": ["short", "standard", "long"],
        }
    if final_offer is None:
        final_offer = {"budget": "medium", "timeline": "standard"}

    return [
        _make_initiate_msg(session_id, content_text),
        _make_server_msg(session_id, "msg-s0", issues, options_per_issue, current_offer=final_offer),
        _make_agent_msg(session_id, "msg-a0", "agent-a", "ACCEPT_OFFER", final_offer),
    ]


# ── validate_input ────────────────────────────────────────────────────────────


class TestValidateInput:
    def test_raises_on_none(self):
        pipeline = SemanticAlignmentValidationPipeline()
        with pytest.raises(ValidationInputError):
            pipeline.validate_input(None)

    def test_raises_on_non_list(self):
        pipeline = SemanticAlignmentValidationPipeline()
        with pytest.raises(ValidationInputError, match="list"):
            pipeline.validate_input({"not": "a list"})

    def test_raises_on_empty_list(self):
        pipeline = SemanticAlignmentValidationPipeline()
        with pytest.raises(ValidationInputError):
            pipeline.validate_input([])

    def test_raises_on_missing_content_text(self):
        pipeline = SemanticAlignmentValidationPipeline()
        trace = _minimal_agreed_trace()
        trace[0]["payload"] = {}
        with pytest.raises(ValidationInputError, match="content_text"):
            pipeline.validate_input(trace)

    def test_raises_on_mismatched_session_ids(self):
        pipeline = SemanticAlignmentValidationPipeline()
        trace = _minimal_agreed_trace()
        trace[1]["semantic_context"]["session_id"] = "different-session"
        with pytest.raises(ValidationInputError, match="session_id"):
            pipeline.validate_input(trace)

    def test_raises_when_no_issues(self):
        pipeline = SemanticAlignmentValidationPipeline()
        trace = _minimal_agreed_trace()
        trace[1]["semantic_context"]["issues"] = []
        with pytest.raises(ValidationInputError, match="issues"):
            pipeline.validate_input(trace)

    def test_raises_when_no_options(self):
        pipeline = SemanticAlignmentValidationPipeline()
        trace = _minimal_agreed_trace()
        trace[1]["semantic_context"]["options_per_issue"] = {}
        with pytest.raises(ValidationInputError, match="options_per_issue"):
            pipeline.validate_input(trace)

    def test_raises_accept_offer_without_outcome(self):
        pipeline = SemanticAlignmentValidationPipeline()
        trace = _minimal_agreed_trace()
        trace[-1]["semantic_context"]["sao_response"]["outcome"] = {}
        with pytest.raises(ValidationInputError, match="outcome"):
            pipeline.validate_input(trace)

    def test_valid_trace_passes(self):
        pipeline = SemanticAlignmentValidationPipeline()
        pipeline.validate_input(_minimal_agreed_trace())  # should not raise


# ── run() — heuristic path (no LLM) ──────────────────────────────────────────


class TestRunHeuristic:
    """Tests for run() with LLM disabled — exercises the heuristic evaluator."""

    def _run_no_llm(self, trace):
        """Run pipeline with LLM provider patched to None."""
        with patch(
            "app.agent.semantic_alignment_validation_pipeline.get_llm_provider",
            return_value=None,
        ):
            pipeline = SemanticAlignmentValidationPipeline()
            return pipeline.run(trace)

    def test_returns_validation_result(self):
        result = self._run_no_llm(_minimal_agreed_trace())
        assert isinstance(result, ValidationResult)

    def test_result_fields_present(self):
        result = self._run_no_llm(_minimal_agreed_trace())
        assert isinstance(result.needs_intervention, bool)
        assert result.severity in ("low", "medium", "high")
        assert 0.0 <= result.alignment_score <= 1.0
        assert isinstance(result.failure_modes, list)
        assert isinstance(result.reasoning, str)
        assert result.recommendation in ("accept", "request_justification", "restart_negotiation", "escalate")

    def test_clean_agreement_produces_result(self):
        """An agreed trace should produce a valid ValidationResult without error."""
        result = self._run_no_llm(_minimal_agreed_trace())
        assert isinstance(result, ValidationResult)
        assert result.severity in ("low", "medium", "high")

    def test_timeout_sets_timed_out_flag(self):
        """A timed-out trace (no ACCEPT_OFFER) should set timed_out=True."""
        session_id = "sess-timeout"
        issues = ["budget", "timeline"]
        options_per_issue = {
            "budget": ["low", "medium", "high"],
            "timeline": ["short", "standard", "long"],
        }
        offer = {"budget": "low", "timeline": "long"}
        # Build a trace where the last response is REJECT — simulating timeout
        trace = [
            _make_initiate_msg(session_id, "Negotiate."),
            # Server message carries current_offer in payload so new_format_records is non-empty
            _make_server_msg(
                session_id, "msg-s0", issues, options_per_issue,
                current_offer=offer, round_idx=1,
            ),
            # Agent rejects — outcome=None is valid for REJECT_OFFER
            {
                **_make_agent_msg(session_id, "msg-a0", "agent-a", "REJECT_OFFER", offer),
                "semantic_context": {
                    "session_id": session_id,
                    "issues": [],
                    "options_per_issue": {},
                    "sao_state": None,
                    "sao_response": {"response": "REJECT_OFFER", "outcome": None, "data": None},
                },
            },
        ]
        result = self._run_no_llm(trace)
        assert result.timed_out is True

    def test_accept_recommendation_when_no_intervention(self):
        result = self._run_no_llm(_minimal_agreed_trace())
        if not result.needs_intervention:
            assert result.recommendation == "accept"

    def test_escalate_recommendation_when_high_severity(self):
        """Patch ACSE evaluator to return high severity and check recommendation."""
        from app.agent.acse.models import AlignmentEvaluation, Severity

        bad_eval = AlignmentEvaluation(
            aligned=False,
            alignment_score=0.2,
            issue_scores={},
            cognitive_alignment=0.3,
            failure_modes=["SM-2: budget: violates constraint"],
            needs_intervention=True,
            severity=Severity.HIGH,
            reasoning="Critical failure.",
        )
        with patch(
            "app.agent.semantic_alignment_validation_pipeline.get_llm_provider",
            return_value=None,
        ), patch(
            "app.agent.acse.evaluator.SemanticAlignmentEvaluator.evaluate",
            return_value=bad_eval,
        ):
            pipeline = SemanticAlignmentValidationPipeline()
            result = pipeline.run(_minimal_agreed_trace())

        assert result.needs_intervention is True
        assert result.severity == "high"
        assert result.recommendation == "escalate"

    def test_alignment_score_in_range(self):
        result = self._run_no_llm(_minimal_agreed_trace())
        assert 0.0 <= result.alignment_score <= 1.0

    def test_failure_modes_are_strings(self):
        result = self._run_no_llm(_minimal_agreed_trace())
        for fm in result.failure_modes:
            assert isinstance(fm, str)


# ── run() — SSTP trace extraction ────────────────────────────────────────────


class TestSstpExtraction:
    """Tests for _extract_from_sstp_trace to verify correct parsing."""

    def _extract(self, trace):
        with patch(
            "app.agent.semantic_alignment_validation_pipeline.get_llm_provider",
            return_value=None,
        ):
            pipeline = SemanticAlignmentValidationPipeline()
            return pipeline._extract_from_sstp_trace(trace)

    def test_mission_goal_extracted(self):
        trace = _minimal_agreed_trace(content_text="Build a cloud platform.")
        mission_goal, _, _, _ = self._extract(trace)
        assert mission_goal == "Build a cloud platform."

    def test_issues_extracted(self):
        issues = ["scope", "quality"]
        options = {"scope": ["core", "extended"], "quality": ["basic", "premium"]}
        trace = _minimal_agreed_trace(issues=issues, options_per_issue=options)
        _, extracted_issues, _, _ = self._extract(trace)
        assert extracted_issues == issues

    def test_options_extracted(self):
        issues = ["scope", "quality"]
        options = {"scope": ["core", "extended"], "quality": ["basic", "premium"]}
        trace = _minimal_agreed_trace(issues=issues, options_per_issue=options)
        _, _, extracted_options, _ = self._extract(trace)
        assert extracted_options == options

    def test_final_agreement_from_accept_offer(self):
        final_offer = {"budget": "high", "timeline": "short"}
        trace = _minimal_agreed_trace(final_offer=final_offer)
        _, _, _, neg_trace = self._extract(trace)
        assert neg_trace.final_agreement == final_offer

    def test_timedout_when_no_accept(self):
        trace = _minimal_agreed_trace()
        trace[-1]["semantic_context"]["sao_response"]["response"] = "REJECT_OFFER"
        trace[-1]["semantic_context"]["sao_response"]["outcome"] = None
        # Server msg already has current_offer in payload (new format) → new_format_records non-empty
        _, _, _, neg_trace = self._extract(trace)
        assert neg_trace.timedout is True

    def test_rounds_built_from_payload(self):
        """Server "respond" messages with payload.current_offer yield round records."""
        final_offer = {"budget": "medium", "timeline": "standard"}
        session_id = "sess-test"
        issues = ["budget", "timeline"]
        options_per_issue = {
            "budget": ["low", "medium", "high"],
            "timeline": ["short", "standard", "long"],
        }
        trace = [
            _make_initiate_msg(session_id, "Negotiate."),
            _make_server_msg(
                session_id, "msg-s0", issues, options_per_issue,
                current_offer=final_offer, round_idx=1,
            ),
            _make_agent_msg(session_id, "msg-a0", "agent-a", "ACCEPT_OFFER", final_offer),
        ]
        _, _, _, neg_trace = self._extract(trace)
        assert len(neg_trace.rounds) >= 1
        assert neg_trace.rounds[0].offer == final_offer


# ── run() — fixture-based smoke tests ────────────────────────────────────────


class TestRunWithFixtures:
    """Smoke tests using real fixture traces (heuristic path only)."""

    @pytest.fixture(params=list(FIXTURES_DIR.glob("*.json")) if FIXTURES_DIR.exists() else [])
    def fixture_trace(self, request):
        path = request.param
        if path.name == "gold.json":
            pytest.skip("gold.json is not a trace fixture")
        with open(path) as f:
            return json.load(f)

    def test_pipeline_runs_without_error(self, fixture_trace):
        with patch(
            "app.agent.semantic_alignment_validation_pipeline.get_llm_provider",
            return_value=None,
        ):
            pipeline = SemanticAlignmentValidationPipeline()
            result = pipeline.run(fixture_trace)
        assert isinstance(result, ValidationResult)
        assert 0.0 <= result.alignment_score <= 1.0
        assert result.severity in ("low", "medium", "high")
        assert result.recommendation in (
            "accept", "request_justification", "restart_negotiation", "escalate"
        )
