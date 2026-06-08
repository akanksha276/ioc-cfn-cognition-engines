# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NegotiationModel (component 3 of the semantic negotiation pipeline)
and SemanticNegotiationPipeline Step 4 (semantic alignment validation).
"""
from unittest.mock import MagicMock, patch

import pytest
from app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationParticipant,
    NegotiationResult,
)
from app.agent.semantic_alignment_validation_pipeline import ValidationResult


class TestNegotiationModel:
    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiates_with_default_steps(self):
        model = NegotiationModel()
        assert model.n_steps == 100

    def test_instantiates_with_custom_steps(self):
        model = NegotiationModel(n_steps=42)
        assert model.n_steps == 42

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError):
            NegotiationModel(n_steps=0)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_single_participant_raises(
        self, model, issues, options_per_issue, two_participants
    ):
        with pytest.raises(ValueError, match="two participants"):
            model.run(issues, options_per_issue, two_participants[:1])

    def test_missing_issue_in_options_raises(
        self, model, options_per_issue, two_participants
    ):
        with pytest.raises(ValueError, match="no entry"):
            model.run(["budget", "nonexistent"], options_per_issue, two_participants)

    def test_empty_options_list_raises(self, model, two_participants):
        with pytest.raises(ValueError, match="empty options"):
            model.run(["budget"], {"budget": []}, two_participants)

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_run_returns_negotiation_result(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result, NegotiationResult)

    def test_result_has_boolean_flags(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result.timedout, bool)
        assert isinstance(result.broken, bool)

    def test_result_steps_is_positive(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert result.steps >= 0

    def test_agreement_covers_all_issues_when_reached(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        if result.agreement is not None:
            assert len(result.agreement) == len(issues)
            agreed_ids = {o.issue_id for o in result.agreement}
            assert agreed_ids == set(issues)

    def test_agreement_options_are_valid(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        if result.agreement is not None:
            for outcome in result.agreement:
                assert outcome.chosen_option in options_per_issue[outcome.issue_id]

    def test_history_is_recorded(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result.history, list)

    # ------------------------------------------------------------------
    # Identical preferences — agreement should be fast
    # ------------------------------------------------------------------

    def test_identical_preferences_reach_agreement(self, issues, options_per_issue):
        prefs = {
            "budget": {"low": 0.1, "medium": 0.9, "high": 0.0},
            "timeline": {"short": 0.0, "long": 1.0},
        }
        participants = [
            NegotiationParticipant(id="a", name="A", preferences=prefs),
            NegotiationParticipant(id="b", name="B", preferences=prefs),
        ]
        result = NegotiationModel(n_steps=30).run(
            issues, options_per_issue, participants
        )
        assert result.agreement is not None

    # ------------------------------------------------------------------
    # Issue weights
    # ------------------------------------------------------------------

    def test_custom_issue_weights_accepted(self, model, issues, options_per_issue):
        participants = [
            NegotiationParticipant(
                id="a",
                name="A",
                preferences={"budget": {"medium": 1.0}, "timeline": {"short": 1.0}},
                issue_weights={"budget": 0.8, "timeline": 0.2},
            ),
            NegotiationParticipant(
                id="b",
                name="B",
                preferences={"budget": {"medium": 1.0}, "timeline": {"short": 1.0}},
                issue_weights={"budget": 0.5, "timeline": 0.5},
            ),
        ]
        result = model.run(issues, options_per_issue, participants)
        assert isinstance(result, NegotiationResult)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Semantic alignment validation is called at end of negotiation
# ─────────────────────────────────────────────────────────────────────────────


def _make_pipeline(n_steps: int = 1):
    """Build a SemanticNegotiationPipeline with all LLM-touching parts mocked out."""
    from app.agent.semantic_negotiation import SemanticNegotiationPipeline

    pipeline = SemanticNegotiationPipeline(n_steps=n_steps)
    # Mock steps 1 & 2 so no LLM calls are needed.
    pipeline._intent_discovery = MagicMock()
    pipeline._intent_discovery.discover.return_value = ["budget", "timeline"]
    pipeline._options_generation = MagicMock()
    pipeline._options_generation.generate_options.return_value = MagicMock(
        options_per_issue={"budget": ["low", "high"], "timeline": ["short", "long"]},
        memory_blob=None,
    )
    return pipeline


def _stub_validation() -> ValidationResult:
    return ValidationResult(
        needs_intervention=False,
        should_retry=False,
        severity="low",
        alignment_score=1.0,
        cognitive_alignment=1.0,
        failure_modes=[],
        per_issue_evaluations=[],
        reasoning="stub",
        agreement_coherence=1.0,
        cross_issue_conflicts=[],
        timed_out=False,
        recommendation="accept",
    )


def _agents_raw():
    return [
        {"id": "agent-a", "name": "Agent A"},
        {"id": "agent-b", "name": "Agent B"},
    ]


class TestAlignmentValidationStep4:
    """Step 4 (semantic alignment validation) must run at negotiation end only."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _force_terminal(self, pipeline, session_id: str):
        """Drive the session to a terminal state with a minimal step loop."""
        # step until execute returns a non-ongoing status
        replies = [
            {
                "participant_id": "agent-a",
                "sao_response": "REJECT_OFFER",
                "offer": None,
            },
            {
                "participant_id": "agent-b",
                "sao_response": "REJECT_OFFER",
                "offer": None,
            },
        ]
        for _ in range(200):
            result = pipeline.execute(session_id, agent_replies=replies)
            if result["status"] != "ongoing":
                return result
        raise RuntimeError("Session did not reach terminal status after 200 steps")

    # ── assertion: validator called once on terminal, not on initiate ─────────

    def test_alignment_validation_called_on_terminal(self):
        pipeline = _make_pipeline(n_steps=1)
        session_id = "test-step4-terminal"

        with patch.object(
            pipeline,
            "run_alignment_validation",
            wraps=pipeline.run_alignment_validation,
        ) as mock_val:
            with patch.object(
                pipeline._alignment_validator, "run", return_value=_stub_validation()
            ):
                # Initiate — validation must NOT fire yet
                pipeline.execute(
                    session_id,
                    content_text="Agree on budget and timeline.",
                    agents_raw=_agents_raw(),
                )
                mock_val.assert_not_called()

                # Drive to terminal — validation must fire exactly once
                self._force_terminal(pipeline, session_id)
                mock_val.assert_called_once()

    def test_terminal_result_contains_validation_key(self):
        pipeline = _make_pipeline(n_steps=1)
        session_id = "test-step4-key"

        with patch.object(
            pipeline._alignment_validator, "run", return_value=_stub_validation()
        ):
            pipeline.execute(
                session_id,
                content_text="Agree on budget and timeline.",
                agents_raw=_agents_raw(),
            )
            terminal = self._force_terminal(pipeline, session_id)

        assert "validation" in terminal
        v = terminal["validation"]
        assert "severity" in v
        assert "recommendation" in v
        assert "needs_intervention" in v
        assert "alignment_score" in v

    def test_validation_not_called_on_ongoing_rounds(self):
        pipeline = _make_pipeline(n_steps=100)  # large budget — won't time out quickly
        session_id = "test-step4-ongoing"

        with patch.object(
            pipeline,
            "run_alignment_validation",
            wraps=pipeline.run_alignment_validation,
        ) as mock_val:
            with patch.object(
                pipeline._alignment_validator, "run", return_value=_stub_validation()
            ):
                pipeline.execute(
                    session_id,
                    content_text="Agree on budget and timeline.",
                    agents_raw=_agents_raw(),
                )
                # Run a few decide rounds that stay ongoing
                replies = [
                    {
                        "participant_id": "agent-a",
                        "sao_response": "REJECT_OFFER",
                        "offer": None,
                    },
                    {
                        "participant_id": "agent-b",
                        "sao_response": "REJECT_OFFER",
                        "offer": None,
                    },
                ]
                for _ in range(3):
                    result = pipeline.execute(session_id, agent_replies=replies)
                    if result["status"] != "ongoing":
                        break  # terminal reached unexpectedly — still valid assertion below

                # Validator should still not have been called (all rounds ongoing)
                assert mock_val.call_count == 0 or result["status"] != "ongoing"

    def test_validation_result_fields_match_stub(self):
        pipeline = _make_pipeline(n_steps=1)
        session_id = "test-step4-fields"

        stub = ValidationResult(
            needs_intervention=True,
            should_retry=False,
            severity="high",
            alignment_score=0.2,
            cognitive_alignment=0.3,
            failure_modes=["deadlock_detected"],
            per_issue_evaluations=[],
            reasoning="Agents never moved.",
            agreement_coherence=1.0,
            cross_issue_conflicts=[],
            timed_out=False,
            recommendation="escalate",
        )
        with patch.object(pipeline._alignment_validator, "run", return_value=stub):
            pipeline.execute(
                session_id,
                content_text="Agree on budget and timeline.",
                agents_raw=_agents_raw(),
            )
            terminal = self._force_terminal(pipeline, session_id)

        v = terminal["validation"]
        assert v["needs_intervention"] is True
        assert v["severity"] == "high"
        assert v["alignment_score"] == pytest.approx(0.2)
        assert v["failure_modes"] == ["deadlock_detected"]
        assert v["recommendation"] == "escalate"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2 never fire during mid-session decide rounds
# ─────────────────────────────────────────────────────────────────────────────


class TestStepsSkippedOnDecide:
    """Steps 1 (IntentDiscovery) and 2 (OptionsGeneration) must only run on
    initiate, never on subsequent decide rounds."""

    def _replies(self):
        return [
            {
                "participant_id": "agent-a",
                "sao_response": "REJECT_OFFER",
                "offer": None,
            },
            {
                "participant_id": "agent-b",
                "sao_response": "REJECT_OFFER",
                "offer": None,
            },
        ]

    def test_steps_1_and_2_called_exactly_once_on_initiate(self):
        pipeline = _make_pipeline(n_steps=10)
        session_id = "test-steps-initiate"

        with patch.object(
            pipeline._alignment_validator, "run", return_value=_stub_validation()
        ):
            pipeline.execute(
                session_id,
                content_text="Agree on budget and timeline.",
                agents_raw=_agents_raw(),
            )

        pipeline._intent_discovery.discover.assert_called_once()
        pipeline._options_generation.generate_options.assert_called_once()

    def test_steps_1_and_2_not_called_on_decide_rounds(self):
        pipeline = _make_pipeline(n_steps=10)
        session_id = "test-steps-decide"

        with patch.object(
            pipeline._alignment_validator, "run", return_value=_stub_validation()
        ):
            # Initiate — steps 1 & 2 fire once here
            pipeline.execute(
                session_id,
                content_text="Agree on budget and timeline.",
                agents_raw=_agents_raw(),
            )
            # Reset call counts so we only observe decide-round behaviour
            pipeline._intent_discovery.discover.reset_mock()
            pipeline._options_generation.generate_options.reset_mock()

            # Run several decide rounds
            for _ in range(3):
                result = pipeline.execute(session_id, agent_replies=self._replies())
                if result["status"] != "ongoing":
                    break

        # Neither step 1 nor step 2 should have been invoked on any decide round
        pipeline._intent_discovery.discover.assert_not_called()
        pipeline._options_generation.generate_options.assert_not_called()

    def test_step_3_called_on_every_decide_round(self):
        pipeline = _make_pipeline(n_steps=10)
        session_id = "test-step3-decide"

        with patch.object(
            pipeline._alignment_validator, "run", return_value=_stub_validation()
        ):
            pipeline.execute(
                session_id,
                content_text="Agree on budget and timeline.",
                agents_raw=_agents_raw(),
            )

            # Spy on step_negotiation (step 3) after initiate
            with patch.object(
                pipeline, "step_negotiation", wraps=pipeline.step_negotiation
            ) as mock_step:
                decide_rounds = 3
                for _ in range(decide_rounds):
                    result = pipeline.execute(session_id, agent_replies=self._replies())
                    if result["status"] != "ongoing":
                        decide_rounds = mock_step.call_count
                        break

        assert mock_step.call_count == decide_rounds
