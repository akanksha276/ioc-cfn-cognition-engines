# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the semantic negotiation agent."""
import pytest
from app.agent.negotiation_model import NegotiationModel, NegotiationParticipant


@pytest.fixture
def model() -> NegotiationModel:
    """A NegotiationModel with a tight step budget for fast tests."""
    return NegotiationModel(n_steps=50)


@pytest.fixture
def two_participants() -> list[NegotiationParticipant]:
    """Two participants with opposing preferences over a shared issue set."""
    return [
        NegotiationParticipant(
            id="agent_a",
            name="Agent A",
            preferences={
                "budget": {"low": 0.9, "medium": 0.5, "high": 0.1},
                "timeline": {"short": 0.8, "long": 0.2},
            },
        ),
        NegotiationParticipant(
            id="agent_b",
            name="Agent B",
            preferences={
                "budget": {"low": 0.1, "medium": 0.5, "high": 0.9},
                "timeline": {"short": 0.3, "long": 0.7},
            },
        ),
    ]


@pytest.fixture
def issues() -> list[str]:
    return ["budget", "timeline"]


@pytest.fixture
def options_per_issue() -> dict[str, list[str]]:
    return {
        "budget": ["low", "medium", "high"],
        "timeline": ["short", "long"],
    }
