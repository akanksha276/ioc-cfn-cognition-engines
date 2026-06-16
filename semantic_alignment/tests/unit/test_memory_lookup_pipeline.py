# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared-memory lookup through :meth:`SemanticNegotiationPipeline.discover_and_generate`.

Similar spirit to ``test_callback_agents.py`` (stack-level scenario), but as **pytest** with
**no live fabric**: ``httpx.Client`` is patched so the first mission pass sees empty memory
and the second pass receives a synthetic summary of the first run's issues and options.

Run from repo root (``ioc-cfn-cognitive-agents``):

    poetry run pytest semantic_alignment/tests/unit/test_memory_lookup_pipeline.py -v

Or (set ``PYTHONPATH`` like CI):

    PYTHONPATH=semantic_alignment/app python semantic_alignment/tests/unit/test_memory_lookup_pipeline.py

Requires no Azure keys when LLM calls are stubbed below.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from app.agent.intent_discovery import fetch_shared_memory_for_intent_discovery
from app.agent.semantic_alignment import SemanticNegotiationPipeline

_MISSION = (
    "We need to align on budget and timeline for the Q2 release before Friday."
)


def _fake_llm_intent(_prompt: str) -> str:
    return json.dumps(
        {
            "negotiable_entities": [
                {"term": "budget", "reasoning": "trade-off between cost and scope"},
                {"term": "timeline", "reasoning": "deadline pressure vs quality"},
            ]
        }
    )


def _fake_llm_options(prompt: str) -> str:
    # Same option shape regardless of memory blob; tests focus on HTTP memory rounds.
    del prompt  # deterministic output
    return json.dumps(
        {
            "options_per_term": [
                {"term": "budget", "options": ["low", "medium", "high"]},
                {"term": "timeline", "options": ["aggressive", "balanced"]},
            ]
        }
    )


class _MemoryHttpHarness:
    """Tracks POSTs and returns round-1 vs round-2 JSON bodies for shared-memories query."""

    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.response_bodies: list[dict[str, Any]] = []
        self._first_snapshot: dict[str, Any] | None = None

    def _make_response(self, body: dict[str, Any]) -> MagicMock:
        self.response_bodies.append(body)
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = body

        def _raise() -> None:
            return None

        r.raise_for_status.side_effect = _raise
        return r

    def build_client_class(self):
        harness = self

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _FakeClient:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def post(self, path: str, **kwargs: Any) -> MagicMock:
                # Use **kwargs so the request body does not shadow the stdlib ``json`` module.
                payload = kwargs.get("json")
                harness.post_calls.append({"path": path, "json": payload or {}})
                n = len(harness.post_calls)
                # Two issues per options pass → two POSTs per discover_and_generate memory lookup.
                posts_first_round = 2
                if n <= posts_first_round:
                    return harness._make_response(
                        {
                            "message": "no_prior_negotiation_memory",
                            "response_id": f"mem-1-{n}",
                        }
                    )
                prior = harness._first_snapshot or {}
                msg = json.dumps(
                    {
                        "prior_issues": prior.get("issues"),
                        "prior_options": prior.get("options"),
                    }
                )
                return harness._make_response(
                    {"message": msg, "response_id": f"mem-2-{n}"}
                )

        return _FakeClient


@pytest.fixture
def pipeline_stubbed_llm() -> SemanticNegotiationPipeline:
    p = SemanticNegotiationPipeline(n_steps=5)
    p._intent_discovery._llm = _fake_llm_intent
    p._options_generation._llm = _fake_llm_options
    return p


def test_discover_and_generate_twice_memory_second_round_includes_first_results(
    pipeline_stubbed_llm: SemanticNegotiationPipeline,
) -> None:
    harness = _MemoryHttpHarness()
    fabric = {
        "fabric_node_base_url": "http://fabric.test",
        "workspace_id": "ws-mem-test",
        "mas_id": "mas-mem-test",
        "agent_names": ["Agent A", "Agent B"],
    }

    with patch(
        "app.agent.http_repo.httpx.Client",
        harness.build_client_class(),
    ):
        issues_1, options_1, _mem1 = pipeline_stubbed_llm.discover_and_generate(
            _MISSION,
            **fabric,
        )
        harness._first_snapshot = {"issues": list(issues_1), "options": dict(options_1)}

        issues_2, options_2, _mem2 = pipeline_stubbed_llm.discover_and_generate(
            _MISSION,
            **fabric,
        )

    assert issues_1 == ["budget", "timeline"]
    assert set(options_1.keys()) == {"budget", "timeline"}
    assert issues_2 == issues_1
    assert options_2 == options_1

    assert len(harness.post_calls) == 4
    for i in range(4):
        assert harness.post_calls[i]["json"]["intent"]
        assert _MISSION in harness.post_calls[i]["json"]["intent"]
    assert "budget" in harness.post_calls[0]["json"]["intent"]
    assert "timeline" in harness.post_calls[1]["json"]["intent"]

    # Second round: first POST for issue "budget" includes prior snapshot in the body.
    round2 = json.loads(harness.response_bodies[2]["message"])
    assert round2["prior_issues"] == ["budget", "timeline"]
    assert round2["prior_options"] == options_1


def test_fetch_shared_memory_for_intent_discovery_matches_query_shape() -> None:
    """Standalone helper: exactly one POST (combined intent), same path as options lookup."""
    harness = _MemoryHttpHarness()

    with patch(
        "app.agent.http_repo.httpx.Client",
        harness.build_client_class(),
    ):
        out = fetch_shared_memory_for_intent_discovery(
            "http://fabric.test",
            "ws-x",
            "mas-y",
            _MISSION,
            agent_names=["A"],
            timeout=5.0,
        )

    assert len(harness.post_calls) == 1
    assert harness.post_calls[0]["path"].endswith(
        "/api/workspaces/ws-x/multi-agentic-systems/mas-y/shared-memories/query"
    )
    assert out["evidence_message"] == "no_prior_negotiation_memory"
    assert _MISSION in harness.post_calls[0]["json"]["intent"]
    assert "A" in harness.post_calls[0]["json"]["intent"]
    assert out["source"] == "fabric_node_shared_memories_query"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
