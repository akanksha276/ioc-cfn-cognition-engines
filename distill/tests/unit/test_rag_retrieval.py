# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from distill.app.agent.rag_retrieval import retrieve_rag_top_k


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_repo():
    assert await retrieve_rag_top_k(None, "intent", 3, "rid") == []


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_no_search_method():
    class NoSearch:
        pass

    assert await retrieve_rag_top_k(NoSearch(), "intent", 3, "rid") == []


@pytest.mark.asyncio
async def test_retrieve_normalizes_and_display_line(monkeypatch):
    class FakeRepo:
        async def search_similar_rag(self, **kwargs):
            assert kwargs["embedded_text"] == "q"
            assert kwargs["top_k"] == 2
            assert kwargs["request_id"] == "r1"
            return [
                {
                    "embedded_text": "  hello  ",
                    "timestamp": "t1",
                    "domain": "d1",
                    "score": "0.5",
                }
            ]

    hits = await retrieve_rag_top_k(FakeRepo(), "q", 2, "r1", embedding_vector=[0.1, 0.2])
    assert len(hits) == 1
    assert hits[0]["embedded_text"] == "hello"
    assert hits[0]["score"] == 0.5
    assert "[1]" in hits[0]["display_line"]
    assert "hello" in hits[0]["display_line"]
