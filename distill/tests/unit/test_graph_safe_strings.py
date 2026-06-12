# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from distill.app.services.distillation_job import _codin_display_name, _graph_safe_single_line


def test_graph_safe_single_line_strips_newlines():
    raw = "1. First line\n2. Second line\n3. Third"
    out = _graph_safe_single_line(raw)
    assert "\n" not in out
    assert "First line" in out
    assert "Second line" in out


def test_codin_display_name_is_short():
    name = _codin_display_name("website_selector_agent", "667b1296-fd13-42e9-a958-7c03e078e4af", "Summary")
    assert name.startswith("Summary-website_selector_agent-")
    assert "\n" not in name
    assert len(name) <= 256


def test_graph_safe_single_line_collapses_multiline_description():
    raw = (
        "1. Website_expert_agent uses tools.\n"
        "2. It is equipped with LLM capabilities.\n"
        "3. Website_expert_agent can scrape websites."
    )
    out = _graph_safe_single_line(raw, max_len=8192)
    assert "\n" not in out
    assert "1." in out and "2." in out and "3." in out
