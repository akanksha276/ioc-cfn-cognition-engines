# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from distill.app.agent import llm_distill as llm_mod
from distill.app.config import settings as settings_mod


def _resp(arguments=None, finish_reason="stop", refusal=None, with_tool_call=True):
    tool_calls = None
    if with_tool_call:
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(arguments=arguments if arguments is not None else "{}")
            )
        ]
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(tool_calls=tool_calls, refusal=refusal),
    )
    return SimpleNamespace(choices=[choice])


def test_run_distillation_llm_retries_then_success(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_RETRIES", 3)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_BACKOFF_SEC", 1)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MODEL", "openai/gpt-4o-mini")

    calls = {"n": 0}

    def fake_completion(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return _resp(
            arguments=json.dumps(
                {
                    "distilled_description": "distilled",
                    "summarized_context": "summary",
                }
            )
        )
    monkeypatch.setattr(llm_mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _x: None)

    out = llm_mod.run_distillation_llm("sys", "usr")
    assert out.distilled_description == "distilled"
    assert calls["n"] == 3


def test_run_distillation_llm_fails_after_retries(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_BACKOFF_SEC", 1)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MODEL", "openai/gpt-4o-mini")

    def fake_completion(**_kwargs):
        raise RuntimeError("always bad")
    monkeypatch.setattr(llm_mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _x: None)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        llm_mod.run_distillation_llm("sys", "usr")


def test_run_distillation_llm_retries_on_empty_payload(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MAX_BACKOFF_SEC", 1)
    monkeypatch.setattr(settings_mod.settings, "CODI_LLM_MODEL", "openai/gpt-4o-mini")

    calls = {"n": 0}

    def fake_completion(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp(
                arguments=json.dumps(
                    {"distilled_description": "", "summarized_context": ""}
                )
            )
        return _resp(
            arguments=json.dumps(
                {"distilled_description": "", "summarized_context": "ok"}
            )
        )
    monkeypatch.setattr(llm_mod.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _x: None)

    out = llm_mod.run_distillation_llm("sys", "usr")
    assert out.summarized_context == "ok"
    assert calls["n"] == 2
