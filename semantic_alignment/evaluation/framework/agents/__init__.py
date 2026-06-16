# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Pluggable negotiation agents for the evaluation framework.

All agents share the same ``POST /decide`` contract so they work identically
with both environments:

* **direct_env** — :class:`~evaluation.framework.runner.EvaluationRunner`
  drives SAO rounds itself via ``BatchCallbackRunner``.
* **callback_env** — the external negotiation server calls ``/decide`` on
  every SAO round.

Agent types
-----------
* :class:`~evaluation.framework.agents.llm_agent.LLMAgent` —
  LLM-driven agent whose persona and prompt style are configurable from
  external YAML.

Loading from YAML
-----------------
Use :func:`~evaluation.framework.agents.agent_loader.load_agents_from_yaml` to
instantiate a list of agents from a YAML file::

    from evaluation.framework.agents import load_agents_from_yaml
    agents = load_agents_from_yaml("my_agents.yaml")

YAML schema::

    agents:
      - id: mediator
        type: llm
        prefer_low: true
        persona: >
          You are a neutral mediator.  Prefer outcomes that maximise social
          welfare.  Concede steadily and accept any offer above 0.4 utility.
        prompt_mode: english   # "sstp" | "english"
"""

from .agent_loader import AgentSpec, load_agents_from_yaml
from .agent_server import make_decide_app, start_agent_server, wait_for_server
from .base_agent import BaseAgent
from .llm_agent import LLMAgent

__all__ = [
    "BaseAgent",
    "LLMAgent",
    "AgentSpec",
    "load_agents_from_yaml",
    "make_decide_app",
    "start_agent_server",
    "wait_for_server",
]
