# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven agent factory.

:func:`load_agents_from_yaml` reads an ``agents:`` list from a YAML file
and returns ready-to-use :class:`~evaluation.framework.agents.base_agent.BaseAgent`
instances that can be put straight into a registry dict (keyed by agent id)
or passed to :func:`~evaluation.framework.agents.agent_server.start_agent_server`.

YAML schema
-----------

.. code-block:: yaml

    agents:
      - id: mediator
        type: llm
        prefer_low: true
        persona: "You are a neutral mediator who seeks win-win agreements."
        prompt_mode: english    # "english" | "sstp"

All fields except ``id`` and ``type`` are optional (sensible defaults apply).

See Also
--------
* :class:`AgentSpec` — mirrors the YAML schema as a dataclass.
* :class:`~evaluation.framework.agents.llm_agent.LLMAgent`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── sys.path safety ───────────────────────────────────────────────────────────
_repo_root = str(Path(__file__).resolve().parents[4])  # ioc-cfn-cognitive-agents/
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from .base_agent import BaseAgent  # noqa: E402
from .llm_agent import LLMAgent  # noqa: E402


@dataclass
class AgentSpec:
    """Schema for a single agent entry in the YAML ``agents:`` list.

    Attributes:
        id: ``participant_id`` used as the registry key and for matching
            inbound SSTP ``participant_id`` fields.
        type: ``"llm"``.
        prefer_low: When ``True`` the agent ranks lower-indexed options
            higher (buyer perspective).  Defaults to ``True``.
        params: Free-form dict of extra parameters (reserved for future use).
        persona: System-level persona text injected into every LLM prompt.
            Only used when ``type == "llm"``.
        prompt_mode: ``"english"`` (plain narrative) or ``"sstp"``
            (raw JSON envelope).  Defaults to ``"english"``.
        url: Optional remote URL.  When set the agent keeps its spec but is
            not instantiated locally (the evaluation runner forwards calls to
            the remote endpoint instead).  Reserved for future use.
    """

    id: str
    type: str = "llm"
    prefer_low: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    persona: Optional[str] = None
    prompt_mode: str = "english"
    url: Optional[str] = None


def _spec_from_dict(raw: Dict[str, Any]) -> AgentSpec:
    return AgentSpec(
        id=raw["id"],
        type=raw.get("type", "llm"),
        prefer_low=raw.get("prefer_low", True),
        params=raw.get("params") or {},
        persona=raw.get("persona"),
        prompt_mode=raw.get("prompt_mode", "english"),
        url=raw.get("url"),
    )


def _instantiate(spec: AgentSpec) -> BaseAgent:
    agent_type = spec.type.lower()
    if agent_type == "llm":
        return LLMAgent(
            agent_id=spec.id,
            prefer_low=spec.prefer_low,
            persona=spec.persona,
            prompt_mode=spec.prompt_mode,
        )
    raise ValueError(
        f"Unknown agent type '{spec.type}' for agent '{spec.id}'. "
        "Supported types: 'llm'."
    )


def load_agents_from_yaml(path: str) -> List[BaseAgent]:
    """Parse *path* and return a list of instantiated agents.

    The file must contain a top-level ``agents:`` key whose value is a
    sequence of agent spec dicts (see module docstring for full schema).

    Agents whose ``url`` field is set are still instantiated locally — the
    remote URL is currently informational only and reserved for future
    evaluation runner support.

    Args:
        path: Filesystem path to a YAML file.

    Returns:
        List of :class:`~evaluation.framework.agents.base_agent.BaseAgent`
        instances in the same order as the YAML list.

    Raises:
        FileNotFoundError: When *path* does not exist.
        KeyError: When a required ``id`` field is missing.
        ValueError: When an unsupported agent ``type`` is encountered.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {yaml_path}")

    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    raw_agents: List[Dict[str, Any]] = data.get("agents", [])
    agents: List[BaseAgent] = []
    for raw in raw_agents:
        spec = _spec_from_dict(raw)
        agents.append(_instantiate(spec))

    return agents
