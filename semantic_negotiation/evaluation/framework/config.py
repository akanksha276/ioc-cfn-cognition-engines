# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the general evaluation framework.

Three top-level configurable parts
-----------------------------------
1. **Agents** — :class:`AgentConfig` list — who is negotiating and with what strategy.
2. **Environment** — :class:`EnvironmentConfig` — LLM settings, server ports,
   and timeouts.
3. **Missions** — :class:`MissionsConfig` — negotiation problems to evaluate,
   loaded from a YAML file or provided inline.

They are combined in :class:`EvaluationConfig` and passed to
:class:`~evaluation.framework.runner.EvaluationRunner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Agents
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration for one negotiation agent.

    Attributes:
        agent_id: Unique identifier; used as ``participant_id`` in every
            SSTP message.  Must be unique across the agents list.
        prefer_low: When ``True`` the agent prefers lower-indexed options
            (index 0 = best outcome).  When ``False`` it prefers
            higher-indexed options (last index = best outcome).
            Drives the default per-issue utility function used both by the
            in-process agent and when building
            :class:`~app.agent.negotiation_model.NegotiationParticipant`
            preference weights.
        params: Concession knobs.  Recognised keys:

            * ``exponent`` *(float, default 2.0)* — Boulware concession
              curve shape: ``aspiration(t) = max(min_reservation, 1−t^exponent)``.
              Values > 1 mean slow initial concession; 1 = linear.
            * ``min_reservation`` *(float, default 0.0)* — hard utility floor;
              the agent never accepts below this value.

        url: HTTP base URL of an **external** agent server that is already
            running and exposes a ``POST /decide`` endpoint following the
            SSTP ``List[SSTPNegotiateMessage] → List[SSTPNegotiateMessage]``
            contract.  When ``None`` the runner starts the agent in-process.

    Example::

        AgentConfig("buyer",  prefer_low=True,  params={"exponent": 3.0, "min_reservation": 0.1})
        AgentConfig("seller", prefer_low=False)
        AgentConfig("ext_agent", url="http://agent-host:9000")
    """

    agent_id: str
    prefer_low: bool = True  # True = prefer index-0, False = prefer last
    params: Dict[str, Any] = field(default_factory=dict)
    url: Optional[str] = None  # None → run in-process


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Environment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EnvironmentConfig:
    """Runtime environment settings for one evaluation run.

    Attributes:
        llm_base_url: OpenAI-compatible base URL used for optional
            intent/options discovery steps.  Falls back to the
            ``OPENAI_BASE_URL`` environment variable when empty.
        llm_api_key: API key sent as ``Authorization: Bearer …``.
            Falls back to ``OPENAI_API_KEY`` when empty.
        llm_model: Deployment / model name for completion requests.
            Falls back to ``OPENAI_MODEL`` when empty.
        agent_port: TCP port for the in-process callback agent FastAPI server.
            Default ``8094`` avoids clashing with the negotiation server
            (``8089``), test agents (``8091``/``8092``), and the CaSiNo eval
            server (``8093``).
        neg_timeout: Per-round HTTP request timeout in seconds for calls from
            ``BatchCallbackRunner`` to the agent server.
    """

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    agent_port: int = 8094
    neg_timeout: float = 30.0
    neg_server: str = "http://localhost:8089"  # used by callback mechanism only


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Missions dataset
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissionEntry:
    """One negotiation problem.

    Attributes:
        name: Human-readable label; also used as the trace folder slug.
        content_text: Free-text description of what the parties are
            negotiating over.  Sent verbatim to an optional
            ``IntentDiscovery`` / ``OptionsGeneration`` step when
            ``issues`` / ``options_per_issue`` are not pre-populated.
        n_steps: Maximum SAO rounds before a timeout is declared.
            Pass ``0`` to trigger automatic sizing via
            :func:`~app.agent.batch_callback_runner.compute_n_steps`.
            Works for both direct and callback mechanisms.
        issues: Ordered list of issue identifiers.  May be empty when the
            YAML file omits them (the runner will use the keys of
            ``options_per_issue`` if available, otherwise raise).
        options_per_issue: Mapping ``{issue: [opt, …]}`` with options
            ordered from cheapest/smallest (index 0) to most
            expensive/largest.  May be empty if discovery is delegated to
            the LLM pipeline.
    """

    name: str
    content_text: str
    n_steps: int = 30
    issues: List[str] = field(default_factory=list)
    options_per_issue: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MissionsConfig:
    """Source of missions to evaluate.

    Exactly one of *path* or *missions* must be provided.

    Attributes:
        path: Path to a YAML file following the same schema as
            ``missions.yaml`` — a top-level ``missions`` list whose
            entries carry ``name``, ``content_text``, and optional
            ``n_steps``, ``issues``, ``options_per_issue`` keys.
        missions: Inline :class:`MissionEntry` list.  Takes precedence
            over *path* when non-empty.

    Example YAML (``my_missions.yaml``)::

        missions:
          - name: "Price Negotiation"
            content_text: >
              Two parties need to agree on price and warranty terms.
            issues: [price, warranty]
            options_per_issue:
              price:   ["$100", "$120", "$150", "$200"]
              warranty: ["6 months", "12 months", "24 months"]
            n_steps: 40
    """

    path: Optional[str] = None
    missions: List[MissionEntry] = field(default_factory=list)

    def load(self) -> List[MissionEntry]:
        """Return the resolved list of :class:`MissionEntry` objects.

        Inline missions take precedence over the YAML path.

        Raises:
            ValueError: When neither *path* nor *missions* is set.
            FileNotFoundError: When *path* does not exist.
        """
        if self.missions:
            return list(self.missions)

        if self.path:
            raw = yaml.safe_load(Path(self.path).read_text(encoding="utf-8"))
            entries: List[MissionEntry] = []
            for m in raw.get("missions", []):
                entries.append(
                    MissionEntry(
                        name=m["name"],
                        content_text=m["content_text"],
                        n_steps=int(m.get("n_steps", 30)),
                        issues=list(m.get("issues", [])),
                        options_per_issue={
                            k: list(v)
                            for k, v in m.get("options_per_issue", {}).items()
                        },
                    )
                )
            return entries

        raise ValueError(
            "MissionsConfig requires either 'path' (YAML file) or 'missions' (inline list)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Combined top-level config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    """Top-level configuration for one evaluation run.

    Combines the three configurable parts:

    * :attr:`agents` — list of :class:`AgentConfig` — *who* is negotiating
    * :attr:`environment` — :class:`EnvironmentConfig` — *how* the
      infrastructure is wired up
    * :attr:`missions` — :class:`MissionsConfig` — *what* they negotiate over

    Example::

        config = EvaluationConfig(
            agents=[
                AgentConfig("buyer",  prefer_low=True),
                AgentConfig("seller", prefer_low=False),
            ],
            environment=EnvironmentConfig(agent_port=8094),
            missions=MissionsConfig(path="missions.yaml"),
        )
        results = EvaluationRunner(config).run()
    """

    agents: List[AgentConfig]
    environment: EnvironmentConfig
    missions: MissionsConfig
    mechanism: str = "direct"  # "direct" | "callback"

    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig":
        """Load an :class:`EvaluationConfig` from a YAML file.

        Expected structure::

            environment:
              agent_port: 8092
              neg_timeout: 120.0

            agents:
              - agent_id: agent-a
                prefer_low: true
                params:
                  exponent: 2.0
                  min_reservation: 0.0

            missions:
              path: path/to/missions.yaml   # OR inline list below
              # missions:
              #   - name: "My Mission"
              #     content_text: "..."
              #     n_steps: 30
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        env_raw = raw.get("environment", {})
        environment = EnvironmentConfig(
            llm_base_url=env_raw.get("llm_base_url", ""),
            llm_api_key=env_raw.get("llm_api_key", ""),
            llm_model=env_raw.get("llm_model", ""),
            agent_port=int(env_raw.get("agent_port", 8094)),
            neg_timeout=float(env_raw.get("neg_timeout", 30.0)),
            neg_server=env_raw.get("neg_server", "http://localhost:8089"),
        )

        agents: List[AgentConfig] = []
        for a in raw.get("agents", []):
            agents.append(
                AgentConfig(
                    agent_id=a["agent_id"],
                    prefer_low=bool(a.get("prefer_low", True)),
                    params=dict(a.get("params", {})),
                    url=a.get("url"),
                )
            )

        missions_raw = raw.get("missions", {})
        if isinstance(missions_raw, dict):
            missions_path = missions_raw.get("path")
            inline = missions_raw.get("missions", [])
        else:
            missions_path = None
            inline = []

        if missions_path:
            # Resolve relative path against the config file's directory
            missions_path = str(Path(path).parent / missions_path)
            missions = MissionsConfig(path=missions_path)
        elif inline:
            missions = MissionsConfig(
                missions=[
                    MissionEntry(
                        name=m["name"],
                        content_text=m["content_text"],
                        n_steps=int(m.get("n_steps", 30)),
                        issues=list(m.get("issues", [])),
                        options_per_issue={
                            k: list(v)
                            for k, v in m.get("options_per_issue", {}).items()
                        },
                    )
                    for m in inline
                ]
            )
        else:
            raise ValueError(
                f"Config file '{path}' must have a 'missions.path' or inline 'missions.missions' list."
            )

        mechanism: str = raw.get("mechanism", "direct")

        return cls(
            agents=agents,
            environment=environment,
            missions=missions,
            mechanism=mechanism,
        )
