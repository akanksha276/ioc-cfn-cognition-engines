# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""General evaluation runner framework.

Three configurable parts
------------------------
* :class:`~evaluation.framework.config.AgentConfig` list — *who* is negotiating
* :class:`~evaluation.framework.config.EnvironmentConfig` — *how* the infrastructure
  is wired up (LLM, ports, timeouts)
* :class:`~evaluation.framework.config.MissionsConfig` — *what* they are negotiating over

Quickstart::

    from evaluation.framework.config import AgentConfig, EnvironmentConfig, MissionsConfig, EvaluationConfig
    from evaluation.framework.runner import EvaluationRunner

    config = EvaluationConfig(
        agents=[
            AgentConfig("buyer",  prefer_low=True),
            AgentConfig("seller", prefer_low=False),
        ],
        environment=EnvironmentConfig(agent_port=8094),
        missions=MissionsConfig(path="missions.yaml"),
    )
    results = EvaluationRunner(config).run()
    print(results.summary())
"""

from .config import AgentConfig, EnvironmentConfig, EvaluationConfig, MissionEntry, MissionsConfig
from .runner import EvaluationResult, EvaluationRunner, MissionResult

__all__ = [
    "AgentConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "MissionEntry",
    "MissionsConfig",
    "EvaluationResult",
    "EvaluationRunner",
    "MissionResult",
]
