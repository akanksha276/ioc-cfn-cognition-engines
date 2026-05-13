# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
TraceStateBuilder — summarises the negotiation trace into per-issue position data.
InteractionSignalExtractor — computes instability signals from the trace state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .models import InteractionSignals, NegotiationTrace, TraceState
from .utils import mean, normalized_index_distance


class TraceStateBuilder:
    def build(self, trace: NegotiationTrace, issues: List[str]) -> TraceState:
        issue_focus_counts: Dict[str, int] = defaultdict(int)
        issue_positions_by_agent: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        all_mentions_by_agent: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        for rr in trace.rounds:
            all_mentions_by_agent[rr.proposer_id].append(rr.offer)
            for issue in issues:
                value = rr.offer.get(issue)
                if value:
                    issue_focus_counts[issue] += 1
                    issue_positions_by_agent[issue][rr.proposer_id].append(value)

        return TraceState(
            issue_focus_counts=dict(issue_focus_counts),
            issue_positions_by_agent={
                k: dict(v) for k, v in issue_positions_by_agent.items()
            },
            all_mentions_by_agent=dict(all_mentions_by_agent),
            final_agreement=trace.final_agreement,
            total_rounds=trace.total_rounds,
        )


class InteractionSignalExtractor:
    """
    Supporting instability signals extracted from trace dynamics.
    These are NOT the main semantic validator outputs.
    """

    def extract(
        self,
        trace_state: TraceState,
        options_per_issue: Dict[str, List[str]],
    ) -> InteractionSignals:
        positional_instability: Dict[str, Dict[str, float]] = defaultdict(dict)
        oscillation_rate: Dict[str, Dict[str, float]] = defaultdict(dict)
        per_issue_divergence: Dict[str, float] = {}

        for issue_id, by_agent in trace_state.issue_positions_by_agent.items():
            options = options_per_issue.get(issue_id, [])

            for agent_id, positions in by_agent.items():
                dists: List[float] = []
                reversals = 0

                for i in range(1, len(positions)):
                    d = normalized_index_distance(positions[i - 1], positions[i], options)
                    if d is not None:
                        dists.append(d)

                for i in range(2, len(positions)):
                    if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]:
                        reversals += 1

                positional_instability[issue_id][agent_id] = round(mean(dists), 4)
                oscillation_rate[issue_id][agent_id] = round(
                    reversals / max(1, len(positions) - 2), 4
                )

            # Cross-agent divergence at final state
            final_positions = [
                positions[-1]
                for positions in by_agent.values()
                if positions
            ]
            pair_dists = [
                d
                for i in range(len(final_positions))
                for j in range(i + 1, len(final_positions))
                for d in [normalized_index_distance(final_positions[i], final_positions[j], options)]
                if d is not None
            ]
            per_issue_divergence[issue_id] = round(mean(pair_dists), 4)

        overall_components: List[float] = []
        for issue_id in positional_instability:
            overall_components.extend(positional_instability[issue_id].values())
        overall_components.extend(per_issue_divergence.values())
        for issue_id in oscillation_rate:
            overall_components.extend(oscillation_rate[issue_id].values())

        return InteractionSignals(
            positional_instability={k: dict(v) for k, v in positional_instability.items()},
            per_issue_divergence=per_issue_divergence,
            oscillation_rate={k: dict(v) for k, v in oscillation_rate.items()},
            overall_instability_score=round(mean(overall_components), 4),
        )
