# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic negotiation pipeline - top-level orchestrator.

Wires together:

1. :class:`~app.agent.intent_discovery.IntentDiscovery` - extracts negotiable issues.
2. :class:`~app.agent.options_generation.OptionsGeneration` - generates options per issue.
3. :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` - runs the turn-by-turn
   SAO negotiation, driven externally via ``/initiate`` + ``/decide``.

Callers that need the resolved issue space for envelopes (e.g. SSTP
``semantic_context``) should use :meth:`SemanticNegotiationPipeline.run`, which
returns :class:`SemanticPipelineRun` with ``issues`` and ``options_per_issue``.
HTTP initiate builds :class:`~app.agent.negotiation_model.NegotiationParticipant`
lists only *after* options exist (server-side prefs are derived per option list),
so the API uses ``run(..., after_options=...)`` rather than pre-filling
``self.participants``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .batch_callback_runner import compute_n_steps
from .intent_discovery import IntentDiscovery
from .negotiation_model import NegotiationParticipant, NegotiationResult
from .options_generation import OptionsGeneration

logger = logging.getLogger(__name__)


class SemanticNegotiationError(RuntimeError):
    """Base exception for semantic_negotiation library errors."""


class SemanticNegotiationInputError(SemanticNegotiationError, ValueError):
    """Raised when caller inputs are invalid."""


class SemanticNegotiationSessionNotFoundError(SemanticNegotiationError, KeyError):
    """Raised when a session_id cannot be found."""


class SemanticNegotiationExecutionError(SemanticNegotiationError):
    """Raised when pipeline execution fails for unexpected reasons."""


@dataclass(frozen=True)
class SemanticPipelineRun:
    """Bundle returned by :meth:`SemanticNegotiationPipeline.run`.

    Holds the negotiation outcome plus the exact issue list and option map used,
    so HTTP or tracing layers can populate SSTP ``semantic_context`` without
    re-invoking intent discovery or options generation.
    """

    result: NegotiationResult
    issues: List[str]
    options_per_issue: Dict[str, List[str]]
    participants: List[NegotiationParticipant]


__all__ = [
    "SemanticNegotiationPipeline",
    "SemanticPipelineRun",
    "SemanticNegotiationError",
    "SemanticNegotiationInputError",
    "SemanticNegotiationSessionNotFoundError",
    "SemanticNegotiationExecutionError",
    # Re-exported for convenience
    "NegotiationParticipant",
    "IntentDiscovery",
    "OptionsGeneration",
    "compute_n_steps",
]


class SemanticNegotiationPipeline:
    """Orchestrates the semantic negotiation flow.

    Provides three public methods that map to the three pipeline components:

    1. :meth:`discover_and_generate` - Components 1+2 (IntentDiscovery + OptionsGeneration).
    2. :meth:`start_negotiation` - Component 3 seed: creates a session and returns
       the first batch of agent messages.
    3. :meth:`step_negotiation` - Component 3 advance: applies one batch of replies
       and returns the next messages or the final result.

    Args:
        n_steps: Maximum SAO rounds for the negotiation.
        enable_local_trace: When ``True``, commit and error envelopes are written
            to ``<cwd>/neg_trace/<session_id>/sstp_message_trace.json``.  Disabled
            by default; set to ``True`` only for local debugging.
    """

    def __init__(self, n_steps: int = 100, *, enable_local_trace: bool = False) -> None:
        self.n_steps = n_steps
        self.enable_local_trace = enable_local_trace
        self._intent_discovery = IntentDiscovery()
        self._options_generation = OptionsGeneration()
        # session_id → (runner, sess)
        self._sessions: Dict[str, tuple] = {}
        logger.info("SemanticNegotiationPipeline initialized n_steps=%d", n_steps)

    def start_negotiation(
        self,
        issues: List[str],
        options_per_issue: Dict[str, List[str]],
        participants: List[NegotiationParticipant],
        session_id: str,
        n_steps: int | None = None,
    ) -> tuple:
        """Seed a turn-by-turn SAO session (Component 3, step 0).

        Returns ``(runner, sess, first_round_messages)`` where *runner* and
        *sess* must be passed back to :meth:`step_negotiation` on every
        subsequent ``/decide`` call.

        Raises:
            SemanticNegotiationInputError: If provided issues/options/participants
                are invalid.
            SemanticNegotiationExecutionError: For unexpected failures in the
                underlying runner.
        """
        from ..agent.batch_callback_runner import BatchCallbackRunner

        try:
            eff = n_steps if n_steps is not None else self.n_steps
            logger.info(
                "start_negotiation session_id=%s issues=%d participants=%d n_steps=%d",
                session_id,
                len(issues),
                len(participants),
                eff,
            )
            runner = BatchCallbackRunner(n_steps=eff)
            sess, first_round_messages = runner.start(
                issues=issues,
                options_per_issue=options_per_issue,
                participants=participants,
                session_id=session_id,
            )
            logger.info(
                "start_negotiation complete session_id=%s first_messages=%d",
                session_id,
                len(first_round_messages),
            )
            return runner, sess, first_round_messages
        except (KeyError, ValueError, TypeError) as exc:
            raise SemanticNegotiationInputError(
                f"Failed to start negotiation for session_id='{session_id}'"
            ) from exc
        except Exception as exc:
            raise SemanticNegotiationExecutionError(
                f"Unexpected error starting negotiation for session_id='{session_id}'"
            ) from exc

    def step_negotiation(
        self,
        runner: Any,
        sess: Any,
        agent_replies: List[Dict[str, Any]],
    ) -> tuple:
        """Advance the SAO by one batch of agent replies (Component 3, step N).

        Returns the same ``(status, next_messages, result)`` triple as
        :meth:`~app.agent.batch_callback_runner.BatchCallbackRunner.step`.

        Raises:
            SemanticNegotiationInputError: If the runner/session/replies are invalid.
            SemanticNegotiationExecutionError: For unexpected runner errors.
        """
        try:
            logger.info(
                "step_negotiation replies=%d",
                len(agent_replies),
            )
            status, next_messages, result = runner.step(sess, agent_replies)
            logger.info(
                "step_negotiation status=%s next_messages=%s",
                status,
                0 if next_messages is None else len(next_messages),
            )
            return status, next_messages, result
        except (KeyError, ValueError, TypeError) as exc:
            raise SemanticNegotiationInputError("Failed to step negotiation") from exc
        except Exception as exc:
            raise SemanticNegotiationExecutionError(
                "Unexpected error stepping negotiation"
            ) from exc

    def discover_and_generate(
        self,
        content_text: str,
        *,
        workspace_id: str | None = None,
        mas_id: str | None = None,
        fabric_node_base_url: str | None = None,
        agent_names: List[str] | None = None,
    ) -> tuple[List[str], Dict[str, List[str]], Optional[str]]:
        """Run only Components 1 and 2 and return ``(issues, options_per_issue, options_memory_blob)``.

        *options_memory_blob* is the JSON string retrieved from fabric and fed into
        the options LLM when the memory strategy runs; ``None`` for LLM-only or when
        no negotiable entities.

        Use this from the turn-by-turn ``/initiate`` endpoint to get issues and
        options without triggering the NegMAS negotiation (Component 3).

        When *fabric_node_base_url*, *workspace_id*, and *mas_id* are all set,
        options are produced with the **memory + LLM** strategy: each lookup calls
        the cognition fabric node's shared-memories **query** route, which runs
        evidence gathering server-side; the returned message is passed into the
        options prompt. Optional *agent_names* is attached to that query as
        ``additional_context.negotiation_agent_names``. Otherwise the default
        **LLM-only** strategy is used.

        Raises:
            SemanticNegotiationInputError: If intent discovery / option generation
                inputs are invalid.
            SemanticNegotiationExecutionError: For unexpected failures in dependent
                components.
        """
        try:
            use_fabric = bool(fabric_node_base_url and workspace_id and mas_id)
            logger.info(
                "discover_and_generate content_len=%d use_fabric=%s",
                len(content_text or ""),
                use_fabric,
            )
            issues = self._intent_discovery.discover(
                sentence=content_text,
                agent_names=agent_names,
                fabric_node_base_url=fabric_node_base_url,
                workspace_id=workspace_id,
                mas_id=mas_id,
            )
            if hasattr(issues, "negotiable_entities"):
                issues = issues.negotiable_entities
            gen_out = self._options_generation.generate_options(
                issues,
                content_text,
                agent_names=agent_names,
                fabric_node_base_url=fabric_node_base_url,
                workspace_id=workspace_id,
                mas_id=mas_id,
            )
            logger.info(
                "discover_and_generate done issues=%d options_keys=%d memory_blob=%s",
                len(issues),
                len(gen_out.options_per_issue),
                gen_out.memory_blob is not None,
            )
            return issues, gen_out.options_per_issue, gen_out.memory_blob
        except (KeyError, ValueError, TypeError) as exc:
            raise SemanticNegotiationInputError(
                "Failed to discover issues and generate options"
            ) from exc
        except Exception as exc:
            raise SemanticNegotiationExecutionError(
                f"Unexpected error during issue discovery/options generation: {exc}"
            ) from exc

    async def async_execute(
        self,
        session_id: str,
        *,
        n_steps: int | None = None,
        content_text: str = "",
        agents_raw: List[Dict[str, Any]] | None = None,
        initiate_message: Dict[str, Any] | None = None,
        agent_replies: List[Dict[str, Any]] | None = None,
        commit_message_id: str = "",
        workspace_id: str | None = None,
        mas_id: str | None = None,
        fabric_node_base_url: str | None = None,
        agent_names: List[str] | None = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.execute,
            session_id,
            n_steps=n_steps,
            content_text=content_text,
            agents_raw=agents_raw,
            initiate_message=initiate_message,
            agent_replies=agent_replies,
            commit_message_id=commit_message_id,
            workspace_id=workspace_id,
            mas_id=mas_id,
            fabric_node_base_url=fabric_node_base_url,
            agent_names=agent_names,
        )

    def execute(
        self,
        session_id: str,
        *,
        n_steps: int | None = None,
        content_text: str = "",
        agents_raw: List[Dict[str, Any]] | None = None,
        initiate_message: Dict[str, Any] | None = None,
        agent_replies: List[Dict[str, Any]] | None = None,
        commit_message_id: str = "",
        workspace_id: str | None = None,
        mas_id: str | None = None,
        fabric_node_base_url: str | None = None,
        agent_names: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Unified entry point for both ``/initiate`` and ``/decide``.

        - **New session** (``session_id`` not yet registered): runs Components 1+2
          then seeds the SAO session via :meth:`start_negotiation`.  Requires
          *content_text* and *agents_raw*.
        - **Existing session** (``session_id`` already registered): advances the SAO
          by one batch via :meth:`step_negotiation`.  Requires *agent_replies*.
          Raises ``KeyError`` if the session is not found (caller should return 404).

        Returns a dict with ``status`` and either ``messages`` (ongoing) or
        ``result`` (terminal — ``agreed``, ``broken``, or ``timeout``).

        On **initiate**, the response includes ``options_memory_blob`` (JSON string
        from fabric memory lookup when used, else ``null``).

        On **initiate**, optional ``workspace_id``, ``mas_id``,
        ``fabric_node_base_url``, and ``agent_names`` are forwarded to
        :meth:`discover_and_generate`. When the first three are all set, options
        use fabric-node evidence (shared-memories query) before the LLM.
        ``agent_names`` is sent as ``additional_context.negotiation_agent_names``
        on that query.

        Raises:
            SemanticNegotiationSessionNotFoundError: When stepping an unknown session.
            SemanticNegotiationInputError: When required inputs are missing/invalid.
            SemanticNegotiationExecutionError: For unexpected failures.
        """
        try:
            if session_id not in self._sessions:
                logger.info("execute initiate session_id=%s", session_id)
                # ── Initiate ──────────────────────────────────────────────
                if agents_raw is None:
                    raise SemanticNegotiationInputError(
                        f"agents information is required to initiate a session"
                    )
                issues, options_per_issue, options_memory_blob = (
                    self.discover_and_generate(
                        content_text,
                        workspace_id=workspace_id,
                        mas_id=mas_id,
                        fabric_node_base_url=fabric_node_base_url,
                        agent_names=agent_names,
                    )
                )
                if n_steps:
                    # Caller explicitly provided a non-zero budget — use it as-is.
                    effective_n = n_steps
                else:
                    # Compute a dynamic budget from negotiation complexity.
                    # If settings.negotiation_n_steps > 0 it acts as a hard cap
                    # so operators can still control the upper bound via env var.
                    dynamic_n = compute_n_steps(
                        n_agents=len(agents_raw),
                        n_issues=len(issues),
                        options_per_issue=options_per_issue,
                    )
                    effective_n = (
                        min(dynamic_n, self.n_steps) if self.n_steps > 0 else dynamic_n
                    )
                participants = [
                    NegotiationParticipant(id=a["id"], name=a["name"])
                    for a in agents_raw
                ]
                runner, sess, messages = self.start_negotiation(
                    issues,
                    options_per_issue,
                    participants,
                    session_id,
                    n_steps=effective_n,
                )
                if initiate_message:
                    sess.sstp_message_trace.insert(0, initiate_message)
                sess.content_text = content_text
                sess.agents_negotiating = [a["id"] for a in (agents_raw or [])]
                self._sessions[session_id] = (runner, sess)
                logger.info(
                    "execute initiated session_id=%s issues=%d n_steps=%d",
                    session_id,
                    len(issues),
                    effective_n,
                )
                return {
                    "status": "initiated",
                    "session_id": session_id,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                    "options_memory_blob": options_memory_blob,
                    "n_steps": effective_n,
                    "round": 1,
                    "messages": messages,
                }

            # ── Decide ────────────────────────────────────────────────────
            logger.info(
                "execute decide session_id=%s replies=%d",
                session_id,
                len(agent_replies or []),
            )
            try:
                runner, sess = self._sessions[session_id]
            except KeyError as exc:
                raise SemanticNegotiationSessionNotFoundError(session_id) from exc

            status, next_messages, result = self.step_negotiation(
                runner, sess, agent_replies or []
            )
        except SemanticNegotiationError as exc:
            self._save_error_commit_to_disk(session_id, str(exc))
            raise
        except (KeyError, ValueError, TypeError) as exc:
            self._save_error_commit_to_disk(session_id, str(exc))
            raise SemanticNegotiationInputError(
                f"Failed to execute pipeline for session_id='{session_id}'"
            ) from exc
        except Exception as exc:
            self._save_error_commit_to_disk(session_id, str(exc))
            raise SemanticNegotiationExecutionError(
                f"Unexpected error executing pipeline for session_id='{session_id}'"
            ) from exc
        if status == "ongoing":
            logger.info(
                "execute ongoing session_id=%s round=%s",
                session_id,
                getattr(sess, "sao_step", "?"),
            )
            # Detect if the last outgoing message is already a commit envelope.
            # This can happen when the runner emits a commit as part of its final
            # propose/respond cycle before the terminal step is reached.
            if next_messages:
                last_msg = next_messages[-1]
                last_kind = (
                    last_msg.kind if hasattr(last_msg, "kind") else last_msg.get("kind")
                )
                if last_kind == "commit":
                    pass  # TODO validate later
            return {
                "status": "ongoing",
                "session_id": session_id,
                "round": sess.sao_step + 1,
                "messages": next_messages,
            }
        # Terminal — build commit envelope and clean up session.
        logger.info(
            "execute terminal session_id=%s status=%s",
            session_id,
            status,
        )
        del self._sessions[session_id]
        participant_id_by_name = {p.name: p.id for p in sess.participants}
        try:
            commit = self.build_commit_envelope(
                result,
                sess.issues,
                participant_id_by_name,
                session_id,
                commit_message_id,
                content_text=sess.content_text,
                agents_negotiating=sess.agents_negotiating,
                options_per_issue=sess.options_per_issue,
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._save_error_commit_to_disk(session_id, str(exc))
            raise SemanticNegotiationExecutionError(
                f"Failed to build commit envelope for session_id='{session_id}'"
            ) from exc
        return {
            "status": status,
            "session_id": session_id,
            "round": sess.sao_step,
            "result": result,
            "issues": sess.issues,
            "participant_id_by_name": participant_id_by_name,
            "final_result": commit,
        }

    def build_commit_envelope(
        self,
        result: Any,
        issues: List[str],
        participant_id_by_name: Dict[str, str],
        session_id: str,
        message_id: str,
        *,
        content_text: str = "",
        agents_negotiating: List[str] | None = None,
        options_per_issue: Dict[str, List[str]] | None = None,
    ) -> Any:
        """Build an ``SSTPCommitMessage`` from a terminal :class:`NegotiationResult`.

        Appends the serialised commit envelope to ``result.sstp_message_trace``
        so the complete dialogue history (initiate → all rounds → commit) is
        captured in one list.

        Returns the ``SSTPCommitMessage`` instance.

        Raises:
            SemanticNegotiationExecutionError: If required protocol/schema imports
                are unavailable or the commit envelope cannot be built.
        """
        logger.info(
            "build_commit_envelope session_id=%s issues=%d message_id_set=%s",
            session_id,
            len(issues),
            bool(message_id),
        )
        try:
            from datetime import datetime, timezone

            from protocol.sstp import SSTPCommitMessage
            from protocol.sstp._base import (
                LogicalClock,
                Origin,
                PolicyLabels,
                Provenance,
            )
            from protocol.sstp.commit import NegotiateCommitSemanticContext

            from ..api.schemas import (
                AgentDecision,
                NegotiationOutcomeResponse,
                NegotiationTrace,
                RoundOffer,
            )
        except Exception as exc:
            raise SemanticNegotiationExecutionError(
                "Failed to import commit-envelope dependencies"
            ) from exc

        try:
            # ── Rebuild negotiation trace ────────────────────────────────
            def _resolve_id(name: str) -> str:
                if name in participant_id_by_name:
                    return participant_id_by_name[name]
                for pname, pid in participant_id_by_name.items():
                    if name.startswith(pname):
                        return pid
                return name

            decisions_map: Dict[int, List[Any]] = getattr(result, "round_decisions", {})
            next_proposer_map: Dict[int, Any] = getattr(
                result, "round_next_proposer", {}
            )
            rounds: List[Any] = []
            for idx, (sao_step, name, offer_tuple) in enumerate(result.history):
                offer: Dict[str, str] = {}
                if offer_tuple is not None:
                    for i, issue_id in enumerate(issues):
                        offer[issue_id] = str(offer_tuple[i])
                round_num = idx + 1
                # decisions_map is keyed by 1-based round number.
                # The decisions for history entry at index idx are the agents'
                # responses to that offer, recorded in round_decisions[idx + 1].
                # (sao_step + 1 would be 0 for the server's initial row, but
                # those decisions are stored at key 1, not 0.)
                raw_decs = decisions_map.get(idx + 1, [])
                decisions = [
                    AgentDecision(
                        participant_id=d["participant_id"],
                        action=d["action"],
                        offer=d.get("offer"),
                    )
                    for d in raw_decs
                ]
                # sao_step=-1 for the server's initial offer (key 0 = first SAO proposer);
                # for a counter-offer made in SAO round R, key R contains the next proposer.
                rounds.append(
                    RoundOffer(
                        round=round_num,
                        proposer_id=_resolve_id(name),
                        next_proposer_id=next_proposer_map.get(sao_step + 1),
                        offer=offer,
                        decisions=decisions,
                    )
                )

            final_agreement = None
            if result.agreement is not None:
                final_agreement = [
                    NegotiationOutcomeResponse(
                        issue_id=o.issue_id, chosen_option=o.chosen_option
                    )
                    for o in result.agreement
                ]

            trace = NegotiationTrace(
                rounds=rounds,
                final_agreement=final_agreement,
                timedout=result.timedout,
                broken=result.broken,
            )

            total_rounds = len(rounds)
            _agreed = result.agreement is not None
            _outcome = (
                "agreement"
                if _agreed
                else ("broken" if result.broken else "disagreement")
            )
            _status_str = (
                "agreed" if _agreed else ("broken" if result.broken else "timeout")
            )

            # ── Build SSTPCommitMessage ──────────────────────────────────
            commit = SSTPCommitMessage(
                kind="commit",
                message_id=message_id,
                dt_created=datetime.now(timezone.utc).isoformat(),
                origin=Origin(actor_id="negotiation-server", tenant_id=session_id),
                semantic_context=NegotiateCommitSemanticContext(
                    session_id=session_id,
                    outcome=_outcome,
                    final_agreement=(
                        [o.model_dump() for o in trace.final_agreement]
                        if trace.final_agreement
                        else None
                    ),
                    content_text=content_text or None,
                    agents_negotiating=agents_negotiating or None,
                    issues=issues or None,
                    options_per_issue=options_per_issue or None,
                ),
                payload_hash="0" * 64,
                policy_labels=PolicyLabels(
                    sensitivity="internal",
                    propagation="restricted",
                    retention_policy="default",
                ),
                provenance=Provenance(sources=[], transforms=[]),
                payload={
                    "status": _status_str,
                    "session_id": session_id,
                    "total_rounds": total_rounds,
                    "trace": trace.model_dump(mode="json", exclude={"final_agreement"}),
                },
                state_object_id=session_id,
                parent_ids=[message_id],
                logical_clock=LogicalClock(type="lamport", value=total_rounds),
                merge_strategy="add",
                confidence_score=1.0 if _agreed else 0.0,
                risk_score=0.0 if _agreed else 1.0,
                ttl_seconds=86400,
            )
        except Exception as exc:
            raise SemanticNegotiationExecutionError(
                f"Failed to build commit envelope for session_id='{session_id}'"
            ) from exc

        # Append the commit to the full message trace.
        if isinstance(getattr(result, "sstp_message_trace", None), list):
            result.sstp_message_trace.append(commit.model_dump(mode="json"))

        if self.enable_local_trace:
            try:
                base_dir = Path.cwd() / "neg_trace"
                out_dir = base_dir / session_id
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "sstp_message_trace.json"

                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(commit.model_dump(mode="json"), fh, indent=2)
                logger.info(
                    "[%s] SSTP trace written: %s (commit only)",
                    session_id,
                    out_path,
                )
            except OSError as exc:
                logger.warning("[%s] failed to write SSTP trace: %s", session_id, exc)

        return commit

    def _save_error_commit_to_disk(self, session_id: str, error_message: str) -> None:
        """Write a minimal error-commit JSON when the pipeline raises unexpectedly.

        No-op unless :attr:`enable_local_trace` is ``True``.
        """
        if not self.enable_local_trace:
            return
        try:
            error_commit = {
                "kind": "commit",
                "semantic_context": {
                    "schema_id": "urn:ioc:schema:negotiate:commit:v1",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "outcome": "error",
                    "error_message": error_message,
                },
                "payload": {
                    "status": "error",
                    "session_id": session_id,
                },
            }
            base_dir = Path.cwd() / "neg_trace"
            out_dir = base_dir / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "sstp_message_trace.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(error_commit, fh, indent=2)
            logger.info("[%s] Error commit written: %s", session_id, out_path)
        except OSError as write_exc:
            logger.warning(
                "[%s] failed to write error commit: %s", session_id, write_exc
            )

    def release_session(self, session_id: str) -> None:
        """Remove a session from the internal store (e.g. on error clean-up)."""
        self._sessions.pop(session_id, None)
