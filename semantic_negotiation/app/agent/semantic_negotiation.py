# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Semantic negotiation pipeline - top-level orchestrator.

Wires together:

1. :class:`~app.agent.intent_discovery.IntentDiscovery` - extracts negotiable issues.
2. :class:`~app.agent.options_generation.OptionsGeneration` - generates options per issue.
3. :class:`~app.agent.batch_callback_runner.BatchCallbackRunner` - runs the turn-by-turn
   SAO negotiation, driven externally via ``/initiate`` + ``/decide``.
4. :class:`~app.agent.semantic_alignment_validation_pipeline.SemanticAlignmentValidationPipeline`
   — validates the final agreement before it is committed.

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
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .batch_callback_runner import compute_n_steps
from .intent_discovery import IntentDiscovery
from .negotiation_model import NegotiationParticipant, NegotiationResult
from .options_generation import OptionsGeneration
from .semantic_alignment_validation_pipeline import (
    SemanticAlignmentValidationPipeline,
    ValidationResult,
)
from .token_tracker import TokenAccumulator

# CFN-compatible traces: normalize a throwaway copy for Step 4 only (see module docstring).
from .validation_trace_adapter import adapt_sstp_trace_for_alignment_validation
from ..config.settings import settings

logger = logging.getLogger(__name__)


SessionStoreKey = tuple[str | None, str | None, str]


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
        self._alignment_validator = SemanticAlignmentValidationPipeline()
        # (workspace_id, mas_id, session_id) → (runner, sess, retry_count)
        self._sessions: Dict[SessionStoreKey, tuple] = {}
        logger.info("SemanticNegotiationPipeline initialized n_steps=%d", n_steps)

    @staticmethod
    def _session_key(
        session_id: str,
        workspace_id: str | None = None,
        mas_id: str | None = None,
    ) -> SessionStoreKey:
        """Build the scoped in-memory key for a negotiation session."""
        return (workspace_id, mas_id, session_id)

    def run_alignment_validation(
        self,
        trace: List[Dict[str, Any]],
        status: str = "",
    ) -> Optional[ValidationResult]:
        """Run Step 4 — semantic alignment validation on the completed negotiation trace.

        Called immediately after the SAO reaches a terminal state and before
        :meth:`build_commit_envelope`.  Returns a :class:`ValidationResult`
        on success, or ``None`` if the pipeline raises an unexpected exception
        (logged as a warning; the commit path is not blocked).

        Args:
            trace: The ``sstp_message_trace`` list of dicts accumulated during
                the negotiation session.

        Returns:
            :class:`ValidationResult` on success, ``None`` on failure.
        """
        logger.info("run_alignment_validation trace_len=%d", len(trace))
        try:
            validation = self._alignment_validator.run(trace, agreed=(status == "agreed"))
            logger.info(
                "run_alignment_validation done severity=%s score=%.2f "
                "needs_intervention=%s recommendation=%s timed_out=%s",
                validation.severity,
                validation.alignment_score,
                validation.needs_intervention,
                validation.recommendation,
                validation.timed_out,
            )
            if validation.heuristic_scores is not None:
                logger.info(
                    "run_alignment_validation heuristic_scores overall_instability=%.4f "
                    "per_issue_divergence=%s positional_instability=%s oscillation_rate=%s",
                    validation.heuristic_scores.overall_instability_score,
                    validation.heuristic_scores.per_issue_divergence,
                    validation.heuristic_scores.positional_instability,
                    validation.heuristic_scores.oscillation_rate,
                )
            return validation
        except Exception as exc:
            logger.warning(
                "run_alignment_validation failed — skipping validation: %s", exc
            )
            return None

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

    def _should_retry(
        self,
        validation: Optional[ValidationResult],
        retry_count: int,
    ) -> bool:
        """Return True when SAV recommends retry and budget remains."""
        if validation is None:
            return False
        return validation.should_retry and retry_count < self._alignment_validator._config.retry_max_attempts

    def _run_retry(
        self,
        session_key: SessionStoreKey,
        sess: Any,
        retry_count: int,
        validation: ValidationResult,
    ) -> list:
        """Re-generate options with exclusion guidance and re-seed the SAO."""
        history_parts = []
        for i, rec in enumerate(sess.retry_history or [], start=1):
            opts = rec.get("options_per_issue") or {}
            opts_str = "; ".join(f"{iss}: [{', '.join(o for o in v)}]" for iss, v in opts.items())
            reasoning = ((rec.get("validation") or {}).get("reasoning") or "").strip()
            part = f"Attempt {i}:"
            if opts_str:
                part += f"\n  Options used: {opts_str}"
            if reasoning:
                part += f"\n  Failure: {reasoning}"
            history_parts.append(part)
        current_opts_str = "; ".join(f"{iss}: [{', '.join(v)}]" for iss, v in (sess.options_per_issue or {}).items())
        current_reasoning = (validation.reasoning or "").strip()
        current_part = f"Attempt {retry_count + 1}:"
        if current_opts_str:
            current_part += f"\n  Options used: {current_opts_str}"
        if current_reasoning:
            current_part += f"\n  Failure: {current_reasoning}"
        history_parts.append(current_part)
        failure_context = "\n\n".join(history_parts) or None

        workspace_id, mas_id, _ = session_key
        gen_out = self._options_generation.generate_options(
            sess.issues,
            sess.content_text or "",
            agent_names=getattr(sess, "agent_names", None),
            fabric_node_base_url=settings.cfn_url,
            workspace_id=workspace_id,
            mas_id=mas_id,

            failure_context=failure_context,
        )
        new_options = gen_out.options_per_issue

        runner, new_sess, messages = self.start_negotiation(
            sess.issues,
            new_options,
            sess.participants,
            sess.session_id,
            n_steps=sess.n_steps,
        )
        new_sess.content_text = sess.content_text
        new_sess.agents_negotiating = sess.agents_negotiating
        new_sess.agent_names = getattr(sess, "agent_names", None)
        new_sess.retry_history = sess.retry_history

        self._sessions[session_key] = (runner, new_sess, retry_count + 1)
        logger.warning(
            "RETRY options_after session_id=%s attempt=%d %s",
            sess.session_id,
            retry_count + 1,
            {iss: opts for iss, opts in new_options.items()},
        )
        logger.info(
            "_run_retry complete session_id=%s attempt=%d issues=%d",
            sess.session_id,
            retry_count + 1,
            len(new_options),
        )
        return messages

    def discover_and_generate(
        self,
        content_text: str,
        *,
        workspace_id: str | None = None,
        mas_id: str | None = None,
        fabric_node_base_url: str | None = None,
        agent_names: List[str] | None = None,
        token_accumulator: Optional[TokenAccumulator] = None,
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
                token_accumulator=token_accumulator,
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
                token_accumulator=token_accumulator,
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
        session_key = self._session_key(
            session_id,
            workspace_id=workspace_id,
            mas_id=mas_id,
        )
        try:
            if session_key not in self._sessions:
                # If agent_replies were supplied the caller clearly intended a
                # /decide — the session simply doesn't exist yet (or was already
                # cleaned up).  Raise the dedicated 404 error so routes.py can
                # return the correct status code instead of 400.
                if agent_replies is not None:
                    logger.error(
                        "session not found session_id=%s workspace_id=%s mas_id=%s",
                        session_id,
                        workspace_id,
                        mas_id,
                    )
                    raise SemanticNegotiationSessionNotFoundError(session_id)
                logger.info("execute initiate session_id=%s", session_id)
                if agents_raw is None:
                    raise SemanticNegotiationInputError(
                        "agents information is required to initiate a session"
                    )

                # Create TokenAccumulator for tracking tokens across LLM calls
                token_accumulator = TokenAccumulator()

                # ── Step 1: Discover Issues ────────────────────────────────
                # IntentDiscovery extracts the negotiable entities (issues) from
                # the free-text description of the negotiation context.
                issues = self._intent_discovery.discover(
                    sentence=content_text,
                    agent_names=agent_names,
                    fabric_node_base_url=fabric_node_base_url,
                    workspace_id=workspace_id,
                    mas_id=mas_id,
                    token_accumulator=token_accumulator,
                )
                if hasattr(issues, "negotiable_entities"):
                    issues = issues.negotiable_entities
                logger.info(
                    "execute step1 issues_discovered session_id=%s issues=%s",
                    session_id,
                    issues,
                )

                # ── Step 2: Generate Options for Issues ────────────────────
                # OptionsGeneration produces a candidate list of values for each
                # discovered issue.  When a cognition fabric node is reachable,
                # shared-memory evidence is retrieved first and fed into the LLM
                # prompt; otherwise pure LLM generation is used.
                gen_out = self._options_generation.generate_options(
                    issues,
                    content_text,
                    agent_names=agent_names,
                    fabric_node_base_url=fabric_node_base_url,
                    workspace_id=workspace_id,
                    mas_id=mas_id,
                    token_accumulator=token_accumulator,
                )
                options_per_issue = gen_out.options_per_issue
                options_memory_blob = gen_out.memory_blob
                logger.info(
                    "execute step2 options_generated session_id=%s options_keys=%d memory_blob=%s",
                    session_id,
                    len(options_per_issue),
                    options_memory_blob is not None,
                )

                # ── Step 3: Negotiate ──────────────────────────────────────
                # BatchCallbackRunner runs the SAO (Stacked Alternating Offers)
                # mechanism turn-by-turn.  start_negotiation seeds the session
                # and returns the first round of agent messages; subsequent
                # rounds arrive via /decide → step_negotiation.
                if n_steps is not None:
                    # Caller explicitly provided a budget — use it as-is.
                    effective_n = n_steps
                else:
                    # Compute a dynamic budget from negotiation complexity.
                    # If self.n_steps > 0 it acts as a hard cap so operators
                    # can still control the upper bound via env var.
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
                sess.agent_names = agent_names
                sess.retry_history = []
                self._sessions[session_key] = (runner, sess, 0)
                logger.info(
                    "execute step3 negotiation_started session_id=%s n_steps=%d",
                    session_id,
                    effective_n,
                )
                # Convert accumulated tokens to metadata
                token_metadata = token_accumulator.to_metadata()

                return {
                    "status": "initiated",
                    "session_id": session_id,
                    "issues": issues,
                    "options_per_issue": options_per_issue,
                    "options_memory_blob": options_memory_blob,
                    "n_steps": effective_n,
                    "round": 1,
                    "messages": messages,
                    "token_metadata": token_metadata,
                }

            # ── Step 3 (continue): Decide ──────────────────────────────────
            # Session already exists — advance the SAO by one batch of replies.
            logger.info(
                "execute decide session_id=%s replies=%d",
                session_id,
                len(agent_replies or []),
            )
            try:
                runner, sess, retry_count = self._sessions[session_key]
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
        participant_id_by_name = {p.name: p.id for p in sess.participants}

        # ── Step 4: Semantic alignment validation ─────────────────────────
        # Run before building the commit so callers can act on the
        # recommendation (accept / request_justification / escalate).
        # ``sess.sstp_message_trace`` may omit gateway-style initiate rows and may
        # store bare agent dicts (CFN path). Pass an adapted *copy* so ACSE
        # ``validate_input`` succeeds without changing persisted trace or CFN API.
        validation = self.run_alignment_validation(
            adapt_sstp_trace_for_alignment_validation(
                sess.sstp_message_trace,
                session_id=sess.session_id,
                content_text=sess.content_text or "",
                issues=sess.issues,
                options_per_issue=sess.options_per_issue,
                participants=sess.participants,
                n_steps=sess.n_steps,
            ),
            status=status,
        )
        if validation is not None:
            logger.info(
                "execute step4 alignment_validation complete "
                "session_id=%s severity=%s score=%.2f needs_intervention=%s "
                "recommendation=%s timed_out=%s",
                session_id,
                validation.severity,
                validation.alignment_score,
                validation.needs_intervention,
                validation.recommendation,
                validation.timed_out,
            )
        else:
            logger.warning(
                "execute step4 alignment_validation skipped session_id=%s", session_id
            )

        validation_dict = {
            "needs_intervention": validation.needs_intervention,
            "should_retry": validation.should_retry,
            "severity": validation.severity,
            "alignment_score": validation.alignment_score,
            "cognitive_alignment": validation.cognitive_alignment,
            "agreement_coherence": validation.agreement_coherence,
            "failure_modes": validation.failure_modes,
            "timed_out": validation.timed_out,
            "cross_issue_conflicts": validation.cross_issue_conflicts,
            "reasoning": validation.reasoning,
            "recommendation": validation.recommendation,
            "heuristic_scores": (
                dataclasses.asdict(validation.heuristic_scores)
                if validation.heuristic_scores is not None else None
            ),
        } if validation is not None else None

        # ── Step 4b: Retry if SAV recommends it and budget remains ────────
        if self._should_retry(validation, retry_count):
            attempt_record = {
                "attempt": retry_count + 1,
                "options_per_issue": sess.options_per_issue,
                "final_agreement": (
                    [{"issue_id": o.issue_id, "chosen_option": o.chosen_option}
                     for o in result.agreement]
                    if result.agreement else None
                ),
                "validation": validation_dict,
            }
            sess.retry_history.append(attempt_record)
            logger.warning(
                "RETRY TRIGGERED session_id=%s attempt=%d/%d "
                "severity=%s score=%.3f cognitive=%.3f coherence=%.3f "
                "failure_modes=%s reasoning=%r",
                session_id,
                retry_count + 1,
                self._alignment_validator._config.retry_max_attempts,
                validation.severity if validation else "?",
                validation.alignment_score if validation else 0.0,
                validation.cognitive_alignment if validation else 0.0,
                validation.agreement_coherence if validation else 0.0,
                validation.failure_modes if validation else [],
                (validation.reasoning or "")[:120] if validation else "",
            )
            logger.info(
                "RETRY options_before session_id=%s %s",
                session_id,
                {iss: opts for iss, opts in (sess.options_per_issue or {}).items()},
            )
            messages = self._run_retry(session_key, sess, retry_count, validation)
            return {
                "status": "ongoing",
                "session_id": session_id,
                "round": 1,
                "messages": messages,
            }

        # No more retries — commit and clean up.
        if validation is not None and validation.should_retry and retry_count >= self._alignment_validator._config.retry_max_attempts:
            logger.warning(
                "RETRY BUDGET EXHAUSTED session_id=%s attempts_used=%d severity=%s score=%.3f cognitive=%.3f coherence=%.3f",
                session_id,
                retry_count,
                validation.severity,
                validation.alignment_score,
                validation.cognitive_alignment,
                validation.agreement_coherence,
            )
        elif validation is not None and not validation.should_retry:
            logger.info(
                "NO RETRY NEEDED session_id=%s severity=%s score=%.3f needs_intervention=%s",
                session_id,
                validation.severity,
                validation.alignment_score,
                validation.needs_intervention,
            )

        del self._sessions[session_key]
        retry_history: list = getattr(sess, "retry_history", [])

        # Log the full retry outcome once we have the final validation result.
        # Shows whether the retry loop actually improved semantic alignment.
        if retry_count > 0 and validation is not None and retry_history:
            _SEV_RANK = {"high": 2, "medium": 1, "low": 0}
            first_val = retry_history[0].get("validation") or {}
            initial_sev = (first_val.get("severity") or "?").lower()
            initial_score = first_val.get("alignment_score") or 0.0
            final_sev = (validation.severity or "?").lower()
            final_score = validation.alignment_score

            rank_before = _SEV_RANK.get(initial_sev, -1)
            rank_after  = _SEV_RANK.get(final_sev,   -1)
            if rank_after < rank_before:
                verdict = "SEVERITY_IMPROVED"
            elif rank_after == rank_before and final_score > initial_score:
                verdict = "SCORE_IMPROVED"
            elif rank_after == rank_before:
                verdict = "UNCHANGED"
            else:
                verdict = "WORSENED"

            # Build progression string across all attempts + final
            attempt_scores = [
                f"{(att.get('validation') or {}).get('severity','?').upper()}"
                f"/{(att.get('validation') or {}).get('alignment_score', 0.0):.3f}"
                for att in retry_history
            ]
            progression = "  →  ".join(attempt_scores) + f"  →  {final_sev.upper()}/{final_score:.3f} [final]"

            logger.warning(
                "RETRY OUTCOME session_id=%s attempts=%d verdict=%s "
                "initial=%s/%.3f  final=%s/%.3f  progression: %s",
                session_id,
                retry_count,
                verdict,
                initial_sev.upper(), initial_score,
                final_sev.upper(), final_score,
                progression,
            )

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
                validation=validation_dict,
                retry_history=retry_history,
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
            "validation": validation_dict,
            "retry_history": retry_history,
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
        validation: dict | None = None,
        retry_history: list | None = None,
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
                # Include per-agent reason when present (explainability; not used by SAO).
                decisions = [
                    AgentDecision(
                        participant_id=d["participant_id"],
                        action=d["action"],
                        offer=d.get("offer"),
                        reason=d.get("reason"),
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
                    **({"validation": validation, "retry_history": retry_history or []}
                       if validation else {}),
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

    def release_session(
        self,
        session_id: str,
        workspace_id: str | None = None,
        mas_id: str | None = None,
    ) -> None:
        """Remove a session from the internal store (e.g. on error clean-up)."""
        if workspace_id is not None or mas_id is not None:
            self._sessions.pop(
                self._session_key(
                    session_id,
                    workspace_id=workspace_id,
                    mas_id=mas_id,
                ),
                None,
            )
            return

        stale_keys = [key for key in self._sessions if key[2] == session_id]
        for key in stale_keys:
            self._sessions.pop(key, None)
