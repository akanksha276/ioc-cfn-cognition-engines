# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Options generation — component 2 of the semantic negotiation pipeline.

For each issue identified by component 1 (intent discovery), this component
generates candidate options that the negotiating agents could agree upon.
It has access to the original context as well as each agent and its memory.

Three strategies:
1. LLM-only: LLM proposes plausible meanings using its own judgment.
2. Memory + LLM: Fetch context/preferences via an injected or fabric HTTP lookup, then LLM generates options.
3. Agent query: Ask the agents that sent the message for their interpretation (mocked).

Run as a module (recommended) from ``ioc-cfn-cognitive-agents``::

    PYTHONPATH=semantic_negotiation poetry run python -m app.agent.options_generation

Or after activating the venv, from ``semantic_negotiation``::

    PYTHONPATH=. python -m app.agent.options_generation

Direct ``python path/to/options_generation.py`` is supported: the repo root
``semantic_negotiation/`` is prepended to ``sys.path`` so ``app.*`` imports resolve.

"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ``.../semantic_negotiation/app/agent/this_file.py`` → parent of package ``app``
_semantic_negotiation_root = Path(__file__).resolve().parents[2]
if str(_semantic_negotiation_root) not in sys.path:
    sys.path.insert(0, str(_semantic_negotiation_root))

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
import time

from ..config.utils import litellm_completion_compat

from ..config.settings import settings
from .token_tracker import TokenAccumulator

from app.agent.http_repo import (
    SharedMemoryQueryError,
    SharedMemoryNotFoundError,
    gather_shared_memories_queries,
    issue_labels_from_negotiable_entities,
    run_coro_in_own_loop,
    shared_memories_query_path,
)
from app.config.utils import get_llm_provider

logger = logging.getLogger(__name__)


def _agents_phrases(agent_names: Optional[List[str]]) -> tuple[str, str]:
    """``(agents_line, agents_in_questions)`` for header and question text."""
    if agent_names:
        if isinstance(agent_names, (list, tuple, set)):
            agents_line = ", ".join(str(n) for n in agent_names)
        else:
            agents_line = str(agent_names)
    else:
        agents_line = "(not specified)"
    if agents_line != "(not specified)":
        agents_in_questions = agents_line
    else:
        agents_in_questions = "the negotiating agents (agent names not provided)"
    return agents_line, agents_in_questions


def _memory_query_header(mission: str, ctx: str, agents_line: str) -> str:
    return (
        "Evidence gathering query for semantic negotiation.\n\n"
        f"Mission: {mission}\n"
        f"Context: {ctx}\n"
        f"Agents: {agents_line}\n"
    )


def _queries_for_issue_block(issue: str, agents_in_questions: str) -> str:
    q = (
        f'1. What are the following agents\' {agents_in_questions} interpretations / preferences of the term "{issue}", in the given context?\n'
        f'2. From previous negotiation history, what options for "{issue}" were preferred by '
        f"{agents_in_questions}?\n"
        f'3. What information from memory is relevant to generating better options for "{issue}" '
        f"for {agents_in_questions}?"
    )
    return f"## Issue: {issue}\n{q}"


def build_evidence_lookup_intent_for_issue(
    sentence: str,
    context: Optional[str],
    issue: str,
    agent_names: Optional[List[str]],
) -> str:
    """
    Natural-language intent for a **single** negotiable issue (one shared-memories HTTP body).
    """
    mission = (sentence or "").strip() or "(not specified)"
    ctx = (context or "").strip() or "not specified"
    agents_line, agents_in_questions = _agents_phrases(agent_names)
    header = _memory_query_header(mission, ctx, agents_line)
    body = _queries_for_issue_block(issue, agents_in_questions)
    return f"{header}\n{body}"


def build_evidence_lookup_intent(
    sentence: str,
    context: Optional[str],
    negotiable_entities: Optional[list[Any]],
    agent_names: Optional[List[str]],
) -> str:
    """
    Natural-language intent combining all issues in one string (documentation / debugging).

    Intent discovery uses :func:`app.agent.intent_discovery.build_intent_discovery_shared_memory_intent`
    instead (mission + agents only). Options generation uses
    :func:`build_evidence_lookup_intent_for_issue` per issue with
    :func:`app.agent.http_repo.post_shared_memories_query` in a loop.
    """
    mission = (sentence or "").strip() or "(not specified)"
    ctx = (context or "").strip() or "not specified"
    issues = issue_labels_from_negotiable_entities(negotiable_entities)
    if not issues:
        issues = ["(no specific issue label yet—use mission and context to scope evidence)"]
    agents_line, agents_in_questions = _agents_phrases(agent_names)
    header = _memory_query_header(mission, ctx, agents_line)
    body = "\n\n".join(_queries_for_issue_block(i, agents_in_questions) for i in issues)
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# Mock: ask sending agents for their interpretation (to be replaced later)
# ---------------------------------------------------------------------------

def mock_agent_interpretation_query(
    negotiable_entities: list[Any],
    sentence: str,
    context: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> dict[str, list[str]]:
    """
    Stub for querying agents that sent the message for how they interpret each negotiable entity.

    Returns a mapping from term (str) to list of interpretations (one per responding agent).
    Replace with real agent query later.
    """
    # Mock: one or two fake interpretations per term
    result: dict[str, list[str]] = {}
    for t in negotiable_entities:
        term = getattr(t, "term", str(t))
        if term not in result:
            result[term] = []
        if "inexpensive" in term.lower() or "affordable" in term.lower():
            result[term].extend(["under $15 per person", "mid-range; not fine dining"])
        elif "soon" in term.lower():
            result[term].extend(["within the next 2–3 days", "this week"])
        else:
            result[term].extend([f"interpretation of '{term}' from agent (mock)"])
    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InterpretationOption:
    """A single possible meaning/interpretation for an ambiguous term."""
    value: str
    source: str  # "llm" | "memory_llm" | "agent"


@dataclass
class TermOptions:
    """All generated options for one ambiguous term."""
    term: str
    reasoning: str = ""
    options: list[InterpretationOption] = field(default_factory=list)


@dataclass
class OptionsGenerationResult:
    """Result of generating options for all ambiguous terms."""
    sentence: str
    context: Optional[str] = None
    term_options: list[TermOptions] = field(default_factory=list)
    strategy_used: str = ""  # "llm_only" | "memory_llm" | "agent"

    # Return a mapping from term to list of option strings (e.g. {"firewood": ["...", "..."], ...}) as needed by the negotiation protocol
    def options_by_term(self) -> dict[str, list[str]]:
        """Return options as a mapping from term to list of option strings (for negotiation)."""
        out: dict[str, list[str]] = {}
        for to in self.term_options:
            out[to.term] = [str(o.value) for o in to.options]
        return out


@dataclass
class OptionsGenerationOutput:
    """Public return type for :class:`OptionsGeneration` strategies."""

    options_per_issue: dict[str, list[str]]
    """JSON string of fabric shared-memory lookup (same text passed into the memory LLM), else ``None``."""
    memory_blob: Optional[str] = None

    def options_by_term(self) -> dict[str, list[str]]:
        """Alias for :attr:`options_per_issue` (compat with scripts expecting a mapping method)."""
        return self.options_per_issue


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_LLM_ONLY_PROMPT = """Read the context thoroughly. The context contains the premise or mission of the application and the current conversation. From it, infer the kind of negotiation at hand: negotiation may be about (a) different interpretations of a word or phrase (e.g. what "affordable" or "soon" means), or (b) negotiating the amount or quantity of an entity (e.g. how many units, what level). Deduce which applies—or both—based on the context and on each negotiable entity below.

For each negotiable entity, suggest 2–4 concrete options that agents could negotiate over. If the entity is interpretation-heavy, suggest distinct plausible meanings; if it is quantity- or amount-heavy, suggest plausible quantities, levels, or integers as appropriate—infer from context whether options should be numeric. Use the sentence and context to keep options relevant.

Sentence: "{sentence}"
Context: {context}

Negotiable entities (with reasoning from intent discovery):
{terms_blob}

Output format—this exact JSON structure only:
{{
  "options_per_term": [
    {{
      "term": "exact term from the list",
      "options": ["option 1", "option 2", ...]
    }}
  ]
}}

Output (JSON only):"""

_MEMORY_LLM_PROMPT = """Read the context thoroughly; it contains the premise or mission and additioinal context from the MAS. Infer the kind of negotiation: (a) different interpretations of a word or phrase, or (b) negotiating amount/quantity of an entity—or both. Using the sentence, context, and the following memory/preferences, suggest 2–4 concrete options for negotiation for each negotiable entity. Prefer options that align with stated preferences where relevant. Output ONLY valid JSON.
Go over the context carefully. It contains preferences or past context relevant to the negotiation. Use this knowlede to generate better options.

Sentence: "{sentence}"
Context: {context}

Memory / preferences:
{memory_blob}

Negotiable entities (with reasoning from intent discovery):
{terms_blob}"""

_RECORD_OPTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_options",
        "description": "Record the negotiation options generated for each term.",
        "parameters": {
            "type": "object",
            "properties": {
                "options_per_term": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string", "description": "Exact term from the negotiable entities list"},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "2-4 concrete options agents could negotiate over",
                            },
                        },
                        "required": ["term", "options"],
                    },
                },
            },
            "required": ["options_per_term"],
        },
    },
}


# ---------------------------------------------------------------------------
# Option generator
# ---------------------------------------------------------------------------

class OptionsGeneration:
    """
    Generates interpretation options for ambiguous terms so agents can negotiate.

    Supports three strategies: LLM-only, memory + LLM, and query sending agents (mocked).
    """

    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        *,
        agent_interpretation_query: Optional[Callable[..., dict[str, list[str]]]] = None,
    ):
        """
        Args:
            llm_provider: Unused; kept for backwards-compatible constructor signature.
                          LLM calls now go through litellm using settings.llm_model.
            agent_interpretation_query: Optional agent query function for strategy 3.
        """
        self._agent_query = agent_interpretation_query or mock_agent_interpretation_query
        logger.info("OptionsGeneration initialized")

    def _call_llm_tool(
        self,
        prompt: str,
        negotiable_entities: list[Any],
        source: str,
        token_accumulator: Optional[TokenAccumulator] = None,
    ) -> list[TermOptions]:
        """Call litellm with the record_options tool and parse the result into TermOptions."""
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_RECORD_OPTIONS_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "record_options"}},
        }
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url


        start_time = time.time()
        resp = litellm_completion_compat(**kwargs)
        latency_ms = (time.time() - start_time) * 1000

        if token_accumulator:
            token_accumulator.add(resp.usage)
            token_accumulator.add_latency(latency_ms)
            token_accumulator.set_model(resp.model)


        by_term: dict[str, list[str]] = {}
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            raw = tool_calls[0].function.arguments
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                data = {}
            for item in data.get("options_per_term") or []:
                if isinstance(item, dict) and item.get("term") is not None:
                    t = str(item["term"]).strip()
                    opts = item.get("options") or []
                    by_term[t] = [str(o).strip() for o in opts if o]

        term_options: list[TermOptions] = []
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            opts = by_term.get(term) or []
            term_options.append(TermOptions(
                term=term,
                reasoning=reasoning,
                options=[InterpretationOption(value=o, source=source) for o in opts],
            ))
        return term_options

    def generate_options_llm_only(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
        token_accumulator: Optional[TokenAccumulator] = None,
    ) -> OptionsGenerationOutput:
        """
        Strategy 1: LLM proposes plausible meanings using its own judgment.
        Returns :class:`OptionsGenerationOutput` with ``memory_blob=None``.
        """
        if not negotiable_entities:
            logger.info("generate_options_llm_only: empty negotiable_entities")
            return OptionsGenerationOutput(options_per_issue={}, memory_blob=None)

        logger.info(
            "generate_options_llm_only entity_count=%d",
            len(negotiable_entities),
        )
        terms_blob = self._format_terms_for_prompt(negotiable_entities)
        context_str = context if context else "not specified"
        prompt = _LLM_ONLY_PROMPT.format(sentence=sentence, context=context_str, terms_blob=terms_blob)
        term_options = self._call_llm_tool(prompt, negotiable_entities, source="llm", token_accumulator=token_accumulator)
        result = OptionsGenerationResult(
            sentence=sentence, context=context, term_options=term_options, strategy_used="llm_only",
        )
        out = OptionsGenerationOutput(
            options_per_issue=result.options_by_term(),
            memory_blob=None,
        )
        logger.info(
            "generate_options_llm_only done terms_with_options=%d",
            len(out.options_per_issue),
        )
        return out

    def generate_options_with_memory(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
        agent_names: Optional[List[str]] = None,
        fabric_node_base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
        mas_id: Optional[str] = None,
        token_accumulator: Optional[TokenAccumulator] = None,
    ) -> OptionsGenerationOutput:
        """
        Strategy 2: Resolve memory via *fabric_node_base_url* / *workspace_id* / *mas_id*,
        then pass the result to the LLM.

        If that triple is not set, uses :meth:`generate_options_llm_only`.
        If the HTTP lookup raises :exc:`SharedMemoryQueryError` (any non-success HTTP
        status, connection failure, or timeout from :func:`post_shared_memories_query`),
        falls back to :meth:`generate_options_llm_only`.

        On success, :attr:`OptionsGenerationOutput.memory_blob` is the JSON string
        passed into the memory LLM (fabric lookup payload).
        """
        if not negotiable_entities:
            logger.info("generate_options_with_memory: empty negotiable_entities")
            return OptionsGenerationOutput(options_per_issue={}, memory_blob=None)
        if not (fabric_node_base_url and workspace_id and mas_id):
            logger.info(
                "Memory + LLM requires fabric_node_base_url, workspace_id, and mas_id; "
                "falling back to LLM-only options."
            )
            return self.generate_options_llm_only(
                negotiable_entities, sentence, context, token_accumulator=token_accumulator
            )
        issues = issue_labels_from_negotiable_entities(negotiable_entities)
        if not issues:
            issues = [
                "(no specific issue label yet—use mission and context to scope evidence)"
            ]
        base = fabric_node_base_url.rstrip("/")
        path = shared_memories_query_path(workspace_id, mas_id)
        logger.info(
            "generate_options_with_memory workspace=%s mas=%s issues=%d base=%s",
            workspace_id,
            mas_id,
            len(issues),
            base,
        )
        memory_data: dict[str, Any]
        try:

            async def _fetch_fabric_evidence() -> dict[str, Any]:
                intents = [
                    build_evidence_lookup_intent_for_issue(
                        sentence, context, issue, agent_names
                    )
                    for issue in issues
                ]
                logger.info(
                    "generate_options_with_memory: parallel fabric POSTs path=%s intents=%d "
                    "intent_chars=%d timeout_s=%s",
                    path,
                    len(intents),
                    sum(len(s) for s in intents),
                    120.0,
                )
                rows = await gather_shared_memories_queries(
                    base, path, intents, timeout=120.0
                )
                rows_aligned: list[Any] = list(rows)
                if len(rows_aligned) != len(issues):
                    logger.warning(
                        "generate_options_with_memory: fabric row count mismatch "
                        "issues=%d rows=%d; padding with None or truncating extras.",
                        len(issues),
                        len(rows_aligned),
                    )
                    if len(rows_aligned) < len(issues):
                        rows_aligned.extend([None] * (len(issues) - len(rows_aligned)))
                    else:
                        rows_aligned = rows_aligned[: len(issues)]
                if rows_aligned and all(data is None for data in rows_aligned):
                    raise SharedMemoryQueryError(
                        "All fabric shared-memories queries timed out",
                        status_code=None,
                    )
                by_issue: list[dict[str, Any]] = []
                message_sections: list[str] = []
                response_ids: list[str] = []
                for issue, data in zip(issues, rows_aligned, strict=True):
                    if data is None:
                        msg = "(fabric evidence request timed out)"
                        rid = None
                    else:
                        msg = data.get("message")
                        rid = data.get("response_id")
                    by_issue.append({"issue": issue, "message": msg, "response_id": rid})
                    message_sections.append(
                        f"## Evidence for issue: {issue}\n"
                        f"{msg if msg is not None else '(no message)'}"
                    )
                    if rid is not None:
                        response_ids.append(str(rid))
                return {
                    "evidence_by_issue": by_issue,
                    "evidence_message": "\n\n".join(message_sections),
                    "evidence_response_id": ";".join(response_ids) if response_ids else None,
                    "source": "fabric_node_shared_memories_query",
                }

            memory_data = run_coro_in_own_loop(_fetch_fabric_evidence())
        except (SharedMemoryQueryError, SharedMemoryNotFoundError) as exc:
            logger.warning(
                "generate_options_with_memory: fabric lookup failed "
                "(http_status=%s), falling back to LLM-only: %s",
                exc.status_code,
                exc,
            )
            return self.generate_options_llm_only(
                negotiable_entities, sentence, context, token_accumulator=token_accumulator
            )
        #print(f"Memory data: {memory_data}")
        memory_blob = json.dumps(memory_data, indent=2)
        terms_blob = self._format_terms_for_prompt(negotiable_entities)
        context_str = context if context else "not specified"
        prompt = _MEMORY_LLM_PROMPT.format(
            sentence=sentence, context=context_str, memory_blob=memory_blob, terms_blob=terms_blob,
        )
        term_options = self._call_llm_tool(prompt, negotiable_entities, source="memory_llm", token_accumulator=token_accumulator)
        result = OptionsGenerationResult(
            sentence=sentence, context=context, term_options=term_options, strategy_used="memory_llm",
        )
        out = OptionsGenerationOutput(
            options_per_issue=result.options_by_term(),
            memory_blob=memory_blob,
        )
        logger.info(
            "generate_options_with_memory done terms_with_options=%d",
            len(out.options_per_issue),
        )
        return out

    def generate_options_from_agents(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> OptionsGenerationOutput:
        """
        Strategy 3: Ask the agents that sent the message for their interpretation (mocked).
        Returns :class:`OptionsGenerationOutput` with ``memory_blob=None``.
        """
        if not negotiable_entities:
            logger.info("generate_options_from_agents: empty negotiable_entities")
            return OptionsGenerationOutput(options_per_issue={}, memory_blob=None)

        logger.info(
            "generate_options_from_agents entity_count=%d",
            len(negotiable_entities),
        )
        agent_interpretations = self._agent_query(negotiable_entities, sentence, context, sender_id)
        term_options: list[TermOptions] = []
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            opts = agent_interpretations.get(term) or []
            term_options.append(TermOptions(
                term=term,
                reasoning=reasoning,
                options=[InterpretationOption(value=o, source="agent") for o in opts],
            ))
        result = OptionsGenerationResult(
            sentence=sentence,
            context=context,
            term_options=term_options,
            strategy_used="agent",
        )
        out = OptionsGenerationOutput(
            options_per_issue=result.options_by_term(),
            memory_blob=None,
        )
        logger.info(
            "generate_options_from_agents done terms_with_options=%d",
            len(out.options_per_issue),
        )
        return out

    def generate_options(
        self,
        negotiable_entities: list[Any],
        sentence: str,
        context: Optional[str] = None,
        agent_names: Optional[List[str]] = None,
        *,
        fabric_node_base_url: Optional[str] = None,
        workspace_id: Optional[str] = None,
        mas_id: Optional[str] = None,
        token_accumulator: Optional[TokenAccumulator] = None,
    ) -> OptionsGenerationOutput:
        """
        Generate options for negotiable entities.

        When *fabric_node_base_url*, *workspace_id*, and *mas_id* are all set,
        delegates to :meth:`generate_options_with_memory` (evidence / shared memory
        then LLM). Otherwise uses :meth:`generate_options_llm_only`.
        """
        use_memory = bool(fabric_node_base_url and workspace_id and mas_id)
        logger.info(
            "generate_options entity_count=%d use_memory=%s",
            len(negotiable_entities),
            use_memory,
        )
        if fabric_node_base_url and workspace_id and mas_id:
            return self.generate_options_with_memory(
                negotiable_entities,
                sentence,
                context=context,
                agent_names=agent_names,
                fabric_node_base_url=fabric_node_base_url,
                workspace_id=workspace_id,
                mas_id=mas_id,
                token_accumulator=token_accumulator,
            )
        return self.generate_options_llm_only(negotiable_entities, sentence, context, token_accumulator=token_accumulator)

    def _format_terms_for_prompt(self, negotiable_entities: list[Any]) -> str:
        lines = []
        for t in negotiable_entities:
            term = getattr(t, "term", str(t))
            reasoning = getattr(t, "reasoning", "") or ""
            lines.append(f"- \"{term}\": {reasoning}")
        return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_option_generator() -> None:
    """Run a quick test of all three strategies with mock data."""
    try:
        from agent.intent_discovery import IntentDiscovery
    except ImportError:
        from intent_discovery import IntentDiscovery

    discovery = IntentDiscovery()
    # CaSiNo-style utterance: campsite negotiation over Food, Water, Firewood
    #sentence = "I brought way too much raw meat, which means I'll need some firewood to cook it. I didn't bring enough water and I'm going to do a ton of hiking, so I need to stay hydrated. I already have plenty of food, and I can cook more, which is why I prioritize firewood over this."
    #context = "Two campsite neighbors negotiate for Food, Water, and Firewood packages, based on their individual preferences and requirements"
    sentence = "The people team and the executive leadership need to align on a working arrangement that allows the company to attract the best talent regardless of location, keeps teams genuinely collaborative and productive, protects the culture that makes the organisation successful, and treats all employees fairly regardless of where they choose to work."
    context = ""
    negotiable_entities = discovery.discover(sentence, context=context).negotiable_entities
    if not negotiable_entities:
        print("No negotiable_entities entities found; using mock list for demo (CaSiNo lexicon).")
        negotiable_entities = ["need", "enough", "prioritize"]

    gen = OptionsGeneration()

    agent_names = ["Agent 1", "Agent 2"]
    fabric_node_base_url = "http://localhost:9000"
    workspace_id = "123"
    mas_id = "456"

    print(f"Strategy 0 : Memory + LLM with fallback to LLM-only")
    out0 = gen.generate_options(negotiable_entities, sentence, context, agent_names=agent_names, fabric_node_base_url=fabric_node_base_url, workspace_id=workspace_id, mas_id=mas_id)
    print(out0.options_per_issue, out0.memory_blob is not None)

    '''
    print("Strategy 1: LLM-only")
    out1 = gen.generate_options_llm_only(negotiable_entities, sentence, context)
    for term, options in out1.items():
        print(f"  {term}: {options}")

    print("\nStrategy 2: Memory + LLM (no fabric triple → LLM-only fallback)")
    out2 = gen.generate_options_with_memory(negotiable_entities, sentence, context)
    for term, options in out2.items():
        print(f"  {term}: {options}")

    print("\nStrategy 3: Agent interpretations (mocked)")
    out3 = gen.generate_options_from_agents(negotiable_entities, sentence, context)
    for term, options in out3.items():
        print(f"  {term}: {options}")
    '''


if __name__ == "__main__":
    test_option_generator()

