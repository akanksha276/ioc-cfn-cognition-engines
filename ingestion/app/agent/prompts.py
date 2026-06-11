# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Format-specific system prompts for concept and relationship extraction.

- observe-sdk-otel has its own prompt pair (OTEL trace spans).
- openclaw and locomo share a single prompt pair (general-purpose records).
"""

# ---------------------------------------------------------------------------
# observe-sdk-otel
# ---------------------------------------------------------------------------

OTEL_CONCEPTS_PROMPT = (
    "You are an expert in distributed-systems observability, OpenTelemetry trace analysis, "
    "and knowledge-graph construction.\n\n"
    "Remember that these concepts form the memory of a multi-agent system. So make sure to "
    "include all the concepts that are relevant to the system."
    "You will receive a JSON array of distilled OpenTelemetry trace spans. Each span may "
    "contain fields such as ServiceName, agent_id, model, system_prompt, user_prompt, "
    "functions (with name, description, parameters), tool_calls, and completion.\n\n"
    "Your task is to identify an EXHAUSTIVE list of all important CONCEPTS present in or "
    "implied by the trace data.\n\n"

    "### What counts as a concept\n"
    "A concept is any distinct entity, actor, capability, data artifact, or domain idea "
    "that plays a meaningful role in the traced system. Concepts fall into the following "
    "categories (use exactly these type labels):\n\n"
    "  - **query**   – The original user question or request that initiated the trace.\n"
    "  - **agent**   – An autonomous or semi-autonomous software agent identified by agent_id.\n"
    "  - **service** – A named micro-service or application (ServiceName).\n"
    "  - **llm**     – A large-language-model endpoint identified by its model name.\n"
    "  - **tool**    – An external tool invoked via tool_calls (e.g., search, calculator, API).\n"
    "  - **function** – A function registered on a span (llm.request.functions) that an agent "
    "or LLM can call.\n"
    "  - **request** – A request sent to a service, agents, llm or tool.\n"
    "  - **response** – A response received from a service, agents, llm or tool.\n"
    "  - **output**  – The final answer, response, or artifact produced at the end of the trace.\n"
    "  - **other_concept** – A higher-level domain idea, capability, or data entity mentioned "
    "in system prompts, user prompts, function descriptions, or completions that is important "
    "for understanding what the system does.\n\n"

    "### Extraction instructions\n"
    "1. Scan EVERY span in the payload. Do not skip any.\n"
    "2. For each span extract any concept whose name appears in ServiceName, agent_id, model, "
    "   function names, tool_call names, user_prompt content, system_prompt content, or "
    "   completion content.\n"
    "3. For **query** concepts: distil the core user question from the user_prompt field. "
    "   Use a concise but complete sentence as the name.\n"
    "4. For **output** concepts: distil the final answer from the completion field of the "
    "   last span that has one. Use a concise summary as the name.\n"
    "5. For **other_concept** entries: look inside system prompts and function descriptions "
    "   for important domain terms, methodologies, or capabilities the system is designed around.\n"
    "6. DEDUPLICATE: if the same logical entity appears under slightly different names across "
    "   spans, emit it only once with the most canonical name.\n"
    "7. Every concept MUST have a detailed, informative description (2-4 sentences) explaining "
    "   what it is, what role it plays in the system, and any notable behaviour observed in "
    "   the traces.\n\n"

    "Return ONLY the list of concepts. Do NOT include relationships."
)

OTEL_RELATIONSHIPS_PROMPT = (
    "You are an expert in distributed-systems observability, OpenTelemetry trace analysis, "
    "and knowledge-graph construction.\n\n"
    "Remember that these relationships form the memory of a multi-agent system. So make sure "
    "to relate all the concepts that are relevant to the system."
    "You will receive TWO pieces of information:\n"
    "  1. A list of CONCEPTS (each with name, type, and description) that were previously "
    "     extracted from the trace data.\n"
    "  2. The original JSON array of distilled OpenTelemetry trace spans from which those "
    "     concepts were extracted.\n\n"
    "Your task is to identify ALL meaningful RELATIONSHIPS between the provided concepts.\n\n"

    "### Relationship extraction instructions\n"
    "1. Consider every possible pair of concepts and determine whether the trace data "
    "   evidences a meaningful interaction, dependency, data flow, or semantic link "
    "   between them.\n"
    "2. The 'source' and 'target' of each relationship MUST be exact names from the "
    "   provided concepts list. Do NOT invent new concept names.\n"
    "3. Relationship labels MUST be in UPPER_SNAKE_CASE and should be descriptive verb "
    "   phrases that capture the nature of the interaction (e.g., SENDS_PROMPT_TO, "
    "   INVOKES_TOOL, ORCHESTRATES, DELEGATES_TASK_TO, QUERIES_MODEL, EXECUTES_FUNCTION, "
    "   PRODUCES_OUTPUT, ANSWERS_QUERY, HOSTS_AGENT, REGISTERS_FUNCTION, LEVERAGES, "
    "   DEPENDS_ON).\n"
    "4. Each relationship MUST include a one-sentence description that explains what "
    "   information or control flows between the source and target, grounded in the "
    "   trace evidence.\n\n"

    "### Quality guidelines\n"
    "1. FOCUS on abstract, higher-level relationships rather than low-level implementation "
    "   details.\n"
    "2. MERGE similar or closely related relationships into a single broader relationship "
    "   to avoid redundancy.\n"
    "3. AVOID overlapping relationships that represent the same underlying idea.\n"
    "4. ENSURE each relationship is truly distinct and adds unique informational value.\n"
    "5. PRIORITIZE relationships that represent core architectural patterns, data flows, "
    "   and functional dependencies.\n"
    "6. Every concept should participate in at least one relationship. If a concept is "
    "   completely isolated, reconsider whether a relationship was missed.\n\n"

    "Return ONLY the list of relationships."
)

# ---------------------------------------------------------------------------
# openclaw / locomo  (shared)
# ---------------------------------------------------------------------------

GENERAL_CONCEPTS_PROMPT = (
    "You are an expert in multi-agent systems, conversational AI, dialogue analysis, "
    "and knowledge-graph construction.\n\n"
    "Remember that these concepts form the memory of a multi-agent system. Make sure to "
    "include all the concepts that are relevant to the system.\n\n"
    "You will receive a JSON array of records. Each record may contain fields describing "
    "agents, speakers, actions, utterances, documents, entities, topics, intents, tools, "
    "inputs, outputs, context, and metadata.\n\n"
    "Your task is to identify an EXHAUSTIVE list of all important CONCEPTS present in or "
    "implied by the data.\n\n"

    "### What counts as a concept\n"
    "A concept is any distinct entity, actor, capability, data artifact, or domain idea "
    "that plays a meaningful role in the system. Concepts fall into the following "
    "categories (use exactly these type labels):\n\n"
    "  - **query**    – The original user question or request that initiated the workflow.\n"
    "  - **agent**    – An autonomous or semi-autonomous software agent.\n"
    "  - **speaker**  – A participant in a conversation (user, assistant, or system).\n"
    "  - **service**  – A named service or application component.\n"
    "  - **llm**      – A large-language-model endpoint identified by its model name.\n"
    "  - **tool**     – An external tool invoked by an agent or system.\n"
    "  - **function** – A callable function exposed by the system.\n"
    "  - **document** – A document, contract, clause, or file being processed.\n"
    "  - **entity**   – A named entity (person, organization, location, date, time, etc.) "
    "extracted from or mentioned in the data.\n"
    "  - **topic**    – A subject or theme discussed in a conversation or document.\n"
    "  - **intent**   – A user intent or goal expressed in the interaction.\n"
    "  - **fact**     – A factual statement or piece of knowledge exchanged.\n"
    "  - **output**   – The final answer, response, or artifact produced.\n"
    "  - **other_concept** – Any other higher-level domain idea, capability, mention of date or time or data "
    "entity that is important for understanding what the system does.\n\n"

    "### Extraction instructions\n"
    "1. Scan EVERY record in the payload. Do not skip any.\n"
    "2. Extract concepts from all available fields: agent names, speakers, actions, "
    "   utterances, topics, intents, entities, dates and times, temporal references, document identifiers, input/output "
    "   text, tool names, context, and metadata.\n"
    "3. For **query** concepts: distil the core user question or request.\n"
    "4. For **output** concepts: distil the final answer or result.\n"
    "5. For **fact** concepts: extract key factual claims or knowledge exchanged including dates and times and temporal references.\n"
    "6. DEDUPLICATE: if the same logical entity appears under slightly different names, "
    "   emit it only once with the most canonical name.\n"
    "7. Every concept MUST have a detailed, informative description (2-4 sentences) "
    "   explaining what it is, what role it plays, and any notable behaviour observed.\n\n"

    "Return ONLY the list of concepts. Do NOT include relationships."
)

GENERAL_RELATIONSHIPS_PROMPT = (
    "You are an expert in multi-agent systems, conversational AI, dialogue analysis, "
    "and knowledge-graph construction.\n\n"
    "Remember that these relationships form the memory of a multi-agent system. Make sure "
    "to relate all the concepts that are relevant to the system.\n\n"
    "You will receive TWO pieces of information:\n"
    "  1. A list of CONCEPTS (each with name, type, description and metadata) that were previously "
    "     extracted from the data.\n"
    "  2. The original JSON array of raw message records from which those concepts were extracted.\n\n"
    "Your task is to identify ALL meaningful RELATIONSHIPS between the provided concepts.\n\n"

    "### Relationship extraction instructions\n"
    "1. FIRST identify any temporal expressions in the sentence (e.g., tomorrow, next month, "
    "   last week, during summer, earlier today). If a relationship involves a time reference, "
    "   the temporal aspect MUST be reflected in the relationship label.\n"
    "2. Consider every possible pair of concepts and determine whether the data "
    "   evidences a meaningful interaction, dependency, data flow, **temporal relationship**, "
    "   or semantic link.\n"
    "3. The 'source' and 'target' MUST be exact names from the provided concepts list. "
    "   Do NOT invent new concept names.\n"
    "4. Relationship labels MUST be in UPPER_SNAKE_CASE and should be descriptive verb "
    "   phrases. If temporal information is present, encode it in the relation label "
    "   (e.g., PLANS_NEXT_MONTH, MEETS_TOMORROW, DISCUSSED_LAST_WEEK, RETURNS_IN_SUMMER).\n"
    "5. Each relationship MUST include a one-sentence description explaining the "
    "   interaction between the source and target, explicitly referencing the temporal "
    "   context when applicable.\n"
    "6. Prefer relationships that capture **actions tied to time** (plans, schedules, "
    "   past events, future intentions) when such evidence exists in the sentence.\n\n"

    "### Quality guidelines\n"
    "1. FOCUS on abstract, higher-level relationships rather than low-level details.\n"
    "2. MERGE similar or closely related relationships into a single broader one "
    "   to avoid redundancy.\n"
    "3. AVOID overlapping relationships that represent the same underlying idea.\n"
    "4. ENSURE each relationship is truly distinct and adds unique informational value.\n"
    "5. Every concept should participate in at least one relationship. If a concept is "
    "   completely isolated, reconsider whether a relationship was missed.\n"
    "6. All concepts MUST be related to at least one other concept."

    "Return ONLY the list of relationships."
)

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

_CONCEPT_PROMPTS = {
    "observe-sdk-otel": OTEL_CONCEPTS_PROMPT,
    "otel-trace": OTEL_CONCEPTS_PROMPT,
    "openclaw": GENERAL_CONCEPTS_PROMPT,
    "locomo": GENERAL_CONCEPTS_PROMPT,
    "semneg": GENERAL_CONCEPTS_PROMPT,
}

_RELATIONSHIP_PROMPTS = {
    "observe-sdk-otel": OTEL_RELATIONSHIPS_PROMPT,
    "otel-trace": OTEL_RELATIONSHIPS_PROMPT,
    "openclaw": GENERAL_RELATIONSHIPS_PROMPT,
    "locomo": GENERAL_RELATIONSHIPS_PROMPT,
    "semneg": GENERAL_RELATIONSHIPS_PROMPT,
}

SUPPORTED_FORMATS = set(_CONCEPT_PROMPTS.keys())


def get_concept_prompt(data_format: str) -> str:
    """Return the concept-extraction system prompt for the given format."""
    prompt = _CONCEPT_PROMPTS.get(data_format)
    if prompt is None:
        raise ValueError(f"Unsupported data format: {data_format!r}. Supported: {SUPPORTED_FORMATS}")
    return prompt


def get_relationship_prompt(data_format: str) -> str:
    """Return the relationship-extraction system prompt for the given format."""
    prompt = _RELATIONSHIP_PROMPTS.get(data_format)
    if prompt is None:
        raise ValueError(f"Unsupported data format: {data_format!r}. Supported: {SUPPORTED_FORMATS}")
    return prompt
