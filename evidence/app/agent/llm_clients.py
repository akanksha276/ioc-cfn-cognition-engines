# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pydantic import BaseModel

import litellm

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMTokenMetadata:
    """Token usage metadata from an LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_ms: float
    cost_usd: Optional[float]
    timestamp: str

# Global counter of successful LLM chat calls
_LLM_CALL_COUNT = 0
_MAX_RETRIES = 5
_T = TypeVar("_T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Pydantic response models for structured LLM output
# ---------------------------------------------------------------------------

class JudgeResponse(BaseModel):
    selected: List[int]
    sufficient: bool
    reason: str


class PathScore(BaseModel):
    index: int
    score: float


class RankerResponse(BaseModel):
    scores: List[PathScore]


class EntityItem(BaseModel):
    name: str


class EntityExtractorResponse(BaseModel):
    entities: List[EntityItem]


class ResponseGeneratorResponse(BaseModel):
    answer: str


class DecompositionItem(BaseModel):
    index: int
    sentence: str
    entities: List[str]


class DecomposerResponse(BaseModel):
    items: List[DecompositionItem]


def get_llm_call_count() -> int:
    return _LLM_CALL_COUNT


def _inc_llm_call_count() -> None:
    global _LLM_CALL_COUNT
    _LLM_CALL_COUNT += 1


def _llm_creds() -> dict:
    out: dict = {}
    if settings.LLM_API_KEY:
        out["api_key"] = settings.LLM_API_KEY
    if settings.LLM_BASE_URL:
        out["base_url"] = settings.LLM_BASE_URL
    return out


def _model_to_tool_schema(response_model: Type[_T]) -> dict:
    """Convert a Pydantic model to a litellm function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": response_model.__name__,
            "description": f"Return a structured {response_model.__name__} response.",
            "parameters": response_model.model_json_schema(),
        },
    }


class _LLMBaseClient:
    """
    Shared litellm client utilities.
    Subclasses use _call_chat_structured(...) for all LLM interactions.
    """

    def __init__(self, temperature: float, client_label: str):
        self.temperature = temperature
        self._client_label = client_label
        logger.info(
            "[%s] init | model='%s' | api_key=%s | base_url=%s",
            client_label,
            settings.LLM_MODEL,
            "set" if settings.LLM_API_KEY else "(missing)",
            settings.LLM_BASE_URL or "(default)",
        )

    def _call_chat_structured(self, system: str, user: str, response_model: Type[_T]) -> _T:
        """
        Invoke the LLM via litellm tool_calls with a Pydantic schema.
        Raises on empty/filtered/refused responses.
        """
        parsed, _ = self._call_chat_structured_with_tokens(system, user, response_model)
        return parsed

    def _call_chat_structured_with_tokens(
        self, system: str, user: str, response_model: Type[_T]
    ) -> Tuple[_T, LLMTokenMetadata]:
        """
        Invoke the LLM via litellm tool_calls with a Pydantic schema.
        Returns (parsed_response, token_metadata) tuple.
        Raises on empty/filtered/refused responses.
        """
        tool = _model_to_tool_schema(response_model)
        kwargs: dict = {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": response_model.__name__}},
            "temperature": self.temperature,
            **_llm_creds(),
        }

        logger.debug(
            "[LLM._call_chat_structured] request | model=%s | response_model=%s",
            settings.LLM_MODEL, response_model.__name__,
        )

        # Track timing
        start_time = time.time()

        # Call LLM
        resp = litellm.completion(**kwargs)
        _inc_llm_call_count()

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage
        usage = resp.usage
        token_metadata = LLMTokenMetadata(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=resp.model,
            latency_ms=latency_ms,
            cost_usd=None,  # Can be calculated if needed
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Calculate cost (optional)
        try:
            token_metadata.cost_usd = litellm.completion_cost(completion_response=resp)
        except Exception:
            pass  # Cost calculation is optional

        choice = resp.choices[0] if resp.choices else None
        finish_reason = getattr(choice, "finish_reason", None) if choice else None

        if finish_reason == "content_filter":
            raise RuntimeError(
                f"LLM response blocked by content filter (finish_reason={finish_reason!r})."
            )

        tool_calls = choice.message.tool_calls if choice and choice.message else None
        if not tool_calls:
            refusal = getattr(choice.message, "refusal", None) if choice and choice.message else None
            if refusal:
                raise RuntimeError(f"LLM refused to respond: {refusal}")
            raise RuntimeError(
                f"LLM returned no tool call (finish_reason={finish_reason!r}). "
                "Likely content filter or token limit issue."
            )

        raw = tool_calls[0].function.arguments
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON in tool arguments: {raw!r}") from exc

        parsed = response_model(**data)

        logger.debug(
            "[LLM._call_chat_structured] response | finish_reason=%s | tokens=%d | latency=%.2fms",
            finish_reason, token_metadata.total_tokens, latency_ms,
        )
        return parsed, token_metadata


class EvidenceJudge(_LLMBaseClient):
    """
    LLM client for selecting most relevant paths and declaring sufficiency.
    Raises on failure after retries.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="EvidenceJudge")

    def select_paths_and_check_sufficiency(
        self, question: str, candidate_paths: List[str], select_k: int
    ) -> Tuple[List[int], bool, str]:
        logger.info("[EvidenceJudge] Judge invoked | candidates=%d | select_k=%d", len(candidate_paths), select_k)
        if not candidate_paths:
            logger.info("[EvidenceJudge] No candidates provided; returning empty.")
            return [], False, ""

        system = (
            "You are an evidence-based reasoning judge selecting the most relevant knowledge paths to answer a query. "
            "Your task is NOT keyword matching, but logical evaluation of whether each path provides evidence "
            "for answering the question.\n\n"
            "Rules:\n"
            '- "selected" must be an array of 0-based integers referring to the shown candidates.\n'
            '- "reason" must be a concise single sentence (max ~20 words) justifying the sufficiency decision.'
        )
        numbered = "\n".join([f"{i}. {p}" for i, p in enumerate(candidate_paths)])
        user = f"Question: {question or '(none)'}\n\nCandidate paths:\n{numbered}\n\nSelect top {select_k} paths."

        last_error: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._call_chat_structured(system, user, JudgeResponse)
                clean = [i for i in result.selected if 0 <= i < len(candidate_paths)]
                clean = clean[:select_k]
                reason = result.reason.strip().splitlines()[0] if result.reason else ""
                logger.info("[EvidenceJudge] Result: selected=%s | sufficient=%s | reason=%r", clean, result.sufficient, reason)
                return clean, result.sufficient, reason
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    wait = min(2 ** (attempt - 1), 16)
                    logger.warning("[EvidenceJudge] Attempt %d/%d failed: %s. Retrying in %ds...", attempt, _MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("[EvidenceJudge] All %d attempts failed.", _MAX_RETRIES)
        raise RuntimeError(
            f"[EvidenceJudge] select_paths_and_check_sufficiency failed after {_MAX_RETRIES} attempts"
        ) from last_error

    async def async_select_paths_and_check_sufficiency(
        self, question: str, candidate_paths: List[str], select_k: int
    ) -> Tuple[List[int], bool, str]:
        return await asyncio.to_thread(self.select_paths_and_check_sufficiency, question, candidate_paths, select_k)


class EvidenceRanker(_LLMBaseClient):
    """
    LLM client dedicated to ranking paths by importance in [0, 1].
    Raises on failure after retries.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="EvidenceRanker")

    def rank_paths(self, question: str, candidate_paths_repr: List[str]) -> Dict[int, float]:
        if not candidate_paths_repr:
            return {}

        system = (
            "You are ranking knowledge paths by how much they contribute to answering a question. "
            "Score each candidate on a 0.0 to 1.0 scale (1.0 = highly useful, 0.0 = irrelevant).\n\n"
            "Rules:\n"
            '- Each "index" is a 0-based integer for a shown candidate.\n'
            '- Each "score" must be a number in [0, 1].\n'
            "- Return a score for every candidate."
        )
        numbered = "\n".join([f"{i}. {p}" for i, p in enumerate(candidate_paths_repr)])
        user = f"Question: {question or '(none)'}\n\nCandidate paths:\n{numbered}\n\nRank all items."

        last_error: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._call_chat_structured(system, user, RankerResponse)
                scores: Dict[int, float] = {}
                for item in result.scores:
                    if 0 <= item.index < len(candidate_paths_repr):
                        sc = max(0.0, min(1.0, item.score))
                        scores[item.index] = sc
                if not scores:
                    raise ValueError("LLM returned no valid scores")
                return scores
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    wait = min(2 ** (attempt - 1), 16)
                    logger.warning("[EvidenceRanker] Attempt %d/%d failed: %s. Retrying in %ds...", attempt, _MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("[EvidenceRanker] All %d attempts failed.", _MAX_RETRIES)
        raise RuntimeError(
            f"[EvidenceRanker] rank_paths failed after {_MAX_RETRIES} attempts"
        ) from last_error

    async def async_rank_paths(self, question: str, candidate_paths_repr: List[str]) -> Dict[int, float]:
        return await asyncio.to_thread(self.rank_paths, question, candidate_paths_repr)


class ResponseGenerator(_LLMBaseClient):
    """
    LLM client that generates a final user-facing response from evidence only.
    Raises on failure after retries.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="ResponseGenerator")

    def generate_final_response(
        self,
        intent: str,
        symbolic_paths: List[str],
        verdict: str,
        rag_snippets: Optional[List[Dict[str, Any]]] = None,
        return_tokens: bool = False,
    ) -> str | Tuple[str, Optional[LLMTokenMetadata]]:
        rag_snippets = rag_snippets or []
        if not intent and not symbolic_paths and not verdict and not rag_snippets:
            if return_tokens:
                return "Insufficient Evidence", None
            return "Insufficient Evidence"

        base_rules = (
            "You are a response generator. Your ONLY job is to turn the provided evidence and verdict "
            "into a clear, direct answer to the user's intent.\n\n"
            "STRICT RULES:\n"
        )
        if rag_snippets:
            base_rules += (
                "- Use ONLY the information in the EVIDENCE block below (graph symbolic paths, verdict, and RAG chunks). "
                "Do not add any fact from your training.\n"
            )
        else:
            base_rules += (
                "- Use ONLY the information in the EVIDENCE block below. Do not add any fact from your training.\n"
            )
        base_rules += (
            "- Do not remove or contradict any part of the verdict or the listed paths.\n"
            '- If the evidence does not support an answer to the intent, set answer to: '
            '"The evidence does not support an answer to this question."\n'
            "- Keep the response concise. Reflect the judge's conclusion.\n\n"
            "NEGATIVE EVIDENCE:\n"
            "- When the question asks whether something is true (e.g. \"Would X do Y?\") and there is NO evidence "
            "supporting it but there IS evidence supporting a contrasting alternative, you may answer in the negative "
            "(e.g. \"Likely no\", \"No\") and cite the evidence for the alternative (e.g. \"she wants to be a counselor\").\n\n"
            "TEMPORAL REASONING RULES:\n"
            "- Relationships may encode relative time (e.g., YESTERDAY, TOMORROW, NEXT_MONTH, LAST_WEEK).\n"
            "- Each relationship includes a timestamp indicating when the statement occurred.\n"
            "- Use the timestamp as the reference point to convert any relative time expression into an absolute calendar date.\n\n"
            "TEMPORAL NORMALIZATION RULE:\n"
            "- ALWAYS convert relative time expressions into absolute dates when possible.\n"
            "- NEVER answer using relative expressions such as \"yesterday\", \"tomorrow\", \"next month\", \"last week\", etc.\n"
            "- The final answer MUST contain the resolved calendar time (e.g., \"7 May 2023\", \"June 2023\", \"14 Aug 2022\")."
        )
        system = base_rules
        evidence_block = "EVIDENCE:\n"
        if verdict:
            evidence_block += f"Verdict: {verdict.strip()}\n"
        if symbolic_paths:
            evidence_block += "Symbolic paths (graph):\n" + "\n".join(
                f"- {p}" for p in symbolic_paths if (p or "").strip()
            )
            evidence_block += "\n"
        if rag_snippets:
            evidence_block += "Retrieved chunks (RAG):\n"
            for i, sn in enumerate(rag_snippets):
                line = (sn.get("display_line") or "").strip()
                if not line:
                    t = (sn.get("embedded_text") or "").strip()
                    if not t:
                        continue
                    line = f"[{i + 1}] {t}"
                evidence_block += f"- {line}\n"
        evidence_block += "END EVIDENCE"

        user = f"User intent: {intent or '(none)'}\n\n{evidence_block}\n\nGenerate a short answer using only the evidence above."

        last_error: BaseException | None = None
        last_token_metadata: Optional[LLMTokenMetadata] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if return_tokens:
                    result, token_metadata = self._call_chat_structured_with_tokens(system, user, ResponseGeneratorResponse)
                    last_token_metadata = token_metadata
                else:
                    result = self._call_chat_structured(system, user, ResponseGeneratorResponse)
                answer = (result.answer or "").strip()
                if not answer:
                    raise ValueError("LLM returned empty answer")
                if return_tokens:
                    return answer, last_token_metadata
                return answer
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    wait = min(2 ** (attempt - 1), 16)
                    logger.warning("[ResponseGenerator] Attempt %d/%d failed: %s. Retrying in %ds...", attempt, _MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("[ResponseGenerator] All %d attempts failed.", _MAX_RETRIES)
        raise RuntimeError(
            f"[ResponseGenerator] generate_final_response failed after {_MAX_RETRIES} attempts"
        ) from last_error

    async def async_generate_final_response(
        self,
        intent: str,
        symbolic_paths: List[str],
        verdict: str,
        rag_snippets: Optional[List[Dict[str, Any]]] = None,
        return_tokens: bool = False,
    ) -> str | Tuple[str, Optional[LLMTokenMetadata]]:
        return await asyncio.to_thread(
            self.generate_final_response, intent, symbolic_paths, verdict, rag_snippets, return_tokens
        )


class EntityExtractor(_LLMBaseClient):
    """
    LLM client for extracting entities from a ReasonerCognitionRequest.
    Raises on failure after retries.
    """

    SYSTEM_PROMPT = (
        "You extract salient entities (proper nouns, products, APIs, teams, systems, key technical terms) "
        "from the user's intent and any provided text.\n\n"
        "Rules:\n"
        '- Each entity must have a non-empty "name" string.\n'
        "- Return all relevant entities found."
    )

    def __init__(self, temperature: float = 0.1):
        super().__init__(temperature=temperature, client_label="EntityExtractor")

    def extract_entities_from_request(self, request) -> List[Dict]:
        intent = request.payload.intent or ""
        texts: List[str] = []
        for rec in request.payload.records or []:
            try:
                rt = rec.record_type.value if hasattr(rec.record_type, "value") else str(rec.record_type)
            except Exception:
                rt = str(rec.record_type)
            if rt == "string" and isinstance(rec.content, str):
                texts.append(rec.content)
            elif rt == "json":
                texts.append(json.dumps(rec.content, ensure_ascii=False, separators=(",", ":")))

        user_prompt = f"INTENT:\n{intent}\n\nTEXT:\n" + ("\n".join(texts) if texts else "(none)")

        last_error: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._call_chat_structured(self.SYSTEM_PROMPT, user_prompt, EntityExtractorResponse)
                out = [{"name": e.name.strip()} for e in result.entities if e.name.strip()]
                if not out:
                    raise ValueError("LLM returned no entities")
                logger.info("[EntityExtractor] LLM extracted entities: %d", len(out))
                return out
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    wait = min(2 ** (attempt - 1), 16)
                    logger.warning("[EntityExtractor] Attempt %d/%d failed: %s. Retrying in %ds...", attempt, _MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("[EntityExtractor] All %d attempts failed.", _MAX_RETRIES)
        raise RuntimeError(
            f"[EntityExtractor] extract_entities_from_request failed after {_MAX_RETRIES} attempts"
        ) from last_error

    async def async_extract_entities_from_request(self, request) -> List[Dict]:
        return await asyncio.to_thread(self.extract_entities_from_request, request)


class QueryDecomposer(_LLMBaseClient):
    """
    LLM client for decomposing a query into numbered, atomic statements with up to two entities.
    Raises on failure after retries.
    """

    SYSTEM_PROMPT = (
        "You will receive a multi-hop question, which is composed of several interconnected queries, along"
        " with a list of topic entities that serve as the main keywords for the question. Your task is to break the"
        " question into simpler parts, using each topic entity once or twice (source and target) and provide a Chain of Thought (CoT) that"
        " shows how the topic entities are related. Note: Each simpler question should explore how"
        " one query connects to others or the answer. The goal is to systematically address each entity to derive"
        " the final answer.\n\n"
        "Rules:\n"
        '- Each item must have a 1-based "index", a "sentence" (the sub-question), and "entities" (up to two entity names referenced by that sentence).\n'
        "- Return at least one item."
    )

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="QueryDecomposer")

    @staticmethod
    def _reorder_entities(sentence: str, raw_entities: List[str], safe_ents: List[str]) -> List[str]:
        """Re-order entities by their first appearance in the sentence, capped at two."""
        final = raw_entities[:2]
        if not safe_ents:
            return final
        lower_sent = sentence.lower()
        located = []
        for e in safe_ents:
            pos = lower_sent.find(e.lower())
            if pos != -1:
                located.append((pos, e))
        if located:
            located.sort(key=lambda t: t[0])
            seen: set[str] = set()
            ordered = []
            for _, e in located:
                if e not in seen:
                    seen.add(e)
                    ordered.append(e)
            final = ordered[:2]
        return final

    def decompose(self, text: str, entities: List[str] | None = None) -> List[Dict]:
        """
        Returns: List[{index:int, sentence:str, entities:[str]}]
        """
        if not (text or "").strip():
            return []

        safe_ents = [str(e).strip() for e in (entities or []) if str(e).strip()]
        system_content = self.SYSTEM_PROMPT
        if safe_ents:
            system_content = (
                self.SYSTEM_PROMPT
                + "\n\nTopic entities (use at most two per statement; avoid reuse):\n- "
                + "\n- ".join(safe_ents)
            )
        user_input = f"Sentence:\n{text}".strip()

        last_error: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._call_chat_structured(system_content, user_input, DecomposerResponse)
                if not result.items:
                    raise ValueError("LLM returned no decomposition items")
                out: List[Dict] = []
                for item in result.items:
                    ents = self._reorder_entities(item.sentence, item.entities, safe_ents)
                    out.append({
                        "index": item.index,
                        "sentence": item.sentence.strip(),
                        "entities": ents,
                    })
                return out
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    wait = min(2 ** (attempt - 1), 16)
                    logger.warning("[QueryDecomposer] Attempt %d/%d failed: %s. Retrying in %ds...", attempt, _MAX_RETRIES, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("[QueryDecomposer] All %d attempts failed.", _MAX_RETRIES)
        raise RuntimeError(
            f"[QueryDecomposer] decompose failed after {_MAX_RETRIES} attempts"
        ) from last_error

    async def async_decompose(self, text: str, entities: List[str] | None = None) -> List[Dict]:
        return await asyncio.to_thread(self.decompose, text, entities)
