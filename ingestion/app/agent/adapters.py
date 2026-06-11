"""Adapters for format-specific payload filtering and compaction."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, List, TypedDict

logger = logging.getLogger(__name__)

try:
    from flatten_dict import flatten
except ImportError:
    def flatten(
        data: dict[str, Any],
        *,
        reducer,
        keep_empty_types: tuple[type, ...] = (dict,),
    ) -> dict[str, Any]:
        """Fallback dict flattener when ``flatten_dict`` is unavailable."""
        flat: dict[str, Any] = {}

        def _walk(node: Any, parent_key: str = "") -> None:
            if isinstance(node, dict):
                if not node and dict in keep_empty_types and parent_key:
                    flat[parent_key] = {}
                    return
                for key, value in node.items():
                    next_key = reducer(parent_key, str(key))
                    _walk(value, next_key)
                return
            flat[parent_key] = node

        _walk(data)
        return flat

class ExtractionAdapter:
    """Adapter that normalizes records into compact, LLM-ready payloads."""

    @staticmethod
    def filter_spans(otel_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep SpanKind 'Client'/'Server' records, plus records with no SpanKind."""
        required = {"Client", "Server"}
        return [r for r in otel_records if r.get("SpanKind") in required or r.get("SpanKind") is None]

    @staticmethod
    def extract_important_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For every filtered span, pull out the fields the LLM needs:
        ServiceName, agent_id, system prompt content, function metadata,
        and the user prompt content.
        """
        func_name_pat = re.compile(r"^llm\.request\.functions\.(\d+)\.name$")
        extracted: List[Dict[str, Any]] = []

        for record in records:
            attrs = record.get("SpanAttributes", {})
            entry: Dict[str, Any] = {
                "ServiceName": record.get("ServiceName"),
                "agent_id": attrs.get("agent_id"),
                "model": attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
            }

            # System prompt
            prompt_role_pat = re.compile(r"^gen_ai\.prompt\.(\d+)\.role$")
            for key in attrs:
                m = prompt_role_pat.match(key)
                if m and attrs[key].lower() == "system":
                    idx = m.group(1)
                    entry["system_prompt"] = attrs.get(f"gen_ai.prompt.{idx}.content", "")
                    break

            # User prompt (original query)
            for key in attrs:
                m = prompt_role_pat.match(key)
                if m and attrs[key].lower() == "user":
                    idx = m.group(1)
                    entry["user_prompt"] = attrs.get(f"gen_ai.prompt.{idx}.content", "")
                    break

            # Functions registered on the span
            functions: List[Dict[str, Any]] = []
            for key in attrs:
                fm = func_name_pat.match(key)
                if fm:
                    fi = fm.group(1)
                    functions.append(
                        {
                            "name": attrs[key],
                            "description": attrs.get(f"llm.request.functions.{fi}.description", ""),
                            "parameters": attrs.get(f"llm.request.functions.{fi}.parameters", ""),
                        }
                    )
            if functions:
                entry["functions"] = functions

            # Tool calls (names only)
            tool_names = []
            for key in attrs:
                if "tool_calls" in key and "name" in key:
                    val = attrs.get(key)
                    if val:
                        tool_names.append(val)
            if tool_names:
                entry["tool_calls"] = tool_names

            # Completion / final output
            completion_pat = re.compile(r"^gen_ai\.completion\.(\d+)\.content$")
            for key in attrs:
                cm = completion_pat.match(key)
                if cm:
                    content = attrs[key]
                    if content and isinstance(content, str) and content.strip():
                        entry["completion"] = content.strip()
                        break

            extracted.append(entry)

        return extracted

    @staticmethod
    def extract_important_fields_openclaw(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pull out relevant fields from OpenClaw records for LLM consumption."""
        tool_call_keys = ("id", "name", "input", "result")

        def _flatten_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
            entry: Dict[str, Any] = {}
            for key in ("userMessage", "thinking", "response", "timestamp"):
                val = turn.get(key)
                if val is not None:
                    entry[key] = val

            raw_calls = turn.get("toolCalls")
            if raw_calls:
                entry["toolCalls"] = [{k: tc[k] for k in tool_call_keys if k in tc} for tc in raw_calls]
            return entry

        extracted: List[Dict[str, Any]] = []
        for record in records:
            turns = record.get("turns") if isinstance(record.get("turns"), list) else None
            if turns is not None:
                for turn in turns:
                    flat = _flatten_turn(turn)
                    if flat:
                        extracted.append(flat)
            else:
                flat = _flatten_turn(record)
                if flat:
                    extracted.append(flat)
        return extracted

    @staticmethod
    def extract_important_fields_locomo(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pull out relevant fields from LoCoMo conversation records."""
        extracted: List[Dict[str, Any]] = []
        for record in records:
            entry: Dict[str, Any] = {}
            for key in ("speaker", "blip_caption", "dia_id", "text", "query", "session_date_time"):
                val = record.get(key)
                if val is not None:
                    entry[key] = val
            if not entry:
                entry = record
            extracted.append(entry)
        return extracted

    @staticmethod
    def _iter_negotiation_messages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize semantic negotiation (semneg) input into a list of direct messages.

        For ``data_format == "semneg"``, we intentionally ignore and filter out:
        - top-level ``semneg_message_trace``
        - nested ``payload.trace`` (including ``payload.trace.semneg_message_trace``)

        Only ``trace.rounds`` is preserved from within the trace structure.
        This keeps the parser focused on direct message-level fields and avoids
        pulling large trace payloads into extraction input.
        """
        messages: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue

            direct_message = "dt_created" in record or "semantic_context" in record
            if direct_message:
                # Preserve rounds if already normalized on a prior pass.
                rounds = record.get("rounds")

                # Pull rounds out of the top-level trace (if present); drop everything else.
                trace = record.get("trace")
                if rounds is None and isinstance(trace, dict):
                    rounds = trace.get("rounds")

                # Also check payload.trace.rounds for records where trace is nested.
                payload = record.get("payload")
                payload_no_trace = None
                if isinstance(payload, dict):
                    payload_trace = payload.get("trace")
                    if rounds is None and isinstance(payload_trace, dict):
                        rounds = payload_trace.get("rounds")
                    payload_no_trace = {k: v for k, v in payload.items() if k != "trace"}

                sanitized = {
                    "dt_created": record.get("dt_created"),
                    "origin": record.get("origin"),
                    "semantic_context": record.get("semantic_context"),
                    "confidence_score": record.get("confidence_score"),
                    "kind": record.get("kind"),
                    "ttl_seconds": record.get("ttl_seconds"),
                }
                sem_ctx = record.get("semantic_context")
                if not isinstance(sem_ctx, dict):
                    sem_ctx = {}
                content_text = sem_ctx.get("content_text")
                if content_text is not None:
                    sanitized["content_text"] = content_text
                agents_negotiating = sem_ctx.get("agents_negotiating")
                if agents_negotiating is not None:
                    sanitized["agents_negotiating"] = agents_negotiating
                outcome = sem_ctx.get("outcome")
                if outcome is not None:
                    sanitized["outcome"] = outcome
                if rounds is not None:
                    sanitized["rounds"] = rounds
                record_payload = record.get("payload") or {}
                validation = record_payload.get("validation")
                retry_history = record_payload.get("retry_history")
                if validation is not None:
                    sanitized["validation"] = validation
                if retry_history is not None:
                    sanitized["retry_history"] = retry_history
                logger.debug(
                    "[adapters] payload keys=%s  validation=%s  retry_history_len=%s",
                    list(record_payload.keys()),
                    validation,
                    len(retry_history) if retry_history else 0,
                )
                if payload_no_trace:
                    sanitized["payload"] = payload_no_trace
                messages.append(sanitized)

        return messages

    @staticmethod
    def extract_important_fields_negotiation(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract only the required fields for semantic negotiation (semneg) records.

        Fields extracted:
        - dt_created
        - semantic_context.issues
        - semantic_context.options_per_issue
        - semantic_context.final_agreement
        - content_text   (from semantic_context.content_text)
        - agents_negotiating  (from semantic_context)
        - outcome        (from semantic_context.outcome)
        - confidence_score
        - kind
        - rounds         (from trace.rounds, when present)
        - validation     (SAV result, when present)
        - retry_history  (list of prior attempt records, when retries occurred)
        """
        extracted: List[Dict[str, Any]] = []
        for message in ExtractionAdapter._iter_negotiation_messages(records):
            semantic_context = message.get("semantic_context")
            if not isinstance(semantic_context, dict):
                semantic_context = {}

            entry: Dict[str, Any] = {
                "dt_created": message.get("dt_created"),
                "semantic_context": {
                    "issues": semantic_context.get("issues", []),
                    "options_per_issue": semantic_context.get("options_per_issue", {}),
                    "final_agreement": semantic_context.get("final_agreement", []),
                },
                "confidence_score": message.get("confidence_score"),
                "kind": message.get("kind"),
                "ttl_seconds": message.get("ttl_seconds"),
            }
            content_text = message.get("content_text")
            if content_text is not None:
                entry["content_text"] = content_text
            agents_negotiating = message.get("agents_negotiating")
            if agents_negotiating is not None:
                entry["agents_negotiating"] = agents_negotiating
            outcome = message.get("outcome")
            if outcome is not None:
                entry["outcome"] = outcome
            rounds = message.get("rounds")
            if rounds is not None:
                entry["rounds"] = rounds
            validation = message.get("validation")
            if validation is not None:
                entry["validation"] = validation
            retry_history = message.get("retry_history")
            if retry_history is not None:
                entry["retry_history"] = retry_history
            extracted.append(entry)

        return extracted

    @staticmethod
    def filter_spans_otel_trace(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep spans with ioa_observe.span.kind in workflow/agent/tool/task."""
        allowed_kinds = {"workflow", "agent", "tool", "task"}
        filtered = []
        for r in records:
            attrs = r.get("attributes") or {}
            kind = attrs.get("ioa_observe.span.kind")
            if kind in allowed_kinds:
                filtered.append(r)
        return filtered

    @staticmethod
    def extract_important_fields_otel_trace(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract important fields from otel-trace span records.

        Handles the NDJSON export format where top-level keys are:
        name, kind, traceId, spanId, attributes, resource, timestamp,
        startTime, endTime, duration, status, events, links.
        """
        extracted: List[Dict[str, Any]] = []

        for record in records:
            attrs = record.get("attributes") or {}
            resource = record.get("resource") or {}
            entry: Dict[str, Any] = {
                "ServiceName": resource.get("service.name") or attrs.get("service.name"),
                "agent_id": attrs.get("gen_ai.agent.id") or attrs.get("gen_ai.agent.name"),
                "model": attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model"),
                "span_kind": attrs.get("ioa_observe.span.kind"),
                "span_name": record.get("name") or record.get("spanName"),
            }

            user_prompt = (
                attrs.get("ioa_observe.entity.input")
                or attrs.get("openclaw.request.input")
                or attrs.get("openclaw.agent.input")
            )
            if user_prompt:
                entry["user_prompt"] = user_prompt

            completion = (
                attrs.get("ioa_observe.entity.output")
                or attrs.get("openclaw.agent.output")
                or attrs.get("openclaw.request.output")
                or attrs.get("openclaw.message.output")
            )
            if completion:
                entry["completion"] = completion

            tool_name = attrs.get("gen_ai.tool.name") or attrs.get("openclaw.tool.name")
            if tool_name:
                entry["tool_calls"] = [tool_name]

            tool_input = attrs.get("openclaw.tool.input")
            if tool_input:
                entry["tool_input"] = tool_input

            tool_output = attrs.get("openclaw.tool.output")
            if tool_output:
                entry["tool_output"] = tool_output

            extracted.append(entry)

        return extracted

    def filter_records(self, records: List[Dict[str, Any]], data_format: str) -> List[Dict[str, Any]]:
        """Apply format-specific filtering."""
        if data_format == "observe-sdk-otel":
            return self.filter_spans(records)
        if data_format == "otel-trace":
            return self.filter_spans_otel_trace(records)
        if data_format == "openclaw":
            turns: List[Dict[str, Any]] = []
            for record in records:
                record_turns = record.get("turns")
                if isinstance(record_turns, list):
                    turns.extend(record_turns)
            return turns if turns else records
        if data_format == "semneg":
            return self._iter_negotiation_messages(records)
        return records

    def build_compact_payload(self, records: List[Dict[str, Any]], data_format: str) -> List[Dict[str, Any]]:
        """Extract important fields using the format-appropriate method."""
        extractors = {
            "observe-sdk-otel": self.extract_important_fields,
            "otel-trace": self.extract_important_fields_otel_trace,
            "openclaw": self.extract_important_fields_openclaw,
            "locomo": self.extract_important_fields_locomo,
            "semneg": self.extract_important_fields_negotiation,
        }
        extractor = extractors.get(data_format, self.extract_important_fields)
        return extractor(records)


def _normalize_timestamp_value(value: Any) -> str:
    """Normalize a timestamp value to string (empty string when unavailable)."""
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _raw_timestamp_for_format(data: dict[str, Any], data_format: str) -> Any:
    """Get raw timestamp from compact data using known format-specific keys."""
    timestamp_keys = {
        "observe-sdk-otel": ("Timestamp", "timestamp", "dt_created", "session_date_time"),
        "otel-trace": ("timestamp", "recordedAt", "startAt"),
        "openclaw": ("timestamp", "Timestamp"),
        "locomo": ("session_date_time", "timestamp", "Timestamp"),
        "semneg": ("dt_created", "timestamp", "Timestamp"),
    }.get(data_format, ("timestamp", "Timestamp", "dt_created", "session_date_time"))

    for key in timestamp_keys:
        if key in data and data.get(key) not in (None, ""):
            return data[key]
    return None


class NestedDictTextDocument(TypedDict):
    """Flattened text plus domain/timestamp metadata."""

    text: str
    metadata: dict[str, str]


class ExtractionAdapterRAG:
    """Nested dict → flattened text and metadata."""

    @staticmethod
    def _nested_dict_to_text_document_single(
        data: dict[str, Any],
        *,
        key_value_delimiter: str = ": ",
        pair_separator: str = " ",
        key_path_separator: str = ": ",
        data_format: str,
    ) -> NestedDictTextDocument:
        """One nested dict → NestedDictTextDocument."""

        def _path(left: str, right: str) -> str:
            """Join path segments."""
            return f"{left}{key_path_separator}{right}" if left else right

        flat = flatten(data, reducer=_path, keep_empty_types=(dict,))
        flat_text = pair_separator.join(
            f"{k}{key_value_delimiter}{v}" for k, v in flat.items()
        )

        timestamp = _normalize_timestamp_value(_raw_timestamp_for_format(data, data_format))

        metadata: dict[str, str] = {
            "domain": data_format,
            "timestamp": timestamp,
        }
        return {
            "text": flat_text,
            "metadata": metadata,
        }

    @staticmethod
    def nested_dict_to_text_document(
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        key_value_delimiter: str = ": ",
        pair_separator: str = " ",
        key_path_separator: str = ": ",
        data_format: str,
    ) -> list[NestedDictTextDocument]:
        """One dict or list of dicts → list of NestedDictTextDocument."""
        kwargs = {
            "key_value_delimiter": key_value_delimiter,
            "pair_separator": pair_separator,
            "key_path_separator": key_path_separator,
            "data_format": data_format,
        }
        if isinstance(data, list):
            return [
                ExtractionAdapterRAG._nested_dict_to_text_document_single(item, **kwargs)
                for item in data
            ]
        return [ExtractionAdapterRAG._nested_dict_to_text_document_single(data, **kwargs)]
