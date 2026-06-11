# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Transforms raw OtelSpan DB rows (from cfn-svc) into ExtractionRecord format
compatible with the otel-trace ingestion adapter.

Port of Go: ioc-cfn-svc/pkg/otelreceiver/ingestion.go
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List


def spans_to_extraction_records(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw OtelSpan DB rows to ExtractionRecord format.

    Input (snake_case, from cfn-svc API):
        {
            "start_time": "2026-05-01T22:59:44.140Z",
            "trace_id": "abc123...",
            "span_id": "def456...",
            "parent_span_id": "",
            "name": "span.name",
            "service_name": "my-service",
            "kind": 1,
            "duration_nano": 92626709381,
            "status_code": 1,
            "status_message": "",
            "attributes": {...},
            "events": [...],
            "links": [...],
            "resource": {...}
        }

    Output (camelCase, ExtractionRecord for otel-trace adapter):
        {
            "timestamp": "2026-05-01T22:59:44.140Z",
            "traceId": "abc123...",
            "spanId": "def456...",
            "parentSpanId": "",
            "name": "span.name",
            "kind": 1,
            "startTime": [epoch_sec, nano_fraction],
            "endTime": [epoch_sec, nano_fraction],
            "duration": [sec, nano_fraction],
            "attributes": {...},
            "status": {"code": 1},
            "events": [...],
            "links": [...],
            "resource": {"service.name": "my-service", ...}
        }
    """
    return [_span_to_extraction_record(span) for span in spans]


def _span_to_extraction_record(span: Dict[str, Any]) -> Dict[str, Any]:
    start_time_str = span.get("start_time", "")
    start_dt = _parse_timestamp(start_time_str)

    start_sec = int(start_dt.timestamp())
    start_nano = start_dt.microsecond * 1000

    duration_nano: int = span.get("duration_nano", 0) or 0
    duration_sec = duration_nano // 1_000_000_000
    duration_nano_frac = duration_nano % 1_000_000_000

    end_total_nano = (start_sec * 1_000_000_000 + start_nano) + duration_nano
    end_sec = end_total_nano // 1_000_000_000
    end_nano_frac = end_total_nano % 1_000_000_000

    attributes = _parse_json_field(span.get("attributes"), dict, {})
    events = _parse_json_field(span.get("events"), list, [])
    links = _parse_json_field(span.get("links"), list, [])
    resource = _parse_json_field(span.get("resource"), dict, {})

    service_name = span.get("service_name", "")
    if service_name and "service.name" not in resource:
        resource["service.name"] = service_name

    status_code = span.get("status_code", 0) or 0
    status_message = span.get("status_message", "")
    status: Dict[str, Any] = {"code": status_code}
    if status_message:
        status["message"] = status_message

    timestamp = start_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{start_dt.microsecond // 1000:03d}Z"

    record: Dict[str, Any] = {
        "timestamp": timestamp,
        "traceId": span.get("trace_id", ""),
        "spanId": span.get("span_id", ""),
        "name": span.get("name", ""),
        "kind": span.get("kind", 0) or 0,
        "startTime": [start_sec, start_nano],
        "endTime": [end_sec, end_nano_frac],
        "duration": [duration_sec, duration_nano_frac],
        "attributes": attributes,
        "status": status,
        "events": events,
        "links": links,
        "resource": resource,
    }

    parent_span_id = span.get("parent_span_id", "")
    if parent_span_id:
        record["parentSpanId"] = parent_span_id

    return record


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO-8601 timestamp string to datetime (UTC)."""
    if not ts:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    iso_ts = ts.strip()
    if iso_ts.endswith("Z"):
        iso_ts = f"{iso_ts[:-1]}+00:00"
    iso_ts = re.sub(r"(\.\d{6})\d+((?:[+-]\d{2}:?\d{2})?)$", r"\1\2", iso_ts)
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    ts = ts.rstrip("Z").rstrip("+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_json_field(value: Any, expected_type: type, default: Any) -> Any:
    if value in (None, ""):
        return default

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    return value if isinstance(value, expected_type) else default
