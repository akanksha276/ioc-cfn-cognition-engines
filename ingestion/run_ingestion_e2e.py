"""Run ingestion extraction end-to-end against a local JSON payload.

This script runs:
1) Concept + relationship extraction
2) Knowledge post-processing (embeddings/dedup)
3) Optional FAISS storage via caching layer

It reads settings from `ingestion/.env` through `ingestion.app.config.settings`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.config.settings import settings
from ingestion.app.dependencies import get_knowledge_processor


def _load_payload(path: Path) -> list[dict[str, Any]]:
    """Load payload JSON and normalize to list[dict]."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    raise ValueError("Payload must be a JSON object or an array of JSON objects.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ingestion end-to-end locally.")
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to payload JSON (object or array of objects).",
    )
    parser.add_argument(
        "--format",
        default="sstp",
        help="Format descriptor (default: sstp).",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional request ID (auto-generated when omitted).",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional Azure/OpenAI endpoint override (e.g. https://llm-proxy.prod.outshift.ai).",
    )
    parser.add_argument(
        "--output",
        default="ingestion_e2e_response.json",
        help="Output path for full JSON response.",
    )
    parser.add_argument(
        "--compact-output",
        default="ingestion_compact_payload.json",
        help="Output path for compact payload (service.py line 1000 equivalent).",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS/caching storage step.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    payload_path = Path(args.payload).expanduser().resolve()
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload file not found: {payload_path}")

    records = _load_payload(payload_path)
    if not records:
        raise ValueError("No valid records found in payload.")

    request_id = args.request_id or f"ingestion-e2e-{uuid.uuid4()}"
    endpoint = (args.endpoint or settings.azure_openai_endpoint or "").strip()

    service = ConceptRelationshipExtractionService(
        azure_endpoint=endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        azure_api_version=settings.azure_openai_api_version,
    )

    # Mirror service.py line 1000 output for debugging/inspection:
    # compact_payload = self._extraction_adapter.build_compact_payload(filtered, data_format)
    data_format = args.format.strip().lower()
    filtered = service._extraction_adapter.filter_records(records, data_format)
    compact_payload = service._extraction_adapter.build_compact_payload(filtered, data_format)
    compact_output_path = Path(args.compact_output).expanduser().resolve()
    with compact_output_path.open("w", encoding="utf-8") as f:
        json.dump(compact_payload, f, indent=2)

    result = asyncio.run(
        service.extract_concepts_and_relationships(
            compact_payload=compact_payload,
            request_id=request_id,
            format_descriptor=args.format,
        )
    )

    processor = get_knowledge_processor()
    result = processor.process(result)

    faiss_info: dict[str, Any] | None = None
    if not args.skip_faiss and settings.enable_faiss_storage:
        from ingestion.app.agent.concept_vector_store import ConceptVectorStore

        vector_store = ConceptVectorStore(
            vector_dimension=settings.faiss_vector_dimension,
            metric=settings.faiss_metric,
        )
        vector_store.store_concepts(result.get("concepts", []))
        faiss_info = vector_store.describe()

    output_path = Path(args.output).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Request ID: {request_id}")
    print(f"Endpoint: {endpoint}")
    print(f"Payload records: {len(records)}")
    print(f"Descriptor: {result.get('descriptor')}")
    print(f"Concepts: {len(result.get('concepts', []))}")
    print(f"Relations: {len(result.get('relations', []))}")
    if faiss_info is not None:
        print(f"FAISS ntotal: {faiss_info.get('ntotal')}")
    print(f"Saved compact payload: {compact_output_path}")
    print(f"Saved response: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
