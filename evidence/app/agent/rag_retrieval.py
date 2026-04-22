# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Async top-k retrieval against the RAG /rag/similarity-search API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _attach_display_line(row: Dict[str, Any], index: int) -> None:
    ts = str(row.get("timestamp") or "").strip()
    domain = str(row.get("domain") or "").strip()
    txt = str(row.get("embedded_text") or "").strip()
    parts: List[str] = [p for p in (ts, domain) if p]
    parts.append(txt)
    row["display_line"] = f"[{index}] " + ", ".join(parts)


def _normalize_hit(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not row or not isinstance(row, dict):
        return None
    text = row.get("embedded_text")
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    out = {k: v for k, v in row.items()}
    out["embedded_text"] = text  # store the stripped canonical value
    try:
        out["score"] = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        out["score"] = 0.0
    return out


async def retrieve_rag_top_k(
    repo: Any,
    intent: str,
    top_k: int,
    request_id: str,
    embedding_vector: Optional[List[float]] = None,
    timeout_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run similarity via POST /rag/similarity-search.
    Each hit includes display_line: ``[n] timestamp, domain, embedded_text``.
    """
    _ = timeout_seconds
    if repo is None or not hasattr(repo, "search_similar_rag") or not (intent or "").strip():
        return []

    try:
        raw = await repo.search_similar_rag(
            embedded_text=str(intent).strip(),
            embedding_vector=embedding_vector or [],
            request_id=request_id,
            top_k=top_k,
            search_metrics="l2",
            filters=[],
        )
    except Exception as e:
        logger.warning("[RAG] /rag/similarity-search failed: %s", e)
        return []

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(raw or [], start=1):
        n = _normalize_hit(r if isinstance(r, dict) else {})
        if n:
            _attach_display_line(n, i)
            out.append(n)
    logger.info("[RAG] retrieval result: %s", out)
    return out
