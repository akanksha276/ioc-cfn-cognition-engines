# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight semantic similarity using the local Granite-30M ONNX model.

Provides a single public function :func:`cosine_similarity` that returns a
0-100 float (matching rapidfuzz's score convention) representing how
semantically similar two strings are.

The ONNX session and tokenizer are loaded **once** on first call (lazy
singleton) to avoid penalising startup time.  Subsequent calls are fast
ONNX inference (~1-3 ms on CPU for short negotiation strings).

Model path: resolved from ``EMBEDDING_MODEL_PATH`` environment variable.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort
    from tokenizers import Tokenizer as _Tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution — uses EMBEDDING_MODEL_PATH env var (set in Dockerfile).
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(os.environ.get("EMBEDDING_MODEL_PATH", "").strip())


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_session: "ort.InferenceSession | None" = None
_tokenizer: "_Tokenizer | None" = None
_load_error: Exception | None = None


def _load() -> tuple["ort.InferenceSession", "_Tokenizer"]:
    """Load the ONNX session and tokenizer (called at most once).

    On failure the error is cached so that subsequent calls return immediately
    without retrying — the caller's ``except`` block handles the silent fallback.
    Raises the cached exception on every call after a failure so ``cosine_similarity``
    can catch it and return -1.0 without logging again.
    """
    global _session, _tokenizer, _load_error  # noqa: PLW0603

    with _lock:
        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer
        if _load_error is not None:
            raise _load_error  # already logged once; caller silently returns -1.0

        try:
            import onnxruntime as ort  # type: ignore[import]
            from tokenizers import Tokenizer  # type: ignore[import]

            onnx_path = _MODEL_DIR / "model.onnx"
            tokenizer_path = _MODEL_DIR / "tokenizer.json"

            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            tok = Tokenizer.from_file(str(tokenizer_path))
            tok.enable_padding()
            tok.enable_truncation(max_length=128)

            sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            _session, _tokenizer = sess, tok
            logger.info("Granite-30M embedding model loaded from %s", _MODEL_DIR)
            return sess, tok

        except Exception as exc:  # noqa: BLE001
            _load_error = exc
            logger.warning(
                "Granite-30M embedding model unavailable (%s). "
                "Offer validation will rely on rapidfuzz tiers only — no action needed.",
                exc,
            )
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cosine_similarity(a: str, b: str) -> float:
    """Return the cosine similarity between *a* and *b* on a **0–100** scale.

    Uses the locally-bundled Granite-30M ONNX model.  The model's pooled
    sentence embedding (output index 1) is used; no mean-pooling needed.

    If the model is unavailable or inference fails for any reason, returns
    ``-1.0`` silently so the caller's tier-5 check (``score >= threshold``)
    is never satisfied and offer validation falls back to rapidfuzz tiers
    without raising or logging additional noise.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Cosine similarity in the range [0, 100], or -1.0 on any error.
    """
    try:
        sess, tok = _load()
    except Exception:  # noqa: BLE001
        # Model unavailable — already logged once at WARNING in _load();
        # return sentinel so tier 5 is silently skipped.
        return -1.0

    try:
        enc = tok.encode_batch([a, b])
        input_ids = np.array([e.ids for e in enc], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        # outputs[1] is the pooled 384-dim sentence embedding (from 1_Pooling layer)
        vec_a, vec_b = outputs[1][0], outputs[1][1]
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cos = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # clamp to [0, 1] (cosine can be slightly negative for dissimilar strings)
        return max(0.0, cos) * 100.0
    except Exception as exc:  # noqa: BLE001
        # Inference error — return sentinel, do not propagate.
        logger.warning("Embedding similarity inference error (will use fuzzy tiers): %s", exc)
        return -1.0


def is_available() -> bool:
    """Return True if the embedding model can be loaded successfully."""
    try:
        _load()
        return True
    except Exception:  # noqa: BLE001
        return False
