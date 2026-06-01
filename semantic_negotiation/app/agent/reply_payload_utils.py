# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for agent SSTP reply payloads (``reason``, round decision records).

Agent replies (accept / reject / counter_offer) may carry a free-text ``reason``
in ``SSTPNegotiateMessage.payload``.  The negotiation server does **not** use
``reason`` for SAO mechanics (only ``action`` and ``offer`` affect outcomes);
it is persisted in ``round_decisions``, ``sstp_message_trace``, and commit
envelopes for explainability and downstream alignment judges.

These helpers normalize ``reason`` (length, HTML stripping, Unicode/control
character hygiene, log-injection defenses) and build consistent
``round_decisions`` rows from unwrapped callback replies.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# Cap stored reason text so traces and commit payloads stay bounded.
MAX_REASON_LEN = 2000

# Collapse runs of whitespace after control/HTML stripping.
_WS_RE = re.compile(r"\s+")
# Remove script/style blocks (including body) before tag stripping.
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
# Fallback when Bleach is not installed in the active environment.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

try:
    import bleach as _bleach

    _bleach_clean = _bleach.clean
except ImportError:
    _bleach_clean = None

_bleach_missing_logged = False


def _strip_html(text: str) -> str:
    """Remove HTML tags and attributes; keep visible text only."""
    global _bleach_missing_logged
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    if _bleach_clean is None:
        if not _bleach_missing_logged:
            _bleach_missing_logged = True
            logger.warning(
                "bleach is not installed in this Python environment; using regex "
                "HTML stripping for agent reason text. Fix: from ioc-cfn-cognitive-agents "
                "run `poetry install`, or from ioc-cognition-fabric-node-svc run "
                "`poetry install` (re-syncs cognition-engine deps), then restart the "
                "service with `poetry run ...`."
            )
        return _HTML_TAG_RE.sub(" ", text)
    return _bleach_clean(text, tags=[], attributes={}, strip=True)


def _strip_unicode_controls(text: str) -> str:
    """Replace Unicode control/format/surrogate chars with a single space.

    Covers log-injection (CR/LF, ANSI escapes), bidi overrides (Cf), and other
    non-printable categories without dropping legitimate letters or punctuation.
    """
    return "".join(
        " " if unicodedata.category(ch).startswith("C") else ch for ch in text
    )


def sanitize_reason(reason: Any) -> str | None:
    """Normalize agent ``reason`` text for logs, traces, and JSON storage.

    Pipeline: reject non-strings → NFKC normalize → strip HTML (Bleach) →
    drop/replace control & format characters → collapse whitespace → length cap.

    Returns ``None`` when the input is not a string or sanitizes to empty text.
    """
    if not isinstance(reason, str):
        return None

    text = unicodedata.normalize("NFKC", reason)
    text = _strip_html(text)
    text = _strip_unicode_controls(text)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return None
    return text[:MAX_REASON_LEN]


def attach_reason(payload: dict[str, Any], reason: Any) -> dict[str, Any]:
    """Set ``payload['reason']`` when *reason* sanitizes to non-empty text.

    Mutates and returns *payload* so agent code can build the reply dict in one
    expression.  Omitting the key when empty keeps backward compatibility with
    clients that do not yet send ``reason``.
    """
    cleaned = sanitize_reason(reason)
    if cleaned:
        payload["reason"] = cleaned
    return payload


def round_decision_from_reply(
    participant_id: str, reply: dict[str, Any]
) -> dict[str, Any]:
    """Build a ``round_decisions`` entry from an unwrapped agent reply.

    Copies ``action``, optional ``offer`` (counter_offer only), and optional
    ``reason`` (any action).  Used by :class:`BatchCallbackRunner` after
    :func:`unwrap_reply` so trace and commit data match the wire payload.
    """
    dec: dict[str, Any] = {
        "participant_id": participant_id,
        "action": reply.get("action", "reject"),
    }
    if dec["action"] == "counter_offer" and "offer" in reply:
        dec["offer"] = reply["offer"]
    cleaned = sanitize_reason(reply.get("reason"))
    if cleaned:
        dec["reason"] = cleaned
    return dec
