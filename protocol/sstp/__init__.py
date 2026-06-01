# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
sstp — Pydantic v2 models for the Semantic State Transfer Protocol (SSTP)
================================================================================
Implements the SSTP envelope spec as a discriminated-union model.

The ``kind`` field is the discriminator. All kinds share a common envelope
(origin, semantic_context, policy_labels, provenance, payload_hash) plus a
set of optional general fields.  ``kind="commit"`` promotes several of those
optional fields to **required**.

Kind taxonomy
-------------
``intent | delegation | knowledge | query | commit | memory_delta | evidence_bundle | negotiate``

For ``kind="negotiate"`` the ``semantic_context`` field is typed as
:class:`NegotiateSemanticContext` which carries a full NegMAS SAO snapshot
(``SAOState``, ``SAOResponse``, optional ``SAONMI``).

Usage
-----
    from protocol.sstp import STPMessage

    # build from a dict / JSON
    msg = STPMessage.model_validate({
        # "protocol": "SSTP",
        "version": "0",
        "kind": "intent",
        "message_id": "01920000-0000-7000-8000-000000000001",
        "dt_created": "2026-02-27T10:00:00Z",
        "origin": {
            "actor_id": "agent:planner-7",
            "tenant_id": "acme",
            "attestation": "sha256:abc123",
        },
        "semantic_context": {
            "schema_id": "urn:ioc:schema:intent:v1",
            "schema_version": "1.0",
            "encoding": "json",
        },
        "payload_hash": "sha256:deadbeef",
        "policy_labels": {
            "sensitivity": "internal",
            "propagation": "forward",
            "retention_policy": "pol-90d",
        },
        "provenance": {"sources": ["urn:doc:abc"], "transforms": []},
        "payload": {"goal": "book flight", "priority": "high"},
    })
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

#: Semantic version of the SSTP protocol package.
#: Bump this when the envelope schema or any kind model changes.
__version__: str = "1.0.0"

# Base models and literals — importable directly from sstp._base or via sstp.*
from ._base import (
    EncodingType,
    LogicalClock,
    MergeStrategy,
    Origin,
    PayloadRef,
    PayloadRefType,
    PolicyLabels,
    PropagationType,
    ProtocolType,
    Provenance,
    SemanticContext,
    SensitivityType,
    _STBaseMessage,
)

# Kind-specific message classes
from .commit import NegotiateCommitSemanticContext, SSTPCommitMessage
from .delegation import DelegationMessage
from .evidence_bundle import EvidenceBundleMessage
from .intent import IntentMessage
from .knowledge import KnowledgeMessage
from .memory_delta import MemoryDeltaMessage
from .negotiate import (
    NEGOTIATE_MESSAGE_JSON_EXCLUDE,
    NegotiateAgentReplyPayload,
    NegotiateSemanticContext,
    SSTPNegotiateMessage,
    dump_negotiate_message_json,
)
from .query import QueryMessage

# ---------------------------------------------------------------------------
# Discriminated union — the single type to use when deserialising any SSTP
# message whose kind is not known in advance.
# ---------------------------------------------------------------------------

STPMessage = Annotated[
    Union[
        IntentMessage,
        DelegationMessage,
        KnowledgeMessage,
        QueryMessage,
        SSTPCommitMessage,
        MemoryDeltaMessage,
        EvidenceBundleMessage,
        SSTPNegotiateMessage,
    ],
    Field(discriminator="kind"),
]
"""
Discriminated union of all SSTP message kinds.

Pydantic selects the concrete model based on the ``kind`` field value.
Use :func:`pydantic.TypeAdapter` or ``model_validate`` on the individual
concrete classes when you already know the kind; use ``STPMessage`` (via
``TypeAdapter``) when kind is unknown at parse time.

Example::

    from pydantic import TypeAdapter
    from protocol.sstp import STPMessage

    adapter = TypeAdapter(STPMessage)
    msg = adapter.validate_python(raw_dict)
"""

__all__ = [
    # Package version
    "__version__",
    # Literals & primitives
    "ProtocolType",
    "SensitivityType",
    "PropagationType",
    "EncodingType",
    "MergeStrategy",
    "PayloadRefType",
    # Sub-models
    "Origin",
    "SemanticContext",
    "PolicyLabels",
    "Provenance",
    "PayloadRef",
    "LogicalClock",
    # Base envelope
    "_STBaseMessage",
    # Kind message classes
    "IntentMessage",
    "DelegationMessage",
    "KnowledgeMessage",
    "QueryMessage",
    "NegotiateCommitSemanticContext",
    "SSTPCommitMessage",
    "MemoryDeltaMessage",
    "EvidenceBundleMessage",
    "NEGOTIATE_MESSAGE_JSON_EXCLUDE",
    "NegotiateAgentReplyPayload",
    "NegotiateSemanticContext",
    "SSTPNegotiateMessage",
    "dump_negotiate_message_json",
    # Union
    "STPMessage",
]
