#!/bin/bash
# test_flow.sh — smoke test for the semantic_alignment step-through API (stateless)
#
# Both /negotiate/initiate and /negotiate/offer-response accept an
# SSTPNegotiateMessage envelope (protocol.sstp.SSTPNegotiateMessage).
#
# Usage:
#   bash test_flow.sh [BASE_URL]
#
# Default BASE_URL: http://localhost:8089

set -e

BASE_URL="${1:-http://localhost:8089}"

python3 - "$BASE_URL" <<'PYEOF'
import sys, json, uuid
from datetime import datetime, timezone
import urllib.request

BASE = sys.argv[1]

# ── SSTP envelope helpers ─────────────────────────────────────────────────────

ORIGIN = {"actor_id": "agent:test-runner", "tenant_id": "ws-1"}
POLICY = {"sensitivity": "internal", "propagation": "forward", "retention_policy": "pol-90d"}
PROVENANCE = {"sources": [], "transforms": []}

def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def negotiate_envelope(session_id: str, payload: dict) -> dict:
    """Wrap negotiation payload in an SSTPNegotiateMessage envelope."""
    return {
        "protocol": "SSTP",
        "version": "0",
        "kind": "negotiate",
        "message_id": str(uuid.uuid4()),
        "dt_created": _now(),
        "origin": ORIGIN,
        "semantic_context": {
            "schema_id": "urn:ioc:schema:negotiate:negmas-sao:v1",
            "schema_version": "1.0",
            "encoding": "json",
            "session_id": session_id,
            "sao_state": None,
        },
        "payload_hash": "sha256:test",
        "policy_labels": POLICY,
        "provenance": PROVENANCE,
        "payload": payload,
    }

PARTICIPANTS = [
    {"id": "a", "name": "Agent A",
     "preferences": {"budget": {"low": 0.9, "medium": 0.5, "high": 0.1},
                     "timeline": {"short": 0.8, "long": 0.2}}},
    {"id": "b", "name": "Agent B",
     "preferences": {"budget": {"low": 0.1, "medium": 0.5, "high": 0.9},
                     "timeline": {"short": 0.3, "long": 0.7}}},
]

INITIATE_PAYLOAD = {
    "issues": ["budget", "timeline"],
    "options_per_issue": {"budget": ["low", "medium", "high"], "timeline": ["short", "long"]},
    "participants": PARTICIPANTS,
}

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def pp(label, obj):
    print(f"\n=== {label} ===")
    print(json.dumps(obj, indent=2))

# ── Test 1: initiate + continue + accept ─────────────────────────────────────
session_id = str(uuid.uuid4())
init = post(
    "/api/v1/negotiate/initiate",
    negotiate_envelope(session_id, INITIATE_PAYLOAD),
)
trace = init["trace"]
pp("POST /negotiate/initiate", {k: v for k, v in init.items() if k != "trace"})
print(f"  trace.rounds count: {len(trace['rounds'])}")

resp1 = post(
    "/api/v1/negotiate/offer-response",
    negotiate_envelope(session_id, {"round": 1, "action": "continue", "trace": trace}),
)
pp("POST /negotiate/offer-response (continue round 1)", resp1)

resp2 = post(
    "/api/v1/negotiate/offer-response",
    negotiate_envelope(session_id, {"round": 2, "action": "accept", "trace": trace}),
)
pp("POST /negotiate/offer-response (accept round 2)", resp2)

# ── Test 2: initiate + reject ─────────────────────────────────────────────────
session_id2 = str(uuid.uuid4())
init2 = post(
    "/api/v1/negotiate/initiate",
    negotiate_envelope(session_id2, INITIATE_PAYLOAD),
)
trace2 = init2["trace"]

reject = post(
    "/api/v1/negotiate/offer-response",
    negotiate_envelope(session_id2, {"round": 1, "action": "reject", "trace": trace2}),
)
pp("POST /negotiate/offer-response (reject round 1)", reject)
assert reject["status"] == "broken", f"Expected broken, got {reject['status']}"

# ── Health ────────────────────────────────────────────────────────────────────
with urllib.request.urlopen(f"{BASE}/health") as r:
    pp("GET /health", json.loads(r.read()))

print("\n✓ All tests passed")
PYEOF

