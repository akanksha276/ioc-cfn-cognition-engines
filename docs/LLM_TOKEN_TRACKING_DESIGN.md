# LLM Token Usage Tracking & Metrics Reporting - Design Document

**Status:** Proposal  
**Author:** Design Analysis  
**Date:** 2026-05-20  
**Version:** 2.0 (Hybrid Approach)

---

## Executive Summary

This document provides a comprehensive design for capturing LLM token usage across all cognitive agents and reporting it to the CFN service's metrics API. The solution uses a **hybrid approach**: token usage is returned in every API response (following industry standards like OpenAI, Anthropic, AWS Bedrock), with optional fire-and-forget metrics reporting to CFN for centralized tracking and analytics.

### Key Design Principles

1. **Return tokens in response** - Users see token usage immediately (industry standard)
2. **Optional centralized tracking** - Fire-and-forget metrics to CFN (non-blocking, graceful degradation)
3. **Zero boilerplate** - No wrappers, no context vars, just return what LiteLLM gives you
4. **Contributor-friendly** - Copy-paste pattern, impossible to forget

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Detailed Design](#detailed-design)
4. [Implementation Plan](#implementation-plan)
5. [Best Practices](#best-practices)
6. [Testing Strategy](#testing-strategy)

---

## Problem Statement

### Current Issues

1. **No Token Usage Visibility** - LLM token consumption is not tracked or reported
2. **Non-Standard API** - Industry standard (OpenAI, Anthropic, AWS) is to return tokens in response
3. **Cost Tracking Gap** - Unable to measure LLM costs per workspace/MAS/agent in real-time
4. **Debugging Difficulty** - Cannot correlate latency with token usage without querying separate systems

### Test Results

From `/docs/temp/FINAL_RESULTS.md`:
- ✅ All cognition flows working (semantic negotiation, knowledge extraction, evidence gathering)
- ❌ **NO token usage returned in any API responses**
- Quote: "⚠️ NO TOKEN USAGE FOUND in response"

### Goals

1. ✅ Return token usage in **every** API response (align with industry standards)
2. ✅ Optional centralized tracking to CFN TimescaleDB (for analytics)
3. ✅ **Zero refactoring** - minimal changes to existing code
4. ✅ **Simple for contributors** - copy-paste pattern, clear and obvious
5. ✅ **Graceful degradation** - works even if CFN metrics API is down

---

## Solution Overview

### Industry Standards

**OpenAI API:**
```json
{
  "choices": [{"message": {"content": "..."}}],
  "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
}
```

**Anthropic API:**
```json
{
  "content": [{"text": "..."}],
  "usage": {"input_tokens": 10, "output_tokens": 20}
}
```

**AWS Bedrock:**
```json
{
  "output": {"message": "..."},
  "usage": {"inputTokens": 15, "outputTokens": 25}
}
```

### Our Approach: Hybrid (Best of Both Worlds)

```
┌──────────────────────────────────────────────────────────────────┐
│ Cognitive Agent Service (ingestion/evidence/semantic_neg)       │
│                                                                  │
│  1. Call LiteLLM                                                │
│     response = litellm.acompletion(model="gpt-4", messages=[..])│
│                                                                  │
│  2. Return tokens in response (PRIMARY)                         │
│     return {                                                     │
│       "response_id": "...",                                      │
│       "data": {...},                                             │
│       "meta": {                                                  │
│         "tokens": {                                              │
│           "prompt": response.usage.prompt_tokens,                │
│           "completion": response.usage.completion_tokens,        │
│           "total": response.usage.total_tokens,                  │
│           "model": response.model                                │
│         },                                                       │
│         "latency_ms": latency                                    │
│       }                                                          │
│     }                                                            │
│                                                                  │
│  3. OPTIONAL: Fire-and-forget metrics to CFN (SECONDARY)        │
│     asyncio.create_task(                                         │
│       metrics_client.post_metric(ws_id, mas_id, agent_id, ...)  │
│     )                                                            │
│     # Non-blocking, gracefully fails if CFN down                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                          │
                          │ (Optional) POST /api/internal/cognition-engine/metrics
                          ↓
┌──────────────────────────────────────────────────────────────────┐
│ CFN Service                                                      │
│  - Receives metrics batch (if enabled)                          │
│  - Stores in TimescaleDB (hypertable)                           │
│  - Provides query API for analytics                             │
└──────────────────────────────────────────────────────────────────┘
```

### Architecture Comparison

**Before (Current Design v1.0 - Complex):**
```
LLM Call → Wrapper → Context Vars → Middleware → Async Flush → CFN
          ↑ Must use wrapper
          ↑ Must pass ws_id/mas_id/agent_id every time
          ↑ Silent failure if wrapper not used
```
**6 components, 3 async boundaries, high complexity**

**After (Hybrid Design v2.0 - Simple):**
```
LLM Call → LiteLLM → Return tokens in response ✅
                        ↓ (optional, non-blocking)
                   Fire-and-forget → CFN
```
**2 components, 1 optional async boundary, low complexity**

### Key Benefits

| Aspect | Hybrid Approach |
|--------|-----------------|
| **User Experience** | 😊 Immediate visibility in response |
| **API Standards** | 😊 Matches OpenAI/Anthropic/AWS |
| **Implementation** | 😊 Simple (no wrappers, no context vars) |
| **Contributor Onboarding** | 😊 Copy-paste pattern |
| **Debugging** | 😊 Single response has all info |
| **Failure Modes** | 😊 Graceful degradation (tokens always returned) |
| **Historical Analysis** | 😊 Optional centralized tracking |
| **Real-time Cost Control** | 😊 Immediate per-request visibility |

---

## Detailed Design

### 1. Response Schema (PRIMARY)

Every cognition engine response includes a `meta` object with token usage.

#### Standard Response Format

```python
{
  "response_id": "string",
  "data": {
    # Domain-specific response data
  },
  "meta": {
    "tokens": {
      "prompt": 150,
      "completion": 200,
      "total": 350,
      "model": "claude-3-5-sonnet-20241022"
    },
    "latency_ms": 1234.56,
    "timestamp": "2026-05-20T10:30:00Z"
  }
}
```

#### Implementation Pattern

**Before (No tokens):**
```python
async def extract_knowledge(request: ExtractionRequest):
    # Extract knowledge using LLM
    response = await litellm.acompletion(
        model="claude-3-5-sonnet-20241022",
        messages=[...]
    )
    
    # Process results
    concepts = parse_concepts(response.choices[0].message.content)
    
    return {
        "response_id": request.request_id,
        "concepts": concepts,
        "relationships": relationships
    }
```

**After (With tokens):**
```python
async def extract_knowledge(request: ExtractionRequest):
    start_time = time.time()
    
    # Extract knowledge using LLM
    response = await litellm.acompletion(
        model="claude-3-5-sonnet-20241022",
        messages=[...]
    )
    
    # Process results
    concepts = parse_concepts(response.choices[0].message.content)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Return with tokens (SIMPLE!)
    return {
        "response_id": request.request_id,
        "concepts": concepts,
        "relationships": relationships,
        "meta": {
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
                "model": response.model,
            },
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }
```

**That's it. No wrappers. No context vars. No middleware. Just return what LiteLLM gives you.**

---

### 2. Centralized Metrics Client (SECONDARY, OPTIONAL)

For services that want centralized tracking (analytics, dashboards, billing), we provide a simple fire-and-forget client.

#### `common/metrics/cfn_client.py`

```python
"""
common/metrics/cfn_client.py

Simple fire-and-forget metrics client for CFN API.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TokenMetric:
    """Token usage metric to post to CFN."""
    
    workspace_id: str
    mas_id: str
    agent_id: str
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: Optional[float] = None
    operation: str = "unknown"


class CFNMetricsClient:
    """
    Fire-and-forget metrics client for CFN.
    
    Features:
    - Non-blocking (uses asyncio.create_task)
    - Graceful degradation (logs errors but never raises)
    - Batching (groups multiple metrics)
    - Retry with exponential backoff
    
    Usage:
        client = CFNMetricsClient("http://localhost:9002")
        
        # Fire-and-forget (doesn't block response)
        await client.post_token_metric(
            workspace_id="ws-123",
            mas_id="mas-456",
            agent_id="agent-789",
            model="gpt-4",
            prompt_tokens=150,
            completion_tokens=200,
            total_tokens=350,
            latency_ms=1234.5
        )
    """

    def __init__(
        self,
        cfn_base_url: str,
        enabled: bool = True,
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        """
        Initialize CFN metrics client.
        
        Args:
            cfn_base_url: Base URL of CFN service (e.g. "http://localhost:9002")
            enabled: Enable/disable metrics posting (for feature flag)
            timeout: HTTP timeout in seconds
            max_retries: Number of retries on failure
        """
        self.cfn_base_url = cfn_base_url.rstrip("/")
        self.endpoint = f"{self.cfn_base_url}/api/internal/cognition-engine/metrics"
        self.enabled = enabled
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def post_token_metric(
        self,
        workspace_id: str,
        mas_id: str,
        agent_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        cost_usd: Optional[float] = None,
        operation: str = "unknown",
    ) -> None:
        """
        Post token metric to CFN (fire-and-forget).
        
        This is non-blocking - it creates an async task and returns immediately.
        Errors are logged but never raised.
        """
        if not self.enabled:
            return

        metric = TokenMetric(
            workspace_id=workspace_id,
            mas_id=mas_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation=operation,
        )

        # Fire-and-forget (non-blocking)
        asyncio.create_task(self._post_metric_internal(metric))

    async def _post_metric_internal(
        self,
        metric: TokenMetric,
        retry_count: int = 0,
    ) -> None:
        """Internal method to post metric with retry logic."""
        try:
            # Build payload
            payload = {
                "workspace_id": metric.workspace_id,
                "mas_id": metric.mas_id,
                "agent_id": metric.agent_id,
                "attributes": {
                    "service": "cognitive_agent",
                    "operation": metric.operation,
                },
                "metrics": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.prompt",
                        "value": metric.prompt_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.completion",
                        "value": metric.completion_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.tokens.total",
                        "value": metric.total_tokens,
                        "attributes": {"model": metric.model},
                    },
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "name": "llm.latency_ms",
                        "value": metric.latency_ms,
                        "attributes": {"model": metric.model},
                    },
                ],
            }

            # Add cost if available
            if metric.cost_usd is not None:
                payload["metrics"].append({
                    "timestamp": metric.timestamp.isoformat(),
                    "name": "llm.cost_usd",
                    "value": metric.cost_usd,
                    "attributes": {"model": metric.model},
                })

            # Post to CFN
            client = await self._get_client()
            response = await client.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 202:
                logger.debug(f"Posted token metrics to CFN: {metric.total_tokens} tokens")
            elif response.status_code >= 500 and retry_count < self.max_retries:
                # Retry on server errors
                logger.warning(
                    f"CFN metrics API returned {response.status_code}, retrying... (attempt {retry_count + 1})"
                )
                await self._backoff(retry_count)
                await self._post_metric_internal(metric, retry_count + 1)
            else:
                logger.error(
                    f"CFN metrics API failed: {response.status_code} - {response.text[:200]}"
                )

        except httpx.TimeoutException:
            if retry_count < self.max_retries:
                logger.warning(f"CFN metrics API timeout, retrying... (attempt {retry_count + 1})")
                await self._backoff(retry_count)
                await self._post_metric_internal(metric, retry_count + 1)
            else:
                logger.error("CFN metrics API timeout after all retries")

        except Exception as e:
            logger.error(f"Failed to post metrics to CFN (non-fatal): {e}")

    async def _backoff(self, retry_count: int):
        """Exponential backoff between retries."""
        delay = min(2 ** retry_count, 10)  # max 10s
        await asyncio.sleep(delay)


# Global instance (optional - can be injected via dependency injection)
_metrics_client: Optional[CFNMetricsClient] = None


def init_metrics_client(cfn_base_url: str, enabled: bool = True) -> CFNMetricsClient:
    """
    Initialize global metrics client.
    Call once during app startup.
    """
    global _metrics_client
    _metrics_client = CFNMetricsClient(cfn_base_url, enabled=enabled)
    return _metrics_client


def get_metrics_client() -> Optional[CFNMetricsClient]:
    """Get global metrics client (or None if not initialized)."""
    return _metrics_client
```

---

### 3. Integration Examples

#### Example 1: Ingestion Service

**File:** `ingestion/app/agent/service.py`

```python
"""
ingestion/app/agent/service.py

Knowledge extraction service with token tracking.
"""
import time
from datetime import datetime, timezone
from typing import Optional

import litellm
from common.metrics.cfn_client import get_metrics_client


async def extract_knowledge(request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract concepts and relationships from telemetry data.
    Returns token usage in response.
    """
    start_time = time.time()
    
    # Build prompt
    prompt = build_extraction_prompt(request.payload.data)
    
    # Call LLM
    llm_response = await litellm.acompletion(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    
    # Parse response
    concepts, relationships = parse_extraction_results(
        llm_response.choices[0].message.content
    )
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Calculate cost (optional)
    try:
        cost_usd = litellm.completion_cost(completion_response=llm_response)
    except Exception:
        cost_usd = None
    
    # OPTIONAL: Post metrics to CFN (fire-and-forget, non-blocking)
    metrics_client = get_metrics_client()
    if metrics_client:
        await metrics_client.post_token_metric(
            workspace_id=request.workspace_id,
            mas_id=request.mas_id,
            agent_id=request.agent_id,
            model=llm_response.model,
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation="knowledge_extraction",
        )
    
    # Return response with tokens (PRIMARY - always included)
    return ExtractionResponse(
        response_id=request.request_id,
        concepts=concepts,
        relationships=relationships,
        meta=TokenUsageMeta(
            tokens=TokenUsage(
                prompt=llm_response.usage.prompt_tokens,
                completion=llm_response.usage.completion_tokens,
                total=llm_response.usage.total_tokens,
                model=llm_response.model,
            ),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
```

#### Example 2: Evidence Service

**File:** `evidence/app/agent/llm_clients.py`

```python
"""
evidence/app/agent/llm_clients.py

Evidence gathering with token tracking.
"""
import time
from datetime import datetime, timezone

import litellm
from common.metrics.cfn_client import get_metrics_client


async def gather_evidence(query_request: QueryRequest) -> QueryResponse:
    """
    Gather evidence and reason over knowledge graph.
    Returns token usage in response.
    """
    start_time = time.time()
    
    # Retrieve relevant context
    context = await retrieve_context(query_request.intent)
    
    # Build reasoning prompt
    prompt = build_reasoning_prompt(query_request.intent, context)
    
    # Call LLM
    llm_response = await litellm.acompletion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Extract answer
    answer = llm_response.choices[0].message.content
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # OPTIONAL: Post to CFN (fire-and-forget)
    metrics_client = get_metrics_client()
    if metrics_client:
        await metrics_client.post_token_metric(
            workspace_id=query_request.workspace_id,
            mas_id=query_request.mas_id,
            agent_id=query_request.agent_id,
            model=llm_response.model,
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
            latency_ms=latency_ms,
            operation="evidence_gathering",
        )
    
    # Return with tokens
    return QueryResponse(
        response_id=query_request.request_id,
        message=answer,
        records=context["records"],
        meta=TokenUsageMeta(
            tokens=TokenUsage(
                prompt=llm_response.usage.prompt_tokens,
                completion=llm_response.usage.completion_tokens,
                total=llm_response.usage.total_tokens,
                model=llm_response.model,
            ),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
```

#### Example 3: Semantic Alignment

**File:** `semantic_alignment/app/agent/sao_engine.py`

```python
"""
semantic_alignment/app/agent/sao_engine.py

SAO negotiation with token tracking.
"""
import time
from datetime import datetime, timezone

import litellm
from common.metrics.cfn_client import get_metrics_client


async def discover_issues(request: NegotiationStartRequest) -> NegotiationStartResponse:
    """
    Discover issues and generate options via LLM.
    Returns token usage in response.
    """
    start_time = time.time()
    
    # Build issue discovery prompt
    prompt = build_issue_discovery_prompt(request.content_text)
    
    # Call LLM
    llm_response = await litellm.acompletion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": ISSUE_DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Parse issues and options
    issues, options = parse_issues_and_options(llm_response.choices[0].message.content)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # OPTIONAL: Post to CFN
    metrics_client = get_metrics_client()
    if metrics_client:
        await metrics_client.post_token_metric(
            workspace_id=request.workspace_id,
            mas_id=request.mas_id,
            agent_id=request.agents[0].id if request.agents else "unknown",
            model=llm_response.model,
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
            latency_ms=latency_ms,
            operation="semantic_alignment_start",
        )
    
    # Return with tokens
    return NegotiationStartResponse(
        session_id=request.session_id,
        issues=issues,
        options=options,
        meta=TokenUsageMeta(
            tokens=TokenUsage(
                prompt=llm_response.usage.prompt_tokens,
                completion=llm_response.usage.completion_tokens,
                total=llm_response.usage.total_tokens,
                model=llm_response.model,
            ),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
```

---

### 4. App Startup Configuration

**File:** `ingestion/app/main.py` (example)

```python
"""
ingestion/app/main.py

FastAPI app with metrics client initialization.
"""
from fastapi import FastAPI
from common.metrics.cfn_client import init_metrics_client
from .config.settings import settings

app = FastAPI()


@app.on_event("startup")
async def startup():
    """Initialize metrics client on startup."""
    if settings.metrics_enabled:
        init_metrics_client(
            cfn_base_url=settings.cfn_base_url,
            enabled=True,
        )
        print(f"✅ Metrics client initialized: {settings.cfn_base_url}")
    else:
        print("⚠️  Metrics client disabled")


@app.on_event("shutdown")
async def shutdown():
    """Clean up metrics client."""
    from common.metrics.cfn_client import get_metrics_client
    
    client = get_metrics_client()
    if client:
        await client.close()
```

**File:** `.env` (configuration)

```bash
# CFN Service
CFN_BASE_URL=http://localhost:9002

# Metrics Configuration
METRICS_ENABLED=true
```

---

### 5. Response Schema Definitions

For consistency across all services, define shared Pydantic models:

**File:** `common/models/responses.py`

```python
"""
common/models/responses.py

Shared response models with token usage.
"""
from pydantic import BaseModel
from typing import Optional


class TokenUsage(BaseModel):
    """Token usage information."""
    
    prompt: int
    completion: int
    total: int
    model: str


class TokenUsageMeta(BaseModel):
    """Metadata including token usage."""
    
    tokens: TokenUsage
    latency_ms: float
    cost_usd: Optional[float] = None
    timestamp: str


# Example usage in service-specific responses:
class ExtractionResponse(BaseModel):
    response_id: str
    concepts: list
    relationships: list
    meta: TokenUsageMeta


class QueryResponse(BaseModel):
    response_id: str
    message: str
    records: list
    meta: TokenUsageMeta


class NegotiationStartResponse(BaseModel):
    session_id: str
    issues: list
    options: list
    meta: TokenUsageMeta
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Days 1-2)

**Day 1: Metrics Client**
- [ ] Create `common/metrics/` package
- [ ] Implement `cfn_client.py` (simple fire-and-forget client)
- [ ] Add unit tests (mock CFN API)
- [ ] Add configuration (`.env`, settings)

**Day 2: Response Models**
- [ ] Create `common/models/responses.py`
- [ ] Define `TokenUsage`, `TokenUsageMeta` models
- [ ] Document usage patterns
- [ ] Add validation tests

### Phase 2: Service Integration (Days 3-5)

**Day 3: Ingestion Service**
- [ ] Update `extract_knowledge()` to return tokens
- [ ] Add optional metrics posting
- [ ] Update API tests to verify tokens in response
- [ ] Deploy to dev environment

**Day 4: Evidence Service**
- [ ] Update `gather_evidence()` to return tokens
- [ ] Add optional metrics posting
- [ ] Update API tests to verify tokens in response
- [ ] Deploy to dev environment

**Day 5: Semantic Alignment Service**
- [ ] Update `discover_issues()` and other LLM calls to return tokens
- [ ] Add optional metrics posting
- [ ] Update API tests to verify tokens in response
- [ ] Deploy to dev environment

### Phase 3: Validation & Rollout (Days 6-7)

**Day 6: Validation**
- [ ] End-to-end testing with test script (`run_full_test.sh`)
- [ ] Verify tokens in all responses
- [ ] Verify metrics in CFN TimescaleDB (if enabled)
- [ ] Performance testing (ensure <1ms overhead)

**Day 7: Documentation & Rollout**
- [ ] Update API documentation (OpenAPI specs)
- [ ] Create contributor guide ("How to add token tracking")
- [ ] Deploy to staging
- [ ] Deploy to production

---

## Best Practices

### 1. Always Return Tokens in Response

✅ **DO:**
```python
return {
    "data": {...},
    "meta": {
        "tokens": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
            "model": response.model,
        },
        "latency_ms": latency_ms,
    }
}
```

❌ **DON'T:**
```python
return {
    "data": {...}
}
# Missing tokens - user has no visibility!
```

### 2. Centralized Metrics Are Optional

✅ **DO:**
```python
# Fire-and-forget, non-blocking
metrics_client = get_metrics_client()
if metrics_client:
    await metrics_client.post_token_metric(...)
    
# Continue - don't wait for metrics to post
return response
```

❌ **DON'T:**
```python
# Don't block response on metrics posting
await metrics_client.post_token_metric(...)  # This blocks!
return response
```

### 3. Error Handling

✅ **DO:**
```python
try:
    cost_usd = litellm.completion_cost(completion_response=llm_response)
except Exception:
    cost_usd = None  # Graceful fallback
```

❌ **DON'T:**
```python
cost_usd = litellm.completion_cost(completion_response=llm_response)
# Might raise exception and break the response!
```

### 4. Configuration

✅ **DO:**
```python
# Use feature flags for metrics posting
if settings.metrics_enabled:
    init_metrics_client(...)
```

❌ **DON'T:**
```python
# Don't hardcode - make it configurable
init_metrics_client("http://localhost:9002")
```

### 5. Testing

✅ **DO:**
```python
def test_extraction_returns_tokens():
    response = await extract_knowledge(request)
    assert "meta" in response
    assert "tokens" in response["meta"]
    assert response["meta"]["tokens"]["total"] > 0
```

❌ **DON'T:**
```python
# Don't test only that it doesn't error
def test_extraction():
    response = await extract_knowledge(request)
    assert response  # Not enough!
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_cfn_client.py
import pytest
from httpx import AsyncClient
from common.metrics.cfn_client import CFNMetricsClient


@pytest.mark.asyncio
async def test_post_token_metric_success(mock_cfn_server):
    """Test successful metrics posting."""
    client = CFNMetricsClient("http://localhost:9002")
    
    await client.post_token_metric(
        workspace_id="ws-123",
        mas_id="mas-456",
        agent_id="agent-789",
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=1234.5,
    )
    
    # Fire-and-forget - need to wait a bit for async task
    await asyncio.sleep(0.1)
    
    # Verify mock server received request
    assert mock_cfn_server.request_count == 1
    assert mock_cfn_server.last_request["workspace_id"] == "ws-123"


@pytest.mark.asyncio
async def test_post_token_metric_graceful_failure():
    """Test graceful failure when CFN is down."""
    client = CFNMetricsClient("http://localhost:9999")  # Wrong port
    
    # Should not raise - fire-and-forget
    await client.post_token_metric(
        workspace_id="ws-123",
        mas_id="mas-456",
        agent_id="agent-789",
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=1234.5,
    )
    
    # No exception raised
```

### Integration Tests

```python
# tests/test_ingestion_service.py
import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_extraction_returns_tokens():
    """Test that extraction response includes token usage."""
    request = {
        "request_id": "test-001",
        "header": {
            "agent_id": "agent-001",
            "workspace_id": "ws-123",
            "mas_id": "mas-456",
        },
        "payload": {
            "metadata": {"format": "observe-sdk-otel"},
            "data": [...],
        },
    }
    
    response = await client.post("/api/knowledge-mgmt/extraction", json=request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify token usage in response
    assert "meta" in data
    assert "tokens" in data["meta"]
    assert data["meta"]["tokens"]["prompt"] > 0
    assert data["meta"]["tokens"]["completion"] > 0
    assert data["meta"]["tokens"]["total"] > 0
    assert data["meta"]["tokens"]["model"] in ["gpt-4", "claude-3-5-sonnet-20241022"]
    assert data["meta"]["latency_ms"] > 0
```

### End-to-End Test

Update `/Users/samuyang/go/src/github.com/ioc-cfn-svc/docs/temp/run_full_test.sh`:

```bash
# Function to check for token usage (UPDATED to expect tokens)
check_token_usage() {
    local response="$1"
    echo ""
    if echo "$response" | grep -q "\"prompt\":\|\"completion\":\|\"total\":"; then
        echo -e "${GREEN}✅ TOKEN USAGE FOUND:${NC}"
        echo "$response" | jq '.meta.tokens' || echo "$response" | grep -E "prompt|completion|total"
    else
        echo -e "${RED}❌ NO TOKEN USAGE FOUND in response (EXPECTED!)${NC}"
        exit 1
    fi
}
```

---

## Success Metrics

### Technical Metrics

- [ ] Token usage returned in **100%** of API responses
- [ ] Metrics delivery success rate >95% (when enabled)
- [ ] Overhead <1ms per request
- [ ] Zero impact on LLM call success rate
- [ ] Graceful degradation when CFN metrics API is down

### Business Metrics

- [ ] Cost visibility per workspace/MAS/agent in real-time
- [ ] Users can see token usage immediately in responses
- [ ] Token consumption trends tracked (when centralized metrics enabled)
- [ ] Performance correlation with token usage visible
- [ ] Matches industry standards (OpenAI, Anthropic, AWS)

---

## Comparison: v1.0 vs v2.0

| Aspect | v1.0 (Wrapper) | v2.0 (Hybrid) |
|--------|----------------|---------------|
| **Complexity** | High (6 components) | Low (2 components) |
| **Contributor Experience** | 😐 Must learn wrapper pattern | 😊 Copy-paste pattern |
| **User Visibility** | 😕 Must query metrics API | 😊 Immediate in response |
| **Industry Standard** | ❌ Non-standard | ✅ Matches OpenAI/Anthropic |
| **Failure Modes** | 😐 Silent if wrapper not used | 😊 Always works |
| **Implementation** | Complex (context vars, middleware) | Simple (return what LiteLLM gives you) |
| **Debugging** | 😐 Multi-step | 😊 Single response |
| **Centralized Analytics** | ✅ Yes | ✅ Yes (optional) |
| **Lines of Code** | ~800 lines | ~300 lines |
| **Onboarding Time** | 1 hour | 5 minutes |

**Winner: v2.0 Hybrid** - Simpler, clearer, matches industry standards, better UX.

---

## Appendix: Metric Names (CFN TimescaleDB)

### Standard Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `llm.tokens.prompt` | gauge | Prompt tokens consumed |
| `llm.tokens.completion` | gauge | Completion tokens consumed |
| `llm.tokens.total` | gauge | Total tokens consumed |
| `llm.latency_ms` | gauge | LLM call latency in milliseconds |
| `llm.cost_usd` | gauge | Estimated cost in USD |
| `llm.errors` | counter | Number of LLM call failures |

### Standard Attributes

| Attribute | Description |
|-----------|-------------|
| `model` | LLM model name |
| `service` | Service name (ingestion/evidence/semantic_alignment) |
| `operation` | Operation name (e.g. "knowledge_extraction") |

---

## Questions & Answers

**Q: Why return tokens in response vs. wrapper approach?**
A: Industry standard (OpenAI, Anthropic, AWS all do this). Immediate visibility. Simpler implementation. Better UX.

**Q: What if CFN metrics API is down?**
A: Tokens still returned in response (PRIMARY). Centralized metrics are fire-and-forget with retry logic, but failures don't block responses.

**Q: Performance impact?**
A: <1ms. Just extracting `response.usage` and adding to JSON response. Metrics posting is fire-and-forget (non-blocking).

**Q: What about cost tracking accuracy?**
A: Uses LiteLLM's built-in cost calculator. Accuracy depends on LiteLLM's price database.

**Q: Can I disable centralized metrics?**
A: Yes, via `METRICS_ENABLED=false` environment variable. Tokens still returned in response.

**Q: Thread-safe?**
A: Yes, fire-and-forget uses `asyncio.create_task()` which is async-safe.

**Q: What if workspace_id/mas_id is missing?**
A: Tokens still returned in response (PRIMARY). Centralized metrics use "unknown" and log warning.

**Q: How do new contributors add token tracking?**
A: Copy-paste pattern from any existing service. Just return what LiteLLM gives you. Impossible to forget.

---

**End of Document**
