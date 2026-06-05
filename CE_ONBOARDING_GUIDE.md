# Cognition Engine (CE) Onboarding Guide

A step-by-step guide to onboard your own Cognition Engine with registration, heartbeats, and metrics.

---

## Table of Contents

1. [What is a CE?](#what-is-a-ce)
2. [Prerequisites](#prerequisites)
3. [Step 1: Create Your CE Service](#step-1-create-your-ce-service)
4. [Step 2: Add CE Registration](#step-2-add-ce-registration)
5. [Step 3: Enable Heartbeats](#step-3-enable-heartbeats)
6. [Step 4: Register Metrics](#step-4-register-metrics)
7. [Step 5: Post Metrics](#step-5-post-metrics)
8. [Step 6: Test Your CE](#step-6-test-your-ce)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## What is a CE?

A **Cognition Engine (CE)** is a specialized service that provides intelligence capabilities to the Multi-Agent System (MAS). Examples:
- Knowledge Management CE: Document ingestion, retrieval, search
- Semantic Negotiation CE: Multi-party coordination, semantic reasoning
- Evidence CE: Evidence extraction and analysis

Each CE:
- **Registers** with CFN (Control Flow Node) on startup
- **Heartbeats** periodically to signal it's online
- **Posts metrics** about operations (LLM tokens, latency, custom metrics)

---

## Prerequisites

1. **Python 3.9+** installed
2. **CFN service running** (default: `http://localhost:9002`)
3. **Common library** installed: `pip install -e common/`
4. **Environment variables** set:
   ```bash
   export CFN_URL="http://localhost:9002"
   export COGNITION_ENGINE_HOST="localhost"
   export COGNITION_ENGINE_PORT="9004"  # Your CE's port
   export CE_HEARTBEAT_INTERVAL_SEC="30"  # Optional, default is 30s
   ```

---

## Step 1: Create Your CE Service

Create a FastAPI service with health endpoint:

```python
# my_ce/app/main.py

from fastapi import FastAPI

app = FastAPI(title="My Custom CE")

@app.get("/health")
async def health():
    """Health check endpoint (required for CFN)."""
    return {"status": "healthy"}

@app.post("/api/my-operation")
async def my_operation(request: dict):
    """Your CE's main operation."""
    # Your logic here
    return {"result": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9004)
```

**Required**: Your CE MUST have a `/health` endpoint that returns HTTP 200 for CFN health checks.

---

## Step 2: Add CE Registration

Use `CELifecycleClient` to register on startup:

```python
# my_ce/app/registration.py

import os
import logging
from common.ce_lifecycle.client import (
    CELifecycleClient,
    CERegistrationRequest,
)
from common.metrics.names import get_llm_metric_names

logger = logging.getLogger(__name__)

# Global: Store CE ID after registration
_ce_id = None
_lifecycle_client = None

async def register_my_ce():
    """Register CE on startup and start heartbeat."""
    global _ce_id, _lifecycle_client

    cfn_url = os.getenv("CFN_URL", "http://localhost:9002")
    ce_host = os.getenv("COGNITION_ENGINE_HOST", "localhost")
    ce_port = int(os.getenv("COGNITION_ENGINE_PORT", "9004"))

    # Create lifecycle client
    _lifecycle_client = CELifecycleClient(
        cfn_base_url=cfn_url,
        enabled=True,  # Set to False to disable registration
        heartbeat_interval_sec=30.0,
        timeout=10.0,
    )

    # Define CE configuration
    ce_config = CERegistrationRequest(
        name="My Custom CE",  # Unique name
        url=f"http://{ce_host}:{ce_port}",
        version="1.0.0",
        kind="custom",  # CE category: knowledge, negotiation, evidence, custom
        subkind="processing",  # CE subcategory: query, ingestion, analysis, etc.
        capabilities=["processing", "analysis"],  # List of capabilities
        metrics=get_llm_metric_names(),  # Metrics you'll report (see Step 4)
        config={
            "model": "openai/gpt-4o",  # Your LLM model
            "max_tokens": 4096,
        },
        mas_config=None,  # MAS-specific config (optional)
        mas_auto_associate=False,  # Auto-associate with all MAS instances
    )

    # Register with CFN
    response = await _lifecycle_client.register(ce_config)

    if response:
        _ce_id = response.ce_id
        action = "created" if response.created else "updated"
        logger.info(
            f"CE registered ({action}): ce_id={response.ce_id}, "
            f"status={response.status}, enabled={response.enabled}"
        )

        # Start heartbeat background task
        _lifecycle_client.start_heartbeat()
        logger.info("Heartbeat task started")

        return response.ce_id
    else:
        logger.warning("CE registration failed")
        return None

async def shutdown_ce():
    """Gracefully shutdown CE and stop heartbeats."""
    global _lifecycle_client

    if _lifecycle_client:
        await _lifecycle_client.close()
        logger.info("CE lifecycle client closed")

def get_ce_id():
    """Get registered CE ID."""
    return _ce_id
```

**Integrate with FastAPI startup**:

```python
# my_ce/app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from my_ce.app.registration import register_my_ce, shutdown_ce

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register CE on startup, shutdown on exit."""
    # Startup
    await register_my_ce()
    yield
    # Shutdown
    await shutdown_ce()

app = FastAPI(title="My Custom CE", lifespan=lifespan)

# Your endpoints here...
```

---

## Step 3: Enable Heartbeats

Heartbeats are **automatically enabled** when you call `client.start_heartbeat()` after registration.

### How Heartbeats Work

1. **Background Task**: Creates an asyncio task that runs in the background
2. **Periodic Ping**: Sends `PUT /api/cognition-engines/{ce_id}/heartbeat` every 30s
3. **Keep-Alive**: Tells Management Plane your CE is "online"
4. **Auto-Recovery**: If a heartbeat fails, logs error and retries on next interval
5. **Graceful Shutdown**: Task is cancelled when you call `client.close()`

### Heartbeat Configuration

```python
# Control heartbeat interval (seconds)
export CE_HEARTBEAT_INTERVAL_SEC="30"  # Default: 30s

# In code
client = CELifecycleClient(
    cfn_base_url=cfn_url,
    enabled=True,
    heartbeat_interval_sec=30.0,  # Seconds between heartbeats
    timeout=10.0,  # HTTP timeout
)
```

### When Heartbeats Stop

If your CE crashes or stops sending heartbeats, Management Plane marks it as **"offline"** after ~2 minutes.

---

## Step 4: Register Metrics

Declare what metrics your CE will report during registration.

### Built-in Metrics (LLM Operations)

If your CE uses LLMs, CFN **automatically captures** these metrics from response metadata:

```python
from common.metrics.names import get_llm_metric_names

metrics=get_llm_metric_names()
# Returns:
# [
#   "llm.tokens.prompt",
#   "llm.tokens.completion",
#   "llm.tokens.total",
#   "llm.latency_ms",
#   "llm.cost_usd"
# ]
```

**How it works**: CFN intercepts your CE's responses, extracts `TokenUsageMeta` from headers, and posts metrics automatically.

### Custom Metrics

For CE-specific metrics (queue depth, cache hits, etc.), register them during CE setup:

```python
from common.metrics.names import get_llm_metric_names

# Combine LLM metrics + custom metrics
metrics = get_llm_metric_names() + [
    "ce.queue.depth",          # Active requests in queue
    "ce.cache.hits",           # Cache hit count
    "ce.cache.misses",         # Cache miss count
    "kb.documents.indexed",    # Documents indexed
    "kb.search.latency_ms",    # Search latency
]

ce_config = CERegistrationRequest(
    name="My Custom CE",
    # ...
    metrics=metrics,  # Register all metrics
)
```

**IMPORTANT**: You MUST register metrics before posting them. CFN validates metric names against this list.

---

## Step 5: Post Metrics

Use `CFNMetricsClient` to post custom metrics:

```python
# my_ce/app/metrics.py

import asyncio
from common.metrics.cfn_client import CFNMetricsClient, MetricDataPoint
from my_ce.app.registration import get_ce_id

# Global metrics client
_metrics_client = None

def init_metrics_client(cfn_url: str):
    """Initialize metrics client after registration."""
    global _metrics_client

    ce_id = get_ce_id()  # Get from registration
    if ce_id:
        _metrics_client = CFNMetricsClient(
            cfn_base_url=cfn_url,
            ce_id=ce_id,
            enabled=True,
        )

async def post_custom_metrics():
    """Post custom CE metrics."""
    if not _metrics_client or not _metrics_client.enabled:
        return

    metrics = [
        MetricDataPoint(
            name="ce.queue.depth",
            value=15.0,
            attributes={"service": "processing"},
        ),
        MetricDataPoint(
            name="ce.cache.hits",
            value=42.0,
        ),
    ]

    await _metrics_client.post_metrics_batch(
        metrics=metrics,
        attributes={"host": "ce-node-1"},  # Batch-level attributes
    )
```

**Initialize after registration**:

```python
# my_ce/app/registration.py

async def register_my_ce():
    # ... registration code ...

    if response:
        _ce_id = response.ce_id

        # Initialize metrics client
        from my_ce.app.metrics import init_metrics_client
        init_metrics_client(cfn_url)

        # Start heartbeat
        _lifecycle_client.start_heartbeat()
```

**Post metrics periodically**:

```python
# In your endpoint
@app.post("/api/my-operation")
async def my_operation(request: dict):
    # Your logic here
    result = process_request(request)

    # Post metrics (fire-and-forget)
    asyncio.create_task(post_custom_metrics())

    return result
```

---

## Step 6: Test Your CE

### 1. Start CFN (if not running)

```bash
cd ioc-cfn-svc
make run
# CFN runs on http://localhost:9002
```

### 2. Start Your CE

```bash
cd my_ce
export CFN_URL="http://localhost:9002"
export COGNITION_ENGINE_HOST="localhost"
export COGNITION_ENGINE_PORT="9004"

python -m my_ce.app.main
```

### 3. Verify Registration

Check logs for:
```
INFO: CE registered (created): ce_id=ce-abc-123, status=online, enabled=True
INFO: Heartbeat task started
```

### 4. Query CFN for Your CE

```bash
# Get all CEs
curl http://localhost:9002/api/cognition-engines

# Get specific CE
curl http://localhost:9002/api/cognition-engines/{ce_id}
```

Expected response:
```json
{
  "id": "ce-abc-123",
  "cfn_id": "cfn-xyz-456",
  "name": "My Custom CE",
  "version": "1.0.0",
  "kind": "custom",
  "status": "online",
  "last_seen": "2026-06-04T10:30:00Z",
  "metrics": ["llm.tokens.prompt", "ce.queue.depth", ...]
}
```

### 5. Check Heartbeats

Wait 30 seconds and check logs:
```
INFO: Heartbeat sent successfully for 'My Custom CE': status=online
```

### 6. Test Metrics

Post custom metrics and query:

```bash
# Query CE metrics
curl "http://localhost:9002/api/cognition-engines/{ce_id}/metrics?start_time=2026-06-04T00:00:00Z&end_time=2026-06-04T23:59:59Z"
```

---

## Common Patterns

### Pattern 1: Multiple CEs in One Service

If you have multiple logical CEs in one service (like gateway with Knowledge + Semantic Negotiation):

```python
ce_configs = [
    CERegistrationRequest(name="CE A", kind="knowledge", ...),
    CERegistrationRequest(name="CE B", kind="negotiation", ...),
]

# Registry to store CE IDs
_ce_registry = {}  # name -> ce_id

for ce_config in ce_configs:
    client = CELifecycleClient(cfn_base_url=cfn_url, ...)
    response = await client.register(ce_config)

    if response:
        _ce_registry[ce_config.name] = response.ce_id
        client.start_heartbeat()  # Each CE has separate heartbeat

# Lookup CE ID by name
def get_ce_id(name: str):
    return _ce_registry.get(name)
```

### Pattern 2: Conditional Registration

Skip registration in dev/testing by not setting CFN_URL:

```python
# Registration only happens if CFN_URL is set
cfn_url = os.getenv("CFN_URL")
if not cfn_url:
    logger.info("CFN_URL not set, CE will run without registration")
    return  # Skip registration

# CFN_URL is set, proceed with registration
client = CELifecycleClient(cfn_base_url=cfn_url)
response = await client.register(ce_config)
```

### Pattern 3: Idempotent Registration

Management Plane deduplicates by `(cfn_id, name, version)`:
- **First call**: Creates CE, returns `created=true`
- **Subsequent calls**: Updates CE, returns `created=false`, returns existing `ce_id`

This makes restarts safe - you get the same `ce_id` back.

### Pattern 4: Metrics with Context

Add attributes to correlate metrics:

```python
await client.post_metrics_batch(
    metrics=[
        MetricDataPoint(
            name="kb.search.latency_ms",
            value=125.5,
            attributes={
                "query_type": "semantic",  # Metric-level
                "index": "documents",
            },
        ),
    ],
    attributes={
        "host": "ce-node-1",  # Batch-level
        "region": "us-west-2",
    },
)
```

Attributes help filter/aggregate in queries.

---

## Troubleshooting

### Registration Fails

**Symptom**: `"CE registration failed: 503 Service Unavailable"`

**Cause**: CFN or Management Plane not reachable

**Fix**:
1. Check CFN is running: `curl http://localhost:9002/health`
2. Check Management Plane URL in CFN config
3. Review CFN logs for connection errors

---

### Heartbeat Stops

**Symptom**: CE shows "offline" in Management Plane

**Cause**: Heartbeat task crashed or CE lost `ce_id`

**Fix**:
1. Check CE logs for heartbeat errors
2. Verify `start_heartbeat()` was called after registration
3. Restart CE (registration is idempotent, will resume heartbeating)

---

### Metrics Rejected

**Symptom**: `400 Bad Request: unregistered_metrics`

**Cause**: You're posting metrics not in your registration

**Fix**:
```python
# Make sure metrics are registered during CE registration
ce_config = CERegistrationRequest(
    name="My CE",
    metrics=[
        "llm.tokens.prompt",
        "ce.custom_metric",  # ✅ Add this
    ],
)
```

Then restart CE to re-register with updated metrics.

---

### Metrics Not Showing in Queries

**Symptom**: `GET /api/cognition-engines/{ce_id}/metrics` returns empty

**Causes**:
1. **Metric names don't match registration** (CFN rejects silently)
2. **Time range is wrong** (check `start_time`/`end_time`)
3. **Metrics client disabled** (`enabled=False`)

**Fix**:
1. Check CFN logs for rejected metrics
2. Verify time range covers when metrics were posted
3. Ensure `CFNMetricsClient(enabled=True, ce_id=...)`

---

### CE ID is None

**Symptom**: `_ce_id` is None after registration

**Cause**: Registration failed or didn't complete

**Fix**:
```python
response = await client.register(ce_config)
if response:
    _ce_id = response.ce_id
else:
    logger.error("Registration failed, check CFN connectivity")
```

---

## Summary Checklist

- [ ] Create FastAPI service with `/health` endpoint
- [ ] Install `common` library: `pip install -e common/`
- [ ] Add `CELifecycleClient` registration on startup
- [ ] Call `start_heartbeat()` after successful registration
- [ ] Register metrics in `CERegistrationRequest.metrics`
- [ ] Initialize `CFNMetricsClient` after getting `ce_id`
- [ ] Post metrics using `post_metrics_batch()`
- [ ] Test registration: `curl http://localhost:9002/api/cognition-engines`
- [ ] Test heartbeat: Check CE shows "online"
- [ ] Test metrics: Query endpoint returns your metrics

---

## Example: Complete CE Service

```python
# my_ce/app/main.py

import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from common.ce_lifecycle.client import CELifecycleClient, CERegistrationRequest
from common.metrics.names import get_llm_metric_names
from common.metrics.cfn_client import CFNMetricsClient

logger = logging.getLogger(__name__)

# Globals
_ce_id = None
_lifecycle_client = None
_metrics_client = None

async def register_ce():
    """Register CE and start heartbeat."""
    global _ce_id, _lifecycle_client, _metrics_client

    cfn_url = os.getenv("CFN_URL", "http://localhost:9002")
    ce_host = os.getenv("COGNITION_ENGINE_HOST", "localhost")
    ce_port = int(os.getenv("COGNITION_ENGINE_PORT", "9004"))

    _lifecycle_client = CELifecycleClient(cfn_base_url=cfn_url, enabled=True)

    ce_config = CERegistrationRequest(
        name="My Custom CE",
        url=f"http://{ce_host}:{ce_port}",
        version="1.0.0",
        kind="custom",
        subkind="processing",
        capabilities=["processing"],
        metrics=get_llm_metric_names() + ["ce.queue.depth"],
        config={"model": "openai/gpt-4o"},
    )

    response = await _lifecycle_client.register(ce_config)
    if response:
        _ce_id = response.ce_id
        logger.info(f"Registered: ce_id={_ce_id}")

        _lifecycle_client.start_heartbeat()
        logger.info("Heartbeat started")

        _metrics_client = CFNMetricsClient(cfn_base_url=cfn_url, ce_id=_ce_id)

async def shutdown_ce():
    """Stop heartbeat and close client."""
    if _lifecycle_client:
        await _lifecycle_client.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await register_ce()
    yield
    await shutdown_ce()

app = FastAPI(title="My Custom CE", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/process")
async def process(request: dict):
    # Your logic here
    return {"result": "success", "ce_id": _ce_id}
```

**Run it**:
```bash
export CFN_URL="http://localhost:9002"
python -m my_ce.app.main
```

---

## Next Steps

1. **Add Custom Metrics**: Define CE-specific metrics in `common/metrics/names.py`
2. **Update CFN Validation**: See [CFN_METRICS_VALIDATION_SPEC.md](CFN_METRICS_VALIDATION_SPEC.md)
3. **Deploy**: Use Docker/K8s, pass `CFN_URL` via env vars
4. **Monitor**: Query metrics endpoint to track CE health

For questions, see [METRICS_ALIGNMENT.md](METRICS_ALIGNMENT.md) or ask the team!
