# Cognition Engine - Developer Guide

A unified knowledge extraction, evidence gathering, and semantic negotiation engine for multi-agent systems.

## Table of Contents

- [Quick Start](#-quick-start) - Installation & environment setup
- [Core Components](#-core-components) - Overview of the three main components
- [Usage Examples](#-usage-examples) - Code examples for common use cases
  - [Semantic Negotiation](#example-1-semantic-negotiation)
  - [FastAPI with Multiple Endpoints](#example-2-fastapi-application-with-multiple-endpoints)
  - [Standalone Scripts](#example-3-standalone-script-knowledge--evidence)
- [Best Practices](#-best-practices) - Cache sharing, embeddings, LLM providers
- [Testing](#-testing) - Mock mode & integration tests
- [Configuration](#️-configuration-reference) - Environment variables
- [API Reference](#-api-reference) - Class and method documentation

---

## 📦 Quick Start

### Installation

```bash
# From Artifactory
pip install cognition-engine --extra-index-url https://your-artifactory-url/artifactory/api/pypi/outshift-pypi/simple

# From source
poetry build
pip install dist/cognition_engine-0.1.0-py3-none-any.whl
```

### Environment Setup

Create a `.env` file with your Azure OpenAI credentials:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Optional: For semantic negotiation, you can also use OpenAI or AWS Bedrock
LLM_PROVIDER=azure-openai  # or 'openai' or 'bedrock'
```

### Embedding Model

The package uses `fastembed` for embeddings (~100MB, auto-downloads on first use):

```python
from fastembed import TextEmbedding
TextEmbedding('BAAI/bge-small-en-v1.5')  # Pre-download if needed
```

---

## 🧩 Core Components

The engine includes three main components:

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Knowledge Extraction** | Extract concepts and relationships from telemetry data | `ConceptRelationshipExtractionService`, `KnowledgeProcessor` |
| **Evidence Gathering** | Retrieve relevant evidence from knowledge cache | `process_evidence()`, `CachingLayer` |
| **Semantic Negotiation** | Multi-issue negotiation between agents | `SemanticNegotiationPipeline`, `IntentDiscovery`, `OptionsGeneration` |

---

## 💡 Usage Examples

### Example 1: Semantic Negotiation

**Use case:** Multi-agent negotiation to reach consensus on multiple issues.

The semantic negotiation pipeline follows a **3-component flow**:

1. **Intent Discovery** - Extract negotiable issues from natural language
2. **Options Generation** - Generate possible values for each issue
3. **Negotiation Model** - Run multi-issue negotiation using NegMAS SAO

#### Standalone Script (Without Server)

```python
import os
from dotenv import load_dotenv
from semantic_negotiation.app.agent.intent_discovery import IntentDiscovery
from semantic_negotiation.app.agent.options_generation import OptionsGeneration
from semantic_negotiation.app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationParticipant,
)

load_dotenv()

# Negotiation scenario
scenario = """
Alice wants to plan a vacation trip. She is flexible on the destination
but prefers somewhere warm. Her budget is limited to $2000 total.
She wants to stay in a hotel with good reviews.

Bob is helping her plan the trip. He suggests considering both the
destination and accommodation type. He thinks an Airbnb might offer
better value than a hotel.
"""

# Component 1: Discover negotiable issues
intent_discovery = IntentDiscovery()
issues = intent_discovery.discover(scenario)
print(f"✓ Discovered {len(issues)} issues: {issues}")
# Output: ['destination', 'budget', 'accommodation type', ...]

# Component 2: Generate options for each issue
options_gen = OptionsGeneration()
options_per_issue = options_gen.generate_options(
    negotiable_entities=issues,
    sentence=scenario
)
print(f"✓ Generated options:")
for issue, options in options_per_issue.items():
    print(f"  {issue}: {options}")
# Output: {'destination': ['Hawaii', 'Florida', ...], ...}

# Component 3: Run negotiation with two participants
alice = NegotiationParticipant(
    id="alice",
    name="Alice",
    preferences={
        issue: {opt: round(1.0 - i / max(len(options) - 1, 1), 3)
                for i, opt in enumerate(options)}
        for issue, options in options_per_issue.items()
    }
)

bob = NegotiationParticipant(
    id="bob",
    name="Bob",
    preferences={
        issue: {opt: round(i / max(len(options) - 1, 1), 3)
                for i, opt in enumerate(options)}
        for issue, options in options_per_issue.items()
    }
)

negotiation_model = NegotiationModel(n_steps=20)
result = negotiation_model.run(
    issues=issues,
    options_per_issue=options_per_issue,
    participants=[alice, bob]
)

print(f"\n✓ Negotiation completed in {result.steps} steps")
print(f"✓ Agreement reached: {result.agreement is not None}")

if result.agreement:
    print("Final agreement:")
    for outcome in result.agreement:
        print(f"  {outcome.issue_id}: {outcome.chosen_option}")
# Output:
#   destination: Hawaii
#   budget: $2000
#   accommodation type: hotel with 8+ rating
```

#### Using the Pipeline (All-in-One)

`SemanticNegotiationPipeline.execute()` is the unified entry point. It branches
automatically based on whether a session already exists:

- **New `session_id`** → runs Components 1+2 (`discover_and_generate`) then seeds
  the SAO session (`start_negotiation`).
- **Known `session_id`** → advances the SAO by one agent-reply batch
  (`step_negotiation`).

```python
from semantic_negotiation.app.agent.semantic_negotiation import SemanticNegotiationPipeline

pipeline = SemanticNegotiationPipeline(n_steps=20)

# ── Initiate (Components 1+2+3 seed) ─────────────────────────────────────
initiate_result = pipeline.execute(
    session_id="session-123",
    content_text=scenario,
    agents_raw=[
        {"id": "alice", "name": "Alice"},
        {"id": "bob",   "name": "Bob"},
    ],
    n_steps=20,
)
# initiate_result keys: status, session_id, issues, options_per_issue, n_steps, round, messages

# ── Decide loop (Component 3, repeated until terminal) ───────────────────
import time
while True:
    # Collect replies from your agents (list of dicts with 'agent_id' + 'bid')
    agent_replies = collect_replies(initiate_result["messages"])

    decide_result = pipeline.execute(
        session_id="session-123",
        agent_replies=agent_replies,
    )

    if decide_result["status"] == "ongoing":
        initiate_result = decide_result  # next messages to dispatch
        continue

    # Terminal: agreed | broken | timeout
    print(f"Status : {decide_result['status']}")
    print(f"Result : {decide_result['result']}")
    break
```

You can also call the sub-methods individually:

```python
# Components 1+2 only
issues, options_per_issue = pipeline.discover_and_generate(scenario)

# Component 3 seed
runner, sess, first_messages = pipeline.start_negotiation(
    issues, options_per_issue, participants, session_id="session-123"
)

# Component 3 advance
status, next_messages, result = pipeline.step_negotiation(runner, sess, agent_replies)
```

#### With FastAPI Server

The server exposes a **turn-by-turn protocol**: one `/initiate` call to seed the
session, then repeated `/decide` calls until a terminal status is returned.

Start the server:

```bash
cd semantic_negotiation
uvicorn app.main:app --port 8089
```

**Step 1 — Initiate** (runs Components 1+2 and seeds the SAO session):

```python
import httpx

client = httpx.Client(base_url="http://localhost:8089")

resp = client.post("/api/v1/negotiate/initiate", json={
    "origin": {"tenant_id": "workspace-1", "actor_id": "mas-1"},
    "semantic_context": {"session_id": "session-123"},
    "message_id": "msg-001",
    "payload": {
        "content_text": scenario,
        "agents": [
            {"id": "alice", "name": "Alice"},
            {"id": "bob",   "name": "Bob"},
        ],
        "n_steps": 20,
    },
})
data = resp.json()
# data["payload"]["messages"] — first batch of SSTP propose messages to deliver
```

**Step 2 — Decide loop** (advance the SAO one batch at a time):

```python
while True:
    resp = client.post("/api/v1/negotiate/decide", json={
        "origin": {"tenant_id": "workspace-1", "actor_id": "mas-1"},
        "semantic_context": {"session_id": "session-123"},
        "message_id": "msg-002",
        "payload": {
            "agent_replies": [
                # Each reply contains the agent's counter-bid
                {"agent_id": "alice", "bid": {"budget": "$2000", "timeline": "2 weeks"}},
                {"agent_id": "bob",   "bid": {"budget": "$2500", "timeline": "1 week"}},
            ]
        },
    })
    payload = resp.json()["payload"]

    if payload["status"] == "ongoing":
        # Deliver payload["messages"] to agents and loop
        continue

    # Terminal — payload["result"] holds the SSTPCommitMessage envelope
    print("Negotiation finished:", payload["status"])
    break
```

When `status` is `"ongoing"` the response `payload.messages` contains the next
batch of SSTP propose messages that your agents must reply to. When it is
`"agreed"`, `"broken"`, or `"timeout"` the server returns a `SSTPCommitMessage`
envelope with the final agreement (or `None` for broken/timeout).

**Key points:**
- ✅ No server required for standalone use — import components directly
- ✅ Supports Azure OpenAI, OpenAI, and AWS Bedrock via `LLM_PROVIDER` env var
- ✅ `pipeline.execute()` is the unified entry point — branches on session existence
- ✅ Server mode is **turn-by-turn**: `/initiate` seeds the session, `/decide` advances it one round at a time
- ✅ Agent participants are identified by `id` and `name`
- ✅ `SSTPCommitMessage` envelope is returned when the negotiation terminates

---

### Example 2: FastAPI Application with Multiple Endpoints

**Use case:** FastAPI app with knowledge extraction and evidence gathering endpoints sharing a unified cache.

```python
from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from evidence.app.agent.evidence import process_evidence
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings
import os

# Create shared cache at app startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize embedding manager (SHARED by all endpoints)
    embedding_manager = EmbeddingManager()

    def embed_fn(text: str):
        out = embedding_manager.generate_embedding(text)
        if out is None:
            raise ValueError("Embedding returned None")
        return out

    # 2. Create ONE cache layer (shared across all requests)
    cache_layer = CachingLayer(
        vector_dimension=384,  # bge-small-en-v1.5 dimension
        metric="l2",
        embed_fn=embed_fn,
    )

    # Store in app state (persists across all requests)
    app.state.cache_layer = cache_layer
    app.state.settings = Settings()

    yield

app = FastAPI(lifespan=lifespan)

# Dependency to inject shared cache
def get_cache_layer(request: Request):
    return request.app.state.cache_layer

def get_settings(request: Request):
    return request.app.state.settings

# Endpoint 1: Knowledge Extraction
@app.post("/api/extraction")
async def extract_knowledge(
    data: dict,
    cache_layer: CachingLayer = Depends(get_cache_layer),
    settings: Settings = Depends(get_settings),
):
    """Extract concepts from telemetry data and store in shared cache."""
    concept_service = ConceptRelationshipExtractionService(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
    )
    vector_store = ConceptVectorStore(cache_layer=cache_layer)  # Uses shared cache
    processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

    # Extract → Process → Store
    result = concept_service.extract_concepts_and_relationships(
        data.get("payload_data"), request_id=data.get("request_id")
    )
    result = processor.process(result)
    vector_store.store_concepts(result.get("concepts", []))

    return {
        "concepts": len(result["concepts"]),
        "relations": len(result["relations"]),
        "cache_size": cache_layer.describe()["ntotal"]
    }

# Endpoint 2: Evidence Gathering
@app.post("/api/evidence")
async def gather_evidence(
    query: dict,
    cache_layer: CachingLayer = Depends(get_cache_layer),
):
    """Gather evidence from the shared cache."""
    from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
    from evidence.app.data.mock_repo import MockDataRepository

    request = ReasonerCognitionRequest(
        header=Header(
            workspace_id=query.get("workspace_id", "default"),
            mas_id=query.get("mas_id", "default"),
            agent_id=query.get("agent_id", "default")
        ),
        request_id=query.get("request_id", "req-001"),
        payload=RequestPayload(intent=query["intent"])
    )

    repo = MockDataRepository()
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)

    return {
        "records": len(response.records),
        "cache_size": cache_layer.describe()["ntotal"]
    }
```

**Key points:**
- ✅ **One `CachingLayer`** created at startup (in `lifespan`)
- ✅ **Stored in `app.state`** (persists across all requests)
- ✅ **Injected via `Depends()`** into both endpoints
- ✅ **Both endpoints share the same cache** - concepts extracted by `/api/extraction` are immediately available to `/api/evidence`

**❌ WRONG: Creating cache in each endpoint**
```python
@app.post("/api/extraction")
async def extract_knowledge(data: dict):
    # ❌ Creates NEW cache every request - data is lost!
    cache_layer = CachingLayer(vector_dimension=384, ...)
    vector_store = ConceptVectorStore(cache_layer=cache_layer)
    # This cache is discarded after the request
```

---

### Example 3: Standalone Script (Knowledge + Evidence)

**Use case:** Standalone Python script for extracting knowledge and gathering evidence.

```python
import asyncio
import os
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings

settings = Settings()

# 1. Initialize embedding manager (SHARED)
embedding_manager = EmbeddingManager()

def embed_fn(text: str):
    out = embedding_manager.generate_embedding(text)
    if out is None:
        raise ValueError("Embedding returned None")
    return out

# 2. Create ONE shared cache layer
cache_layer = CachingLayer(
    vector_dimension=384,
    metric="l2",
    embed_fn=embed_fn,
)

# 3. Extract and store concepts
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)
vector_store = ConceptVectorStore(cache_layer=cache_layer)  # Uses shared cache
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

payload_data = [
    {
        "Timestamp": "2026-03-18 10:00:00.000000000",
        "TraceId": "trace_001",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "authentication.service",
        "SpanKind": "Server",
        "ServiceName": "auth-service",
        "SpanAttributes": {
            "agent_id": "auth_agent",
            "http.method": "POST",
            "http.route": "/login",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 2000000
    }
]

result = concept_service.extract_concepts_and_relationships(
    payload_data, request_id="req-001", format_descriptor="observe-sdk-otel"
)
result = processor.process(result)
vector_store.store_concepts(result.get("concepts", []))

print(f"✓ Extracted {len(result['concepts'])} concepts")
print(f"✓ Cache contains: {cache_layer.describe()['ntotal']} items")

# 4. Gather evidence using the SAME cache
async def gather_evidence():
    repo = MockDataRepository()
    request = ReasonerCognitionRequest(
        header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
        request_id="evidence-001",
        payload=RequestPayload(intent="How does authentication work?")
    )
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"✓ Found {len(response.records)} evidence records")
    return response

asyncio.run(gather_evidence())
```

---

### Example 4: Knowledge Extraction Only

**Use case:** Extract and store concepts without evidence gathering.

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings

settings = Settings()

# Initialize components
embedding_manager = EmbeddingManager()
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.generate_embedding(text),
)

concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)

vector_store = ConceptVectorStore(cache_layer=cache_layer)
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

# Extract and store
payload_data = [{"Timestamp": "...", "TraceId": "...", ...}]
result = concept_service.extract_concepts_and_relationships(payload_data)
result = processor.process(result)
vector_store.store_concepts(result.get("concepts", []))

# Search
similar = vector_store.search_similar(text="authentication", k=3)
for item in similar:
    print(f"{item.get('name', item.get('text'))} (score: {item['score']:.3f})")
```

---

### Example 5: Evidence Gathering Only

**Use case:** Retrieve evidence from an existing cache without extraction.

```python
import asyncio
from evidence.app.agent.evidence import process_evidence
from evidence.app.agent.embeddings import EmbeddingManager
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer

# Initialize cache with existing data
embedding_manager = EmbeddingManager()
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.get_embedding(text)
)

# Populate cache (normally this would be done by ingestion)
cache_layer.store_knowledge(
    text="auth-service: Authentication service",
    metadata={"name": "auth-service", "type": "Service"}
)

# Gather evidence
async def gather_evidence():
    repo = MockDataRepository()
    request = ReasonerCognitionRequest(
        header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
        request_id="evidence-001",
        payload=RequestPayload(intent="How does authentication work?")
    )
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"Found {len(response.records)} records")
    return response

asyncio.run(gather_evidence())
```

---

## ⚡ Best Practices

### 1. Cache Sharing is Critical

When using knowledge extraction + evidence gathering, always share the same cache:

```python
# ✅ DO THIS: Create once, share everywhere
cache_layer = CachingLayer(vector_dimension=384, embed_fn=embed_fn)
vector_store = ConceptVectorStore(cache_layer=cache_layer)
await process_evidence(request, cache_layer=cache_layer)

# ❌ DON'T: Create multiple caches
cache1 = CachingLayer(...)  # Data won't be shared!
cache2 = CachingLayer(...)
```

### 2. Use Production Embeddings

Always use `fastembed` for production-quality semantic search:

```python
from ingestion.app.agent.knowledge_processor import EmbeddingManager

embedding_manager = EmbeddingManager()  # BAAI/bge-small-en-v1.5
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.generate_embedding(text)
)
```

**Why fastembed?**
- ✅ ONNX-based (no PyTorch/CUDA required)
- ✅ Production-quality semantic search
- ✅ Consistent across all services
- ❌ Hash-based embeddings are testing-only

### 3. Enable Embeddings in Processor

```python
# Enable embeddings for vector similarity search
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)
```

Without embeddings enabled, concepts cannot be searched semantically.

### 4. LLM Provider Configuration (Semantic Negotiation)

Choose your LLM provider via `LLM_PROVIDER` environment variable:

```bash
# Azure OpenAI (default)
LLM_PROVIDER=azure-openai
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# OpenAI (direct or via LiteLLM proxy)
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=http://localhost:4000  # Optional: LiteLLM proxy

# AWS Bedrock
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

---

## 🧪 Testing

### Mock Mode (No Credentials)

For CI/CD or local testing without Azure OpenAI:

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService

# Mock mode generates deterministic test data
concept_service = ConceptRelationshipExtractionService(mock_mode=True)
result = concept_service.extract_concepts_and_relationships(payload_data)
```

### Running Tests

```bash
# Mock tests (CI-safe, no credentials required)
pytest tests/integration/test_usage_examples_mock.py -v

# Live tests (requires Azure OpenAI credentials)
pytest tests/integration/test_usage_examples_live.py -v -m ''

# Semantic negotiation integration tests
cd semantic_negotiation && pytest ../test_semantic_negotiation_integration.py -v
```

See [tests/integration/README.md](../tests/integration/README.md) for details.

---

## ⚙️ Configuration Reference

### Environment Variables

```bash
# ─── Azure OpenAI (Knowledge Extraction + Evidence Gathering) ───────────────
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# ─── LLM Provider (Semantic Negotiation) ────────────────────────────────────
LLM_PROVIDER=azure-openai  # Options: azure-openai, openai, bedrock

# Azure OpenAI (if LLM_PROVIDER=azure-openai)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# OpenAI (if LLM_PROVIDER=openai)
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4  # Optional, defaults to gpt-4
OPENAI_BASE_URL=http://localhost:4000  # Optional: for LiteLLM proxy

# AWS Bedrock (if LLM_PROVIDER=bedrock)
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# ─── Optional Settings ──────────────────────────────────────────────────────
CFN_URL=http://localhost:9002               # CFN
EG_MAX_DEPTH=4                              # Evidence graph traversal depth
EG_PATH_LIMIT=20                            # Max evidence paths to explore
EMBEDDING_MODEL_PATH=                       # Local bge-small-en-v1.5 path
ENABLE_EMBEDDINGS=true                      # Enable concept embeddings
ENABLE_DEDUP=true                           # Enable semantic deduplication
SIMILARITY_THRESHOLD=0.95                   # Deduplication threshold
```

---

## 📚 API Reference

### Knowledge Extraction

#### `ConceptRelationshipExtractionService`

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService

service = ConceptRelationshipExtractionService(
    azure_endpoint: str,
    azure_api_key: str,
    azure_deployment: str = "gpt-4o",
    azure_api_version: str = "2024-08-01-preview",
    mock_mode: bool = False,  # Use True for testing without credentials
)

# Extract concepts and relationships from OpenTelemetry data
result = service.extract_concepts_and_relationships(
    payload_data: List[Dict],
    request_id: Optional[str] = None,
    format_descriptor: str = "observe-sdk-otel"
)
# Returns: {"concepts": [...], "relations": [...]}
```

#### `KnowledgeProcessor`

```python
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor

processor = KnowledgeProcessor(
    enable_embeddings: bool = True,   # Enable vector embeddings
    enable_dedup: bool = True,        # Enable semantic deduplication
    similarity_threshold: float = 0.95
)

result = processor.process(extraction_result)
```

### Caching Layer

#### `CachingLayer`

```python
from caching.app.agent.caching_layer import CachingLayer

cache = CachingLayer(
    vector_dimension: int = 384,      # bge-small-en-v1.5 dimension
    metric: str = "l2",               # Distance metric
    embed_fn: Callable[[str], np.ndarray] = None
)

# Store knowledge
cache.store_knowledge(text="...", metadata={"name": "..."})

# Search similar items
results = cache.search_similar(text="query", k=5)

# Get stats
stats = cache.describe()  # {"dimension": 384, "metric": "l2", "ntotal": 100}
```

### Evidence Gathering

#### `process_evidence()`

```python
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload

request = ReasonerCognitionRequest(
    header=Header(workspace_id="...", mas_id="...", agent_id="..."),
    request_id="...",
    payload=RequestPayload(intent="How does authentication work?")
)

response = await process_evidence(
    request,
    repo_adapter=repo,
    cache_layer=cache_layer
)
# Returns: ReasonerCognitionResponse with records
```

### Semantic Negotiation

#### `IntentDiscovery`

```python
from semantic_negotiation.app.agent.intent_discovery import IntentDiscovery

discovery = IntentDiscovery()
issues = discovery.discover(sentence="...", context=None)
# Returns: List[str] - negotiable issues
```

#### `OptionsGeneration`

```python
from semantic_negotiation.app.agent.options_generation import OptionsGeneration

gen = OptionsGeneration()
options = gen.generate_options(
    negotiable_entities=["budget", "timeline"],
    sentence="...",
    context=None
)
# Returns: Dict[str, List[str]] - options per issue
```

#### `SemanticNegotiationPipeline`

Top-level orchestrator.  Use `execute()` for both the initiate and decide phases.

```python
from semantic_negotiation.app.agent.semantic_negotiation import SemanticNegotiationPipeline

pipeline = SemanticNegotiationPipeline(n_steps=100)

# ── Unified entry point ───────────────────────────────────────────────────
result = pipeline.execute(
    session_id: str,
    *,
    # Initiate-only parameters:
    n_steps: int | None = None,           # Override pipeline default for this session
    content_text: str = "",              # Negotiation scenario text
    agents_raw: list | None = None,       # [{"id", "name"}, ...]
    initiate_message: dict | None = None, # SSTP envelope to prepend to the trace
    # Decide-only parameter:
    agent_replies: list | None = None,    # [{"agent_id", "bid"}, ...]
)
# Returns dict with keys: status, session_id, round, …
# Initiate:  + issues, options_per_issue, n_steps, messages
# Ongoing:   + messages
# Terminal:  + result, issues, participant_id_by_name

# ── Sub-method API ────────────────────────────────────────────────────────
# Components 1+2
issues, options_per_issue = pipeline.discover_and_generate(content_text)

# Component 3 — seed
runner, sess, first_messages = pipeline.start_negotiation(
    issues, options_per_issue, participants, session_id, n_steps=None
)

# Component 3 — advance
status, next_messages, result = pipeline.step_negotiation(runner, sess, agent_replies)

# Clean up a session (e.g. on error)
pipeline.release_session(session_id)
```

#### `NegotiationParticipant`

```python
from semantic_negotiation.app.agent.negotiation_model import NegotiationParticipant

# Minimal — id and name only (preferences default to empty dict)
participant = NegotiationParticipant(
    id="agent-1",
    name="Agent 1",
)

# With explicit preferences (used by NegotiationModel standalone path)
participant = NegotiationParticipant(
    id="agent-1",
    name="Agent 1",
    preferences={
        "budget":   {"$1000": 1.0, "$2000": 0.5, "$3000": 0.0},
        "timeline": {"1 week": 0.8, "2 weeks": 0.5},
    },
)
```

#### `NegotiationModel` (low-level / standalone)

Used internally by `BatchCallbackRunner`.  You can also call it directly for
in-process negotiations where participants supply explicit preferences.

```python
from semantic_negotiation.app.agent.negotiation_model import NegotiationModel

model = NegotiationModel(n_steps=20)
result = model.run(
    issues=["budget", "timeline"],
    options_per_issue={"budget": ["$1000", "$2000"], ...},
    participants=[participant1, participant2],
)

# Result attributes:
# - result.agreement: List[NegotiationOutcome] or None
# - result.steps:     int (number of rounds completed)
# - result.timedout:  bool
# - result.broken:    bool
```

---

## 📖 Additional Resources

- **GitHub**: [cisco-eti/ioc-cfn-cognitive-agents](https://github.com/cisco-eti/ioc-cfn-cognitive-agents)
- **Issues**: [Report bugs or request features](https://github.com/cisco-eti/ioc-cfn-cognitive-agents/issues)
- **Integration Tests**: See [tests/integration/README.md](../tests/integration/README.md)
