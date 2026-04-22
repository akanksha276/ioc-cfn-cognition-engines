# Ingestion Cognitive Agent

A cognitive agent that ingests multiple telemetry/message formats, extracts a knowledge graph (concepts + relations), and can also generate RAG chunks for retrieval workflows.

## Project Structure

```
ingestion/
├── app/
│   ├── main.py                 # FastAPI app entry point & /health endpoint
│   ├── dependencies.py         # Dependency injection configuration
│   │
│   ├── api/                    # HTTP layer only
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── agent/                  # Core agent logic
│   │   ├── base.py             # AdapterSDK base class
│   │   ├── adapters.py         # Format filtering + compact payload builders
│   │   ├── ingest_data.py      # Unified orchestration (Graph + optional RAG)
│   │   ├── knowledge_processor.py  # Embedding generation & dedup
│   │   ├── rag.py              # RAG chunking + embedding pipeline
│   │   ├── prompts.py          # Format-specific LLM prompts
│   │   ├── service.py          # Graph extraction services
│   │   └── concept_vector_store.py # Concept vector store adapter
│   │
│   ├── data/                   # Data access abstraction
│   │   ├── base.py             # DataRepository Protocol
│   │   └── mock_repo.py        # File-based mock implementation
│   │
│   └── config/
│       └── settings.py         # Environment settings (Pydantic)
│
└── tests/
    ├── conftest.py             # Shared fixtures
    ├── unit/
    │   ├── test_agent.py       # Unit tests for graph + orchestration services
    │   └── test_rag.py         # Unit tests for RAG pipeline
    └── integration/
        └── test_api.py         # API tests for /api/knowledge-mgmt/extraction
```

## Setup

### Environment Variables

Create a `.env` file in the project root with:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Knowledge Processing Options
ENABLE_EMBEDDINGS=true
ENABLE_DEDUP=true
ENABLE_RAG_INGEST=true
SIMILARITY_THRESHOLD=0.95
EMBEDDING_MODEL_PATH=

# Similarity Search API (concept deduplication)
CFN_URL=http://localhost:9002
SIMILARITY_TOP_K=5
SIMILARITY_METRIC=l2

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8086
LOG_LEVEL=INFO
```

### Running with Poetry

```bash
# Install dependencies
poetry install

# Run the server
poetry run python -m app.main

# Or use uvicorn directly
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload
```

Server starts on `http://localhost:8086`

### Running with Docker

```bash
# Build the image (from project root)
docker build --build-arg GITHUB_TOKEN=$GITHUB_TOKEN -t otel-ingestion-agent .

# Run the container
docker run -p 8086:8086 --env-file .env otel-ingestion-agent

# Or run with inline env vars
docker run -p 8086:8086 \
  -e AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/ \
  -e AZURE_OPENAI_API_KEY=your-api-key \
  -e AZURE_OPENAI_DEPLOYMENT=gpt-4o \
  otel-ingestion-agent
```

## API Endpoints

### Knowledge Extraction

`POST /api/knowledge-mgmt/extraction`

The extraction endpoint. Accepts a structured request envelope with a header, `request_id`, and a payload containing metadata and records.

Processing is:
1. Build compact payload from input format
2. Extract **Graph** (concepts + relationships)
3. Optionally build **RAG chunks**
4. Post-process embeddings/dedup and look up similar concepts via `/concepts/similarity-search`

### Supported Data Formats

- `observe-sdk-otel`: OpenTelemetry span records
- `openclaw`: conversation turns/messages
- `locomo`: conversational session/message records
- `semneg`: semantic negotiation records

```bash
curl -X POST http://localhost:8086/api/knowledge-mgmt/extraction \
  -H "Content-Type: application/json" \
  -d @request.json
```

### File-based Extraction (Dev/Testing)

```bash
curl "http://localhost:8086/api/extract/entities_and_relations/from_file?file_path=/path/to/otel.json&save_output=true"
curl "http://localhost:8086/api/extract/concepts_and_relationships/from_file?file_path=/path/to/otel.json"
```

### Health & Metrics

```bash
curl http://localhost:8086/health
curl http://localhost:8086/api/metrics
```

## Request Format

The extraction endpoint accepts the following envelope:

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "metadata": {
      "format": "observe-sdk-otel"
    },
    "data": [
      {
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "2b6a701a27797f5c",
        "ParentSpanId": "",
        "SpanName": "farm_agent.build_graph.agent",
        "SpanKind": "Client",
        "ServiceName": "corto.farm_agent",
        "SpanAttributes": {
          "agent_id": "farm_agent.build_graph",
          "gen_ai.request.model": "gpt-4"
        },
        "Duration": 21346166
      }
    ]
  }
}
```

| Field | Required | Description |
|---|---|---|
| `header.workspace_id` | Yes | Workspace identifier |
| `header.mas_id` | Yes | MAS identifier |
| `header.agent_id` | No | Agent identifier |
| `request_id` | Yes | Client-supplied ID; echoed back as `response_id` |
| `payload.metadata.format` | Yes | Data format: `observe-sdk-otel`, `openclaw`, `locomo`, or `semneg` |
| `payload.metadata.*` | No | Additional metadata fields become Knowledge Graph labels |
| `payload.data` | Yes | Non-empty array of records matching the declared format |

## Response Format

### Success

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "concepts": [
    {
      "id": "15118c8b99e5813a2239279f0d7fb7c6",
      "name": "website_selector_agent",
      "description": "Agent: website_selector_agent",
      "type": "concept",
      "attributes": {
        "concept_type": "agent",
        "embedding": [[0.042, 0.018, -0.079]]
      }
    }
  ],
  "relations": [
    {
      "id": "91b581b91afb041bdcca33d74ab687c2",
      "node_ids": [
        "15118c8b99e5813a2239279f0d7fb7c6",
        "4e706aec50174e58f15a52a53e6ca4f5"
      ],
      "relationship": "SENDS_PROMPT_TO",
      "attributes": {
        "source_name": "website_selector_agent",
        "target_name": "gpt-4o",
        "summarized_context": "Agent sends inference request to the LLM."
      }
    }
  ],
  "descriptor": "observe-sdk-otel",
  "metadata": {
    "records_processed": 659,
    "concepts_extracted": 10,
    "relations_extracted": 16,
    "dedup_enabled": false,
    "concepts_deduped": 0,
    "relations_deduped": 0
  },
  "rag_chunks": [
    {
      "text": "flattened chunk text",
      "embedding": [[0.042, 0.018, -0.079]],
      "metadata": { "domain": "observe-sdk-otel", "doc_index": 0, "chunk_index": 0 }
    }
  ]
}
```

### Error

When processing fails, the response contains an `error` block instead of concepts/relations:

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "message": "An end-user meaningful error description.",
    "detail": {
      "traceback": "..."
    }
  }
}
```

## Architecture

### Components

- **IngestDataService** (`agent/ingest_data.py`): Main orchestrator used by `/api/knowledge-mgmt/extraction`.
- **ExtractionAdapter** (`agent/adapters.py`): Normalizes each supported format into compact records.
- **ConceptRelationshipExtractionService** (`agent/service.py`): LLM-based graph extraction (concepts + relationships).
- **RagPipeline** (`agent/rag.py`): Optional chunking + embedding stage to produce `rag_chunks`.
- **KnowledgeProcessor** (`agent/knowledge_processor.py`): Embedding enrichment and dedup for graph output.
- **ConceptVectorStore** (`agent/concept_vector_store.py`): Concept vector store adapter.

### Pipeline

```
POST request
  → ExtractionAdapter (format-specific compact payload)
  → IngestDataService
      → Graph extraction (concepts + relationships)
      → Optional RAG chunk generation
  → KnowledgeProcessor (embeddings + dedup)
  → Similarity lookup via /concepts/similarity-search
  → Return response
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and sample data
├── unit/
│   ├── test_agent.py        # Unit tests for graph + orchestration services
│   └── test_rag.py          # Unit tests for RAG pipeline
└── integration/
    └── test_api.py          # API tests for /api/knowledge-mgmt/extraction
```

### Running Tests

```bash
# Run ingestion tests
poetry run pytest ingestion/tests -q

# Unit tests
poetry run pytest ingestion/tests/unit -q

# Integration tests
poetry run pytest ingestion/tests/integration -q
```

### Test Coverage

Tests cover:
- **Ingestion orchestration**: format normalization, graph extraction, optional RAG stage
- **Graph post-processing**: embedding enrichment and deduplication behavior
- **RAG pipeline**: config validation, chunking, metadata, embedding shape
- **API behavior**: request validation, response envelope, similarity lookup fallback

## Development

### Common Commands

```bash
# Install dependencies (from repo root)
poetry install

# Run linting
poetry run ruff check ingestion/

# Format code
poetry run ruff format ingestion/

# Run tests
poetry run pytest ingestion/tests -q
```

### Running Locally

⚠️ **For normal use, run the unified gateway** (see [main README](../README.md))

For standalone development:
```bash
cd ingestion
poetry run uvicorn app.main:app --reload --port 8086
```

### Adding New Extractors

1. Extend `ConceptRelationshipExtractionService` or `TelemetryExtractionService` in `agent/service.py`
2. Add new Pydantic models in `api/schemas.py` if needed
3. Add new endpoints or extend the unified handler in `api/routes.py`

### Customizing Data Sources

1. Implement the `DataRepository` protocol from `data/base.py`
2. Update `dependencies.py` to use your implementation
