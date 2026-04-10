# Evidence Gathering Agent

A FastAPI service that exposes an API for evidence gathering over a knowledge graph. It uses LLM-based entity extraction, single- and multi-entity evidence engines, and optional graph + cache backends. The design is layered and DB-agnostic: you can run with an in-process mock repo, then switch to the mocked DB (Neo4j) and/or an internal caching layer without changing the API.

## Features

- **Structured API**: Request/response with `header`, `request_id`, and `payload` (intent, metadata, additional_context).
- **Layered design**: API (HTTP) → agent logic (evidence, single/multi-entity) → data repository (mock or HTTP).
- **Optional backends**:
  - **Mocked DB** (Neo4j): graph paths, neighbors by id, concepts-by-id.
  - **Caching layer (in-process)**: when the host app sets `app.state.cache_layer` (e.g. unified gateway), used for similar-concept search; not part of the public request body.
- Poetry-managed project, unit and integration tests, Dockerfile.

## Project layout

```
evidence/
├── app/
│   ├── main.py              # FastAPI app, /health, router at /api/knowledge-mgmt
│   ├── api/
│   │   ├── routes.py        # /reasoning/evidence, /graph/paths, /graph/neighbors, /graph/concepts
│   │   └── schemas.py       # ReasonerCognitionRequest/Response, Header, RequestPayload, etc.
│   ├── agent/
│   │   ├── evidence.py      # process_evidence orchestration (entity extraction, decomposition, single/multi-entity)
│   │   ├── single_entity.py # SingleEntityEvidenceEngine, ConceptRepository (cache + graph by name)
│   │   ├── multi_entities.py
│   │   ├── llm_clients.py   # EntityExtractor, QueryDecomposer, EvidenceJudge, EvidenceRanker
│   │   ├── embeddings.py    # EmbeddingManager
│   │   └── utiles.py        # PathFormatter, MMR, etc.
│   ├── data/
│   │   ├── base.py          # DataRepository protocol
│   │   ├── mock_repo.py     # MockDataRepository (default)
│   │   └── http_repo.py     # HttpDataRepository (mocked-db)
│   ├── config/settings.py   # Env-based settings
│   └── dependencies.py      # get_repository_for_reasoning, get_repository, get_cache_layer
├── tests/
│   ├── unit/
│   └── integration/test_api.py
├── pyproject.toml
├── poetry.lock
├── .env.example
└── Dockerfile
```

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

## Quick start (Poetry)

```bash
cd evidence

cp .env.example .env   # optional
poetry install
poetry run uvicorn app.main:app --reload --port 8087
```

Health check:

```bash
curl -s http://localhost:8087/health | jq .
```

## Evidence API

**Endpoint:** `POST /api/knowledge-mgmt/reasoning/evidence`

**Request body:** Structured request with `header`, `request_id`, and `payload`.

| Field | Required | Description |
|-------|----------|-------------|
| `header.workspace_id` | Yes | Workspace identifier |
| `header.mas_id`       | Yes | Multi-agent system identifier |
| `header.agent_id`     | No  | Optional agent identifier |
| `request_id`          | Yes | Request id (echoed as `response_id` in response) |
| `payload.intent`      | Yes | User question / intent (e.g. "What does the concierge_agent do?") |
| `payload.metadata`    | No  | Optional query metadata |
| `payload.additional_context` | No | Extra context list |
| `payload.records`      | No  | Optional pre-existing records |

**Example request:**

```bash
curl -s -X POST http://localhost:8087/api/knowledge-mgmt/reasoning/evidence \
  -H "Content-Type: application/json" \
  -d '{
    "header": {
      "workspace_id": "test-workspace",
      "mas_id": "test-mas"
    },
    "request_id": "demo-1",
    "payload": {
      "intent": "What does the concierge_agent do?"
    }
  }' | jq .
```

**Response:** `header` (echo), `response_id` (from `request_id`), `records` (evidence records), `metadata`, and optionally `error`.

With no data layer or LLM configured, the service still runs and may return empty or fallback evidence; configure mocked DB and/or Azure/LiteLLM for full behavior.

## Graph / DB-facing endpoints

These are under the same prefix and use the same repository (mock or HTTP). Useful for testing or direct graph access.

- **POST** `/api/knowledge-mgmt/graph/paths`  
  Body: `{"source_id":"...", "target_id":"...", "max_depth": 3, "limit": 10, "relations": null}`

- **GET** `/api/knowledge-mgmt/graph/neighbors/{concept_id}`  
  One-hop neighbors by concept id.

- **POST** `/api/knowledge-mgmt/graph/concepts/by_ids`  
  Body: `{"ids": ["id1", "id2"]}`

Example:

```bash
curl -s "http://localhost:8087/api/knowledge-mgmt/graph/neighbors/<concept_id>" | jq .
```

## Configuration

Set in `.env` or export before running. The app loads `.env` at startup.

| Variable | Description |
|----------|-------------|
| `MOCKED_DB_BASE_URL` or `CFN_URL` | Base URL of mocked DB (e.g. `http://localhost:8088`). If unset, in-process mock repo is used. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` | For LLM clients (entity extraction, decomposition, judge, ranker). |
| `EG_MAX_DEPTH`, `EG_PATH_LIMIT` | Tuning for path search (if used). |

## Using the Mocked DB (Neo4j)

The mocked DB is a **separate project** (not in this repo). Use it as a backend for evidence-gathering when you need a real graph instead of the in-process mock.

1. **Get the mocked DB:** Clone the **mocked-db** repo elsewhere, or use the sibling folder `../mocked-db` if it exists in your workspace (same repo root as `ioc-cfn-cognitive-agents`).
2. **Start Neo4j and mocked-db** (see the mocked-db README): run the server on port 8088.
3. **Load KCR data** (and optionally generate embeddings) in mocked-db.
4. **Point the evidence agent** at it and run:

```bash
export MOCKED_DB_BASE_URL=http://localhost:8088
poetry run uvicorn app.main:app --reload --port 8087
```

The agent uses `HttpDataRepository` and calls the mocked DB over HTTP for paths, `neighbors/{concept_id}`, and concepts-by-id. Similar concepts come from the configured cache (FAISS), not from a semantic-similar HTTP call on the repo.

## Similar-concept search (cache_layer)

Similar concepts are resolved only through an **in-process** `cache_layer` object (e.g. FAISS) injected as `app.state.cache_layer`. The **unified gateway** creates one shared `CachingLayer` at startup and attaches it to the evidence sub-app; standalone evidence has no `cache_layer` unless you set it in app lifespan or pass a instance when calling `process_evidence` from tests.

Behavior:

- **No `cache_layer`**: Similar-concept retrieval returns empty anchors until a layer is attached.
- **With `cache_layer` + data layer**: Similar concepts come from `cache_layer.search_similar`; each hit must include **`concept_id`**. The agent calls **`neighbors/{concept_id}`** on the data layer. Optional `text` (`name | description`) is for display only.

The HTTP API contract stays the same; callers do not send cache parameters.

## Switching data sources

`app/data/base.py` defines the repository contract. The default is `MockDataRepository`. To use the mocked DB, set `CFN_URL`. **`get_repository_for_reasoning`** (used by `POST /reasoning/evidence`) returns `HttpDataRepository` scoped with `header.workspace_id` and `header.mas_id`, so outbound graph calls use `/api/workspaces/.../multi-agentic-systems/.../graph/...`. Standalone **`/graph/*`** routes use **`get_repository`**, which returns `HttpDataRepository` with legacy `/api/v1/graph/...`. When **`cache_layer`** is on the app and **`CFN_URL`** is set, **similar concepts** come from in-process FAISS via `ConceptRepository`, and **graph calls** use **`HttpDataRepository`**.

## Tests

```bash
poetry run pytest -q
```

## Docker
```bash
cd evidence
docker build -t evidence-agent .
docker run --rm -p 8087:8087 --env-file .env evidence-agent
```

