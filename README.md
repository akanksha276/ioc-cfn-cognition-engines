# IoC CFN Cognitive Agents

A collection of cognitive agents for processing OpenTelemetry data and evidence gathering for reasoning systems.

## Agents

- **[Ingestion Service](ingestion/)** – Extracts knowledge from OpenTelemetry traces (entities, relations, embeddings).
- **[Evidence Gathering Service](evidence/)** – Retrieves relevant evidence from the knowledge graph (e.g. "What does Miss-Marple do?").
- **[Semantic Alignment Agent](semantic_alignment/)** – Handles multi-party semantic negotiation using NegMAS and SSTP (Semantic State Transfer Protocol).

The evidence and ingestion services retrieve and store knowledge via the CFN service (`CFN_URL`), which routes requests to `ioc-knowledge-memory-svc`.

## Table of Contents

- [IoC CFN Cognitive Agents](#ioc-cfn-cognitive-agents)
  - [Agents](#agents)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Run the gateway with Docker (recommended)](#run-the-gateway-with-docker-recommended)
    - [Run the gateway locally (no Docker)](#run-the-gateway-locally-no-docker)
    - [Run agents individually (development only)](#run-agents-individually-development-only)
    - [Testing Semantic Alignment](#testing-semantic-alignment)
  - [Development](#development)
    - [Prerequisites](#prerequisites)
    - [Environment setup (required for local and Docker)](#environment-setup-required-for-local-and-docker)
    - [Run with CFN Stack](#run-with-cfn-stack)
    - [Testing](#testing)
    - [Code Quality](#code-quality)
  - [Architecture](#architecture)
  - [Troubleshooting](#troubleshooting)
    - [Embedding Model Download Issues](#embedding-model-download-issues)
    - [Docker Build Fails with SSL Errors](#docker-build-fails-with-ssl-errors)
    - [LLM Connection Errors](#llm-connection-errors)
    - [Tests Fail with "Directory not found"](#tests-fail-with-directory-not-found)
  - [Project Structure](#project-structure)
  - [CI/CD Workflow](#cicd-workflow)
    - [Automated Docker Builds](#automated-docker-builds)
      - [Pull Request (Build Validation)](#pull-request-build-validation)
      - [Merge to Main (Latest Release)](#merge-to-main-latest-release)
      - [Tag Push (Versioned Release)](#tag-push-versioned-release)
    - [Using Published Images](#using-published-images)
    - [Multi-Platform Support](#multi-platform-support)
    - [Release Checklist](#release-checklist)
  - [Contributing](#contributing)
  - [License](#license)

---

## Quick Start

### Run the gateway with Docker (recommended)

The gateway serves ingestion, evidence and semantic alignment on **port 9004**. It uses a **`.env` file** at repo root (see [Environment setup](#environment-setup-required-for-local-and-docker)); create it from `.env.example` if needed.

```bash
# From repo root (ensure .env exists)
docker compose up --build
```

Then use the API at `http://localhost:9004`:

| Backend        | Path                                     | Example                                                            |
| -------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Gateway health | `/api/internal/diagnostics/health`       | `GET http://localhost:9004/api/internal/diagnostics/health`        |
| Ingestion      | `/api/knowledge-mgmt/extraction`         | `POST http://localhost:9004/api/knowledge-mgmt/extraction`         |
| Evidence       | `/api/knowledge-mgmt/reasoning/evidence` | `POST http://localhost:9004/api/knowledge-mgmt/reasoning/evidence` |

Prefixed paths also work: `/ingestion/...`, `/evidence/...`.

### Run the gateway locally (no Docker)

Create `.env` from the template and set your credentials (see [Environment setup](#environment-setup-required-for-local-and-docker)):

```bash
cp .env.example .env
```

Then from repo root:

```bash
PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004
```

### Run agents individually (development only)

**For normal use, run the gateway above** (port 9004) — it's the unified entry point for ingestion and evidence.

For development/testing, you can run agents as standalone services:

<details>
<summary><b>Click to expand: Individual agent commands</b></summary>

**Ingestion Agent** (standalone, port 8080):

```bash
cd ingestion
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Evidence Agent** (standalone, port 8087):

```bash
cd evidence
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8087
```

**Semantic Alignment Agent** (independent service, port 8089):

```bash
cd semantic_alignment
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

</details>

**Note:** The gateway (port 9004) is the recommended setup. It runs ingestion + evidence in a single process. The semantic alignment agent is a separate service that runs independently.

### Testing Semantic Alignment

Two test harnesses are available under `semantic_alignment/evaluation/framework/`:

- **`test_via_semantic_alignment_agents_configured.py`** — Spawns a multi-agent system (MAS) in-process and tests semantic alignment directly. Use `--filter` to run a specific mission.

  ```bash
  poetry run python semantic_alignment/evaluation/framework/test_via_semantic_alignment_agents_configured.py
  poetry run python semantic_alignment/evaluation/framework/test_via_semantic_alignment_agents_configured.py --filter "Quick deal"
  ```

- **`test_via_cfn_service.py`** — Calls the full CFN service end-to-end for integration testing.
  ```bash
  poetry run python semantic_alignment/evaluation/framework/test_via_cfn_service.py
  ```

---

## Development

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for containerized development)

### Environment setup (required for local and Docker)

A **`.env` file is required** at **repo root** (`ioc-cfn-cognitive-agents/.env`). Create it from the template:

```bash
cp .env.example .env
```

This single file is used by local development, Docker Compose, and CI/CD workflows.

**Required and optional variables:**

| Variable                    | Required | Description                                                                      |
| --------------------------- | -------- | -------------------------------------------------------------------------------- |
| `LLM_BASE_URL`              | Yes      | LLM endpoint URL (e.g. LiteLLM proxy).                                           |
| `LLM_API_KEY`               | Yes      | LLM API key.                                                                     |
| `LLM_MODEL`                 | Yes      | Model name (e.g. `openai/azure/gpt-4o`).                                         |
| `CFN_URL`                   | Yes      | URL of the CFN service (e.g. `http://localhost:9002`). CEs auto-register if set. |
| `CE_HEARTBEAT_INTERVAL_SEC` | No       | Heartbeat interval in seconds (default: `30`).                                   |
| `COGNITION_ENGINE_HOST`     | No       | Advertised host for this CE (used during registration, default: `localhost`).    |
| `COGNITION_ENGINE_PORT`     | No       | Advertised port for this CE (default: `9004`).                                   |
| `EMBEDDING_MODEL_PATH`      | No       | Path to local `bge-small-en-v1.5` folder. Uses Hugging Face download if unset.   |
| `ENABLE_EMBEDDINGS`         | No       | Enable embedding generation (default: `true`).                                   |
| `ENABLE_DEDUP`              | No       | Enable semantic deduplication (default: `true`).                                 |
| `SIMILARITY_THRESHOLD`      | No       | Dedup threshold 0.0–1.0 (default: `0.95`).                                       |

Each app ignores unknown keys, so the same `.env` can contain variables for multiple services. See [.env.example](.env.example) for a full template.

### Run with CFN Stack

Run the cognitive agents locally while connecting to the full CFN stack.

**1. Start the CFN stack**

Follow the instructions in the [ioc-cfn-mgmt-backend-svc README](https://github.com/cisco-eti/ioc-cfn-mgmt-backend-svc/tree/main) to bring up the full stack using the `full-stack` Docker Compose profile.

Once the stack is up, stop the `ioc-cfn-cognition-engine` container — you'll run this service locally instead:

```bash
cd ioc-cfn-mgmt-backend-svc
docker compose stop ioc-cfn-cognition-engine
```

Also update `COGNITION_ENGINE_SVC_URL` in `docker-compose.yml` under `ioc-cfn-svc` so it can reach your locally-running engine:

```yaml
- COGNITION_ENGINE_SVC_URL=http://host.docker.internal:9004
```

The remaining services and their ports:

| Service                    | Port   | Purpose                        |
| -------------------------- | ------ | ------------------------------ |
| `ioc-cfn-mgmt-plane-svc`   | `9000` | Management plane backend       |
| `ioc-cfn-svc`              | `9002` | Cognition Fabric Node          |
| `ioc-knowledge-memory-svc` | `9003` | Knowledge graph + vector store |

**2. Configure `.env`**

Set these variables to point at the running CFN stack:

```bash
# LLM credentials
LLM_BASE_URL=https://your-litellm-endpoint
LLM_API_KEY=sk-your-api-key
LLM_MODEL=openai/azure/gpt-4o

# Point at the local CFN service
CFN_URL=http://localhost:9002

# CE Registration & Heartbeat (auto-registers with Management Plane)
CE_REGISTRATION_ENABLED=true
CE_VERSION=1.2.3
COGNITION_ENGINE_HOST=localhost
COGNITION_ENGINE_PORT=9004
```

**3. Run the Cognition Engine**

```bash
# From repo root — gateway serves ingestion + evidence + semantic alignment on port 9004
PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004 --reload
```

On startup, the gateway will automatically:

1. Register 2 Cognition Engines with the Management Plane (via CFN):
   - Knowledge Management CE
   - Semantic Alignment CE
2. Start heartbeat background tasks (every 30s) to maintain "online" status

**Expected startup logs:**

```
INFO - Starting CE registration with cfn_url=http://localhost:9002
INFO - CE 'Knowledge Management CE' created: ce_id=abc-123, status=offline
INFO - Heartbeat task started for 'Knowledge Management CE'
INFO - CE 'Semantic Alignment CE' created: ce_id=def-456, status=offline
INFO - Heartbeat task started for 'Semantic Alignment CE'
INFO - Application startup complete
```

Verify connectivity:

```bash
curl http://localhost:9004/api/internal/diagnostics/health
```

### Cognition Engine Registration

The gateway automatically registers itself with the Management Plane on startup using the new CE lifecycle API. This enables:

- **Automatic discovery**: Management Plane knows about all running CEs
- **Health monitoring**: Heartbeats every 30s keep CEs marked as "online"
- **Lifecycle management**: CEs can be enabled/disabled via Management Plane
- **Metrics tracking**: CE operations are tracked and associated with ce_id

**Configuration:**

| Variable                    | Default     | Description                      |
| --------------------------- | ----------- | -------------------------------- |
| `CE_REGISTRATION_ENABLED`   | `true`      | Enable/disable auto-registration |
| `CE_VERSION`                | `1.2.3`     | CE version for registration      |
| `CE_HEARTBEAT_INTERVAL_SEC` | `30`        | Heartbeat interval (seconds)     |
| `COGNITION_ENGINE_HOST`     | `localhost` | Advertised host                  |
| `COGNITION_ENGINE_PORT`     | `9004`      | Advertised port                  |

**Registration Flow:**

```
1. Gateway starts
2. Calls: POST http://cfn:9002/api/cognition-engines
3. CFN injects cfn_id and forwards to Management Plane
4. Management Plane generates ce_id and returns response
5. Gateway stores ce_id and starts heartbeat background task
6. Heartbeat: PUT http://cfn:9002/api/cognition-engines/{ce_id}/heartbeat (every 30s)
7. Management Plane transitions CE status: offline → online
```

**Graceful Degradation:**

If CFN/Management Plane is unavailable:

- Gateway logs warning and continues startup (doesn't crash)
- CEs operate normally but without Management Plane visibility
- No heartbeats sent

**Verification:**

Check registered CEs in Management Plane database:

```sql
SELECT ce_id, name, version, status, last_seen
FROM cognition_engine
WHERE cfn_id = 'your-cfn-id';
```

For detailed testing instructions, see [docs/MANUAL_TESTING_GUIDE.md](docs/MANUAL_TESTING_GUIDE.md).

---

**Original configuration instructions continued:**

Set these variables to point at the running CFN stack:

```bash
# LLM credentials
LLM_BASE_URL=https://your-litellm-endpoint
LLM_API_KEY=sk-your-api-key
LLM_MODEL=openai/azure/gpt-4o

# Point at the local CFN service (CEs auto-register if CFN_URL is set)
CFN_URL=http://localhost:9002

# CE registration settings (optional, defaults shown)
CE_HEARTBEAT_INTERVAL_SEC=30
COGNITION_ENGINE_HOST=localhost
COGNITION_ENGINE_PORT=9004
```

**3. Run the Cognition Engine**

```bash
# From repo root — gateway serves ingestion + evidence + semantic alignment on port 9004
PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004 --reload
```

Verify connectivity:

```bash
curl http://localhost:9004/api/internal/diagnostics/health
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific service tests
cd ingestion && poetry run pytest
cd evidence && poetry run pytest
```

### Code Quality

```bash
# Linting
poetry run ruff check .

# Auto-fix
poetry run ruff check --fix .

# Format
poetry run ruff format .
```

---

## Architecture

The **unified gateway** runs ingestion and evidence in one process (port 9004), connected to the CFN stack:

```
                         MAS (Multi-Agent System)
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    ioc-cfn-svc (9002)   │  Cognition Fabric Node
                    │                         │
                    │  ┌──────────────────┐   │   ┌─────────────────────────┐
                    │  │  Knowledge &     │◄──┼───►│ ioc-knowledge-memory-svc│
                    │  │  Memory routing  │   │   │        (9003)            │
                    │  └──────────────────┘   │   └─────────────────────────┘
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Cognition Engine (9004) │  ← this repo
                    │  ┌──────────────────┐   │
                    │  │ Ingestion        │   │
                    │  │ Evidence         │   │
                    │  │ Semantic Neg.    │   │
                    │  └──────────────────┘   │
                    └─────────────────────────┘
```

---

## Troubleshooting

### Embedding Model Download Issues

**Problem:** First startup hangs or fails with SSL certificate errors when downloading the `bge-small-en-v1.5` model from HuggingFace.

**Solution 1: Let fastembed download automatically (Recommended)**

Remove any corrupted local model directory and let fastembed download fresh:

```bash
# Remove corrupted Git LFS pointer files if they exist
rm -rf bge-small-en-v1.5/

# For corporate SSL certificate issues, add to .env:
HTTPX_VERIFY=false
OPENAI_VERIFY_SSL=false
```

The model (~127MB) downloads to `/tmp/fastembed_cache` on first run. Subsequent runs use the cached version.

**Solution 2: Manual download from cache**

If fastembed already downloaded the model to `/tmp/fastembed_cache`, copy it to the repo:

```bash
# Find the cached model
ls /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/

# Copy to repo root
mkdir -p bge-small-en-v1.5
cp -L /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/model_optimized.onnx bge-small-en-v1.5/
cp -L /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/*.json bge-small-en-v1.5/
cp -L /tmp/fastembed_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/*/vocab.txt bge-small-en-v1.5/
```

Now local dev and Docker builds will use the bundled model (no download needed).

**Solution 3: Git LFS (not recommended)**

If the model files are in a Git LFS repository, you need Git LFS installed:

```bash
brew install git-lfs  # macOS
git lfs install
git lfs pull
```

However, Solution 1 (fastembed auto-download) is cleaner and doesn't bloat your repository.

### Docker Build Fails with SSL Errors

The Dockerfile includes SSL bypass for model downloads. If you still encounter issues:

```dockerfile
# In Dockerfile, ensure these lines exist (already present):
RUN export HF_HUB_DISABLE_SSL_VERIFY=1 && \
    export CURL_CA_BUNDLE="" && \
    /opt/venv/bin/python -c "from fastembed import TextEmbedding; ..."
```

### LLM Connection Errors

**Problem:** `[SSL: CERTIFICATE_VERIFY_FAILED]` when calling the LLM API.

**Solution:** The code automatically disables SSL verification when `HTTPX_VERIFY=false` is set in `.env`:

```bash
# Add to .env
HTTPX_VERIFY=false
```

### Tests Fail with "Directory not found"

**Problem:** After the monorepo refactoring, old test commands reference outdated directory names.

**Solution:** Use the correct directory names:

```bash
# Old (incorrect):
cd ingestion-cognitive-agent && poetry run pytest

# New (correct):
cd ingestion && poetry run pytest
cd evidence && poetry run pytest
```

Or run all tests from the root:

```bash
poetry run pytest
```

---

## Project Structure

```
ioc-cfn-cognitive-agents/
├── pyproject.toml          # Single package definition
├── gateway/                # Unified FastAPI app (port 9004)
│   └── app/
├── ingestion/              # Knowledge extraction service
│   └── app/
├── evidence/               # Evidence gathering service
│   └── app/
└── semantic_alignment/   # Semantic negotiation service (port 8089)
    └── app/
```

---

## CI/CD Workflow

### Automated Docker Builds

The CI pipeline automatically builds and publishes a unified Docker image using GitHub Actions.

#### Pull Request (Build Validation)

When you open a PR:

```bash
git checkout -b feature/my-changes
git push origin feature/my-changes
# Open PR on GitHub
```

**What happens:**

- Runs unit tests
- Builds unified Docker image (validation only, does **not** push to registry)

#### Merge to Main (Latest Release)

When you merge to `main`:

**What happens:**

- Runs unit tests
- Builds and pushes image with `latest` tag to GHCR

**Published image:**

```
ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
```

#### Tag Push (Versioned Release)

To create a production release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

**What happens:**

- Validates tag follows semantic versioning (`vX.Y.Z`)
- Runs unit tests
- Builds and pushes image with version tag to GHCR

**Published image:**

```
ghcr.io/<org>/ioc-cfn-cognitive-agents:v1.0.0
```

**Valid tag formats:** `v1.0.0`, `v2.3.4-alpha.1`, `v1.0.0-beta`, `v3.2.1-rc.2`

### Using Published Images

```bash
docker pull ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
docker run -p 9004:9004 ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
```

### Multi-Platform Support

All images are built for `linux/amd64` and `linux/arm64` (Apple Silicon, ARM servers).

### Release Checklist

1. **Ensure tests pass**: `poetry run pytest`
2. **Update version in code** (if needed)
3. **Create semantic version tag**: `git tag v1.0.0`
4. **Push tag**: `git push origin v1.0.0`
5. **Monitor CI**: Check GitHub Actions for build status
6. **Verify image**: `docker pull ghcr.io/<org>/ioc-cfn-cognitive-agents:v1.0.0`

---

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Ensure tests pass: `poetry run pytest`
4. Push and open a PR (CI will validate build)
5. After merge, `latest` images are auto-published
6. Tag releases with semantic versions for production

## License

[Add your license here]
