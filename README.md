# IoC CFN Cognitive Agents

A collection of cognitive agents for processing OpenTelemetry data and evidence gathering for reasoning systems.

## Agents

- **[Ingestion Service](ingestion/)** – Extracts knowledge from OpenTelemetry traces (entities, relations, embeddings).
- **[Evidence Gathering Service](evidence/)** – Retrieves relevant evidence from the knowledge graph (e.g. “What does Miss-Marple do?”).
- **[Semantic Negotiation Agent](semantic_negotiation/)** – Handles multi-party semantic negotiation using NegMAS and SSTP (Semantic State Transfer Protocol).

The evidence service can use an optional **mocked DB** (Neo4j-backed graph API). For that setup, run the mocked-db service and set `CFN_URL` or `MOCKED_DB_BASE_URL`; see [evidence/README.md](evidence/README.md).

## Quick Start

### Run the gateway with Docker (recommended)

The gateway serves both ingestion and evidence on **port 9004**. It uses a **`.env` file** at repo root (see [Environment setup](#environment-setup)); create it from `.env.example` if needed.

```bash
# From repo root (ensure .env exists with Azure OpenAI credentials, etc.)
docker compose up --build
```

Then use the API at `http://localhost:9004`:

| Backend   | Path | Example |
|-----------|------|--------|
| Gateway health | `/health` | `GET http://localhost:9004/health` |
| Ingestion | `/api/knowledge-mgmt/extraction` | `POST http://localhost:9004/api/knowledge-mgmt/extraction` |
| Evidence  | `/api/knowledge-mgmt/reasoning/evidence` | `POST http://localhost:9004/api/knowledge-mgmt/reasoning/evidence` |

**Confluence paths** (above); prefixed paths also work: `/ingestion/...`, `/evidence/...`.

### Run the gateway locally (no Docker)

**Requirement: a `.env` file** with at least Azure OpenAI credentials (see [Environment setup](#environment-setup) below). Create it from the template:

```bash
cp .env.example .env
# Edit .env and set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, etc.
```

One-time setup so the gateway can import ingestion and evidence:

```bash
./scripts/setup_local_links.sh
```

Then (from repo root):

```bash
PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004
```

Use `http://localhost:9004` as the base URL (see API paths in the Quick Start section above).

### Run agents individually (development only)

**⚠️ For normal use, run the gateway above** (port 9004) – it's the single unified entry point that includes ingestion and evidence.

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

**Semantic Negotiation Agent** (independent service, port 8089):
```bash
cd semantic_negotiation
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

</details>

**Note:** The gateway (port 9004) is the recommended production setup. It runs ingestion + evidence in a single process. The semantic negotiation agent is a separate service that runs independently.

### Testing Semantic Negotiation

Two test harnesses are available under `semantic_negotiation/evaluation/framework/`:

- **`test_via_semantic_neg_agents_configured.py`** — Spawns a multi-agent system (MAS) in-process and tests semantic negotiation directly. Use `--filter` to run a specific mission.
  ```bash
  poetry run python semantic_negotiation/evaluation/framework/test_via_semantic_neg_agents_configured.py
  poetry run python semantic_negotiation/evaluation/framework/test_via_semantic_neg_agents_configured.py --filter "Quick deal"
  ```

- **`test_via_cfn_service.py`** — Calls the full CFN service end-to-end for integration testing.
  ```bash
  poetry run python semantic_negotiation/evaluation/framework/test_via_cfn_service.py
  ```

---

## CI/CD Workflow

### 🔄 Automated Docker Builds

The CI pipeline automatically builds and publishes a unified Docker image using GitHub Actions.

#### **Pull Request** (Build Validation)
When you open a PR:
```bash
git checkout -b feature/my-changes
git push origin feature/my-changes
# Open PR on GitHub
```

**What happens:**
- ✅ Runs unit tests
- ✅ Builds unified Docker image (validation only)
- ❌ Does **NOT** push image to registry
- 🎯 Purpose: Catch Docker build regressions early

#### **Merge to Main** (Latest Release)
When you merge to `main`:
```bash
git checkout main
git pull origin main
git merge feature/my-changes
git push origin main
```

**What happens:**
- ✅ Runs unit tests
- ✅ Builds unified Docker image
- ✅ Pushes with `latest` tag to GHCR

**Published image:**
```
ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
```

#### **Tag Push** (Versioned Release)
To create a production release:
```bash
# Create and push a semantic version tag
git tag v1.0.0
git push origin v1.0.0
```

**What happens:**
- ✅ Validates tag follows semantic versioning (`vX.Y.Z`)
- ✅ Runs unit tests
- ✅ Builds unified Docker image
- ✅ Pushes with version tag to GHCR

**Published image:**
```
ghcr.io/<org>/ioc-cfn-cognitive-agents:v1.0.0
```

**Valid tag formats:**
- `v1.0.0` - Standard release
- `v2.3.4-alpha.1` - Pre-release
- `v1.0.0-beta` - Beta release
- `v3.2.1-rc.2` - Release candidate

**Invalid tags will fail CI:**
- `1.0.0` (missing `v` prefix)
- `v1.2` (incomplete version)
- `release-1` (not semver)

### 📦 Using Published Images

The recommended way to run is the **unified gateway image** (single process, port 9004):

```bash
# Pull and run the unified gateway (ingestion + evidence on port 9004)
docker pull ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
docker run -p 9004:9004 ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
```

Then use `http://localhost:9004` for ingestion and evidence paths (see Quick Start).

### 🏗️ Multi-Platform Support

All images are built for:
- `linux/amd64` (x86_64)
- `linux/arm64` (Apple Silicon, ARM servers)

### 🚀 Release Checklist

1. **Ensure tests pass**: `poetry run pytest`
2. **Update version in code** (if needed)
3. **Create semantic version tag**: `git tag v1.0.0`
4. **Push tag**: `git push origin v1.0.0`
5. **Monitor CI**: Check GitHub Actions for build status
6. **Verify image**: `docker pull ghcr.io/<org>/ioc-cfn-cognitive-agents:v1.0.0`

---

## 📦 Python Package Publishing

The `cognition-engine` package auto-publishes to Artifactory on push to `clawbee` branch.

### Publishing Workflow

```bash
git checkout clawbee
git push origin clawbee
```

**CI automatically:**
- Generates dev version: `0.1.0.dev1`, `0.1.0.dev2`, etc. (PEP 440)
- Builds `.tar.gz` and `.whl` packages
- Publishes to Artifactory using Vault credentials

### Bumping Version

```bash
# Update version in pyproject.toml
poetry version minor  # 0.1.0 → 0.2.0 (or: patch, major)

# Commit and push
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
git push origin clawbee
```

Next publish will be `0.2.0.dev1`, then `0.2.0.dev2`, etc.

### Installing

```bash
pip install cognition-engine --extra-index-url https://<artifactory-url>/artifactory/api/pypi/outshift-pypi/simple
```

**Package includes:** `ingestion`, `evidence`, `gateway` modules
**Usage examples:** [docs/usage.md](docs/usage.md)

---

## Development

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for containerized development)

### Environment setup (required for local and Docker)

A **`.env` file is required** for the gateway (local and Docker) so ingestion and evidence have credentials and options.

**Location:** Place `.env` at **repo root** (`ioc-cfn-cognitive-agents/.env`). This single file is used by:
- Local development (gateway, ingestion, evidence)
- Docker Compose (via `env_file` in compose.yaml)
- CI/CD workflows

**Create from template:**

```bash
# From repo root
cp .env.example .env
# Edit .env and set your values (see below)
```

**Required and optional variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes (for ingestion) | Azure OpenAI endpoint URL. Ingestion needs this for LLM-based extraction. |
| `AZURE_OPENAI_API_KEY` | Yes (for ingestion) | Azure OpenAI API key. |
| `AZURE_OPENAI_DEPLOYMENT` | No | Deployment name (default: `gpt-4o`). |
| `AZURE_OPENAI_API_VERSION` | No | API version (default: `2024-08-01-preview`). |
| `EMBEDDING_MODEL_PATH` | No | Path to local `bge-small-en-v1.5` folder. If set, ingestion and evidence use it instead of downloading from Hugging Face. Use `bge-small-en-v1.5` (relative to repo root) or an absolute path. |
| `ENABLE_EMBEDDINGS` | No | Enable embedding generation (default: `true`). |
| `ENABLE_DEDUP` | No | Enable semantic deduplication (default: `true`). |
| `SIMILARITY_THRESHOLD` | No | Dedup threshold 0.0–1.0 (default: `0.95`). |
| `MOCKED_DB_BASE_URL` | No | Evidence: external mocked DB URL (optional). |
| `MGMT_PLANE_URL`, `COGNITION_ENGINE_*` | No | Management plane / engine registration. |

The same `.env` can contain variables for multiple services; each app ignores unknown keys. See [.env.example](.env.example) for a full template.

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

The **unified gateway** runs ingestion and evidence in one process (port 9004):

```
┌─────────────────────────────────────────────────────────┐
│                   OpenTelemetry Traces                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────────────────┐
         │  Unified Gateway (port 9004)                    │
         │  ┌─────────────────────┐  ┌──────────────────┐ │
         │  │ Ingestion            │  │ Evidence         │ │
         │  │ - Extract entities   │  │ - Query intent   │ │
         │  │ - Generate embeddings│  │ - Path finding   │ │
         │  │ - Build relations    │  │ - Evidence rank  │ │
         │  └──────────┬───────────┘  └────────┬─────────┘ │
         │             └───────────────┬────────┘           │
         └─────────────────────────────┼───────────────────┘
                                       │
                      (optional)       ▼
              ┌───────────────────────────────┐
              │ External graph (e.g. Neo4j)   │
              │ when CFN_URL set  │
              └───────────────────────────────┘
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

### Azure OpenAI Connection Errors

**Problem:** `[SSL: CERTIFICATE_VERIFY_FAILED]` when calling Azure OpenAI API.

**Solution:** The code automatically disables SSL verification when `HTTPX_VERIFY=false` is set in `.env`:

```bash
# Add to .env
HTTPX_VERIFY=false
```

The ingestion and evidence services read this variable and configure `httpx.Client(verify=False)` for the Azure OpenAI client.

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

## Project Structure (Good Practice for Package Publishing)

This monorepo uses Poetry with `package-mode = true` and is ready for publishing to JFrog or PyPI:

```
ioc-cfn-cognitive-agents/
├── pyproject.toml          # Single package definition
├── gateway/                # Unified FastAPI app
│   ├── __init__.py
│   └── app/
├── ingestion/              # Knowledge extraction service
│   ├── __init__.py
│   └── app/
├── evidence/               # Evidence gathering service
│   ├── __init__.py
│   └── app/
└── semantic_negotiation/  # Separate negotiation service
```

**Benefits:**
- ✅ Single `pip install ioc-cfn-cognitive-agents` gets everything
- ✅ Shared dependencies in one `pyproject.toml`
- ✅ Directory names match Python imports (`from ingestion.app...`)
- ✅ No symlink workarounds needed
- ✅ Ready for `poetry build` and `poetry publish`

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
