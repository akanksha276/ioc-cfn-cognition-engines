# Integration Tests

This directory contains integration tests for the cognition engine usage examples.

## Test Files

### `test_usage_examples_mock.py`
Tests usage guide examples using **mock mode** (no Azure OpenAI required).
- Uses `mock_mode=True` for concept extraction
- Generates deterministic test data
- Safe for CI/CD pipelines
- Runs by default

### `test_usage_examples_live.py`
Tests usage guide examples with **real Azure OpenAI API calls**.
- Requires valid Azure OpenAI credentials in `.env`
- Makes actual LLM calls for realistic testing
- Marked with `@pytest.mark.requires_credentials`
- **Skipped by default in CI/CD**

## Running Tests

### Run all integration tests (excludes live tests by default)
```bash
PYTHONPATH=. poetry run pytest tests/integration/ -v
```

### Run only mock tests
```bash
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_mock.py -v
```

### Run live tests with Azure OpenAI credentials
```bash
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_live.py -v -m ''
```

Or to run specific live test:
```bash
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_live.py::TestUsageExamplesLive::test_knowledge_extraction_with_llm -v -s -m ''
```

## CI/CD Configuration

The `pyproject.toml` is configured to skip tests marked with `@pytest.mark.requires_credentials` by default:

```toml
[tool.pytest.ini_options]
addopts = "-v -s --tb=short -m 'not requires_credentials'"
markers = [
    "requires_credentials: marks tests that require Azure OpenAI credentials (skipped by default in CI)",
]
```

This ensures CI pipelines don't attempt to run live tests that require credentials.

## Test Results

### Mock Tests (CI-safe)
- ✅ `test_knowledge_extraction_programmatic` - Tests concept extraction with mock data
- ✅ `test_knowledge_extraction_returns_metadata` - Verifies metadata structure
- ✅ `test_evidence_gathering_programmatic` - Tests evidence gathering flow
- ✅ `test_similarity_search_operations` - Tests similarity search API

### Live Tests (require credentials)
- ✅ `test_knowledge_extraction_with_llm` - Real LLM extraction (7 concepts, 7 relations)
- ✅ `test_evidence_gathering_with_llm` - Real evidence gathering with LLM
- ✅ `test_extraction_metadata_with_llm` - Metadata validation with real LLM

## Environment Setup

For live tests, create a `.env` file:
```bash
cp .env.example .env
# Edit .env and set Azure OpenAI credentials
```

Required variables:
```bash
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```
