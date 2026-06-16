# Semantic Alignment Agent

LLM-driven multi-party negotiation service built on the SAO (Stochastic Alternating Offers)
mechanism.  Agents discover a shared negotiation agenda from free-text content, generate
option sets per issue, then negotiate turn-by-turn until they reach agreement or exhaust
the round budget.

---

## Architecture

```
semantic_alignment/
├── app/
│   ├── main.py                          # FastAPI entry point  (port 8089)
│   ├── api/
│   │   ├── routes.py                    # POST /negotiate/initiate  POST /negotiate/decide
│   │   └── schemas.py                   # Request / response models
│   ├── agent/
│   │   ├── semantic_alignment.py      # Top-level pipeline orchestrator
│   │   ├── intent_discovery.py          # Component 1 — extracts negotiable issues
│   │   ├── options_generation.py        # Component 2 — generates options per issue
│   │   ├── batch_callback_runner.py     # SAO turn-by-turn engine (initiate / decide)
│   │   ├── callback_negotiator.py       # Per-agent SAO negotiator wrapper
│   │   ├── offer_validation.py          # Fuzzy offer snapping (rapidfuzz, 4 tiers)
│   │   ├── embedding_similarity.py      # Optional Granite-30M ONNX tier 5 (disabled by default)
│   │   ├── semantic_alignment_validation_pipeline.py  # Post-agreement validation
│   │   ├── negotiation_model.py         # NegotiationParticipant dataclass
│   │   └── http_repo.py                 # Agent callback HTTP client
│   └── config/                          # Settings (LLM model, API keys, …)
├── evaluation/                          # Evaluation harness — see evaluation/framework/README.md
├── tests/
│   ├── unit/
│   └── integration/
└── run_main.sh                          # One-liner to (re)start the server
```

---

## Pipeline overview

```
content_text
    │
    ▼
IntentDiscovery          (LLM)  → issues [ ]
    │
    ▼
OptionsGeneration        (LLM)  → options_per_issue { }
    │
    ▼
BatchCallbackRunner      (SAO)  → /initiate  →  agents negotiate via /decide rounds
    │
    ▼
SemanticAlignmentValidationPipeline  →  validated agreement
```

Offer validation runs inside `BatchCallbackRunner` on every counter-offer:
- **Tier 1-3** exact / case-insensitive / normalised match
- **Tier 4** rapidfuzz `token_set_ratio` (threshold 80, default)
- **Tier 5** Granite-30M ONNX embedding cosine similarity (opt-in — set `EMBEDDING_ENABLED = True`)

---

## Running the server

```bash
# Terminal 1 — start (or restart) the negotiation server on port 8089
cd semantic_alignment
./run_main.sh
# equivalent: poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

Verify:
```bash
curl -s http://localhost:8089/openapi.json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['title'])"
```

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/negotiate/initiate` | Start a negotiation session — runs IntentDiscovery + OptionsGeneration, returns first-round messages |
| `POST` | `/negotiate/decide` | Submit agent decisions for the current round, advance SAO state |

---

## Configuration

Create `semantic_alignment/.env`:

```dotenv
LLM_API_KEY=sk-...
LLM_BASE_URL=https://litellm.prod.outshift.ai
LLM_MODEL=bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0
JUDGE_MODEL=bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

---

## Running tests

```bash
# Unit tests
PYTHONPATH=semantic_alignment poetry run pytest semantic_alignment/tests/unit -v

# Offer validation tests (fuzzy + embedding tiers)
PYTHONPATH=semantic_alignment poetry run pytest semantic_alignment/tests/unit/test_offer_validation.py -v
```

---

## Evaluation

End-to-end negotiation benchmarks live under `evaluation/framework/`.
See **[evaluation/framework/README.md](evaluation/framework/README.md)** for full details.

Quick start against the CFN service:

```bash
cd semantic_alignment
poetry run python evaluation/framework/test_via_cfn_service.py \
    --cfn-url http://localhost:9002 \
    --workspace-id WSID \
    --mas-id MASID \
    --filter "Quick Deal"
```

