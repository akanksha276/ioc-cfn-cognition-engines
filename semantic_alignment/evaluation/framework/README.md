# Evaluation Framework

End-to-end harness for running and benchmarking LLM-driven multi-party negotiations
across a dataset of missions.  Two complementary workflows are supported:

| Workflow | Script | Purpose |
|---|---|---|
| **GOAL A** – LLM negotiation run | `test_via_semantic_neg_agents_configured.py` | Drive real SAO negotiations via the local semantic-alignment app; save per-mission traces |
| **GOAL A-CFN** – LLM negotiation via CFN service | `test_via_cfn_service.py` | Same as GOAL A but routes through the CFN service (`ioc-cognition-fabric-node-svc`) instead of the local app |
| **GOAL B** – Offline benchmark | `evaluation/eval_pipeline.py` | Score saved traces (or live LLM calls) against gold datasets |

Legacy rule-based scripts (`run_evaluation.py`, `runner.py`, `callback_env.py`) are retained
for direct/callback mechanism testing but are **not** required for the primary LLM evaluation flow.

---

## Directory layout

```
evaluation/
├── eval_pipeline.py                    # GOAL B: benchmark traces vs gold (online + offline)
│
└── framework/
    ├── README.md                       # This file
    ├── missions.yaml                   # All 16 missions (General x 2, Hard x 5, Example x 9)
    ├── agent_configs.yaml              # Per-mission LLM agent wiring (references persona files)
    ├── cfn_service_config.yaml         # CFN connection + rule-based agent defaults for GOAL A-CFN
    ├── test_via_semantic_neg_agents_configured.py   # GOAL A v2: per-mission agents + --filter
    ├── test_via_cfn_service.py                      # GOAL A-CFN: same as v2 but via CFN service
    ├── test_via_semantic_neg_agents.py              # GOAL A v1: original, 3 generic agents
    │
    ├── ground_truth/
    │   ├── hard_convergence_direct_5.json           # Gold options for Hard 01-05
    │   └── example_usecases_direct_9_gold.json           # Gold options for Example 01-09
    │
    ├── personas/                       # Composable agent persona library
    │   ├── README.md                   # Persona authoring guide
    │   ├── preferences/                # Who the agent is + what it wants
    │   │   ├── domain_a.yaml           # Default: cost-conscious, risk-averse
    │   │   ├── domain_b.yaml           # Default: innovation-focused
    │   │   ├── domain_c.yaml           # Default: balanced/pragmatic
    │   └── ex01_*.yaml … ex09_*.yaml  # Mission-specific preferences (3 per Example)
    │   ├── strategies/                 # How the agent negotiates
    │   │   ├── negotiate_v1_0.yaml     # Time-pressure aware (default)
    │   │   ├── negotiate_v1_1.yaml     # Clock-agnostic / preference-driven
    │   │   ├── negotiate_v1_2.yaml     # Derailment / fault-injection
    │   │   └── negotiate_ov.yaml       # Overridable: primary-win / their-win logic
    │   └── profiles/                   # Assembled agent profiles (preference + strategy)
    │       ├── default/                # Generic 3-agent set for Hard + Quick Deal missions
    │       │   ├── agent_a.yaml
    │       │   ├── agent_b.yaml
    │       │   └── agent_c.yaml
    │       ├── ex01_email_automation/
    │       ├── ex02_inbox_thread_workflow/
    │       └── … ex09_ci_cd_release/
    │
    ├── run_evaluation.py               # (legacy) Rule-based evaluation CLI
    ├── evaluation.yaml                 # (legacy) Run config
    ├── config.py                       # EvaluationConfig / AgentConfig dataclasses
    ├── runner.py                       # EvaluationRunner -- in-process SAO mechanism
    ├── agents/
    │   ├── base_agent.py
    │   ├── llm_agent.py
    │   ├── agent_loader.py
    │   └── agent_server.py
    └── mechanisms/
        ├── direct_env.py
        └── callback_env.py
```

---

## Prerequisites

### 1. Python environment

```bash
cd ioc-cfn-cognitive-agents
poetry install
```

### 2. `.env` configuration

Create `semantic_alignment/.env`:

```dotenv
# LLM provider key and proxy
LLM_API_KEY=sk-...
LLM_BASE_URL=https://litellm.prod.outshift.ai

# Generator model (IntentDiscovery + OptionsGeneration + LLMNegotiationAgent)
# Use the exact model ID registered on the proxy (check GET /v1/models)
LLM_MODEL=bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0

# Judge model for eval_pipeline.py (can differ from generator to avoid self-consistency bias)
JUDGE_MODEL=bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

> **Note:** `settings.py` reads `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` -- not `OPENAI_*` names.

### 3. Negotiation server (GOAL A only)

```bash
# Terminal 1
cd semantic_alignment
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
```

Verify: `curl http://localhost:8089/openapi.json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['title'])"`

---

## GOAL A -- Run LLM negotiations

`test_via_semantic_neg_agents_configured.py` registers per-mission domain agents
from `agent_configs.yaml`, then drives the full SAO negotiation loop for each
selected mission.  Traces are saved under `neg_trace/<YYYYMMDD_HHMMSS>/`.

```bash
cd semantic_alignment

# Hard convergence missions (Hard 01-05)
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py --filter hard

# Example use-case missions (Example 01-09)
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py --filter example

# All 16 missions
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py

# Single mission by name substring
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py --filter "Hard 01"

# Custom agent configs or missions file
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py \
    --agent-configs evaluation/framework/agent_configs.yaml \
    --filter hard
```

**Key trace file:** `neg_trace/<timestamp>/<mission_slug>/01_initiate_response.json`
Contains `payload.issues` and `payload.options_per_issue` -- the LLM-discovered
negotiation space used by GOAL B.

---

## GOAL A-CFN -- Run LLM negotiations via the CFN service

`test_via_cfn_service.py` is identical in purpose to the GOAL A script above, but
it calls the **CFN service** (`ioc-cognition-fabric-node-svc`) endpoints instead of the
local semantic-alignment app.  This lets you validate the full deployment path
through the Cognition Fabric Node.

### Prerequisites

1. The CFN service must be running (default `http://localhost:9002`):

   ```bash
   cd ioc-cognition-fabric-node-svc && ./localrun.sh
   ```

2. A valid `workspace_id` and `mas_id` must exist in the management plane.

### Configuration (`cfn_service_config.yaml`)

All connection settings and agent definitions live in `cfn_service_config.yaml`
next to the script.  Edit it once instead of passing CLI flags every time:

```yaml
# CFN service connection
cfn_url: http://localhost:9002
workspace_id: ws1
mas_id: mas1

# Port for the local agent callback server
agent_port: 8092

# Paths (relative to this config file)
missions_file: missions.yaml
agent_configs_file: agent_configs.yaml

# Default rule-based agents — add/remove entries to change the agent count.
# Per-mission entries in agent_configs.yaml still override these.
agents:
  - id: agent-a
    name: Agent A
    prefer_low: true       # prefers cheapest options
    exponent: 2.0          # Boulware concession curve
    min_reservation: 0.0

  - id: agent-b
    name: Agent B
    prefer_low: false      # prefers premium options
    exponent: 2.0
    min_reservation: 0.0

  - id: agent-c
    name: Agent C
    prefer_low: true
    exponent: 1.0          # linear (concedes faster)
    min_reservation: 0.0
```

**Agent priority order:**
1. Mission-specific LLM personas from `agent_configs.yaml` (if entry exists for the mission)
2. Rule-based agents defined under `agents:` in `cfn_service_config.yaml`
3. Default LLM agents from `agent_configs.yaml → default_agents`
4. Built-in fallback (3 generic agents A/B/C)

### Usage

```bash
cd semantic_alignment

# Minimal — pass the three required identifiers and a mission filter
poetry run python evaluation/framework/test_via_cfn_service.py \
    --cfn-url http://localhost:9002 \
    --workspace-id WSID \
    --mas-id MASID \
    --filter "Quick Deal"

# Run all missions
poetry run python evaluation/framework/test_via_cfn_service.py \
    --cfn-url http://localhost:9002 \
    --workspace-id WSID \
    --mas-id MASID

# Filter shortcuts
#   --filter hard        → Hard 01-05
#   --filter example     → Example 01-09
#   --filter "Hard 01"   → single mission by name substring
#   --filter connected   → Connected 01-10 (memory evaluation sequence)
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `cfn_service_config.yaml` | YAML config file (all settings; CLI flags override) |
| `--cfn-url URL` | from config / `http://localhost:9002` | Base URL of the CFN service |
| `--workspace-id ID` | from config | Workspace ID for CFN API routes |
| `--mas-id ID` | from config | Multi-Agentic System ID for CFN API routes |
| `--agent-port PORT` | from config / `8092` | Port for the local agent callback server |
| `--missions-file PATH` | from config / `missions.yaml` | Path to YAML missions file |
| `--agent-configs PATH` | from config / `agent_configs.yaml` | Path to agent persona configs |
| `--filter TEXT` | `None` | Case-insensitive substring filter on mission names |

Traces are saved under `neg_trace/<YYYYMMDD_HHMMSS>/` in the same format as GOAL A.

---

## GOAL B -- Benchmark against gold

`eval_pipeline.py` runs a two-phase LLM-as-judge evaluation:

| Phase | Input | Metric |
|---|---|---|
| Phase 1 - Intent coverage | Predicted entities vs gold issues | `pipeline_recall`, `intent_precision` |
| Phase 2 - Options quality | Generated options vs gold options | `options_precision`, `options_recall`, `options_f1` |

### Offline mode (recommended after GOAL A)

Reads the `01_initiate_response.json` traces saved in GOAL A.  Only judge LLM calls are made.

```bash
cd semantic_alignment

# Hard 01-05
poetry run python -m evaluation.eval_pipeline \
    --dataset  evaluation/framework/ground_truth/hard_convergence_direct_5.json \
    --trace-dir neg_trace/<YYYYMMDD_HHMMSS> \
    --output   neg_trace/<YYYYMMDD_HHMMSS>/eval_hard5_report.json \
    --csv      neg_trace/<YYYYMMDD_HHMMSS>/eval_hard5_report.csv \
    --verbose

# Example 01-09
poetry run python -m evaluation.eval_pipeline \
    --dataset  evaluation/framework/ground_truth/example_usecases_direct_9_gold.json \
    --trace-dir neg_trace/<YYYYMMDD_HHMMSS> \
    --output   neg_trace/<YYYYMMDD_HHMMSS>/eval_example9_report.json \
    --csv      neg_trace/<YYYYMMDD_HHMMSS>/eval_example9_report.csv \
    --verbose
```

### Online mode (no prior GOAL A run needed)

Calls IntentDiscovery + OptionsGeneration LLMs live, then judges the output.

```bash
poetry run python -m evaluation.eval_pipeline \
    --dataset evaluation/framework/ground_truth/hard_convergence_direct_5.json \
    --output  results/eval_hard5_online.json \
    --csv     results/eval_hard5_online.csv \
    --verbose
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--dataset PATH` | *(required)* | Path to gold JSON dataset |
| `--trace-dir DIR` | `None` | Offline mode: directory with GOAL A trace folders |
| `--judge-model MODEL` | env `JUDGE_MODEL` | LiteLLM model for judge calls |
| `--output FILE` | `eval_report.json` | JSON report path |
| `--csv FILE` | `None` | Optional CSV report path |
| `--limit N` | `0` (all) | Cap number of samples evaluated |
| `--verbose` | `False` | Print per-issue judge reasoning |

---

## Agent configuration (`agent_configs.yaml`)

Controls which LLM agents are registered per mission in GOAL A.  Agent personas
are defined as **composable YAML files** under `personas/` rather than inline
strings.  `agent_configs.yaml` wires them together via `persona_file`:

```yaml
default_agents:          # Used when no mission-specific entry exists (e.g. Hard missions)
  - id: agent-a
    name: Agent A
    persona_file: personas/profiles/default/agent_a.yaml

  - id: agent-b
    name: Agent B
    persona_file: personas/profiles/default/agent_b.yaml

  - id: agent-c
    name: Agent C
    persona_file: personas/profiles/default/agent_c.yaml

mission_agents:          # Keyed by mission slug (snake_case of mission name)
  example_01_email_automation:
    - id: delete-agent
      name: DeleteAgent
      persona_file: personas/profiles/ex01_email_automation/delete_agent.yaml
    # ...
```

**Slug matching:** `"Example 01 - Email Automation"` maps to `example_01_email_automation`.
Falls back to `default_agents` if no slug match is found.

**Inline `persona:` strings** are still accepted as a fallback — useful for quick
one-off experiments without creating new persona files.

---

## Personas (`personas/`)

Agent personas are **composed from two parts** declared in a profile file:

```yaml
# personas/profiles/default/agent_a.yaml
persona_parts:
  - personas/preferences/domain_a.yaml   # who the agent is + what it wants
  - personas/strategies/negotiate_v1_0.yaml  # how it negotiates
```

At load time the test script concatenates the two files into a single `persona`
string injected into the agent's system prompt.

### Preference files (`personas/preferences/`)

Define the agent's identity and concession priorities under a `domain:` key.
Three generic preferences ship for the Hard / Quick Deal missions:

| File | Archetype |
|---|---|
| `domain_a.yaml` | Cost-conscious, risk-averse — protects cheapest option longest |
| `domain_b.yaml` | Innovation-focused — prefers advanced / high-capability options |
| `domain_c.yaml` | Balanced / pragmatic — ranks issues by stakeholder impact |

Mission-specific preferences (e.g. `ex01_delete_agent.yaml`) capture role-specific
priorities for each Example use-case (3 files × 9 missions = 27 files).

### Strategy files (`personas/strategies/`)

Define negotiation behaviour under a `negotiate:` key.  Choose a strategy per
profile by updating its `persona_parts` reference:

| File | Time-aware | Derailment | Best for |
|---|---|---|---|
| `negotiate_v1_0.yaml` | ✅ tracks `round/n_steps`, deadline-driven concessions | — | Most missions (default) |
| `negotiate_v1_1.yaml` | ❌ clock-agnostic | — | Ablation / pure preference studies |
| `negotiate_v1_2.yaml` | ✅ identical to v1_0 | ~20% chance per round (off-domain counter) | Fault-injection / robustness testing |
| `negotiate_ov.yaml` | — | — | Single-issue-extremist agents: accept if PRIMARY WIN, else counter with `{primary: YOUR WIN, rest: THEIR WIN}` |

### Profile files (`personas/profiles/`)

Assemble a preference + strategy into a deployable agent profile.  The `default/`
profiles are used for all Hard and Quick Deal missions; Example missions each have their
own subdirectory:

```
profiles/
  default/          agent_a.yaml, agent_b.yaml, agent_c.yaml
  ex01_email_automation/   delete_agent.yaml, archive_agent.yaml, compliance_agent.yaml
  ex02_inbox_thread_workflow/   bob.yaml, julie.yaml
  … (one folder per Example mission through ex09_ci_cd_release/))
```

> See `personas/README.md` for authoring guidance and strategy comparison details.

---

## Gold datasets

| File | Missions | Issues/entry | Key |
|---|---|---|---|
| `ground_truth/hard_convergence_direct_5.json` | Hard 01-05 | 4 | `gold_options` |
| `ground_truth/example_usecases_direct_9_gold.json` | Example 01-09 | 2-5 | `gold_options` |

Schema per entry:

```json
{
  "id": "hard_01",
  "sentence": "The engineering team and the ethics board must agree on ...",
  "context": "AI model deployment governance negotiation between ...",
  "domain": "ai_governance",
  "metadata": {
    "difficulty": "very_hard",
    "num_issues": 4
  },
  "gold_issues": [
    "responsible enough",
    "competitive enough",
    "transparent in how decisions are made",
    "unacceptable liability"
  ],
  "gold_options": {
    "responsible enough": ["Every model requires a completed ethics audit ...", "..."],
    "competitive enough": ["Deployment lead time must match or beat ...", "..."]
  }
}
```

---

## Missions dataset (`missions.yaml`)

All 16 missions in one file.  Use `--filter` to select a subset.

| Filter | Selects |
|---|---|
| `--filter hard` | Hard 01-05 (5 missions) |
| `--filter example` | Example 01-09 (9 missions) |
| `--filter "Hard 01"` | Single mission by name substring |
| *(no filter)* | All 16 missions |

| Category | Missions | Typical `n_steps` |
|---|---|---|
| General | Quick Deal, Cloud Platform | `0` (auto) / `40` |
| Hard convergence | Cross-functional tech/org tensions | `30` |
| Example use-case | Multi-agent task coordination | `50` |

---

## Negotiation outcome evaluation (`--run-log`)

After a GOAL A run, pass `run_log.json` to `eval_pipeline.py` to get aggregated
negotiation outcome metrics — separate from (or combined with) the pipeline quality metrics.

> **Note:** `neg_trace/<timestamp>/` is created automatically by GOAL A.
> Replace `<timestamp>` with the folder name printed when the test script runs
> (e.g. `20260403_143022`).

```bash
# Negotiation outcomes only (no LLM judge calls)
poetry run python -m evaluation.eval_pipeline \
    --run-log  neg_trace/<timestamp>/run_log.json \
    --output   neg_trace/<timestamp>/neg_report.json \
    --neg-csv  neg_trace/<timestamp>/neg_report.csv

# Combined pipeline + negotiation report
poetry run python -m evaluation.eval_pipeline \
    --dataset  evaluation/framework/ground_truth/hard_convergence_direct_5.json \
    --trace-dir neg_trace/<timestamp> \
    --run-log  neg_trace/<timestamp>/run_log.json \
    --output   neg_trace/<timestamp>/full_report.json \
    --csv      neg_trace/<timestamp>/pipeline_report.csv \
    --neg-csv  neg_trace/<timestamp>/neg_report.csv \
    --verbose
```

**Negotiation metrics computed:**

| Metric | Description |
|---|---|
| `agreement_rate` | Fraction of missions that reached consensus |
| `avg_rounds_to_agreement` | Mean SAO rounds for agreed missions |
| `avg_rounds_all` | Mean SAO rounds across all missions |
| `timeout_rate` / `broken_rate` | Fraction that timed out / ended broken |
| `avg_duration_s` | Mean wall-clock seconds per mission |
| `llm_calls_total` | initiate (2/mission) + actual decide calls |
| `decide_prompt_tokens` | Total prompt tokens for agent decide calls |
| `decide_completion_tokens` | Total completion tokens for agent decide calls |
| `decide_estimated_cost_usd` | Estimated USD cost (litellm `completion_cost`) |

### LLM cost tracking

The test script uses a **litellm `success_callback`** to capture real token usage
and cost for every agent decide call:

```
Per mission in run_log.json:
  llm_calls_initiate: 2          # IntentDiscovery + OptionsGeneration (neg server, estimated)
  llm_calls_total:   2 + N       # actual decide calls added
  llm_usage_decide:
    calls: N                     # actual LLM calls made by agents
    prompt_tokens: 12400
    completion_tokens: 840
    total_tokens: 13240
    estimated_cost_usd: 0.0412
    note: "decide-phase only; initiate-phase runs in neg server"
```

> **Note on initiate-phase tokens:** IntentDiscovery and OptionsGeneration run inside
> the neg server process and are not directly measurable from the test script.
> Their call count is fixed at 2 per mission; token counts are not available.

---

## Report output

### Pipeline JSON (`eval_report.json`)

```json
{
  "scorer": "llm",
  "judge_model": "...",
  "generator_model": "...",
  "overall": {
    "avg_pipeline_recall": 0.84,
    "micro_precision": 0.71,
    "micro_recall": 0.68,
    "micro_f1": 0.69,
    "full_coverage_rate": 0.60,
    "n_samples": 5
  },
  "by_domain": { "...": "..." },
  "by_difficulty": { "...": "..." },
  "negotiation": {
    "agreement_rate": 0.80,
    "avg_rounds_to_agreement": 47.5,
    "llm_calls_total": 722,
    "decide_total_tokens": 198400,
    "decide_estimated_cost_usd": 0.612,
    "missions": [ "..." ]
  },
  "samples": [ "..." ]
}
```

### Pipeline CSV (`eval_report.csv`)

One row per sample plus an `OVERALL` summary row.  Dynamic columns for each
discovered issue: `issue_<n>_name`, `issue_<n>_pipeline_recall`,
`issue_<n>_options_f1`, etc.

### Negotiation CSV (`neg_report.csv`)

One row per mission plus an `OVERALL` row.  Columns: `mission`, `agreed`,
`verdict`, `status`, `total_rounds`, `n_agents`, `llm_calls_initiate`,
`llm_calls_decide`, `llm_calls_total`, `prompt_tokens`, `completion_tokens`,
`total_tokens`, `estimated_cost_usd`, `duration_s`, `deal_issue_<n>`, `deal_option_<n>`.

---

## Typical end-to-end run (Hard 5)

```bash
# 0. Start neg server (keep running in Terminal 1)
cd semantic_alignment && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089

# 1. GOAL A -- run negotiations; neg_trace/<timestamp>/ is created automatically
poetry run python evaluation/framework/test_via_semantic_neg_agents_configured.py --filter hard
# -> neg_trace/<timestamp>/   e.g. neg_trace/20260403_143022/

# 2. GOAL B -- combined pipeline + negotiation benchmark (substitute your timestamp)
poetry run python -m evaluation.eval_pipeline \
    --dataset  evaluation/framework/ground_truth/hard_convergence_direct_5.json \
    --trace-dir neg_trace/<timestamp> \
    --run-log  neg_trace/<timestamp>/run_log.json \
    --output   neg_trace/<timestamp>/full_report.json \
    --csv      neg_trace/<timestamp>/pipeline_report.csv \
    --neg-csv  neg_trace/<timestamp>/neg_report.csv \
    --verbose
```

---

## Legacy: Rule-based evaluation

`run_evaluation.py` drives the SAO mechanism with rule-based `BoulwareAgent`s
(no LLM calls).  Useful for quick smoke-tests or reproducible unit-level checks.

```bash
# Callback mode (requires neg server at :8089)
poetry run python evaluation/framework/run_evaluation.py

# Direct mode (self-contained, no server)
# Set mechanism: direct in evaluation.yaml first
poetry run python evaluation/framework/run_evaluation.py --config evaluation.yaml

# Limit to first N missions
poetry run python evaluation/framework/run_evaluation.py --n-missions 2
```

Results are printed to stdout and written to `neg_trace/run_evaluation_summary.json`.
