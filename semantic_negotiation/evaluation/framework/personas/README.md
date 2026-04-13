# Personas

Agent personas are composed from two parts, declared in profile files via `persona_parts`:

```yaml
persona_parts:
  - personas/preferences/<preference_file>.yaml   # who the agent is + what it wants
  - personas/strategies/<strategy_file>.yaml       # how it negotiates
```

---

## Directory layout

```
personas/
  preferences/    — domain-specific identity and concession priorities
  strategies/     — negotiation behaviour rules (version-controlled)
  profiles/
    default/      — generic 3-agent setup used for Hard and Quick Deal missions
    ex01_*/       — mission-specific agent profiles
    ex0N_*/         (each references one preference + one strategy)
```

---

## Strategies

Strategy files live in `personas/strategies/` and define the negotiation protocol
injected into every agent's system prompt. Choose a strategy per profile by updating
the `persona_parts` reference.

### `negotiate_v1_0.yaml`, `negotiate_v1_1.yaml`, `negotiate_v1_2.yaml`

| Feature | `negotiate_v1_0` ← default | `negotiate_v1_1` | `negotiate_v1_2` |
|---|---|---|---|
| Time pressure | Yes — tracks `payload.round / payload.n_steps`, counts rounds remaining, uses deadline to drive concessions | No — clock-agnostic; negotiates on preference alone | Yes — identical to v1_0 |
| Convergence rule | Never bare-reject; always counter; progressively lock issues | Identical to v1_0 | Identical to v1_0 |
| Fallback (no acceptable value) | Propose middle-ground on every issue | Identical to v1_0 | Identical to v1_0 |
| Derailment | None | None | ~20% chance per round (≈1 in 5): agent ignores negotiation and counters with off-domain gibberish values |
| Best for | Most Example missions where deadline sensitivity shapes behaviour | Ablation studies, pure preference-driven testing | Robustness / fault-injection testing — measures how well other agents recover from a noisy peer |

---

### `negotiate_ov.yaml`

Overridable (OV) strategy — explicit **primary-win / their-win** decision logic.

- STEP 1: identifies the agent's PRIMARY WIN option from `options_per_issue`
- STEP 2: accepts immediately if current offer matches PRIMARY WIN on that issue;
  otherwise counters with `{primary issue: YOUR WIN, all others: THEIR WIN}`
- Designed for domain-extremist agents (`domain_*.yaml`) that have a single,
  non-negotiable priority and give away everything else to win on it

Best for: `default/agent_*.yaml` profiles and any agent with a clear single-issue focus.

---

## Adding a new strategy

1. Create `personas/strategies/negotiate_v<X_Y>.yaml` with a `negotiate:` key.
2. Reference it in any profile's `persona_parts`.
3. Document it in this README.
