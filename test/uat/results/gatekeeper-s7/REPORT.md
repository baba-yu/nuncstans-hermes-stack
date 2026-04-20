# S7 Gatekeeper E2E — Results

## Setup
- Bonsai llama-server on GPU (CUDA 12.9, -ngl 99), --parallel 1, -c 16384
- Honcho api+deriver rebuilt with migration `g7h8i9j0k1l2` (status / gate_verdict / gate_decided_at / reclassify_count on `queue`)
- `gatekeeper_daemon.py` running with δ=0.20 / τ=0.75 / REEVAL_AFTER_SEC=90 / REEVAL_MAX=2

## Test scenarios and verdicts

| Scenario | Message | A_score | B_score | importance | correction_of_prior | Verdict | Expected | Match |
|---|---|---:|---:|---:|---|---|---|:---:|
| A literal | "My name is Yuki and I'm a backend engineer in California." | 1.00 | 0.05 | 9 | false | `ready` | `ready` | ✅ |
| B hypothetical | "If I were Napoleon, I would have conquered Russia in winter." | 0.05 | 0.90 | 7 | false | `demoted` | `demoted` | ✅ |
| C ambiguous | "I might be allergic to shellfish, but not sure yet." | 0.80 | 0.20 | 5 | false | `ready` | `pending` or `ready` | ✅ |
| D correction | "Actually, I misspoke — my name is not Yuki, it's Daiki." | 0.80 | 0.90 | 9 | **true** | `ready` (via correction override) | `ready` | ✅ |

All four classification decisions landed as expected.

## Audit trail

- Daemon log: `tail -f /tmp/gatekeeper.log`
  ```
  q=504 init a=1.00 b=0.05 imp=9 corr=False conf=0.98 -> ready (A wins by 0.95)
  q=505 init a=0.05 b=0.90 imp=7 corr=False conf=0.87 -> demoted (B wins by 0.85)
  q=506 init a=0.80 b=0.20 imp=5 corr=False conf=0.76 -> ready (A wins by 0.60)
  q=507 init a=0.80 b=0.90 imp=9 corr=True  conf=0.86 -> ready (correction override (A=0.80, B=0.90, corr=True))
  ```
- Per-message latency: 1.8 s – 8 s (JSON-constrained output with logprobs).
- B (demoted) never made it to the deriver queue's ready set; `napoleon`-related documents = 0 as expected.

## Key fix applied during the run

Initial daemon code rejected D at `|A−B|=0.10 < δ=0.20 → pending`, then on force-commit `A<B → demoted`. That silently dropped corrections. Added two overrides to `gatekeeper_daemon.py`:

1. **Initial decide**: when `correction_of_prior=true` and `A_score ≥ 0.7`, short-circuit to `ready` regardless of B. Rationale: correction phrases ("actually", "I misspoke") naturally raise B (framing marker), but the content is still a literal assertion we want remembered so the deriver can run `supersede_observations`.
2. **Forced commit**: when `correction_of_prior=true`, always land `ready`. Losing a correction silently is worse than over-memorizing.

Both overrides persist the `correction_override: true` flag inside `gate_verdict.audit` for traceability.

## Verdict: PASS

All 4 scenarios hit their expected classifier verdicts after the correction override fix.
