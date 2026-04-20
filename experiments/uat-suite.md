# UAT suite — S7 / S8 / S9 gatekeeper regression tests

Three scenario scripts under `test/uat/scripts/` exercise the gatekeeper → deriver → supersede pipeline end-to-end against a running stack. They hit the live Honcho HTTP API, peek at the Postgres schema to read verdicts, and wait for the deriver container to process the queue. No mocks.

Run these after you change anything under `honcho/src/deriver/*`, the gatekeeper daemon (`scripts/gatekeeper_daemon.py`), the `queue.gate_verdict` / `queue.status` schema, or the `supersede_observations` tool wiring. They are the fast way to confirm the three user-visible invariants (literal messages produce observations, hypotheticals are filtered, explicit corrections retract prior facts) still hold.

## Prerequisites

A running Honcho + Bonsai + Ollama stack. The scripts default to `HONCHO_URL=http://localhost:8000`, `BONSAI_URL=http://localhost:8080`, `OLLAMA_URL=http://localhost:11434`; override any of those via env if you run a non-default setup.

```bash
export HERMES_HOME=/home/baba-y/hermes-stack     # MUST be set — lib.sh reads it
# Optional overrides (defaults shown):
# export HONCHO_URL=http://localhost:8000
# export HONCHO_DIR=$HERMES_HOME/honcho
```

The scripts source `test/uat/scripts/lib.sh`, which derives `HONCHO_DIR`, `RESULTS_DIR`, and a new `UAT_RUN_ID` on each invocation, then provides helpers like `honcho_db` (psql via `docker compose exec`), `post_message`, `setup_ws`, `wait_queue_drain`, and `cleanup_workspace`.

## Invocation

Each script runs standalone — they share no state across runs, each creates its own disposable workspace keyed on `UAT_RUN_ID`, and each cleans up on success.

```bash
bash $HERMES_HOME/test/uat/scripts/s7_gatekeeper_e2e.sh
bash $HERMES_HOME/test/uat/scripts/s8_gatekeeper_deriver_e2e.sh
bash $HERMES_HOME/test/uat/scripts/s9_correction_supersede_e2e.sh
```

Exit code is `0` on success, `1` on any assertion failure. On failure, the workspace is still deleted (so the DB stays clean for the next run), but `test/uat/results/<UAT_RUN_ID>/failures.log` captures every assertion that missed.

## `s7_gatekeeper_e2e.sh` — classifier, 4 scenarios

Posts four synthetic messages against a fresh workspace and asserts the gatekeeper's verdict for each. Exercises the classifier daemon against the four shapes it has to distinguish, without waiting for the deriver for most of the assertions.

Message shapes:

- **A (literal)** — `"My name is Yuki and I'm a backend engineer in California."` — expected `ready` with `importance >= 7`.
- **B (hypothetical)** — `"If I were Napoleon, I would have conquered Russia in winter."` — expected `demoted`; after queue drain, no document should mention Napoleon.
- **C (ambiguous)** — `"I might be allergic to shellfish, but not sure yet."` — either `pending` or `ready` passes (the classifier may land on either bucket; both are legitimate).
- **D (correction)** — `"Actually, I misspoke — my name is not Yuki, it's Daiki."` — expected `ready` with `gate_verdict.correction_of_prior = true`.

After classification, waits up to 6 minutes for the deriver to drain any `ready` rows, then checks the `documents` table: documents containing `yuki` are allowed; documents mentioning `napoleon` are a hard fail.

Expected pass output (tail):

```
PASS: A verdict: got 'ready' (expected)
PASS: B verdict: got 'demoted' (expected)
PASS: C verdict ('pending' acceptable — ambiguous)
PASS: D verdict: got 'ready' (expected)
PASS: D correction_of_prior: got 'true' (expected)
PASS: A importance=8 (>=7)
PASS: queue drained
PASS: no 'napoleon' documents (B demoted as expected)
[..:..:..] S7: OK
```

## `s8_gatekeeper_deriver_e2e.sh` — long literal + long hypothetical, full deriver path

Goes deeper than S7: posts a long (>200 tokens) fact-dense literal message first so the deriver actually fires (`REPRESENTATION_BATCH_MAX_TOKENS = 200` gate), waits for the deriver to process it, and reads the representation back via `/v3/workspaces/$WS/peers/$PEER/representation`.

Phase 1 — literal seed `"I'm Alice. I love matcha lattes every morning. I go rock climbing at the Gravity Gym every Sunday..."` triplicated to comfortably clear the token threshold:

- Gatekeeper verdict should be `ready`
- Deriver drains the queue within 5 minutes
- At least one document lands in `documents`
- The representation text matches at least 3 of: `alice`, `matcha`, `climb`, `bonsai`, `kyoto`, `engineer`, `postgres`, `miso`

Phase 2 — long hypothetical starting `"If I were Napoleon Bonaparte..."`:

- Gatekeeper verdict should be `demoted`
- No new document should appear in the next 30 seconds
- Zero documents should mention `napoleon`

Expected pass output (tail):

```
PASS: A gatekeeper verdict: ready
PASS: queue drained
PASS: observations created: 3
PASS: representation contains 6 seed keywords
PASS: B gatekeeper verdict: demoted
PASS: demoted message produced no new observations
PASS: no document mentions Napoleon
[..:..:..] S8: OK
```

## `s9_correction_supersede_e2e.sh` — literal + correction round trip, soft-delete side effects

The heaviest of the three: confirms the Q2-follow-up wiring where a correction message triggers a secondary tool-mode LLM call that soft-deletes retracted observations with `deleted_reason = 'superseded'` and a non-empty `supersede_reason` in `internal_metadata`.

Phase 1 — post a long literal introduction about Emma, including `"I live in Kyoto"`. Wait for deriver drain, then record how many live Kyoto documents exist (`KYOTO_BEFORE`).

Phase 2 — post a correction quadrupled to clear the token gate: `"Actually I need to correct something important. I do NOT live in Kyoto — I moved to Osaka last month..."`.

- Gatekeeper verdict should be `ready`
- `gate_verdict.correction_of_prior` should be `true`
- Deriver drains within 5 minutes, plus 8 seconds of grace so the supersede tool-mode pass can fire

Phase 3 — verify side effects in `documents`:

- At least one row has `deleted_at IS NOT NULL` and `internal_metadata->>'deleted_reason' = 'superseded'`
- If `KYOTO_BEFORE >= 1`, at least one soft-deleted document should mention `kyoto` (strong check — deriver output isn't 100% deterministic, so if phase 1 happened not to produce a Kyoto observation the strong check is skipped and only the weaker "something was superseded" assertion needs to hold)
- A sample `supersede_reason` value is recorded and non-empty

Expected pass output (tail):

```
PASS: A verdict=ready
PASS: A queue drained
PASS: observations created for A: 4
PASS: B verdict=ready
PASS: B correction_of_prior=true
PASS: B queue drained
PASS: at least one observation was superseded
PASS: Kyoto claim specifically superseded
PASS: supersede_reason recorded
[..:..:..] S9: OK
```

## When to run

- After editing anything under `honcho/src/deriver/` (enqueue, prompts, tools handler).
- After changing the gatekeeper daemon `scripts/gatekeeper_daemon.py` or its classifier prompt.
- After schema edits that touch `queue.status` or `queue.gate_verdict`.
- After changing the `supersede_observations` tool definition in `DERIVER_TOOLS` or the CRUD soft-delete path.
- As a smoke on any PR that touches the fork's memory-formation hot path before merging to `baba-yu/nuncstans-honcho` dev.

Typical runtime: S7 ~2-4 min, S8 ~6-8 min, S9 ~8-12 min, depending on deriver throughput.

## Results and logs

Each invocation writes to `test/uat/results/<UAT_RUN_ID>/`:

| File | Contents |
|---|---|
| `run.log` | All `log`/`pass`/`fail` lines interleaved with timestamps, plus the post-run SQL dumps each scenario appends (queue-row status, document snippets). |
| `passes.log` | One line per successful assertion. |
| `failures.log` | One line per missed assertion (only present if a scenario failed). |
| `<scenario>_ollama_ps.log` | Sampled `/api/ps` output during the scenario (used by the `s2_ollama_chat.sh` / `ollama_watch` helpers in `lib.sh`; the S7/S8/S9 scripts don't generate these by themselves, so they're usually empty for gatekeeper runs). |

`UAT_RUN_ID` defaults to `uat-YYYYMMDD-HHMMSS`; set it explicitly if you want grouped results across related scripts:

```bash
export UAT_RUN_ID="gk-regression-$(date +%H%M%S)"
bash $HERMES_HOME/test/uat/scripts/s7_gatekeeper_e2e.sh
bash $HERMES_HOME/test/uat/scripts/s8_gatekeeper_deriver_e2e.sh
bash $HERMES_HOME/test/uat/scripts/s9_correction_supersede_e2e.sh
# -> test/uat/results/gk-regression-HHMMSS/run.log  (accumulates all three)
```

On failure, the `run.log` tail plus the last few SQL dumps are usually enough to identify whether the classifier misfired, the deriver skipped a `ready` row, or the supersede pass failed to soft-delete.
