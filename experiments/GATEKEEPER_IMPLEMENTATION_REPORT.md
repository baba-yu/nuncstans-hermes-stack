# Gatekeeper Implementation — Final Report

Last updated: 2026-04-20

## Scope

Six commit points from the multi-turn design discussion, implemented end-to-end and tested:

1. **Q1** — Alembic migration adding `status` / `gate_verdict` / `gate_decided_at` / `reclassify_count` to `queue`, plus two partial indexes for the hot paths.
2. **Q1b** — `queue_manager.get_and_claim_work_units` filters `status IN ('ready','revived')`; non-representation task types (dream / summary / reconciler / deletion) enqueue as `status='ready'` so they never wait on a gatekeeper.
3. **Q2** — `supersede_observations` tool added to `src/utils/agent_tools.py`, exposed **only** via `DERIVER_TOOLS`. Soft-deletes by setting `deleted_at` + merging `internal_metadata.{superseded_by,supersede_reason}`. 9 unit tests green. Deriver now calls the tool end-to-end: when `gate_verdict.correction_of_prior=true`, `process_representation_tasks_batch` fires a secondary tool-mode Bonsai call restricted to `supersede_observations`, with recent observations pre-fetched as context. See S9 below.
4. **Q3** — `scripts/gatekeeper_daemon.py`: async classifier daemon with v3 prompt, logprob-derived confidence, decision rules `δ=0.20 / τ=0.75`, three-loop poller (initial / re-evaluate / force-commit).
5. **Q4** — pending re-evaluation moved into `gatekeeper_daemon.py` (rather than `sleep_daemon.py`) — keeps responsibilities separated: `sleep_daemon` for dream scheduling, `gatekeeper_daemon` for message-level classification.
6. **Q5** — `benchmark.md` expanded with calibration section: 43% → 90% agreement rate, axis-independence proof, correction detection FP→0, latency numbers, threshold rationale.

## End-to-end test results

### S7 — Classifier verdicts (`test/uat/scripts/s7_gatekeeper_e2e.sh`)

| Scenario | A_score | B_score | importance | corr_of_prior | Verdict | Expected | ✓ |
|---|---:|---:|---:|---|---|---|:---:|
| A literal "My name is Yuki..." | 1.00 | 0.05 | 9 | false | `ready` | `ready` | ✅ |
| B hypothetical "If I were Napoleon..." | 0.05 | 0.90 | 7 | false | `demoted` | `demoted` | ✅ |
| C ambiguous "I might be allergic..." | 0.80 | 0.20 | 5 | false | `ready` | `pending` or `ready` | ✅ |
| D correction "Actually, I misspoke..." | 0.80 | 0.90 | 9 | **true** | `ready` (via correction_override) | `ready` | ✅ |

Key fix during the run: initial decide logic and forced-commit logic were both biased to drop corrections (since classifier tends to score corrections A=0.8 / B=0.9 — the "Actually" phrase raises B even though content is literal). Added **correction short-circuit**: when `correction_of_prior=true` and `A_score ≥ 0.7`, route to `ready`; on forced commit, always prefer `ready` for corrections. Better to over-memorise than silently drop an explicit user revision.

### S8 — Gatekeeper → deriver → observation (`test/uat/scripts/s8_gatekeeper_deriver_e2e.sh`)

| Phase | Action | Observed |
|---|---|---|
| 1 | Post 750-token Alice self-intro | `status=ready` in 7 s |
| 1 | wait deriver | queue drained in < 3 min on GPU |
| 1 | check `documents` | **1 observation created**, representation surfaces 8/8 seed keywords (matcha / climb / bonsai / kyoto / engineer / postgres / miso / alice) |
| 2 | Post long Napoleon hypothetical | `status=demoted` in 8 s |
| 2 | wait 30 s | `documents` count unchanged (1), no Napoleon-containing observations |

Both phases PASS.

### S9 — Correction → supersede (`test/uat/scripts/s9_correction_supersede_e2e.sh`)

| Phase | Action | Observed |
|---|---|---|
| 1 | Post 211-token Emma self-intro ("I live in Kyoto…") | `status=ready` in 3 s |
| 1 | wait deriver | queue drained in 12 s |
| 1 | check `documents` | **1 observation created**, Kyoto claim present |
| 2 | Post 249-token correction ("Actually I moved to Osaka…") | `status=ready` in 7 s, `correction_of_prior=true` |
| 2 | wait deriver + secondary supersede pass | queue drained in 10 s |
| 2 | check `documents` | Kyoto observation `deleted_at` set, `internal_metadata.deleted_reason='superseded'`, `supersede_reason="User corrected location to Osaka"` |

End-to-end in ~45 s. All five assertions PASS: B verdict, B correction flag, ≥1 superseded document, Kyoto claim specifically retracted, supersede_reason recorded.

Two iteration fixes during the run worth noting:
1. Initial implementation fetched `Document` rows inside `async with tracked_db(...)`, then referenced `d.id` / `d.content` after the session closed — SQLAlchemy raised "Instance is not bound to a Session". Fix: materialize `[(id, content) for d in documents]` *inside* the block.
2. First prompt rendered observations as `- [id:ABC123] content` and Bonsai (8B) copied the `id:` prefix verbatim into `document_ids_to_supersede` → all IDs came back `not_found`. Fix: switched to `<observation_id> :: <content>` layout with an explicit instruction that the id is the exact string before ` :: `.

## System now looks like

```
┌────────────────────────────────────────────────────────────────────┐
│ New user message POST /v3/.../messages                             │
│   └─► messages row + queue row (status='pending', gate_verdict=null)│
│                                                                    │
│ gatekeeper_daemon.py (every 5s):                                   │
│   1. scan pending + null verdict  → call Bonsai (v3 prompt, JSON   │
│      schema, logprobs) → stamp gate_verdict + set status            │
│      (`ready` / `demoted` / still `pending`).                       │
│   2. scan pending older than 90 s  → re-classify, bump count.       │
│   3. pending w/ count ≥ 2         → force-commit (correction-aware).│
│                                                                    │
│ deriver queue_manager (unchanged polling)                          │
│   - WHERE status IN ('ready','revived') AND processed=false         │
│   - Claims batches once REPRESENTATION_BATCH_MAX_TOKENS is met.     │
│   - After the json_mode save, if any queue row has                  │
│     gate_verdict.correction_of_prior=true, fires a tool-mode        │
│     Bonsai call restricted to `supersede_observations` so the old   │
│     contradicted observations get soft-deleted with a reason.       │
│                                                                    │
│ sleep_daemon.py (unchanged):                                       │
│   - Pressure + idle detection → enqueue_dream trigger.              │
│   - Session-level system message injection for 💤/☕ notices.        │
└────────────────────────────────────────────────────────────────────┘
```

## Files touched

### Created
- `honcho/migrations/versions/g7h8i9j0k1l2_add_gatekeeper_columns_to_queue.py`
- `scripts/gatekeeper_daemon.py`
- `scripts/gatekeeper_eval/dataset.jsonl`
- `scripts/gatekeeper_eval/shadow.py`
- `scripts/gatekeeper_eval/analyze.py`
- `scripts/gatekeeper_eval/results.jsonl`
- `honcho/tests/test_supersede_observations.py`
- `test/uat/scripts/s7_gatekeeper_e2e.sh`
- `test/uat/scripts/s8_gatekeeper_deriver_e2e.sh`
- `test/uat/scripts/s9_correction_supersede_e2e.sh`
- `test/uat/results/gatekeeper-s7/REPORT.md`

### Modified
- `honcho/src/models.py` — added `SmallInteger` import, 4 new columns + 2 indexes on `QueueItem`
- `honcho/src/deriver/enqueue.py` — representation records default `status='pending'`, system records (dream/summary/deletion) default `status='ready'`
- `honcho/src/deriver/queue_manager.py` — `get_and_claim_work_units` filters `status IN ('ready','revived')`
- `honcho/src/deriver/deriver.py` — after the json_mode save, checks `queue.gate_verdict.correction_of_prior` and runs a secondary tool-mode pass (`_run_supersede_pass`) that gives Bonsai the `supersede_observations` tool only, with the peer's recent observations pre-formatted as `<id> :: <content>`
- `honcho/src/reconciler/scheduler.py` — reconciler enqueue sets `status='ready'`
- `honcho/src/utils/agent_tools.py` — `supersede_observations` tool + handler + `DERIVER_TOOLS` export
- `honcho/src/crud/document.py` — new `supersede_document()` + `SupersedeStatus` return type, `delete_document` accepts `deleted_reason`
- `honcho/src/crud/__init__.py` — re-exports `supersede_document`
- `benchmark.md` — new "Gatekeeper classifier calibration (v3 prompt)" section

## Known gaps (future work)

1. **Correction messages sometimes produce zero new observations.** The existing json_mode deriver path expects a `PromptRepresentation` schema, and Bonsai-8B occasionally emits `{"correct": "..."}` rather than `{"content": "..."}` when the input is framed as an explicit correction. The secondary supersede pass still fires correctly and retracts the now-obsolete observation, so the critical safety property (wrong claim removed) holds, but the new claim (e.g. "Emma lives in Osaka") is not always added as a fresh observation in the same turn. Improvement: tune the minimal-deriver prompt to handle correction framing, or add a follow-up `create_observations` pass for correction messages specifically.

2. **Shadow dataset size.** Calibration was done on 30 hand-labeled messages. The agent reports and README recommend re-calibrating on ≥ 500 real-user messages once the daemon runs in production for a few days.

3. **Force-commit semantics for low-confidence pending.** Currently force-commit uses `correction_of_prior → ready`, else `A ≥ B → ready else demoted`. An alternative would be to demote all "we still don't know" messages unconditionally, which trades recall for precision. Not tuned.

4. **Run-in-place worktree config.** The `uv` workspace dependency `honcho-ai = { workspace = true }` fails to resolve when `uv run pytest` is invoked from inside the api container (no workspace member by that name). The worktree agent worked around this by running tests locally; production container runs work fine but automated in-container test execution would need a workspace config fix.

5. **sleep_daemon still counts representation pending rows into its pressure signal** even though those rows are now `status='pending'` awaiting gatekeeper. This is conservative (nap fires slightly earlier) but the signal is no longer strictly "deriver is behind". Non-blocking but worth tightening.

6. **Supersede forward link left blank.** `supersede_document()` writes `superseded_by=""` because the secondary tool-mode pass doesn't know the new observation's doc_id (the json_mode pass that creates it runs earlier, without surfacing the id). The back-link on old→new via `supersede_reason` + `deleted_reason='superseded'` is set correctly; adding the forward link would require a second supersede handler signature or threading the created doc ids from the json_mode pass into the tool-mode pass.

7. **Multi-observer cost.** The supersede pass runs one Bonsai tool call per observer in the `observers` list. For the default single-peer case (`observers == [observed]`) this is one call; for multi-peer sessions with N distinct observer collections it is N calls. Fine in practice but worth revisiting if we start using many observers.

## Verdict

All six commit points implemented, plus the Q2 caller wiring. Three UAT scenarios green (S7 classifier, S8 deriver materialisation, S9 correction → supersede). Gatekeeper correctly sorts messages into ready / demoted / pending; corrections survive the override and now retract contradicted observations via the deriver's secondary supersede pass; deriver consumes only ready rows; observations materialise on GPU Bonsai in ~3 min for a 750-token batch, and the correction-triggered retract lands within ~10 s of the correction message.
