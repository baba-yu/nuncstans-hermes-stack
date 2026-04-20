# Hermes-Bonsai-Combo Acceptance Test Plan (UAT)

Last updated: 2026-04-17
Target stack: `hermes-stack` (Ollama on GPU + Bonsai-8B on CPU + Honcho on Docker)
Reader expectation: a human tester should be able to execute this document top-to-bottom and decide pass/fail without additional research.

## 0. Goal and scope

This UAT is intended to prove **that the three local inference tiers cooperate end-to-end**. No cloud API or Claude API may be used at any step (no external LLM call may occur).

Concrete acceptance criteria:

| # | Claim to be proven | Tier that should be hit | Tier that must NOT be hit |
|---|---|---|---|
| G1 | User-facing chat inference lands on Ollama (GPU) | Ollama `glm-4.7-flash` | Bonsai |
| G2 | Memory formation (deriver) lands on Bonsai (CPU) | Bonsai `bonsai-8b` | Ollama (as an LLM) |
| G3 | Memory recall (dialectic) lands on Bonsai | Bonsai `bonsai-8b` | Ollama (as an LLM) |
| G4 | Cross-session memory recall works per peer | Honcho + Bonsai | — |

Ollama embedding calls (G2/G3 side-effect) are permitted. As an **LLM**, Ollama is only used for chat.

## 1. Known preconditions

By the start of this UAT, the following has been verified in prior work (see root `REPORT.md`):

- Embedding pipeline works (no 401/400 errors).
- Deriver → Bonsai path successfully wrote one observation to `documents`.
- `POST /v3/workspaces/.../peers/.../representation` returns the raw observation.
- Settings in place: `WORKERS=1`, `REPRESENTATION_BATCH_MAX_TOKENS=1024`, `MAX_EMBEDDING_TOKENS=2048`, `MAX_INPUT_TOKENS=8000`, `MAX_OUTPUT_TOKENS=1500`, `Vector(768)` patch applied.
- Bonsai throughput: 10–15 tokens/sec on CPU. One deriver turn budgets 3–5 minutes.
- Because `REPRESENTATION_BATCH_MAX_TOKENS=1024`, **a single short message will NOT start the deriver immediately**. Use a fact-packed, >1024-token prompt when the deriver path must fire.

## 2. Preflight

### 2.1 Environment variables

```bash
export HERMES_HOME="$HOME/hermes-stack"
export HONCHO_DIR="$HERMES_HOME/honcho"
export OLLAMA_URL="http://localhost:11434"
export BONSAI_URL="http://localhost:8080"
export HONCHO_URL="http://localhost:8000"
export UAT_RUN_ID="uat-$(date +%Y%m%d-%H%M%S)"
```

### 2.2 Service liveness

- Bonsai `/v1/models` returns an entry whose `id == "bonsai-8b"`.
- Ollama `/api/tags` contains both `glm-4.7-flash:latest` and `openai/text-embedding-3-small:latest`.
- Honcho `/openapi.json` returns 200.

### 2.3 Container-level config sanity

`docker compose exec api python3 -c "from src.config import settings; ..."` should print:

```
EMBEDDING_PROVIDER = 'openrouter'
OPENAI_COMPATIBLE_BASE_URL = 'http://host.docker.internal:11434/v1'
VLLM_BASE_URL = 'http://host.docker.internal:8080/v1'
DERIVER.PROVIDER = vllm / bonsai-8b
DERIVER.MAX_OUTPUT_TOKENS = 1500
DERIVER.MAX_INPUT_TOKENS = 8000
DERIVER.REPRESENTATION_BATCH_MAX_TOKENS = 1024
APP.MAX_EMBEDDING_TOKENS = 2048
VECTOR_STORE.DIMENSIONS = 768
```

### 2.4 Backup slots must be stripped

`grep -E "^BACKUP_(PROVIDER|MODEL)" config.toml` must print **nothing**.

### 2.5 pgvector schema width

```sql
SELECT atttypmod FROM pg_attribute
WHERE attrelid = 'message_embeddings'::regclass AND attname = 'embedding';
```

must return `768`. (pgvector stores the vector dimension directly in `atttypmod`.)

## 3. Traffic-assertion strategy

| Observation point | Method |
|---|---|
| Bonsai hit count | diff `bonsai.log` for `print_timing` lines added during the scenario |
| Ollama chat hit | poll `/api/ps` during + after each scenario; `glm-4.7-flash` in the models list means chat ran |
| Ollama embedding hit | same `/api/ps` path, but for `openai/text-embedding-3-small` |
| Honcho queue progress | query `/v3/workspaces/{ws}/queue/status` and inspect DB |

Each scenario snapshots `BONSAI_MARK=$(wc -l < bonsai.log)` at start and diffs on exit.

## 4. Run order

| # | Scenario | Bonsai turns | Expected time |
|---|---|---|---|
| S1 | Pass-through smoke | 0 | 1 min |
| S2 | Ollama direct chat | 0 | 1 min |
| S3 | Mid-size embedding regression | 0 | 1 min |
| S4 | Deriver → Bonsai observation | 1+ | 3–5 min |
| S5 | Dialectic (peer.chat) → Bonsai | 1+ | 2–4 min |
| S6 | Cross-session recall | 1+ | 3–5 min |

Total: ~15–25 minutes. Serial only — `WORKERS=1` means parallelism just backs up the queue.

## 5. Scenario acceptance (summary)

### S1 — Pass-through smoke
- Workspace / peer / session creation all return 2xx.
- Short message POST bumps `message_embeddings` by 1.
- Zero Ollama chat calls.

### S2 — Ollama direct chat
- `POST /v1/chat/completions` against `glm-4.7-flash` returns generated content or reasoning.
- Zero Bonsai calls; Ollama chat hit count increments by 1.

### S3 — Mid-size embedding regression
- Posting ~1500-token content adds one `message_embeddings` row.
- Zero 401/400 errors in api logs.

### S4 — Deriver → Bonsai observation (**UAT anchor 1**)
- Seed with ~1500-token fact-heavy content.
- Queue goes pending/in-progress → drains within 5 minutes.
- `documents` contains one or more rows for the workspace.
- `POST /peers/alice/representation` surfaces at least one of the seed keywords (`matcha`, `climb`, `bonsai`, `bike`, `Kyoto`, `PostgreSQL`, `miso`, `alice`, `engineer`).
- Bonsai `print_timing` count ≥ 1; Ollama chat hit count = 0.

### S5 — Dialectic → Bonsai
- `POST /peers/alice/chat {"reasoning_level":"minimal"}` answers with S4's facts.
- Bonsai ≥ 1; Ollama chat = 0.

### S6 — Cross-session recall (**UAT anchor 2**)
- Create a new session, post a neutral one-liner (no seeded facts).
- `POST /peers/alice/chat {"session_id":"<new>"}` and `/representation {"session_id":"<new>","search_query":"hobbies"}` surface keywords from S4.
- Bonsai ≥ 1; Ollama chat = 0.

## 6. Cumulative negative assertions

After S4/S5/S6:

- Ollama chat hits during those three scenarios = **0** (ignoring S2's legitimate call).
- Bonsai `print_timing` total ≥ 3.
- Ollama `/v1/embeddings` hits ≥ 3.

## 7. Teardown

After each run, delete the scenario workspaces and archive logs to `test/uat/results/$UAT_RUN_ID/`. `cleanup_workspace()` in `lib.sh` handles FK-ordered deletes.

## 8. Final report template

Write to `test/uat/results/$UAT_RUN_ID/REPORT.md`:

```
UAT Run: $UAT_RUN_ID
Date: ...
Preflight: [ ] 2.2 [ ] 2.3 [ ] 2.4 [ ] 2.5
Scenarios: [ ] S1 [ ] S2 [ ] S3 [ ] S4 [ ] S5 [ ] S6
Traffic:
 - Ollama chat hits during S4-S6: _ (expected 0)
 - Bonsai print_timing total:      _ (expected >= 3)
 - Ollama embedding hits total:    _ (expected >= 3)
Final: PASS / FAIL
```
