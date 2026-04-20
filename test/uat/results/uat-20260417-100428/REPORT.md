# Final UAT Report

**UAT Run:** `uat-20260417-100428`
**Date:** 2026-04-17 10:04:28 – 10:12:51 JST (≈ 8 minutes)
**Result:** **PASS**
**Stack under test:** Hermes-Bonsai-Combo (Ollama on GPU × Bonsai on RAM × Honcho on Docker)
**Models (all local, no cloud/Claude API):**
- Ollama `glm-4.7-flash` (chat, VRAM) + `openai/text-embedding-3-small` alias of `nomic-embed-text` (embedding)
- Bonsai-8B (GGUF, CPU) served by `llama-server`

## 1. Scenario results

| # | Scenario | Status | Elapsed | Key observations |
|---|---|:---:|---:|---|
| 00 | Preflight | PASS | 2s | All 9 checks green |
| S1 | Pass-through smoke | PASS | 11s | Short POST → +1 embedding row, Bonsai untouched |
| S2 | Ollama direct chat | PASS | 11s | `glm-4.7-flash` returned content + reasoning; Bonsai untouched |
| S3 | Mid-size embedding regression | PASS | 22s | 1601-token POST embedded cleanly, no 401/400 |
| S4 | Deriver → Bonsai observation | PASS | 255s | 1 observation saved; representation matched 9 seed keywords |
| S5 | Dialectic → Bonsai | PASS | 133s | Dialectic produced the hobby list; 4 seed keywords matched |
| S6 | Cross-session recall | PASS | 68s | 8 seed keywords recalled from a new session |
| 99 | Final cumulative assertions | PASS | 1s | Zero Ollama chat leaks; Bonsai hit 6×; 5 document rows |

**Total wall-clock time:** 503 seconds (~ 8 minutes)

## 2. Acceptance criteria

| ID | Requirement | Result | Evidence |
|---|---|:---:|---|
| G1 | User chat inference lands on Ollama (GPU) | **PASS** | S2 loaded `glm-4.7-flash`; no chat calls occurred during S4–S6 |
| G2 | Memory formation (deriver) lands on Bonsai (CPU) | **PASS** | S4: Bonsai `print_timing` +2, documents +1, representation matched 9/9 seeds |
| G3 | Memory recall (dialectic) lands on Bonsai | **PASS** | S5: Bonsai `print_timing` +1, dialectic output contained the observed facts |
| G4 | Cross-session memory recall | **PASS** | S6 reconstructed `matcha / climb / bonsai / bike / kyoto / postgres / miso / engineer` from a new session |

## 3. Sample responses (excerpts)

### S4 representation (deriver output)

```
## Explicit Observations
[2026-04-17 17:05:15] My name is Alice. I love matcha lattes every morning.
I go rock climbing at the Gravity Gym every Sunday. I ride a red road bike
to the gym. I collect bonsai trees, especially junipers. I live in Kyoto.
I work as a backend engineer. I prefer PostgreSQL over MySQL. My cat is
named Miso. ...
```

### S5 dialectic (peer.chat minimal — produced by Bonsai)

```
Alice has the following hobbies:

1. Rock climbing at the Gravity Gym every Sunday.
2. Riding a red road bike to the gym.
3. Collecting bonsai trees, especially junipers.
4. Liking matcha lattes every morning.
```

### S6 cross-session chat (recalled from a new session — produced by Bonsai)

```
Alice loves matcha lattes every morning. She goes rock climbing at the
Gravity Gym every Sunday and rides a red road bike to the gym. She
collects bonsai trees, especially junipers. She lives in Kyoto and works
as a backend engineer. She prefers PostgreSQL over MySQL. Her cat is
named Miso.
```

## 4. Traffic observations

| Metric | Value | Verdict |
|---|---:|---|
| Bonsai `print_timing` total | 6 (≥ 3) | PASS |
| Ollama chat loaded in S4–S6 windows | 0 | PASS (no leaks) |
| `documents` (observations) total | 5 | PASS |
| `message_embeddings` total | 10 | PASS |
| Chat models seen on `/api/ps` during S1–S6 | 0 (excluding the S2 window) | PASS |

## 5. Preflight snapshot

```
EMBEDDING_PROVIDER         = 'openrouter'
OPENAI_COMPATIBLE_BASE_URL = 'http://host.docker.internal:11434/v1'
VLLM_BASE_URL              = 'http://host.docker.internal:8080/v1'
DERIVER.PROVIDER/MODEL     = vllm / bonsai-8b
DERIVER.MAX_OUTPUT_TOKENS  = 1500
DERIVER.MAX_INPUT_TOKENS   = 8000
DERIVER.REPR_BATCH         = 1024
APP.MAX_EMBEDDING_TOKENS   = 2048
VECTOR_STORE.DIMENSIONS    = 768
BACKUP_PROVIDER            = None
pgvector schema            = Vector(768) ✓
hermes CLI                 = Hermes Agent v0.10.0 (2026.4.16) ✓
```

## 6. Iteration history

| Iter | Failure | Fix |
|---|---|---|
| 1 | preflight 2.5 `pgvector dim = 764` | `atttypmod - 4` → `atttypmod` (pgvector's typmod is the dim itself) |
| 1 | S2 `Ollama returned empty content` | `glm-4.7-flash` is a thinking model; bumped `max_tokens=8` → 128 and count either `content` or `reasoning` as generation |
| 2 | S4 `column d.collection_name does not exist` | This Honcho build stores `workspace_name` directly on `documents`; no `collections` join needed |
| 2 | teardown FK ordering | Delete child tables before parents; broaden `active_queue_sessions` prefix match to include `summary:` / `dream:` |
| 3 | none | All scenarios PASS |

See `test/uat/fixes/FIXES.md` for details.

## 7. Artifacts

```
test/uat/
├── plan/
│   └── PLAN.md                     ... acceptance test plan
├── scripts/
│   ├── lib.sh                      ... shared library
│   ├── 00_preflight.sh             ... preconditions
│   ├── s1_smoke.sh                 ... S1
│   ├── s2_ollama_chat.sh           ... S2
│   ├── s3_embed_regression.sh      ... S3
│   ├── s4_deriver.sh               ... S4 (UAT anchor 1)
│   ├── s5_dialectic.sh             ... S5
│   ├── s6_cross_session.sh         ... S6 (UAT anchor 2)
│   ├── 99_final_assertions.sh      ... cumulative assertions
│   ├── run_all.sh                  ... orchestrator
│   └── watch_memory.sh             ... live pipeline watcher for manual verification
├── fixes/
│   └── FIXES.md                    ... iteration fix log
└── results/uat-20260417-100428/
    ├── REPORT.md                   ... this document
    ├── full_run.log                ... orchestrator log
    ├── run.log                     ... scenario-level log
    ├── scenarios.tsv               ... pass/fail table
    ├── passes.log, failures.log
    ├── config_inside_container.json
    ├── s4_deriver_representation.txt
    ├── s5_dialectic_content.txt
    ├── s6_cross_session_chat_content.txt
    └── s6_cross_session_representation.txt
```

## 8. Verdict

> **PASS** — Ollama (GPU) and Bonsai (CPU) cooperate end-to-end via Honcho. Zero cloud / Claude API calls were made at any step.
