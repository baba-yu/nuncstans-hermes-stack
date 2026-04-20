# Hermes-Bonsai-Combo Investigation & Fix Report

Record of the investigation, debugging, and fixes that got Honcho → Bonsai-8B working after following the README left it broken.

## 1. Symptom and entry point

- Built the stack as the README described: Bonsai-8B (`llama-server` on :8080, CPU) + Ollama (`glm-4.7-flash` / `nomic-embed-text` on :11434) + Honcho (Docker Compose).
- Every user message from Hermes caused the Honcho `api` container to fail with **401 Authentication failed** in the embedding path.
- Snapshot of the failure as recorded in `error.md`:

```
src.embedding_client - WARNING - Embedding batch failed (attempt 1/3) ... 401 Authentication failed
src.embedding_client - ERROR - Error processing batch after all retries
openai.AuthenticationError: 401 - {'error': 'Authentication failed'}
```

- Bonsai's `/slots` endpoint showed real LLM calls landing (the deriver/dialectic LLM path was actually reaching Bonsai). Only the embedding path was failing; no vectors were being persisted.

## 2. Investigation

### 2.1 Environment sanity check

| Object | Observation |
|---|---|
| `models/Bonsai-8B.gguf` | 1.1 GB, architecture `qwen3` (a Qwen3-8B fine-tune). Supported in bonsai-llama.cpp (`src/llama-arch.cpp:37`). |
| `bonsai-llama.cpp/build/bin/llama-server` | Built (13.5 MB), healthy (`/v1/models` → alias `bonsai-8b`). |
| `ollama list` | `openai/text-embedding-3-small:latest` (digest `0a109f422b47`) already aliased to `nomic-embed-text:latest` via `ollama cp`. |
| `~/hermes-stack/honcho/.env` | Contained three `LLM_EMBEDDING_*` lines and one `OPENAI_COMPATIBLE_BASE_URL` line (no `LLM_` prefix). |

### 2.2 Reading Honcho's embedding client

`src/embedding_client.py`'s `__init__` has exactly **three branches**:

```python
if self.provider == "gemini":
    ...
elif self.provider == "openrouter":
    base_url = settings.LLM.OPENAI_COMPATIBLE_BASE_URL or "https://openrouter.ai/api/v1"
    self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    self.model = "openai/text-embedding-3-small"         # <-- hardcoded
else:  # openai
    self.client = AsyncOpenAI(api_key=api_key)           # <-- no base_url override
    self.model = "text-embedding-3-small"                # <-- hardcoded
```

And `src/config.py:201-216` declares `LLMSettings`:

```python
class LLMSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")
    ...
    OPENAI_COMPATIBLE_API_KEY: str | None = None
    OPENAI_COMPATIBLE_BASE_URL: str | None = None
    EMBEDDING_PROVIDER: Literal["openai", "gemini", "openrouter"] = "openai"
    # NOTE: there is NO EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL field.
```

From these two reads:

1. **`LLM_EMBEDDING_API_KEY`, `LLM_EMBEDDING_BASE_URL`, `LLM_EMBEDDING_MODEL` are NOT fields on `LLMSettings`** — the comment in upstream `honcho-self-hosted/config.toml:51-54` is misleading. Putting those vars in `.env` does nothing.
2. **Only the `openrouter` branch honours a custom base URL.** Using it requires `EMBEDDING_PROVIDER = "openrouter"` in `config.toml` plus **`LLM_OPENAI_COMPATIBLE_API_KEY` / `LLM_OPENAI_COMPATIBLE_BASE_URL`** in `.env` (the `LLM_` prefix is mandatory).
3. **The embedding model name is hardcoded to `"openai/text-embedding-3-small"`**, so the local server must expose that exact name (done via `ollama cp nomic-embed-text openai/text-embedding-3-small`).

The user's `.env` had:
```
OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:11434/v1   # no LLM_ prefix → ignored
```
So `settings.LLM.OPENAI_COMPATIBLE_BASE_URL` was `None`, the client fell back to `"https://openrouter.ai/api/v1"` and authenticated there with the sham key `"ollama"`, producing the 401. That's the direct root cause.

### 2.3 Hardcoded pgvector schema

Once the 401 was removed, the next failure would be a vector-dimension mismatch. The pgvector columns are hardcoded to `Vector(1536)` (OpenAI's `text-embedding-3-small` width):

- `src/models.py:281` — `message_embeddings.embedding`
- `src/models.py:389` — `documents.embedding`
- `migrations/versions/a1b2c3d4e5f6_initial_schema.py:366`
- `migrations/versions/917195d9b5e9_add_messageembedding_table.py:31`
- `migrations/versions/119a52b73c60_support_external_embeddings.py:45, 53`

`[vector_store] DIMENSIONS` is only consulted by LanceDB (`src/vector_store/lancedb.py:96`); pgvector ignores it. Inserting a 768-dim `nomic-embed-text` vector raises `ValueError: expected 1536 dimensions, not 768`, which cascades into `PendingRollbackError`.

### 2.4 Second landmine: `MAX_EMBEDDING_TOKENS`

Honcho's default `[app] MAX_EMBEDDING_TOKENS = 8192` is calibrated for OpenAI's 8192-token embedding limit. `nomic-embed-text`'s native context is **2048**. Messages over 2048 tokens are sent as one un-chunked payload, and Ollama rejects them with `400 - the input length exceeds the context length`. Reproduced with a 2230-token message.

### 2.5 Bonsai context too small

After embeddings were unblocked, the deriver started hitting `400 - the input length exceeds the context length` against Bonsai. The deriver agent iterates tool calls:

- Call 1: 2550-token prompt + 4096-token output (default `MAX_OUTPUT_TOKENS`) → 6645 tokens total.
- Call 2: 6645 + tool result (hundreds of tokens) → over 8192 → llama.cpp returns 400.

Actual bonsai.log:

```
slot print_timing: id 2 | task 11 |
prompt eval time =  71568 ms /  2550 tokens
       eval time = 398554 ms /  4096 tokens (10.28 tok/s)
      total time = 470122 ms /  6646 tokens
slot release: ... stop processing: n_tokens = 6645, truncated = 0
```

The README's `-c 8192` is the next wall after embedding.

### 2.6 Misconfigured backup provider

`config.toml` had `BACKUP_PROVIDER = "custom"` / `BACKUP_MODEL = "bonsai-8b"` on every deriver / dialectic / summary / dream block. When primary (`vllm` = Bonsai) fails, the stack fails over to the backup. But the `custom` slot's base URL points at Ollama (embeddings). Ollama doesn't serve `bonsai-8b`, so the fallback raises `NotFoundError`, which the deriver wraps in `RetryError`:

```
deriver - WARNING - Final retry attempt 3/3: switching from vllm/bonsai-8b to backup custom/bonsai-8b
deriver - ERROR - Error processing representation batch for work unit representation:probe2:s1:bob: RetryError[<Future ... raised NotFoundError>]
```

Local-only mode should be single-provider; the upstream `honcho-self-hosted/setup.sh` also strips the backup lines in local mode (`sed -i '/^BACKUP_PROVIDER/d; /^BACKUP_MODEL/d'`).

## 3. Fixes

### 3.1 `~/hermes-stack/honcho/.env`

```diff
-LLM_EMBEDDING_API_KEY=ollama
-LLM_EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1
-LLM_EMBEDDING_MODEL=nomic-embed-text
 LLM_OPENAI_COMPATIBLE_API_KEY=ollama
-OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:11434/v1
+LLM_OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:11434/v1
```

### 3.2 `~/hermes-stack/honcho/config.toml`

```diff
 EMBED_MESSAGES = true
-MAX_EMBEDDING_TOKENS = 8192
+MAX_EMBEDDING_TOKENS = 2048
 ...
 [deriver]
-MAX_OUTPUT_TOKENS = 4096
-MAX_INPUT_TOKENS = 23000
+MAX_OUTPUT_TOKENS = 1500
+MAX_INPUT_TOKENS = 8000
```

`EMBEDDING_PROVIDER = "openrouter"` stays as-is (scaffold default already matches).

Remove all backup slots (local deployment is single-provider):

```bash
sed -i '/^BACKUP_PROVIDER/d; /^BACKUP_MODEL/d' ~/hermes-stack/honcho/config.toml
```

### 3.2.1 Start Bonsai with a bigger context

```bash
./build/bin/llama-server \
  -m $HOME/hermes-stack/models/Bonsai-8B.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 0 -c 16384 --alias bonsai-8b \
  > $HOME/hermes-stack/bonsai.log 2>&1 &
```

### 3.3 pgvector schema patch

Drop the existing DB volume first, then patch the source and rebuild:

```bash
cd ~/hermes-stack/honcho
docker compose down -v
sed -i 's/Vector(1536)/Vector(768)/g' \
  src/models.py \
  migrations/versions/a1b2c3d4e5f6_initial_schema.py \
  migrations/versions/917195d9b5e9_add_messageembedding_table.py \
  migrations/versions/119a52b73c60_support_external_embeddings.py
docker compose up -d --build
```

`--build` bakes the change into the api/deriver images and Alembic recreates the tables at the new width.

## 4. Verification

### 4.1 Container-level settings

```bash
docker compose exec api python3 -c "from src.config import settings; \
  print('EMBEDDING_PROVIDER=', repr(settings.LLM.EMBEDDING_PROVIDER)); \
  print('OPENAI_COMPATIBLE_BASE_URL=', repr(settings.LLM.OPENAI_COMPATIBLE_BASE_URL))"
# EMBEDDING_PROVIDER= 'openrouter'
# OPENAI_COMPATIBLE_BASE_URL= 'http://host.docker.internal:11434/v1'
```

### 4.2 Embedding succeeds

POSTing 11-, 1118-, and 1444-token messages drove `message_embeddings` from 0 → 1 → 2 → 3 with no 401/400 in `api` logs. Only the 2230-token message failed before `MAX_EMBEDDING_TOKENS = 2048` was applied. Post-fix, the same message goes through (chunked).

```sql
SELECT id, session_name, LEFT(content, 50) FROM message_embeddings ORDER BY id;
-- 1 | s1 | I love matcha and go rock climbing every Sunday.
-- 2 | s1 | I love matcha ... (1118 tokens)
-- 3 | s1 | I love matcha ... (1444 tokens, carol/probe3)
```

### 4.3 Deriver → Bonsai pipeline

`REPRESENTATION_BATCH_MAX_TOKENS = 1024` means small messages just accumulate (it can look stalled). For a message over 1024 tokens the flow is:

- A row lands in `queue` with key `representation:{ws}:{sess}:{peer}`.
- The deriver claims it and writes `active_queue_sessions`.
- Bonsai's `/slots` flips the relevant slot to `is_processing=true`.
- `llama-server` hits 500–800% CPU.
- `bonsai.log` progresses through `prompt processing progress ... progress = 0.80 … 1.00` → generation.

Final check: POST a 1444-token message for `carol` in workspace `probe3`:

```
⚡ PERFORMANCE - minimal_deriver_4_carol
  Llm Call Duration       95182  ms
  Total Processing Time   96077  ms
  Observation Count           1  count
```

- `queue.processed = t`, `error = null`
- 1 explicit observation row in `documents`
- `POST /v3/workspaces/probe3/peers/carol/representation` returns `"## Explicit Observations\n\n[2026-04-17 16:15:12] I love matcha ..."`

The full **embedding → deriver → Bonsai → observation → representation** roundtrip works live.

## 5. README changes

The following sections were out of sync with the implementation and have been updated (see `git diff README.md`):

- **Step 2 (Ollama)** — add `ollama cp nomic-embed-text openai/text-embedding-3-small` alias step.
- **Step 3 `.env`** — drop the ignored `LLM_EMBEDDING_*` vars, switch to `LLM_OPENAI_COMPATIBLE_*` (with the mandatory `LLM_` prefix), and add the background explanation.
- **Step 3 `config.toml`** — set `[app] MAX_EMBEDDING_TOKENS = 2048`; lower deriver `MAX_OUTPUT_TOKENS` / `MAX_INPUT_TOKENS` so tool iterations fit Bonsai's context; strip `BACKUP_*` lines.
- **Step 3 patches on top of the scaffold** — grew from 2 items to 3; adds the `Vector(1536)` → `Vector(768)` sed and the need for `docker compose down -v` before rebuild.
- **Start and verify** — switched to `docker compose up -d --build`; added an in-container settings check.
- **Troubleshooting** — added four entries: 401 authentication, model-not-found, input-length-exceeds-context-length, dimension mismatch.

## 6. Known follow-ups

1. **Upstream `config.toml` comment is misleading.** `elkimek/honcho-self-hosted`'s scaffold (`config.toml:51-54`) implies `LLM_EMBEDDING_*` are wired — they aren't. Worth an upstream PR/issue, though this README works around it.
2. **`embedding_client.py` design.** Hardcoded model name + base-URL override only in the `openrouter` branch is unfriendly to local-first deployments. A `plastic-labs/honcho` PR that lets the embedding provider take an independent `base_url` / `model` (matching the config comment that already describes that shape) would be valuable.
3. **pgvector column width is hardcoded.** `src/models.py` should read `Vector(settings.VECTOR_STORE.DIMENSIONS)` and Alembic migrations should parameterise the width. Upstream candidate.
4. **Bonsai on CPU is slow.** A 2550-token prompt takes minutes including generation. Long sessions can backlog the deriver. `--parallel 1` is fine; the default prompt cache (`--cache-ram 8192` = 8 GiB) keeps amortisation reasonable.

## 7. References

- `error.md` — initial snapshot captured at the 401 stall.
- `README.md` — revised in lockstep with this report.
- `bonsai.log` — llama-server runtime log.
- `bonsai-build.log` — llama.cpp build log.
- `git log` — full change set corresponding to this report.
