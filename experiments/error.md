# Error notes — Honcho embedding 401

Snapshot of where we paused. Resume from "Next steps" when picking this back up.

## Stack state at pause

| Component | Status |
|---|---|
| Bonsai-8B (`llama-server` on :8080, CPU) | Running, healthy. `/v1/models` returns the `bonsai-8b` alias. `/slots` shows real traffic (`is_processing: true`) when Honcho's dialectic / deriver fires. |
| Ollama (:11434, GPU) | Running. `glm-4.7-flash` and `nomic-embed-text` pulled. Ollama-side aliases: `openai/text-embedding-3-small` → same digest as `nomic-embed-text` (created with `ollama cp`). `/v1/embeddings` returns a 768-dim vector when hit directly. |
| Honcho (docker compose in `~/hermes-stack/honcho`) | api / deriver / postgres / redis all up and healthy. `docker compose up -d --force-recreate` applied after each config change. |
| Hermes Agent | Installed, memory provider set to `honcho`. Has been used once (workspace `hermes`, session `baba-y` appears in logs). |

## Primary error

Every message POST to Honcho (both via API directly and via Hermes) triggers the same chain in the `api` container:

```
src.embedding_client - WARNING - Embedding batch failed (attempt 1/3), retrying in 1s:
  Error code: 401 - {'error': 'Authentication failed'}
src.embedding_client - WARNING - Embedding batch failed (attempt 2/3), retrying in 2s: ...
src.embedding_client - ERROR - Error processing batch after all retries
openai.AuthenticationError: Error code: 401 - {'error': 'Authentication failed'}
src.crud.message - ERROR - Failed to generate message embeddings for N messages ...
```

And on the dialectic side (also api container):

```
src.dialectic.core - WARNING - Failed to prefetch observations:
  Error code: 401 - {'error': 'Authentication failed'}
```

Deriver still does some work — `/slots` on llama-server shows real LLM calls landing on Bonsai. But the message-embedding path never succeeds, so vectors are never stored, and observation prefetch fails.

## Root cause

`src/embedding_client.py` in the plastic-labs/honcho tree at `~/hermes-stack/honcho/` only supports three providers:

```python
if self.provider == "gemini":
    ...
elif self.provider == "openrouter":
    api_key = settings.LLM.OPENAI_COMPATIBLE_API_KEY
    base_url = settings.LLM.OPENAI_COMPATIBLE_BASE_URL or "https://openrouter.ai/api/v1"
    self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    self.model = "openai/text-embedding-3-small"   # hardcoded
else:  # openai
    api_key = settings.LLM.OPENAI_API_KEY
    self.client = AsyncOpenAI(api_key=api_key)     # NO base_url override
    self.model = "text-embedding-3-small"          # hardcoded
```

Consequences:

- The `LLM_EMBEDDING_API_KEY`, `LLM_EMBEDDING_BASE_URL`, `LLM_EMBEDDING_MODEL` env vars that the scaffolding documentation (and the comment in `config.toml`) claim are used are **not read by `embedding_client.py` at all**. The comment is wrong or stale.
- `provider = "openai"` hits `api.openai.com` with whatever key we give — so our `not-needed` key produces the 401.
- `provider = "openrouter"` is the only path that honours a custom `base_url`, and it reads it from `settings.LLM.OPENAI_COMPATIBLE_BASE_URL` — which in turn is populated from an env var whose exact name still needs to be confirmed (see Next steps).
- The model name is hardcoded per branch, so any custom endpoint must expose a model named exactly `openai/text-embedding-3-small` (handled via `ollama cp nomic-embed-text openai/text-embedding-3-small`).

## Attempts and outcomes

1. **Initial setup** — followed README with `EMBEDDING_PROVIDER = "openrouter"` (as shipped by elkimek/honcho-self-hosted) and `LLM_EMBEDDING_*` env vars pointing at Ollama. → 401 on every embed call. Ollama itself is fine, so the client never went to Ollama.
2. **Switched `EMBEDDING_PROVIDER = "openai"`** and added `OPENAI_BASE_URL` / `OPENAI_COMPATIBLE_BASE_URL` to `.env`. → still 401. The `openai` branch passes no `base_url` to `AsyncOpenAI`, so the client goes to `api.openai.com` with our fake key.
3. **Reverted to `EMBEDDING_PROVIDER = "openrouter"`**, created `ollama cp nomic-embed-text openai/text-embedding-3-small`, set `LLM_OPENAI_COMPATIBLE_API_KEY=ollama` and `OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:11434/v1`. → still 401. This means `settings.LLM.OPENAI_COMPATIBLE_BASE_URL` is empty, so the client still falls through to `https://openrouter.ai/api/v1`.

Direct smoke test that succeeded: Ollama answers `/v1/embeddings` with the aliased model name and a `Bearer ollama` token.

```bash
curl -s -X POST http://localhost:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer ollama' \
  -d '{"model":"openai/text-embedding-3-small","input":"hi"}'
# -> returns a 768-dim embedding
```

So the problem is purely about which env var name Pydantic Settings maps to `settings.LLM.OPENAI_COMPATIBLE_BASE_URL`.

## Next steps when resuming

1. Confirm what Python actually sees inside the api container:

   ```bash
   cd ~/hermes-stack/honcho
   docker compose exec api python3 -c "from src.config import settings; \
     print('provider=', repr(settings.LLM.EMBEDDING_PROVIDER)); \
     print('compat_url=', repr(settings.LLM.OPENAI_COMPATIBLE_BASE_URL)); \
     print('compat_key=', repr(settings.LLM.OPENAI_COMPATIBLE_API_KEY))"
   ```

2. If `compat_url` is empty / `None`, the settings class almost certainly uses `env_prefix="LLM_"`. Add the prefixed variant to `.env`:

   ```
   LLM_OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:11434/v1
   ```

   Then `docker compose up -d --force-recreate api deriver` and re-run step 1.

3. Cross-check by reading the settings class directly:

   ```bash
   grep -nE "OPENAI_COMPATIBLE_BASE_URL|env_prefix|class LLM" ~/hermes-stack/honcho/src/config.py
   ```

4. When the 401 is gone, confirm the full pipeline by posting a message and looking at `alice`'s representation:

   ```bash
   curl -s -X POST http://localhost:8000/v3/workspaces/probe/sessions/s1/messages \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"peer_id":"alice","content":"matcha and rock climbing"}]}'
   sleep 60
   curl -s -X POST http://localhost:8000/v3/workspaces/probe/peers/alice/representation \
     -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
   ```

## Open questions / follow-ups

- The README's `.env` recipe (`LLM_EMBEDDING_*`) is misleading because `embedding_client.py` never reads those vars. Once the correct env-var name is confirmed, the README should be updated.
- The model-name hardcoding in `embedding_client.py` means any local embedding backend must be aliased to `openai/text-embedding-3-small`. Worth a note in the troubleshooting section.
- Consider opening an upstream issue or PR on plastic-labs/honcho to honour `LLM_EMBEDDING_BASE_URL` / `LLM_EMBEDDING_MODEL` as the comment in `config.toml` already claims.
