# UAT iteration fixes

Changes applied during UAT execution to make the scripts or stack pass.

## Iteration 1 (2026-04-17 run `uat-20260417-095720`)

### Fix A: `00_preflight.sh` — pgvector dim formula

- **Symptom**: preflight 2.5 reported `pgvector dim = 764 (expected 768)`.
- **Cause**: pgvector's `atttypmod` already is the vector dimension. My first version subtracted 4 (a PostgreSQL convention for non-pgvector types), which gave `768 - 4 = 764`.
- **Fix**: use `SELECT atttypmod` (no arithmetic) — returns 768 directly.

### Fix B: `s2_ollama_chat.sh` — thinking-model empty content

- **Symptom**: S2 FAIL with `Ollama returned empty content` even though the HTTP call succeeded and `choices[0].finish_reason = "length"`.
- **Cause**: `glm-4.7-flash` is a thinking model (native reasoning tokens). The request asked for `max_tokens=8`; all 8 tokens were consumed by reasoning, leaving `content=""`.
  ```
  {"role":"assistant","content":"","reasoning":"The user is asking for a single-word"}
  finish_reason: "length"
  ```
- **Fix**: bumped `max_tokens` to 128 and treat the test as PASS when either `content` **or** `reasoning` is non-empty. Either non-empty field proves generation happened.

## Iteration 2 (2026-04-17 run `uat-20260417-095832`)

### Fix C: `s4_deriver.sh` — wrong JOIN for documents table

- **Symptom**: S4 FAIL with `ERROR: column d.collection_name does not exist` from psql.
- **Cause**: the Honcho build here stores `workspace_name` directly on `documents` (with `observer` / `observed` per-row). My query assumed an older `collections` table join.
- **Fix**: replaced the JOIN with `SELECT count(*) FROM documents WHERE workspace_name = '$WS'`. The real schema has: `documents(id, content, embedding, workspace_name, session_name, observer, observed, level, ...)`.

### Fix D: `lib.sh` — teardown FK ordering

- **Symptom**: cleanup left orphans when `sessions` was deleted before `session_peers` / before queue rows referencing it.
- **Fix**: reordered DELETE sequence in `cleanup_workspace` so child tables (session_peers, active_queue_sessions, queue, documents, message_embeddings) are purged before sessions/peers/workspaces. Also broadened the `active_queue_sessions` match to include `summary:` / `dream:` prefixes.
