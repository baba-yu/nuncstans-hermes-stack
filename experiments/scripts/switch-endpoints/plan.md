> This is an English translation of the Japanese source plan at `/home/baba-y/.claude/plans/honcho-hermes-tender-metcalfe.md`. If they diverge, the Japanese file is authoritative.

# `scripts/switch-endpoints.py` — Conversational endpoint + model switcher

## Context

Switching the Honcho and Hermes backends currently requires hand-editing 10 places in `honcho/config.toml`, running `hermes config set` three times, and recreating the Honcho container — all done manually. A slip silently stalls the deriver / dialectic. We want to consolidate this into a single conversational tool.

On top of that, when **the chat model itself is being swapped** (for example, moving from the current qwen3.6-35b to llama3.1-70b), it is not enough to replace the endpoint and model name. You also need to retune `-c` (context length), `-ot` (MoE offload), `-ngl` (GPU layer count), and a whole set of parameters that move with the model's characteristics, such as Honcho's `GET_CONTEXT_MAX_TOKENS`. **If the embedding model's dimensionality changes, a pgvector DB migration is also required.** The switcher takes care of all of this in one place.

## Inventory of model-linked parameters

### A. Parameters that need adjusting when the chat model changes

| Location | Key | What it affects | How it is chosen |
|---|---|---|---|
| `scripts/llama-services.sh` | `-hf <spec>` / `-m <path>` | Which model gets loaded | User input |
| Same | `--alias <name>` | The model name exposed on the API side | Derived from the spec (a short form like `qwen3.6-35b`) |
| Same | `-c <ctx>` | KV cache size | **auto**: `min(n_ctx_train, 131072)` (capped by the VRAM budget) |
| Same | `-ngl <N>` | Number of layers placed on the GPU | **auto**: params ≤ 15B → 99 / larger → ask interactively |
| Same | `-ot "ffn_(up\|down\|gate)_exps=CPU"` | CPU offload for MoE experts | **auto**: added for MoE, omitted for dense |
| Same | `--reasoning off` | Suppress reasoning mode on qwen3-family models | **auto**: added only when arch is qwen3* |
| Same | `-ctk q8_0 -ctv q8_0` / `-fa on` / `--jinja` / `--parallel 2` | Shared performance/compatibility flags | Left as-is (model-independent) |
| `honcho/config.toml` | `[app] GET_CONTEXT_MAX_TOKENS` | Max context handed to the dialectic | **auto**: capped at `c - 20_000` (reserving room for output & system) |
| Same | `[dialectic] MAX_INPUT_TOKENS` | Same as above | Kept in sync by the same logic |
| Same | `[deriver.model_config] max_output_tokens` | The model's output ceiling | **auto**: 1500–8192 depending on the model (can be raised for things like dense 70B) |
| Same | Each `model_config.model` | The model name on the API side | Must match `--alias` |

### B. Parameters that need adjusting when the embedding model changes

| Location | Key | What it affects | How it is chosen |
|---|---|---|---|
| `scripts/llama-services.sh` | `-m <embed_blob>` or the ollama-side model name | The embedding model itself | User input |
| Same | `--alias openai/text-embedding-3-small` | The alias Honcho can resolve the model by | Left as-is (Honcho references it by hard-coded name internally) |
| `honcho/config.toml` | `[embedding] VECTOR_DIMENSIONS` | Embedding dimensionality | **auto**: `n_embd` from `/v1/models` / `embedding_length` from ollama |
| Same | `[vector_store] DIMENSIONS` | pgvector column dimensionality | Must always match VECTOR_DIMENSIONS |
| Same | `[embedding] MAX_INPUT_TOKENS` | Max tokens fed into a single embedding call | **auto**: matched to `n_ctx_train` |
| Same | `[embedding.model_config.model` | The model name on the API side | Must match `--alias` |
| **Postgres** | Column definitions like `documents.embedding vector(768)` | DB schema | **A migration is required if DIM changes** |

**Risk of changing DIM**: a change like 768 → 1024 breaks the pgvector column type, and Honcho will refuse to start. Handling it automatically would require `ALTER TABLE ... ALTER COLUMN embedding TYPE vector(1024)` plus discarding existing data, or re-applying Honcho's Alembic migrations. **For the MVP, the rule is "if the user picks a choice that changes DIM, warn and abort"**. Destructive operations are split off into a separate procedure.

## Means of auto-detection

| Source | How to query | What you get |
|---|---|---|
| llama-server | `GET /v1/models` | `meta.n_ctx_train`, `meta.n_embd`, `meta.n_params`, `meta.size` |
| ollama | `POST /api/show {"model":...}` | `model_info.general.architecture`, `<arch>.context_length`, `<arch>.embedding_length`, `<arch>.expert_count` |

A wrapper `probe_model_meta(endpoint, model) -> ModelMeta` absorbs the differences between the two. It returns a unified schema:

```python
@dataclass
class ModelMeta:
    n_ctx_train: int
    n_embd: int
    n_params: int | None
    is_moe: bool | None           # ollama なら確定、llama-server なら None
    arch: str | None              # "qwen35moe", "llama", etc.
```

MoE detection:
- ollama: `expert_count > 0`, or `architecture` contains `moe`
- llama-server: cannot be determined → confirm interactively (default is to keep the current value)

## Chosen approach

### Script layout

A single file, `/home/baba-y/nuncstans-hermes-stack/scripts/switch-endpoints.py`. Dependencies are declared via PEP 723 metadata, with a `#!/usr/bin/env -S uv run --script` shebang. Dependencies: `tomlkit` / `ruamel.yaml` / `httpx` / `questionary`.

### Minor refactor of `llama-services.sh` (so the script can rewrite it safely)

Right now, model specs are **hard-coded inside the shell script** (`HF_CHAT_SPEC` L26, `CHAT_ALIAS` L27, `-c 131072` L126, `-ot ...` L131, etc.). Having switch-endpoints sed these in place would be fragile. Instead:

- Extract them into a new `scripts/llama-services.conf` (sourced by bash):
  ```bash
  # 現状の値
  CHAT_HF_SPEC="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"
  CHAT_ALIAS="qwen3.6-test"
  CHAT_CTX=131072
  CHAT_NGL=99
  CHAT_IS_MOE=1           # 1=付与, 0=外す → -ot を動的に組み立て
  CHAT_REASONING_OFF=1    # qwen3 系のみ 1
  CHAT_PARALLEL=2
  EMBED_BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
  EMBED_ALIAS="openai/text-embedding-3-small"
  ```
- At the top of `llama-services.sh`, `source "$ROOT/scripts/llama-services.conf"`, and rewrite `start_chat` / `start_embed` so they compose their command lines from these variables. The MoE / reasoning flags are built conditionally:
  ```bash
  extra=()
  (( CHAT_IS_MOE )) && extra+=(-ot "ffn_(up|down|gate)_exps=CPU")
  (( CHAT_REASONING_OFF )) && extra+=(--reasoning off)
  nohup "$LLAMA_BUILD/llama-server" -hf "$CHAT_HF_SPEC" ... -c "$CHAT_CTX" -ngl "$CHAT_NGL" "${extra[@]}" ...
  ```
- `switch-endpoints.py` only ever rewrites this conf file (with a dedicated helper that updates `KEY=VALUE` pairs via regex).

### Three switch axes

| Axis | How to change it | What to restart |
|---|---|---|
| **Honcho chat endpoint/model** | 9 blocks of `config.toml` | `docker compose up -d --force-recreate api deriver` |
| **Honcho embedding endpoint/model** | 1 block of `config.toml`, plus `VECTOR_DIMENSIONS` / `MAX_INPUT_TOKENS` if needed | Same as above (abort on DIM change) |
| **Swapping the llama-server model itself** (only when the target is llama-server) | `scripts/llama-services.conf` | `./scripts/llama-services.sh restart` |
| **Hermes chat endpoint/model** | `hermes config set model.base_url` / `.default` | Not needed (the Hermes CLI reads its config on every launch) |

For the "change the llama-server model" case:
1. switch-endpoints asks for the target model (an HF spec or an ollama model id)
2. `probe_model_meta` retrieves ctx / dim / MoE
3. It proposes recommended values (`-c` / `-ngl` / MoE / reasoning) and asks for confirmation
4. It rewrites `llama-services.conf` → updates the dependent keys in `config.toml` → restarts both

### URL translation

Let the user pick things from a host-side view (`localhost`); rewrite them to `host.docker.internal` only when writing to `config.toml`. Hermes runs on the host, so its URL is left unchanged.

### Safety net: snapshots and rollback

Scattering per-file `.bak` files across three locations makes restore painful when something fails — a human has to cross-reference which backups belong to the same write. Instead, **create a single snapshot directory, copy all the relevant files coherently before any write, and auto-prune it with LRU**.

**Snapshot layout:**

```
~/.local/state/hermes-stack/endpoint-snapshots/
├── 20260423-151230.12345/           ← タイムスタンプ.PID
│   ├── config.toml                    (honcho/config.toml の copy)
│   ├── llama-services.conf            (if exists)
│   ├── hermes-config.yaml             (~/.hermes/config.yaml の copy)
│   └── manifest.json
│       {
│         "created_at": "2026-04-23T15:12:30+09:00",
│         "user_choices": { "honcho_chat": {...}, "honcho_embed": {...}, "hermes": {...}, "llama_model": {...} },
│         "files_snapshotted": ["config.toml", "llama-services.conf", "hermes-config.yaml"],
│         "planned_restarts": ["docker compose ... api deriver", "llama-services.sh restart"],
│         "status": "snapshot_only" | "applied" | "applied_with_errors" | "rolled_back",
│         "errors": [...],
│         "previous_snapshot": "20260423-145500.11234"   // chain for audit
│       }
├── 20260423-145500.11234/
├── ...
```

**Behavior:**

1. **Immediately before any write**, always call `create_snapshot()`: copy the three target files and write the initial manifest with `status="snapshot_only"`.
2. Once every write and every service restart has succeeded, update the manifest to `status="applied"`.
3. If something fails midway, promote `auto_rollback(snapshot)`: use `os.replace` to atomically put the snapshot's files back where they came from, update the manifest to `status="rolled_back"`, and have the user manually rerun any restart that failed (only guidance, no auto-execution).
4. If a post-write `compose up` or `llama-services restart` fails, it counts as a rollback trigger (i.e. roll the files back too). Whether this should be the default (no confirmation) is arguable, but **"the write succeeded but the service will not come up" is the most dangerous state**, so the policy is: auto-rollback + restart services by default, and if that also fails, hand off to manual.

**LRU pruning:**

- At startup, sort `~/.local/state/hermes-stack/endpoint-snapshots/` by `mtime`, and delete the oldest entries if there are more than 11.
- The actual rule is **delete when count exceeds 10**, not 11: if `len(dirs) > 10`, `rm -rf` `dirs[:-10]`.
- Everything other than `status="applied"` (i.e. already rolled back, or aborted before any write) lives in the same bucket. They are kept for audit purposes.

**Subcommands:**

- `switch-endpoints.py` (no arguments): the new switching flow
- `switch-endpoints.py --rollback`: restore from the latest snapshot (with confirmation)
- `switch-endpoints.py --list-snapshots`: list 10 entries along with their manifests (applied status, a summary of user_choices, and the affected files)
- `switch-endpoints.py --restore <snapshot-id>`: restore from a specific snapshot
- `switch-endpoints.py --dry-run`: print everything without writing

**Atomicity of the writes themselves (per-file level):**

- Each file write is an atomic rename via tempfile + `os.replace`
- Combining that with the snapshot means that even in a "file A was written but B wasn't" scenario, we can still recover to a consistent state
- Before writing, the diff is shown with `difflib.unified_diff` → two-stage confirmation

## UX flow

```
== 現在の設定 ==
  Honcho chat  :  http://host.docker.internal:8080/v1  qwen3.6-test  (ctx 131072, MoE)
  Honcho embed :  http://host.docker.internal:8081/v1  openai/text-embedding-3-small  (dim 768)
  Hermes       :  http://localhost:8080/v1  qwen3.6-test
  llama-server :  unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL  (-c 131072 -ot MoE)

== A: Honcho チャット ==
  [1] llama-server (:8080) — 現サーバ, 切替なし
  [2] ollama       (:11434)
  [3] 任意 URL
  [4] llama-server にロードするモデルを変える ← 選ぶと D に飛ぶ
  → モデルピッカ(/v1/models 動的)

== B: Honcho embedding ==
  同様。ただし DIM が現行(768)と異なる選択肢を選ぶと大きな警告
  「この変更には DB のマイグレーションが必要です。中止してください。」で中止。

== C: Hermes ==
  [1] Honcho chat と同じ
  [2] 別 URL を指定
  [3] 変更しない

== D: llama-server のモデル差し替え(A で 4 を選んだ時のみ) ==
  spec 入力: unsloth/llama-3.1-70B-Instruct-GGUF:Q4_K_M
  → probe (ollama pull + /api/show, or llama.cpp --dry)
  → 検出: ctx=131072, arch=llama, MoE=no, params=70B
  → 推奨値提示:
       -c 131072  (trained ctx と同値)
       -ngl 40    (70B なので部分オフロード提案、VRAM から計算)
       MoE off, reasoning off
     各値について [accept/override]
  → conf ファイル差分表示

== 変更サマリ ==            （全軸の before/after）
続行しますか？ (y/N)
== config.toml 差分 ==      unified diff
== llama-services.conf 差分 ==
この差分で書き込みますか？
→ 書き込み → Hermes → 再起動確認(2 段: llama-services / honcho compose)
→ 完了、バックアップパス表示
```

## Failure modes

| Situation | Response |
|---|---|
| `/v1/models` unreachable | Fall back to manual input |
| ollama `/api/show` returns 404 | The model has not been pulled → suggest `ollama pull` (just print the command, do not run it) |
| Embedding DIM would change | Abort, and print manual migration steps |
| `hermes` CLI is missing | Skip the Hermes section |
| llama-server restart fails | **auto-rollback** → revert to the previous conf and retry the restart → if that also fails, point the user to `llama-services.sh logs chat` and record `applied_with_errors` in the manifest |
| `compose up` fails | **auto-rollback** → put config.toml back to its old values and retry `compose up` → if that also fails, print manual commands |
| `hermes config set` fails | Restore the Hermes YAML from the snapshot, but leave the other files in their successful state (partial rollback, Hermes side only) |
| Ctrl-C interrupt | Zero side effects before the final confirmation. Once the write phase has begun, call `auto_rollback` from a `finally` block |
| ENOSPC on the snapshot location | Warn and abort the switch (we will not proceed without an undo safety net we can trust) |

## Files involved

**Newly created:**
- `/home/baba-y/nuncstans-hermes-stack/scripts/switch-endpoints.py`
- `/home/baba-y/nuncstans-hermes-stack/scripts/llama-services.conf` (the current values factored out)

**Modified:**
- `/home/baba-y/nuncstans-hermes-stack/scripts/llama-services.sh` (source the `.conf` at the top, replace the literals with variable references)

**Written by the script:**
- `honcho/config.toml` (up to 10 blocks, plus the linked updates to `GET_CONTEXT_MAX_TOKENS` / `MAX_INPUT_TOKENS`)
- `~/.hermes/config.yaml` (via `hermes config set`)
- `scripts/llama-services.conf` (only when swapping the llama-server model)

**Restarted (after user confirmation):**
- `./scripts/llama-services.sh restart` (when the llama-server model or the embed model is swapped)
- `docker compose -f honcho/docker-compose.yml up -d --force-recreate api deriver` (when `config.toml` is changed)

## Function layout

```python
# probing
probe_models(url) -> list[str] | None              # /v1/models
probe_model_meta(endpoint, model) -> ModelMeta     # /v1/models or /api/show
derive_chat_params(meta) -> ChatParams             # -c, -ngl, is_moe, reasoning_off 自動提案
derive_embed_params(meta) -> EmbedParams           # dim, max_input_tokens

# snapshot / rollback
create_snapshot(files: list[Path], choices: dict) -> Snapshot
finalize_snapshot(snap: Snapshot, status: str, errors: list = []) -> None
list_snapshots() -> list[Snapshot]                 # mtime ソート降順
prune_snapshots(keep: int = 10) -> int             # 戻り値: 削除件数
restore_snapshot(snap: Snapshot) -> None           # 各ファイルを os.replace で戻す
auto_rollback(snap: Snapshot, reason: str) -> None # finalize + restore + 案内

# config writers (すべて snapshot 作成後にのみ呼ばれる)
update_honcho_toml(chat, embed, caps, dry_run) -> diff
update_llama_conf(params, dry_run) -> diff
update_hermes_config(endpoint, model, dry_run) -> None

# restart orchestration
restart_llama(dry_run) -> subprocess.CompletedProcess
restart_honcho_compose(dry_run) -> subprocess.CompletedProcess

# helpers
to_docker_url(url) -> str
ollama_context_ceiling() -> int                     # systemctl cat → OLLAMA_CONTEXT_LENGTH 読み取り

# orchestration
cmd_switch() -> None       # デフォルト
cmd_rollback() -> None     # --rollback
cmd_restore(snap_id) -> None  # --restore
cmd_list() -> None         # --list-snapshots
main() -> None             # argparse ディスパッチ
```

## Verification plan

1. **Dry-run**: `./scripts/switch-endpoints.py --dry-run`. For each scenario, just check the diff:
   - Switch to ollama: `base_url` is `:11434` in all 9 blocks, and `model` is the picked ID
   - Swap the llama-server model (70B dense): `-c`, `CHAT_IS_MOE=0`, and `CHAT_NGL` are updated in `llama-services.conf`
   - Embed choice that changes DIM: a warning is shown and the operation is aborted
2. **Sandbox**: an escape hatch via `HONCHO_TOML_OVERRIDE` / `LLAMA_CONF_OVERRIDE` env vars that redirect writes to temp files so the real files are not touched
3. **One shot with a real model**: run `qwen3.6-test` → `qwen3.6-test` from `/v1/models` (i.e. the same model) and verify that the script exits as a no-op (safety-valve test)
4. **Hermes round-trip**: confirm via `hermes config show` that `base_url` / `default` have been updated
5. **compose**: verify it took effect with `docker compose logs --tail=20 deriver | grep base_url`
6. **llama-server**: verify with `ss -tlnp | grep 8080` that the old pid died and a new pid came up, and that the alias exposed at `/v1/models` is now the new model name
7. **Rollback (automatic)**: deliberately have it write a bad setting (e.g. an unreachable `base_url`) so that `compose up` fails → confirm that `auto_rollback` fires, all three files are restored to their old values, and compose and llama-server recover with the old settings
8. **Rollback (manual)**: list snapshots with `switch-endpoints.py --list-snapshots`, then revert to an arbitrary point in time via `--restore <id>`
9. **LRU pruning**: run 11 switches and verify that the oldest snapshot has been removed
10. **Snapshot retention**: after a `--rollback`, verify that the snapshot directory itself is still kept (with `status=rolled_back`) as an audit trail

## Known gotchas when picking an Ollama endpoint (from experiments/bottleneck.md)

When the user switches the Hermes or Honcho endpoint to **ollama (:11434)**, there are two pitfalls specific to the OpenAI-compatible API. **The switcher is obligated to warn about them.**

### Gotcha 1: Silent truncation via `OLLAMA_CONTEXT_LENGTH`

- Ollama's `/v1/chat/completions` (the OpenAI-compatible endpoint) has **no way to pass `num_ctx` per request**. Only the service-wide environment variable `OLLAMA_CONTEXT_LENGTH` exists, and its default is 4096.
- This machine is currently configured via its override.conf to **`OLLAMA_CONTEXT_LENGTH=65536`**. If Honcho or Hermes send a prompt longer than 65536, it is **silently truncated**, with no warning in the logs (the prompt is cut midway, tail characters are eaten, or the model loops indefinitely).
- **Switcher's responsibility**: the moment ollama is picked, read the current `OLLAMA_CONTEXT_LENGTH` via `systemd-cat` or `systemctl cat ollama.service`, and cap Honcho's `[app] GET_CONTEXT_MAX_TOKENS` / `[dialectic] MAX_INPUT_TOKENS` so they do not exceed it. If they do, warn and propose lowering them.

### Gotcha 2: qwen3-family reasoning tokens leaving `content` empty

- On qwen3.6:35b-class models, if the API call does not set `think` explicitly, **every token flows into invisible reasoning and `message.content` comes back empty** (llama.cpp #20099).
- The OpenAI-compatible API has no `think` field, so Hermes's `provider: ollama-launch` client cannot pass it.
- **Fix**: create a Modelfile on the ollama side that bakes in `PARAMETER think false`, and use that as a separate alias (e.g. `qwen3.6:35b-nothink`). Otherwise Hermes cannot call with `think` off, and you get 26-second empty responses per turn.
- **Switcher's responsibility**: if arch is `qwen3*` and the target is ollama, display a warning recommending "create a `think false` variant via a Modelfile before using it". Do not auto-generate the Modelfile — just print the manual procedure.

### Gotcha 3: Differences in tool-call support

- ollama's OpenAI-compatible layer has model-dependent tool-call support, and Hermes's skill tool invocations become flaky under it. llama-server's `--jinja` + tool_call template is more stable.
- **Switcher's responsibility**: when pointing Hermes at ollama, print a note saying "if tool-use matters, llama-server is recommended".

## Timing headroom for Honcho's asynchronous components (the `--parallel` family)

Honcho has a queue-driven async architecture, so slowing the model down lengthens wait times and can cause cascading stalls. Here is the full picture of **queue/timing parameters that should be reviewed alongside a model change**:

| Location | Key | Role | How to think about it on a model change |
|---|---|---|---|
| `scripts/llama-services.sh` → `.conf` | `--parallel <N>` | llama-server parallel slot count (currently 2) | 2 is the baseline for "Hermes foreground + deriver background". On a 70B where VRAM is tight, 1. On an MoE with room to spare, 3 is fine |
| `honcho/config.toml` | `[deriver] WORKERS` | deriver worker count (currently 1) | Only meaningful if you have also raised llama-server's `--parallel`. Make sure WORKERS × inference time stays within STALE_SESSION_TIMEOUT |
| Same | `[deriver] POLLING_SLEEP_INTERVAL_SECONDS` | Queue polling interval (currently 1.0s) | On a slow model (>10s/call), stretch this to 2.0 or so to reduce CPU load |
| Same | `[deriver] STALE_SESSION_TIMEOUT_MINUTES` | Session-lock expiry (currently 5 min) | For a model like 70B where one turn takes minutes, raise to 10 |
| Same | `[db] POOL_TIMEOUT` | DB connection wait (currently 30s) | Our policy (per CLAUDE.md) is that the deriver does not hold DB sessions for long, so normally leave it alone |
| Same | `[dream] IDLE_TIMEOUT_MINUTES` | Idle time before dream fires (currently 10 min) | Model-independent, but on a slow model, dream may eat the foreground |
| env in `scripts/gatekeeper_daemon.py` | `GK_POLL_INTERVAL_SEC` | gatekeeper polling (currently a 5s default, not exported) | Measure and tune when the chat model changes |
| `~/.hermes/config.yaml` | `agent.gateway_timeout` (currently 1800s), `agent.restart_drain_timeout`, `tools.*.timeout` | Various Hermes-side timeouts | Prevents the tool timeout from firing first when a slow model is selected. The switcher only warns — these are Hermes CLI settings, so basically hands-off |

**Switcher's responsibility**: from the selected chat model's `n_params`, automatically decide whether it is a "slow model" (dense 70B or above, or has a high CPU offload ratio), and of the above, propose recommended values and ask for confirmation for just the two that matter: `--parallel` (LLAMA) and `STALE_SESSION_TIMEOUT_MINUTES` (Honcho). The rest are left untouched (documented in the guide).

## Out of scope

- Support for external endpoints like OpenRouter / Anthropic (for `config.toml` alone this is just a `base_url` swap, but unifying `api_key` management under `hermes config` or `.env` is a separate discussion)
- Per-subsystem individual overrides (e.g. a different model just for dream)
- The **destructive DB migration** that a change in embedding DIM entails (the plan is to split this off as a separate procedure like `scripts/reset-pgvector-dim.sh`)
- Automatic `ollama pull` of models (suggestion only — never run automatically)
