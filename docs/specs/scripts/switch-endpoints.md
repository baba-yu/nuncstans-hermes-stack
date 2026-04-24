# switch-endpoints.py

## Purpose

`scripts/switch-endpoints.py` is the single entry point for moving the
hermes-stack between LLM backends. Three files and one container pair have
to stay coherent when a model changes: `honcho/config.toml` (9 chat model
blocks + 1 embedding block, plus four token / dimension caps),
`scripts/llama-services.conf` (launch flags for the two llama-server
processes), and `~/.hermes/config.yaml` (Hermes CLI's default model and
base URL). Editing them by hand invariably drifts — e.g. Honcho's
`GET_CONTEXT_MAX_TOKENS` gets left pointing at a ctx that the new chat
model cannot support, or `embedding.VECTOR_DIMENSIONS` silently desyncs
from pgvector's column type.

The script walks the operator through four axes (Honcho chat, Honcho
embed, Hermes, llama-server model), snapshots every affected file before
writing, applies changes atomically, restarts Honcho (`api` + `deriver`)
and `llama-services.sh` in the right order, and auto-rolls-back on any
failure.

## Usage

```bash
# Interactive switch (chat + hermes + llama axes; embed axis skipped)
./scripts/switch-endpoints.py

# Include the embed axis as well (destructive: DIM change forces pgvector migration)
./scripts/switch-endpoints.py --with-embed

# Compute diffs, print them, do not write or restart anything
./scripts/switch-endpoints.py --dry-run

# List the last 10 snapshots (+ manifest summaries)
./scripts/switch-endpoints.py --list-snapshots

# Restore the most recent snapshot
./scripts/switch-endpoints.py --rollback

# Restore a specific snapshot
./scripts/switch-endpoints.py --restore 20260423-151230.12345
```

The default flow touches three axes: **A** Honcho chat, **C** Hermes, **D**
llama-server model. Axis **B** (Honcho embed) is opt-in because changing
the embedding model's output dim forces a destructive pgvector migration
that this script will not perform for you. Pass `--with-embed` when you
know what you are doing.

Inside the interactive flow each axis offers the same menu:
`llama-server` / `ollama` / custom URL / keep current. The Honcho-chat
axis also offers "change the llama-server model itself" which pivots into
axis D (rewriting `llama-services.conf`).

## Options & env vars

| Flag / env                      | Default                                                            | What it controls                                                                           |
| ------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `--dry-run`                     | off                                                                | Compute diffs, print them, write nothing, restart nothing, no snapshot taken               |
| `--with-embed`                  | off                                                                | Include axis B (Honcho embed). DIM mismatch aborts this axis only, leaves others unchanged |
| `--rollback`                    | —                                                                  | Restore from the most recent snapshot                                                      |
| `--restore <id>`                | —                                                                  | Restore from a specific snapshot id (see `--list-snapshots`)                               |
| `--list-snapshots`              | —                                                                  | List the 10 most recent snapshots with their manifest summaries                            |
| `HERMES_STATE_DIR`              | `$HOME/.local/state/nuncstans-hermes-stack`                        | Root for `endpoint-snapshots/` (and the dir shared with `llama-services.sh`)               |
| `HONCHO_TOML_OVERRIDE`          | `<repo>/honcho/config.toml`                                        | Path to the Honcho TOML the script edits                                                   |
| `LLAMA_CONF_OVERRIDE`           | `<repo>/scripts/llama-services.conf`                               | Path to the llama-services conf the script edits                                           |
| `HERMES_YAML_OVERRIDE`          | `~/.hermes/config.yaml`                                            | Path to the Hermes CLI config the script edits                                             |

Non-configurable constants worth knowing:

- `CHAT_CTX_HARD_CAP = 131_072` — even if a model reports `n_ctx_train`
  bigger than 128k, the script caps `-c` at 131072. Raise deliberately;
  past this point memory and per-turn latency both blow up.
- `CTX_HEADROOM = 20_000` — reserved for output + system tokens when the
  script computes Honcho's input caps (see below).
- `SNAPSHOT_KEEP = 10` — LRU threshold; older snapshots are pruned at
  startup.

## Specs & assumptions

### 4 interactive axes

| Axis | File                                                        | What is written                                                                     |
| ---- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| A    | `honcho/config.toml` (9 chat `model_config` blocks)         | `model`, `overrides.base_url`                                                       |
| B    | `honcho/config.toml` (`embedding.model_config`)             | `model`, `overrides.base_url`; dim/max-input caps co-moved (see below)              |
| C    | `~/.hermes/config.yaml`                                     | `model.base_url`, `model.default`, plus `providers.<model.provider>.{api,default_model,models[]}` — all four are **force-synced** every run to prevent the display-vs-runtime bifurcation that Hermes v0.10 otherwise exhibits |
| D    | `scripts/llama-services.conf`                               | `CHAT_HF_SPEC`, `CHAT_ALIAS`, `CHAT_CTX`, `CHAT_NGL`, `CHAT_IS_MOE`, `CHAT_REASONING_OFF`, `CHAT_PARALLEL` |

Default flow: A + C + D. B is opt-in (`--with-embed`); justification is
that swapping embed DIM without also migrating pgvector leaves the stack
in a broken state, and this script does not touch the database.

### Snapshot envelope

- Location: `${HERMES_STATE_DIR:-~/.local/state/nuncstans-hermes-stack}/endpoint-snapshots/<YYYYMMDD-HHMMSS>.<pid>/`
- Contents: `config.toml`, `llama-services.conf`, `hermes-config.yaml` (any
  that exist) + `manifest.json`.
- Manifest schema:
  ```json
  {
    "created_at": "ISO-8601 with tz",
    "user_choices": { "honcho_chat": {...}, "honcho_embed": {...}, "hermes": {...},
                      "llama_chat_params": {...}, "llama_embed_params": {...},
                      "caps": {...} },
    "files_snapshotted": ["config.toml", "llama-services.conf", ...],
    "planned_restarts": ["docker compose ... api deriver", "scripts/llama-services.sh restart"],
    "status": "snapshot_only | applied | applied_with_errors | rolled_back",
    "errors": [],
    "previous_snapshot": "<id or null>"
  }
  ```
- LRU: at startup, any snapshots beyond the 10 most recent are deleted.
- Auto-rollback triggers: any `FatalError`, any `docker compose` non-zero
  exit, any `llama-services.sh restart` non-zero exit, or `KeyboardInterrupt`
  after writes have begun. Rollback is atomic per file (copy into a
  sibling tmp, `os.replace` onto destination). Service restarts are **not**
  re-invoked by the rollback — the manifest notes this and the operator is
  told to re-run restarts by hand.

### Chat parameter derivation (axis D)

From `ModelMeta` (populated by `/v1/models` or ollama `/api/show`):

| Output field     | Rule                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `ctx` (`-c`)     | `min(n_ctx_train, 131072)` if metadata known; otherwise keep current                             |
| `ngl` (`-ngl`)   | `99` for `n_params ≤ 15B`; otherwise keep current (operator is re-prompted when metadata thin)   |
| `is_moe`         | `True` if ollama `expert_count > 0` or arch name contains "moe"                                  |
| `reasoning_off`  | `True` when `arch.lower().startswith("qwen3")`                                                   |
| `parallel`       | `2` by default; `1` for dense models with `n_params ≥ 35B` (non-MoE)                             |

If the probe can't fetch metadata (e.g. the new HF spec hasn't been
pulled yet), the script falls back to asking the operator for each
value with the current values as defaults.

### Honcho cap co-movement

When chat `ctx` changes (new llama-server model, or new chat endpoint
with known `n_ctx_train`), the script computes
`target = max(ctx - 20000, 1024)` and writes it into both
`app.GET_CONTEXT_MAX_TOKENS` and `dialectic.MAX_INPUT_TOKENS` — **even
if the operator had manually set higher values previously**. This is
intentional (keeping them pinned to the runtime ctx minus output headroom)
but has drift behaviour: any hand-tuned cap the operator put in the TOML
is silently overwritten on the next switch. If you need a hand-tuned cap
to survive a switch, re-apply it after the switch completes.

When the embed axis runs and meta is available, the script also writes
`embedding.VECTOR_DIMENSIONS`, `vector_store.DIMENSIONS`, and (if
`n_ctx_train` is known) `embedding.MAX_INPUT_TOKENS`.

### URL translation rule

For `honcho/config.toml` writes, `localhost` / `127.0.0.1` / `0.0.0.0`
are rewritten to `host.docker.internal` so the Honcho container can
reach the host-side llama-server / ollama. For the Hermes YAML, the
host form is preserved — `hermes` runs on the host, not in a container,
so `host.docker.internal` there would break resolution.

### Hermes provider sync (always on)

Hermes v0.10 stores its LLM endpoint in two places:

- Top level `model.{base_url,default}` — drives the session-start header
  display
- `providers.<model.provider>.{api,default_model,models[]}` — drives
  the actual runtime request path (`api` is the URL hit,
  `default_model` is the id sent, `models[]` is the `hermes model`
  picker catalogue)

If the two layers disagree, the session banner will show one model
while inference actually runs against another — the split happens
silently at session boot. Writing only `model.*` (e.g. via
`hermes config set`) does not propagate to the provider entry.

To prevent this the script **always** force-syncs all four fields in
the provider entry whenever the Hermes axis is written. There is no
opt-out — the split-layer behaviour is a correctness hazard, not a
preference. Sync steps:

1. `hermes config set model.base_url <URL>` and
   `hermes config set model.default <MODEL>` — writes the display
   fields.
2. ruamel re-read of the YAML, then force-write of
   `providers.<model.provider>.api = <URL>`,
   `providers.<model.provider>.default_model = <MODEL>`, and
   `models[].insert(0, <MODEL>)` if not already present (deduped).

One side effect: the `models[]` list grows monotonically across runs
as you try new models. Prune by hand in the YAML if it gets noisy; the
runtime does not care.

### ollama-specific pitfalls (warned on)

1. **`OLLAMA_CONTEXT_LENGTH` silent truncation.** The script greps
   `systemctl cat ollama.service` for `OLLAMA_CONTEXT_LENGTH=<N>`. If it
   finds one, Honcho caps are further capped at `min(ctx, N) - 20000`.
   If the env is unset, ollama defaults to 4096 and prompts past that are
   silently truncated server-side — the script warns loudly in that case.
2. **qwen3 `think` flag.** The OpenAI-compat API does not expose the
   think toggle. To suppress thinking tokens on ollama, create a
   `Modelfile` with `PARAMETER think false`, `ollama create` a new alias,
   and point Honcho at the new alias. The script flags this whenever
   the selected model's arch starts with `qwen3`.
3. **Tool-call stability.** ollama's tool-call support varies by model
   and release; for tool-heavy workflows prefer llama-server, which keeps
   the jinja template and tool-call path consistent with upstream.

### Embed DIM mismatch guard

If the selected embed model reports a different `n_embd` than the
current `embedding.VECTOR_DIMENSIONS`, the script aborts the embed axis
only (leaves chat / hermes / llama axes intact) and prints a message
telling the operator that a pgvector migration is required. It will not
attempt the migration itself.

### Compose restart

`honcho/docker-compose.override.yml` uses `ports: !reset []` on the
`database` and `redis` services so they don't collide with the
host-managed `llm-postgres` / `llm-redis`. That override is only
auto-merged by compose when no `-f` is passed; once `-f` is explicit we
have to list both files. The script invokes:

```
docker compose -f honcho/docker-compose.yml \
               -f honcho/docker-compose.override.yml \
               up -d --force-recreate --no-deps api deriver
```

`--no-deps` keeps database/redis untouched.

### Single-instance assumption

Snapshots and state live under a single `HERMES_STATE_DIR`. Running two
concurrent switch flows against the same repo is not supported — they
will race on the snapshot id's pid suffix only, and their writes on the
three config files will interleave. If you need parallel switches (why?),
point each invocation at a different state dir and at separate
`HONCHO_TOML_OVERRIDE` / `LLAMA_CONF_OVERRIDE` / `HERMES_YAML_OVERRIDE`
paths.

## Outlook

The `HERMES_STATE_DIR` override is a stopgap for running a second Hermes
instance on the same host. It covers snapshot location but not the
config files themselves, so a real second instance would still need
per-instance TOML/conf/YAML paths. The right fix is to containerize
Hermes so each instance gets its own filesystem, network, and
llama-server siblings — see
[`docs/specs/scripts/llama-services.md`](llama-services.md) for the same
recommendation from the other side.

Future work on the script itself is deliberately small:

- Pgvector migration helper — currently out of scope; the embed axis
  aborts rather than attempting it.
- Snapshot diff inspector (`--diff <id1> <id2>`) — the manifest has
  enough structure to make this trivial if it turns out to be wanted.
- Remove the cap-overwrite drift behaviour in favour of "only bump down,
  never bump up" — waiting on more operational data on whether the
  drift is actually painful in practice.
