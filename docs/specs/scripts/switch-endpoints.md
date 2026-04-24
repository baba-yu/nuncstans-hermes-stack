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
| A    | `honcho/config.toml` (9 chat `model_config` blocks)         | `model`, `overrides.base_url`; also co-writes `GK_LLM_URL` / `GK_LLM_MODEL` in `scripts/llama-services.conf` so the gatekeeper classifier follows the chat engine (see "Gatekeeper follows chat engine" below) |
| B    | `honcho/config.toml` (`embedding.model_config`)             | `model`, `overrides.base_url`; dim/max-input caps co-moved (see below)              |
| C    | `~/.hermes/config.yaml`                                     | `model.base_url`, `model.default`, plus `providers.<model.provider>.{api,default_model,models[]}` — all four are **force-synced** every run to prevent the display-vs-runtime bifurcation that Hermes v0.10 otherwise exhibits |
| D    | `scripts/llama-services.conf`                               | `CHAT_HF_SPEC`, `CHAT_ALIAS`, `CHAT_CTX`, `CHAT_NGL`, `CHAT_IS_MOE`, `CHAT_REASONING_OFF`, `CHAT_PARALLEL` |

Default flow: A + C + D, plus a lightweight "move embed to the same
engine?" offer (Y/n) whenever Axis A changes engine and Axis B would
otherwise stay on the old one. That short-form prompt keeps the chat
and embed engines in step without requiring the user to reason about
dim/model ids. `--with-embed` still exists for the full Axis B picker
(endpoint + model both chosen explicitly); justification for making
that mode opt-in is unchanged — swapping embed DIM without also
migrating pgvector leaves the stack in a broken state, and this script
does not touch the database.

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

### Gatekeeper follows chat engine

`scripts/gatekeeper_daemon.py` (fork-specific queue classifier) reads
`GK_LLM_URL` / `GK_LLM_MODEL` from env, which `scripts/llama-services.sh`
exports from its conf file. Whenever Axis A changes the Honcho chat
endpoint, this script also rewrites those two conf keys to match — the
classifier and the main chat path always share the same engine. The
rewrite strips the trailing `/v1` because the daemon appends
`/v1/chat/completions` itself.

This couples three things to Axis A: Honcho chat, gatekeeper classifier,
and (via `needs_llama_restart`) the `llama-services.sh restart` step.
It is intentional — the alternative ("user picks gk separately") is a
footgun in a single-host deployment because the classifier fails
silently when the endpoint it points at is stopped.

Operators who genuinely want a dedicated lightweight classifier on a
separate URL/model can hand-edit `GK_LLM_URL` / `GK_LLM_MODEL` in
`scripts/llama-services.conf`. The switcher will rewrite them on the
next Axis A run, so that workflow requires either pinning a custom conf
and never re-running Axis A, or automating the re-application after
each run.

### Embed engine-match offer (short form, default flow)

When Axis A changes to a different engine (llama-server ↔ ollama) and
the user did *not* pass `--with-embed`, the script offers a one-question
"also move embed to the same engine?" prompt (default Yes). Accepting
it maps the engine to the canonical 768-dim nomic-embed-text endpoint
(`:8081` `openai/text-embedding-3-small` for llama-server, `:11434`
`openai/text-embedding-3-small:latest` for ollama — both resolve to the
same GGUF blob so no DIM change and no pgvector migration). Declining
leaves the embed axis untouched; `--with-embed` is still the way to
change model or dim explicitly.

The side effect worth knowing: if the user accepts and also stops the
llama-server chat (Axis A moved to ollama), the llama-server chat
(`:8080`) becomes truly unused — see "Outlook" below for the VRAM
reclaim workflow.

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

### Container-visible config via bind mount

Honcho's upstream Dockerfile bakes `honcho/config.toml` into
`/app/config.toml` at image build time, so a switch-endpoints run
that edits the host file silently does not reach the running `api` /
`deriver` containers — `--force-recreate` reuses the baked image.
Symptom: the switcher reports "all writes succeeded" but the deriver
keeps calling the pre-bake endpoints and the dialectic path never
sees the new model.

`honcho/docker-compose.override.yml` solves this with a read-only
bind mount on both services:

```yaml
services:
  api:
    volumes:
      - ./config.toml:/app/config.toml:ro
  deriver:
    volumes:
      - ./config.toml:/app/config.toml:ro
```

With the mount in place, a `--force-recreate` is enough to pick up
config.toml changes — no image rebuild.

`atomic_write` in this script writes with mode `0644` (not the
tempfile-default 0600) so the container user can read the file. The
default 0600 was historically permissive enough on the host but broke
read access inside the container (UID mismatch).

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

### Engine consistency (as of this version)

- The gatekeeper classifier now tracks the Honcho chat engine (ollama
  or llama-server) automatically on every Axis A run — there is one
  source of truth for "which engine is driving the stack."
- If an operator genuinely wants a dedicated lightweight classifier on
  a separate URL/model, they can hand-edit `GK_LLM_URL` / `GK_LLM_MODEL`
  in `scripts/llama-services.conf`. That override survives until the
  next Axis A run, at which point the switcher rewrites both keys back
  in sync with the chat endpoint. This is an intentional trade: the
  default is safe (no silent classifier breakage on engine switch) at
  the cost of a small amount of churn for the minority workflow.
- The llama-services lifecycle is now per-service and scoped by the
  axis changes (`_llama_lifecycle_plan` in the switcher):
  - **chat llama-server** — restarted only when `llama_chat_params`
    actually changed; started (idempotent) when Axis A moves chat
    back onto llama-server; **stopped** when Axis A moves chat to
    ollama (no caller left → VRAM reclaimed automatically).
  - **embed llama-server** — same three cases keyed off Axis B /
    `llama_embed_params`.
  - **gk daemon** — restarted whenever `honcho_chat` changed
    (`GK_LLM_URL` / `GK_LLM_MODEL` need to be re-read from the conf).
  - **ollama model unload** — opt-in for the reverse direction. The
    switcher knows which ollama models were in use before the switch
    (across chat / embed / hermes axes) and are not in use after, and
    will interactively ask whether to POST `/api/generate keep_alive=0`
    for each of them. **Default is No** — keeping the user's
    `OLLAMA_KEEP_ALIVE=-1` policy intact, preserving prompt/KV caches,
    and avoiding the ~10-30s reload cost for a 30B-class model on a
    quick switch-back. When VRAM reclaim genuinely matters (tight GPU
    or long-term engine change), answer Yes at the prompt, or pass
    `--unload-ollama` on the command line to skip the confirm. The
    ollama systemd service is never touched either way — stopping the
    whole service requires sudo and is out of scope. Manual unload
    (for ad-hoc reclaim outside a switch) is one line:
    ```bash
    curl -sfS -X POST http://localhost:11434/api/generate \
      -d '{"model":"<id>","keep_alive":0,"prompt":"","stream":false}'
    # or: ollama stop <id>
    ```
  So a run that points chat (and optionally embed) at ollama stops
  the corresponding llama-server processes without manual follow-up,
  and a run that moves chat off ollama releases the ollama model's
  VRAM without manual follow-up. The user still sees one confirmation
  describing the full plan — e.g.
  "stop {chat}; ollama unload {qwen3.6:27b}; restart {gk}" — before
  anything executes.
- If you preferred the old "always stop + start all three" behavior,
  or you want to manually intervene, the per-target subcommands on
  `llama-services.sh` (`start|stop|restart {all|chat|embed|gk}`) are
  the lower-level primitive the switcher drives.

Future work on the script itself is deliberately small:

- Pgvector migration helper — currently out of scope; the embed axis
  aborts rather than attempting it.
- Snapshot diff inspector (`--diff <id1> <id2>`) — the manifest has
  enough structure to make this trivial if it turns out to be wanted.
- Remove the cap-overwrite drift behaviour in favour of "only bump down,
  never bump up" — waiting on more operational data on whether the
  drift is actually painful in practice.
