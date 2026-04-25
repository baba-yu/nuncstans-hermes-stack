# llama-services.sh

## Purpose

`scripts/llama-services.sh` is the process supervisor for the three
host-side daemons that make up the hermes-stack's local LLM plane:

- the **chat llama-server** on `:8080` (OpenAI-compatible, serves the
  main chat model and is what Hermes / Honcho / gatekeeper all point at),
- the **embedding llama-server** on `:8081` (OpenAI-compatible
  `/embeddings`, aliased as `openai/text-embedding-3-small` so Honcho's
  hardcoded embedding client finds it), and
- the **gatekeeper daemon** (classifies representation queue rows; needs
  the chat server up to answer, so it is bundled into this script's
  lifecycle rather than run separately).

Parameters for the two llama-servers live in
[`scripts/llama-services.conf`](../../../scripts/llama-services.conf);
[`scripts/switch-endpoints.py`](switch-endpoints.md) is the automated
writer for that file. This script reads it on every start.

## Usage

```bash
# Start / stop / restart everything — optional target picks a single service.
./scripts/llama-services.sh start          # all = embed + chat + gk
./scripts/llama-services.sh start embed    # only the embedding server
./scripts/llama-services.sh start chat     # only the chat server
./scripts/llama-services.sh start gk       # only the gatekeeper daemon

./scripts/llama-services.sh stop           # reverse order: gk → chat → embed
./scripts/llama-services.sh stop chat      # stop one (useful for VRAM reclaim
                                           # when chat is no longer in use)

./scripts/llama-services.sh restart        # stop all + start all
./scripts/llama-services.sh restart gk     # restart one

./scripts/llama-services.sh status         # pid / port / health per service

./scripts/llama-services.sh logs chat      # tail one log. Targets: chat | embed | gk
```

All subcommands are idempotent; `start` checks the tracked pid file and
the port before launching. If a port is bound but no tracked pid exists,
the script refuses to start that service (rather than killing a possibly
unrelated process) and tells the operator to investigate.

The per-target variants are specifically there so operators can reclaim
VRAM after a `switch-endpoints` run has moved Honcho chat (and
transitively the gatekeeper) off llama-server onto ollama — at that
point `./scripts/llama-services.sh stop chat` frees the `:8080` VRAM
without touching the embedding server or the daemon.

Typical first-time start:

```bash
./scripts/llama-services.sh start
# [llama-services] starting embedding server on :8081
# [llama-services]   embedding server ready (pid …)
# [llama-services] starting chat server on :8080 (alias=qwen3.6-test, ctx=131072, …)
# [llama-services]   chat server ready (pid …)
# [llama-services] starting gatekeeper daemon
# [llama-services]   gatekeeper running (pid …)
```

## Options & env vars

### Environment variables

| Env                | Default                                         | Effect                                                                                 |
| ------------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------- |
| `HERMES_STATE_DIR` | `$HOME/.local/state/nuncstans-hermes-stack`     | Directory for pid files, log files, and `endpoint-snapshots/` (shared with switch-endpoints). |
| `LD_LIBRARY_PATH`  | (extends with `/usr/local/cuda/lib64`)          | Exported before each llama-server launch so CUDA libs resolve.                         |

The script also exports a small environment to the gatekeeper daemon
on launch:

| Exported to gatekeeper | Source                                    |
| ---------------------- | ----------------------------------------- |
| `GK_LLM_URL`           | `$GK_LLM_URL` from the conf (default `http://localhost:8080`; kept in sync with the Honcho chat endpoint by `switch-endpoints.py` on every Axis A run) |
| `GK_LLM_MODEL`         | `$GK_LLM_MODEL` from the conf (default `$CHAT_ALIAS`; also kept in sync by the switcher) |
| `HERMES_HOME`          | repo root (resolved from the script path) |
| `HONCHO_DIR`           | `$HERMES_HOME/honcho`                     |

Other `GK_*` environment variables (`GK_DELTA`, `GK_TAU`,
`GK_POLL_INTERVAL_SEC`, `GK_REEVAL_AFTER_SEC`, `GK_REEVAL_MAX`,
`GK_BATCH_LIMIT`, `GK_MAX_OUTPUT_TOKENS`) are **not** exported — the
daemon falls back to its own defaults. To override them, either edit
the script, wrap `start_gatekeeper` in a helper, or run the daemon by
hand. See [`gatekeeper_daemon.md`](gatekeeper_daemon.md) for the
defaults.

### `llama-services.conf` keys

The conf is `bash`-sourced (simple `KEY=VALUE`, double-quoted string
values). Every key below has a safe default baked into the shell
script; missing keys fall back to those defaults rather than erroring.
`switch-endpoints.py` is the automated writer; hand-edits are fine but
watch out: the Python rewriter only preserves comments and blank lines,
not inline comments after values.

| Key                   | Default                                                                         | Affects                                                                          |
| --------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `CHAT_HF_SPEC`        | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL`                                       | `-hf repo:quant` or `-m /path` (leading `/` triggers `-m`)                       |
| `CHAT_ALIAS`          | `qwen3.6-test`                                                                  | `--alias` advertised by `/v1/models`; must match Honcho/Hermes `model`           |
| `CHAT_CTX`            | `131072`                                                                        | `-c <n>` context window; must be ≤ `n_ctx_train`                                 |
| `CHAT_NGL`            | `99`                                                                            | `-ngl` GPU layers (99 = all)                                                     |
| `CHAT_IS_MOE`         | `1`                                                                             | `1` adds `-ot "ffn_(up|down|gate)_exps=CPU"` (MoE expert CPU offload)            |
| `CHAT_REASONING_OFF`  | `1`                                                                             | `1` adds `--reasoning off` (qwen3 suppresses thinking tokens)                    |
| `CHAT_PARALLEL`       | `2`                                                                             | `--parallel` concurrent slots; dense ≥ 35B → 1 recommended                       |
| `EMBED_BLOB`          | `/usr/share/ollama/.ollama/models/blobs/sha256-970aa74c…`                       | `-m` path to the embedding GGUF                                                  |
| `EMBED_ALIAS`         | `openai/text-embedding-3-small`                                                 | `--alias`; do not change unless also updating Honcho's embedding code path       |
| `EMBED_NGL`           | `99`                                                                            | `-ngl` for the embedding server                                                  |
| `GK_LLM_URL`          | `http://localhost:8080`                                                         | Base URL (no `/v1`) the gatekeeper daemon sends classifier requests to; `switch-endpoints.py` Axis A keeps this in step with the Honcho chat endpoint automatically |
| `GK_LLM_MODEL`        | `qwen3.6-test`                                                                  | Model id sent in the classifier request; also auto-synced by the switcher       |

Non-configurable flags also passed to the chat server:
`-fa on -ctk q8_0 -ctv q8_0 --jinja`. Change these by editing the
script itself.

## Specs & assumptions

### Single-instance per host

This is the central assumption. Every pid file and log file is derived
from `$HERMES_STATE_DIR`, and the ports (8080, 8081) are hardcoded. A
second `llama-services.sh` invocation against the same state dir will
see the existing pids and exit idempotently — it will not launch a
second chat server. Running a parallel instance against different ports
requires either:

- an alternate state dir:
  ```bash
  HERMES_STATE_DIR=/var/lib/hermes-alt ./scripts/llama-services.sh start
  # but you would also need to edit the port numbers, and to point that
  # instance's gatekeeper at a different chat URL — this script does not
  # parametrize those.
  ```
- or full containerization, which is the recommended path.

### State dir layout

```
$HERMES_STATE_DIR/
├── chat-server.log
├── chat-server.pid
├── embed-server.log
├── embed-server.pid
├── gatekeeper.log
├── gatekeeper.pid
└── endpoint-snapshots/         # written by switch-endpoints.py
    └── <ts>.<pid>/
```

### Start / stop ordering

`start` runs `embed → chat → gatekeeper`. Reason: the gatekeeper's
first poll may fire within seconds of launch and it calls the chat
server on `:8080`, so the chat server has to be healthy first. Embed
is started first because the chat server's cold-load path warms up
the embedding client elsewhere in the stack.

`stop` reverses: `gatekeeper → chat → embed`. This keeps the gatekeeper
from hitting a missing chat server mid-classification.

### Health checks

- Chat: `GET http://127.0.0.1:8080/health`, 900s timeout on first start
  (covers a cold HF download for a large quant), 2s per attempt.
- Embed: `GET http://127.0.0.1:8081/health`, 60s timeout.
- Gatekeeper: no HTTP health endpoint; the script sleeps 2 seconds and
  checks the pid is still alive. If the daemon crashes during
  initialisation, `start` dies here.

### Pitfalls that affect this script

These are all driven by the conf file (and therefore by
`switch-endpoints.py`), but worth flagging:

- `CHAT_CTX` must be ≤ the model's `n_ctx_train`. Too large → llama-server
  refuses to start and the `wait_health` loop eventually times out with
  a generic message; check `logs chat` for the actual error.
- `CHAT_IS_MOE=1` with a dense model adds `-ot` flags that no tensor
  matches; llama-server logs a warning but still starts. Prefer setting
  it correctly via `switch-endpoints.py`.
- `CHAT_REASONING_OFF=1` only has an effect on qwen3-family models.
  Harmless on others.
- Ollama-specific pitfalls (context truncation, qwen3 think flag, tool
  call stability) do not affect this script directly, but they apply
  when you point `switch-endpoints.py` at an ollama endpoint — see
  [`switch-endpoints.md`](switch-endpoints.md) for the full list.

### Upgrade notes

**2026-04-23.** The default state dir was renamed from
`~/.local/state/hermes-stack/` to
`~/.local/state/nuncstans-hermes-stack/` to match the repo name. To
migrate an existing install:

```bash
mv ~/.local/state/hermes-stack ~/.local/state/nuncstans-hermes-stack
```

Atomic rename, safe while services are running: open file descriptors
keep writing to the moved inodes. The pid files themselves only contain
pid numbers (no absolute paths), so they remain valid across the rename.
Alternatively, set `HERMES_STATE_DIR=~/.local/state/hermes-stack` to
keep the old location.

## Outlook

If you need two or more Hermes instances on one host, containerize
Hermes rather than extending this script. Three reasons:

1. **VRAM arbitration.** Two llama-servers on the same GPU is already
   a non-trivial resource decision; hand-rolling a bash supervisor for
   it will fight with nvidia's own accounting. A container runtime plus
   an NVML-aware scheduler is the right layer.
2. **Independent restart lifecycle.** Today `restart` is all-or-nothing.
   Isolating that per-instance in bash means one pid file per service
   per instance, which is a path toward reimplementing `systemd` badly.
3. **Isolated log streams.** Per-instance `$HERMES_STATE_DIR` works for
   files but the journald / docker-logs path is a cleaner story for
   aggregation, retention, and rotation.

Do not bolt process-supervisor logic onto this script. Its job is the
single-instance happy path: start three things, stop three things, tail
one log. Anything beyond that belongs one level up (systemd, docker
compose, k8s, whatever).

### Engine consistency (as of this version)

- The gatekeeper classifier is now tied to the Honcho chat engine
  (ollama or llama-server) through `GK_LLM_URL` / `GK_LLM_MODEL` in the
  conf. `switch-endpoints.py` rewrites both keys on every Axis A run —
  one source of truth, no silent classifier breakage on engine switch.
- Operators who want a dedicated lightweight classifier on a separate
  URL/model can hand-edit the two `GK_*` keys in `llama-services.conf`.
  That override survives until the next Axis A run; after that the
  switcher puts them back in sync with the chat endpoint.
- After `switch-endpoints.py` moves Honcho chat and (via the embed
  engine-match offer) Honcho embed to ollama, the llama-server chat on
  `:8080` has no caller. Reclaim the VRAM with
  `./scripts/llama-services.sh stop chat`; embed and gk keep running.
  The per-target subcommands (`start`/`stop`/`restart {chat|embed|gk}`)
  exist specifically for this flow.
