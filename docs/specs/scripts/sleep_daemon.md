# sleep_daemon.py

## Purpose

`scripts/sleep_daemon.py` is a memory-consolidation pressure monitor.
It polls Honcho (`HONCHO_URL`, default `http://localhost:8000`) every
`POLL_INTERVAL_SEC`, watches three signals on the active workspace, and
triggers a "nap" (dream / memory consolidation via the deriver) when
any of them fires:

- workspace idle for ≥ `IDLE_TIMEOUT_MINUTES`,
- pending queue rows ≥ `PENDING_THRESHOLD`,
- pending message tokens ≥ `TOKEN_THRESHOLD`.

Naps are rate-limited by `MIN_MINUTES_BETWEEN_NAPS`. On nap, the daemon
posts a bilingual "going to sleep" notification into the Honcho session
(so the user sees it in Hermes the next time they look), enqueues the
dream through the deriver container, waits for completion, and posts an
"awake" message.

The dream model itself is **not** chosen by this daemon. It comes from
`honcho/config.toml`'s `[dream.deduction_model_config]` and
`[dream.induction_model_config]`. In the current single-endpoint stack
both point at the shared qwen3.6 chat llama-server on `:8080`. Earlier
two-endpoint deployments split the dream off onto a separate
memory-specialised model (Bonsai-8B on a different port); that split is
archived in `experiments/bonsai-archive.md`.

## Usage

The daemon is intended to run as part of the Hermes runtime (systemd
user service, process-supervisor child, or a wrapper around `hermes`).
It is not bundled into `scripts/llama-services.sh` because its
lifecycle is tied to whatever front-end / session driver you run, not
to the llama-server processes.

Foreground, for debugging:

```bash
# Against the default local Honcho, default workspace.
python3 /home/baba-y/nuncstans-hermes-stack/scripts/sleep_daemon.py

# Tighter thresholds (dream more often).
PENDING_THRESHOLD=5 TOKEN_THRESHOLD=500 IDLE_TIMEOUT_MINUTES=3 \
python3 scripts/sleep_daemon.py

# Separate workspace / AI peer id.
HONCHO_WS=experiments AI_PEER=hermes-alt \
python3 scripts/sleep_daemon.py
```

The daemon has no CLI arguments.

## Options & env vars

| Env                          | Default                    | Effect                                                                                  |
| ---------------------------- | -------------------------- | --------------------------------------------------------------------------------------- |
| `HERMES_HOME`                | `~/hermes-stack`           | Repo root; used to compute `HONCHO_DIR`, `HERMES_CONFIG`, and the state file path       |
| `HONCHO_URL`                 | `http://localhost:8000`    | Honcho API base URL (host-side, not container-side)                                     |
| `HONCHO_WS`                  | `hermes`                   | Honcho workspace name                                                                   |
| `AI_PEER`                    | `hermes`                   | The AI peer id that posts "sleeping" / "awake" notifications                            |
| `POLL_INTERVAL_SEC`          | `30`                       | Seconds between polls                                                                   |
| `IDLE_TIMEOUT_MINUTES`       | `10`                       | Idle time before a nap can fire on the idle signal                                      |
| `PENDING_THRESHOLD`          | `10`                       | Pending queue rows that will trigger a nap                                              |
| `TOKEN_THRESHOLD`            | `1000`                     | Pending message tokens that will trigger a nap                                          |
| `MIN_MINUTES_BETWEEN_NAPS`   | `30`                       | Rate-limit between naps                                                                 |
| `OLLAMA_URL`                 | `http://localhost:11434`   | Only used by the "detect currently loaded model" fallback; added to the docstring on 2026-04-23 |

Derived paths (not env-driven):

- `HERMES_CONFIG` = `~/.hermes/config.yaml` — read to detect the chat
  model Hermes is using (for the notification text).
- `CONFIG_TOML`   = `$HERMES_HOME/honcho/config.toml` — referenced as
  the authoritative source of the dream model config.
- `STATE_FILE`    = `$HERMES_HOME/.sleep_daemon_state.json` — tracks
  last nap time so `MIN_MINUTES_BETWEEN_NAPS` survives restarts.

## Specs & assumptions

### Nap trigger logic

On every poll:

1. Load Honcho state for workspace `HONCHO_WS`: last user message time,
   pending queue count, pending message token sum.
2. If any of `(now − last_user_msg) ≥ IDLE_TIMEOUT_MINUTES`,
   `pending_rows ≥ PENDING_THRESHOLD`, or
   `pending_tokens ≥ TOKEN_THRESHOLD` — and the last nap finished more
   than `MIN_MINUTES_BETWEEN_NAPS` ago — fire a nap.
3. Otherwise sleep `POLL_INTERVAL_SEC` and repeat.

### Nap sequence

1. **Detect chat model.** Read `model.default` from
   `~/.hermes/config.yaml`. If that file is missing, fall back to
   `GET $OLLAMA_URL/api/ps` for the currently loaded ollama model.
   This value is used only for the human-readable notification text —
   the dream itself is driven by `honcho/config.toml`.
2. **Inject system message** into the active Honcho session: bilingual
   "going to sleep for consolidation" message, tagged `AI_PEER`.
3. **Enqueue the dream.** Shells into the deriver container to invoke
   the dream path (deduction + induction), which will in turn call
   whatever endpoint `honcho/config.toml`'s dream blocks point at.
4. **Wait for completion**, poll-driven.
5. **Inject awake message**, again via `AI_PEER`.
6. Update `STATE_FILE` with the new `last_nap_at` timestamp.

### Coupling

- **Honcho API** at `HONCHO_URL` — reads workspace state, posts
  notifications.
- **Deriver container** (via `docker compose`) — runs the dream itself.
- **Honcho config.toml** — authoritative for the dream model. Changing
  this daemon's env will not change what model actually does the
  dreaming; use `scripts/switch-endpoints.py` (or hand-edit
  `honcho/config.toml`) for that.
- **Hermes CLI config** (`~/.hermes/config.yaml`) — only for the
  notification text; if Hermes isn't installed, the daemon falls back
  to ollama.
- **Ollama `/api/ps`** — only for the fallback path above; unreachable
  ollama does not break the daemon, it just leaves the notification
  model-less.

### Single-instance assumption

The daemon has **no locking**. Two `sleep_daemon.py` processes against
the same Honcho workspace would both observe the same triggers, both
decide to nap, both enqueue a dream, both inject "sleeping" and "awake"
messages. The user would see duplicated notifications and the deriver
would do the consolidation work twice. The `STATE_FILE` is not a lock —
it is only consulted to rate-limit naps within one process's lifetime,
and concurrent writers will clobber each other.

Run one `sleep_daemon.py` per `HONCHO_WS`. Period.

### Pitfalls

- **Nap never fires.** Check `POLL_INTERVAL_SEC` is not larger than
  your idle window; check Honcho's pending queue is actually
  populated; check `STATE_FILE` isn't pinning `last_nap_at` to the
  future (e.g. after a clock skew).
- **Dream uses the wrong model.** This daemon does not control that —
  fix it in `honcho/config.toml` via
  [`switch-endpoints.py`](switch-endpoints.md), then restart the
  Honcho `deriver` container so it picks up the new config.
- **Notification text shows no model.** `~/.hermes/config.yaml` is
  absent *and* `OLLAMA_URL` is unreachable. Cosmetic; the nap itself
  still runs.

## Outlook

If you need to run this daemon against multiple Honcho workspaces on
one host, containerize it — one container per workspace, with its own
`HONCHO_URL` / `HONCHO_WS` / `STATE_FILE` and its own share of the
deriver's compose project. Do not extend this script to multiplex
across workspaces.

The same containerization argument as
[`llama-services.md`](llama-services.md) applies: isolation of state
files, independent restart lifecycles, and clean log streams per
workspace. Duplicate naps / race conditions against a single Honcho
workspace are the acute failure mode, so the containerization cost is
paid as soon as you have a second workspace in play.
