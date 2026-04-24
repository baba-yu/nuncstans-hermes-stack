# gatekeeper_daemon.py

## Purpose

`scripts/gatekeeper_daemon.py` is an async classifier that moves rows
out of Honcho's representation queue's `pending` state. Every
representation task produced by the deriver starts as `pending`; the
daemon polls the queue, calls an OpenAI-compatible classifier LLM with
a JSON-schema-constrained prompt, and writes one of four outcomes back
into `gate_verdict` + `status`:

- `ready` — deriver picks it up and commits it as an observation,
- `demoted` — never promoted (explicit non-literal / hypothetical / sarcasm),
- `pending` (stay) — margin too small, wait for re-evaluation,
- `pending` (stay) — logprob confidence under `TAU`, wait regardless of margin.

Today the classifier endpoint is the shared qwen3.6 chat llama-server on
`:8080` (managed by `scripts/llama-services.sh`, which also starts this
daemon). Historically the classifier was a separate Bonsai-8B instance
on a different port; the `BONSAI_URL` / `BONSAI_MODEL` env vars are
still accepted as deprecated aliases.

## Usage

The daemon is normally started by `scripts/llama-services.sh start`
(after the chat server is healthy) and stopped by
`scripts/llama-services.sh stop`. For debugging / tuning it can be run
directly:

```bash
# Run in the foreground with live output.
GK_LLM_URL=http://localhost:8080 \
GK_LLM_MODEL=qwen3.6-test \
HERMES_HOME=/home/baba-y/nuncstans-hermes-stack \
HONCHO_DIR=/home/baba-y/nuncstans-hermes-stack/honcho \
python3 /home/baba-y/nuncstans-hermes-stack/scripts/gatekeeper_daemon.py

# Swap the classifier model for a shadow evaluation.
GK_LLM_URL=http://localhost:8080 \
GK_LLM_MODEL=my-experimental-alias \
python3 scripts/gatekeeper_daemon.py

# Tighter decision rules (faster promotions, lower confidence floor).
GK_DELTA=0.15 GK_TAU=0.70 python3 scripts/gatekeeper_daemon.py
```

The daemon has no CLI arguments; every knob is an env var.

## Options & env vars

| Env                     | Default                 | Effect                                                                             |
| ----------------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| `HERMES_HOME`           | `~/hermes-stack`        | Repo root; used to locate `HONCHO_DIR` and call `docker compose` there             |
| `HONCHO_DIR`            | `$HERMES_HOME/honcho`   | Working dir for `docker compose exec -T database psql …` calls                     |
| `GK_LLM_URL`            | `http://localhost:8080` | OpenAI-compat classifier endpoint (no `/v1` suffix in the default)                 |
| `GK_LLM_MODEL`          | `qwen3.6-test`          | Model id to pass; must match the server's `--alias`                                |
| `GK_POLL_INTERVAL_SEC`  | `5`                     | Seconds between queue polls                                                        |
| `GK_DELTA`              | `0.20`                  | Margin threshold on `A_score − B_score` for ready/demoted decisions                |
| `GK_TAU`                | `0.75`                  | Logprob confidence floor; verdicts below this stay pending                         |
| `GK_REEVAL_AFTER_SEC`   | `90`                    | Seconds before a pending row is eligible for re-classification                     |
| `GK_REEVAL_MAX`         | `2`                     | Max re-classification attempts before forcing a commit on the last verdict         |
| `GK_BATCH_LIMIT`        | `10`                    | Max rows pulled per poll                                                           |
| `GK_MAX_OUTPUT_TOKENS`  | `400`                   | Classifier response cap                                                            |
| `BONSAI_URL`            | —                       | **Deprecated alias** for `GK_LLM_URL`                                              |
| `BONSAI_MODEL`          | —                       | **Deprecated alias** for `GK_LLM_MODEL`                                            |

`llama-services.sh` exports only `GK_LLM_URL`, `GK_LLM_MODEL`,
`HERMES_HOME`, and `HONCHO_DIR`. Both `GK_LLM_URL` and `GK_LLM_MODEL`
are read from `scripts/llama-services.conf` (they are no longer
hardcoded in the shell script), which means `switch-endpoints.py` can
keep them in sync with the Honcho chat endpoint automatically — and
does, on every Axis A run. See
[`switch-endpoints.md`](switch-endpoints.md) § "Gatekeeper follows
chat engine" for the rationale and the hand-edit escape hatch.

All other `GK_*` values (`GK_DELTA`, `GK_TAU`, polling / re-eval /
batch / output-tokens) fall back to the daemon's defaults listed
above. To override them in the normal lifecycle, either edit the
script, wrap `start_gatekeeper`, or kill the daemon and relaunch it
by hand with your env.

## Specs & assumptions

### Decision rules

With A_score = literal-self-reference, B_score = non-literal framing
strength, both on [0.0, 1.0]:

| Condition                                              | Outcome                             |
| ------------------------------------------------------ | ----------------------------------- |
| `A − B ≥ GK_DELTA` **and** `logprob_conf ≥ GK_TAU`     | `status='ready'`                    |
| `B − A ≥ GK_DELTA` **and** `logprob_conf ≥ GK_TAU`     | `status='demoted'`                  |
| `|A − B| < GK_DELTA`                                    | stay `pending`                      |
| `logprob_conf < GK_TAU` (regardless of margin)         | stay `pending`                      |

Rows stuck in `pending` are re-classified after `GK_REEVAL_AFTER_SEC`.
After `GK_REEVAL_MAX` attempts the daemon forces a commit: `ready` if
the last verdict had `A ≥ B`, else `demoted`. Forced verdicts are
stamped with `forced: true` in `gate_verdict` for later auditing.

Only rows with `status='pending'` and `task_type='representation'` are
touched. Everything else is ignored.

### Classifier prompt

The prompt is `CLASSIFIER_VERSION = "gatekeeper-v3"`, frozen after a
shadow calibration that produced 90% agreement with hand-labelled
ground truth. A / B axes are independent (a message can be both
literal and non-literal, e.g. sarcasm about a real event); importance
correlates weakly with A (r = 0.32) and is kept as its own dimension.
All four scores (`A_score`, `B_score`, `importance`,
`correction_of_prior`) are requested as a single JSON object via
schema-constrained output.

### DB access path

The daemon does not open a direct DB connection. It shells out to:

```
docker compose exec -T database psql -U honcho -d honcho
```

run from `$HONCHO_DIR`. Consequences:

- `DATABASE_URL` is **not** required.
- Honcho's compose stack must be up (`database` service healthy).
- Two gatekeeper instances against the same `database` container would
  both try to `UPDATE` the same queue rows; the daemon has no row-level
  locking beyond what PostgreSQL gives for free on the `UPDATE`. Don't.

### Single-instance assumption

No pid file, no advisory lock, no coordination. Running two daemons
against the same Honcho workspace will produce duplicate classifier
calls (wasted LLM budget) and racy writes to `gate_verdict`. If you
need more classifier throughput, raise `GK_BATCH_LIMIT`; if you need
isolation between workspaces, containerize one daemon per workspace
(see Outlook).

### Coupling to `llama-services.sh`

The daemon's lifecycle is bundled into `scripts/llama-services.sh`:

- `start` launches it last (after the chat server is healthy at
  `:8080`), writes its pid to `$HERMES_STATE_DIR/gatekeeper.pid`, and
  redirects stdout+stderr to `$HERMES_STATE_DIR/gatekeeper.log`.
- `stop` kills it first (before the chat server) so it doesn't see a
  missing classifier mid-poll.
- `logs gk` tails the log.
- The daemon itself writes no pid / log files — that's the shell's job.

### Classifier endpoint pitfalls

Since the classifier is the shared chat llama-server, the same three
ollama caveats described in
[`switch-endpoints.md`](switch-endpoints.md) apply when pointing the
daemon at ollama instead: `OLLAMA_CONTEXT_LENGTH` silent truncation,
qwen3 `think` flag (missing from OpenAI-compat API), and variable
tool-call support. The daemon does not call tools, but does rely on
structured-output / JSON-schema support, which is more stable on
llama-server than on ollama.

## Outlook

- **Model swap.** Changing classifier model is just
  `GK_LLM_URL` / `GK_LLM_MODEL`. There's no other coupling — the prompt
  is frozen, the schema is frozen, and the daemon doesn't care what's
  behind the URL as long as it speaks OpenAI chat-completions with
  logprobs. In practice these two keys are driven by
  `switch-endpoints.py` to follow the Honcho chat engine (one source
  of truth for the active engine). Hand-editing the conf keys is the
  escape hatch when you want a dedicated small classifier — the
  switcher will rewrite them on the next Axis A run, so either pin
  the conf and avoid rerunning Axis A, or accept that the override
  is a per-session override.
- **Shadow evaluation.** The harness for running a candidate
  classifier against labelled queue snapshots lives under
  `scripts/gatekeeper_eval/`. Workflow: freeze a corpus, run the
  current model + candidate against it, diff the verdicts, look at
  the disagreements by hand. Calibration of `GK_DELTA` / `GK_TAU`
  should come out of the same harness.
- **Multi-instance / multi-workspace.** Use one container per
  workspace; give each container its own `HERMES_HOME` pointing at a
  repo/compose-project pair and let the container runtime handle
  isolation. Do not try to bolt workspace-awareness into this script.
