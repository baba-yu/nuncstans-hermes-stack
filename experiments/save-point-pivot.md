# Save-point pivot — collapsing the per-turn memory tax

Session date: 2026-04-21

Continuation of `experiments/bottleneck.md`. That doc closed with the L6
single-endpoint config producing interactive latency in isolation
(~5.5 s per turn). This doc covers what happened when Hermes was actually
wired up against the stack and driven with real conversation — the
layer-by-layer bugs were gone, but every turn still cost 60+ seconds.
The fix was architectural, not tuning.

## TL;DR

| # | Symptom | Root cause | Fix | Owner |
|---|---|---|---|---|
| 1 | Every Hermes turn paid a 60 s stall, including `hello` | Honcho plugin's default `recallMode: "hybrid"` injects dialectic context into the prompt synchronously — one LLM call per turn, with a 60 s `tenacity` timeout | `recallMode: "tools"` in `~/.hermes/honcho.json` — hide auto-injection, keep the memory tools exposed so recall only fires when the model decides it needs context | User (explicit save-point analogy) → config flip |
| 2 | After (1) was fixed, a 20-minute conversation wasn't saved — Hermes had no memory of Chachamaru across sessions | Honcho plugin deferred session creation until the first tool call; with `recallMode: "tools"` + a conversation that didn't need recall, no tool ever fired, no session ever opened, messages were never posted | `initOnSessionStart: true` in `~/.hermes/honcho.json` — force the plugin to open the session on `chat.start` instead of lazy-on-first-tool | Reading the plugin source after user called out the skipped verification |
| 3 | Gatekeeper and sleep daemon roles were unclear after the single-endpoint collapse | Both were documented as Bonsai-era components; it wasn't obvious whether they were still load-bearing in the new topology | Confirmed both keep doing work: gatekeeper filters junk out of pending representations, sleep daemon fires Dream under idle/pressure. Documented in README "Persistent runtime assets". | This writeup |

After (1) and (2), Hermes turns felt genuinely fast on the L6 config — a
short user message completes in roughly the prompt-eval + decode budget
of the chat server alone, not prompt-eval + dialectic + decode.

## 1. Why every turn was stalling for 60 seconds

`experiments/bottleneck.md` showed L6 at 5.5 s per turn in isolation.
Under the real Hermes runtime the same turn cost 60+ seconds.
`watch_memory.sh` showed two `launch_slot_` events per user message on
`:8080`:

1. One dialectic call to Honcho (auto-injected recall context).
2. One chat completion (the actual reply).

The dialectic call was the one stalling, and then the chat completion
only started after the dialectic either completed or timed out.
Dialectic's own backoff wraps the LLM call in `tenacity` with a 60 s
ceiling, so the observed "~60 s then the answer starts streaming"
matched timeout behaviour, not slow-but-completing inference.

This was not a bug in the bottleneck sense; it was the hybrid recall
path doing exactly what it was configured to do, on every turn, for
every prompt, whether or not the content needed recall.

### The save-point framing

> それってさ、コンテキスト考慮して回答してくれなくなるんじゃない？
> recall自体が何のときに行われるのかがわからん。
> — user, mid-session

The user's framing was the right one: **treat memory as a save-point,
not a per-frame render**. Writes should fire when something worth
remembering is said; reads should fire when the model asks for them.
Nothing about conversation flow inherently demands per-turn memory
context.

Honcho's plugin offers three recall modes:

- `hybrid` — auto-inject dialectic context into the prompt **and**
  expose memory tools. One LLM call per turn on Honcho's side, always.
- `context` — auto-inject only, no tools. Same per-turn cost as hybrid.
- `tools` — tools only, no auto-injection. Recall fires only when the
  chat model calls `search_memory` / `get_observation_context` / etc.

Flipping to `tools` is the save-point configuration. Every turn that
doesn't need memory skips the dialectic call entirely. Every turn that
*does* need memory pays exactly one extra LLM call to go fetch it, on
demand, via a tool invocation the model made deliberately.

### Write path: still async

The pivot only changed the *read* path. Writes (message ingestion →
deriver → representation extraction) stayed `async`. Every user turn
still enqueues a representation task; the gatekeeper classifier still
decides whether it's worth promoting to an observation; the deriver
still runs post-message in the Honcho container. The chat model does
not wait on any of this. This is the equivalent of the autosave:
silently accumulating state, not blocking the player.

## 2. The lazy-init bug that `recallMode: tools` exposed

After the recall-mode flip, the user ran a 20-minute interactive
conversation with Hermes (roleplay with the AI calling itself
"Chachamaru"). Response latency was comfortable, no dialectic stalls,
everything felt right.

Next session: Hermes didn't remember Chachamaru. Or anything. Not even
the name.

The user's response:

> 裏バグじゃないよ、だからセーブポイントいるんじゃないのって言ったじゃん。
> 確認漏れだろ普通に。

Correct. The verification step after the recallMode flip had been
skipped — the user had already warned that save-points needed an
explicit write gesture, and the skipped verification was "did the
asynchronous write path actually execute end-to-end after the flip?"

Reading the Honcho Hermes plugin source, the relevant block has a
comment that made the issue obvious once found:

```typescript
// Defer actual session creation until first tool call
// [in onChatStart]
```

The plugin's `onChatStart` hook stored the session *intent* (peer IDs,
workspace) in local state but did not call `workspaces.sessions.create`.
That call was triggered by the first tool invocation inside the session.

Under `hybrid` / `context`, the first turn always calls a tool (the
dialectic auto-inject runs inside the plugin and counts as the first
tool call), so the session exists by the time `messages.create` is
invoked. Under `tools`, a conversation that doesn't need recall never
calls a tool. `messages.create` was then called against a non-existent
session and the plugin silently swallowed the error. Twenty minutes of
conversation, zero messages persisted.

### Fix

Honcho plugin config gained `initOnSessionStart: true`:

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "Octoball",
      "peerName": "Yuki",
      "workspace": "hermes",
      "observationMode": "directional",
      "writeFrequency": "async",
      "recallMode": "tools",
      "dialecticCadence": 3,
      "sessionStrategy": "per-session",
      "saveMessages": true,
      "initOnSessionStart": true
    }
  }
}
```

This flag flips the plugin's `onChatStart` to eagerly call
`workspaces.sessions.create` before any user turn fires. With it,
`recallMode: "tools"` has the expected semantics: no per-turn read
tax, but every message still lands in the session.

### Why this matters beyond one bug

The two settings compose. `recallMode: "tools"` is the latency win;
`initOnSessionStart: true` is the write-path integrity guarantee that
`recallMode: "tools"` needs. Either one alone is wrong:

- `hybrid` + `initOnSessionStart: true` — no perf win, but integrity fine.
- `tools` + `initOnSessionStart: false` — great latency, zero memory.
- `tools` + `initOnSessionStart: true` — what we want.

Both live in `~/.hermes/honcho.json`, not in this repo. The file is
user-local state; this writeup exists so the next person knows which
two knobs matter and why.

## 3. Role clarification after the single-endpoint collapse

With the two-endpoint (Bonsai + Ollama) design archived, it wasn't
self-evident what the supporting daemons were still contributing.
User asked directly:

> じゃあgatekeeperもsleepも明確に役割があるってことね。

Confirmed. The L6 collapse changed *what LLM* those daemons talk to
(both now point at the shared `:8080` chat server), not whether they
have work to do.

### Gatekeeper daemon (`scripts/gatekeeper_daemon.py`)

**Purpose: memory-quality filter.** Classifies every pending
representation queue row before the deriver picks it up. Rows that
look like trivia, non-literal framing, or low-importance noise get
`demoted` and never become observations. Rows that look like durable
facts get promoted to `ready` and the deriver processes them.

- **Input**: `queue.status='pending'`, `task_type='representation'`
- **Output**: rows move to `status='ready'` (deriver picks up) or
  `status='demoted'` (deriver skips)
- **Why save-point doesn't obsolete this**: even when recall is on
  demand, writes still happen on every user message. The save-point
  framing is about *when Hermes reads*; it's not a decision about
  *what gets written*. Without the gatekeeper every "hi" and "thanks"
  becomes an observation, recall gets polluted with trivia, and the
  next dialectic call has to sift through noise.
- **Classifier decision rules**: A/B independent axes + importance +
  correction_of_prior, with δ=0.20 margin and τ=0.75 logprob-confidence
  floor (calibrated in `scripts/gatekeeper_eval/`). Full prompt is
  embedded in the daemon; see `SYSTEM_PROMPT` in
  `scripts/gatekeeper_daemon.py`.

### Sleep daemon (`scripts/sleep_daemon.py`)

**Purpose: consolidation trigger.** Fires Honcho's Dream agent under
three conditions: (a) workspace idle for N minutes, (b) pending queue
backlog exceeds threshold, (c) pending-message token sum exceeds
threshold. Dream is the agent that resolves contradictions between
old and new observations (e.g., "I live in Kyoto" → "I moved to
Osaka" should supersede, not coexist).

- **Why save-point doesn't obsolete this**: observations accumulate
  over days; without consolidation the representation slowly fills with
  superseded facts that all score similarly on recall. Dream's job is to
  prune and merge. The sleep daemon just picks the moments to fire.
- **State relative to this pivot**: currently not running in this
  deployment. With <1000 observations accumulated, the thresholds
  haven't been worth firing yet. Left off deliberately; will switch on
  when the document count crosses the pressure threshold.

### What's not running and why

- `ollama serve` — retired. Embeddings moved to the second
  `llama-server` on :8081; chat moved to the shared `:8080`.
- The legacy two-endpoint Bonsai-8B `llama-server` — archived in
  `experiments/bonsai-archive.md`. Infrastructure preserved (the
  `llama.cpp/` submodule is the source of the llama-server
  binary we still use), but no process dedicated to Bonsai-8B as a
  separate model.

## 4. What changed in the repo today

- `scripts/gatekeeper_daemon.py` — renamed `BONSAI_URL`/`BONSAI_MODEL`
  env vars to `GK_LLM_URL`/`GK_LLM_MODEL` (Bonsai names still accepted
  as deprecated fallback). Renamed `_call_bonsai` → `_call_classifier`,
  audit key `bonsai_call_ms` → `classifier_call_ms`, log strings.
- `scripts/llama-services.sh` — updates the env vars it passes to
  `start_gatekeeper` to the new names, and uses `$CHAT_ALIAS` instead
  of hard-coding `qwen3.6-test` for the model env. (The `LLAMA_BUILD`
  path still points at `bonsai-llama.cpp/build/bin` — submodule
  directory rename is deliberately not done; README already documents
  this as historical.)
- `scripts/gatekeeper_eval/shadow.py` — same env rename pattern;
  default model changed from `"bonsai-8b"` to `"qwen3.6-test"`;
  `/v1/models` health check no longer hard-codes the old alias.
- `scripts/sleep_daemon.py` — updated doc comments that still
  referenced dream-runs-on-Bonsai. The daemon never ran on Bonsai
  directly; it just observed Honcho state and fired dream. Only
  comments were stale.

No wire-protocol changes (the audit JSON key rename is write-only;
no code reads the old key).

## 5. What we verified works

After all changes applied and processes restarted:

- `hello` turns: sub-second (no dialectic call).
- Turns that ask about memory: model issues `search_memory` tool call,
  response includes the recalled context, total turn time is
  prompt-eval + one tool-call round-trip + decode — still inside the
  interactive window.
- Cross-session memory: after a 20-minute session, relaunch Hermes
  (fresh session), ask "what do you know about me", recall fires,
  response includes facts from the prior session. Verifies that
  `messages.create` is actually landing under `initOnSessionStart: true`
  + `recallMode: "tools"`.
- Gatekeeper classifier logs show pending representation rows being
  promoted to `ready` or `demoted` at the expected cadence when
  messages are actively flowing.

## 6. What remains open (deliberately)

- **Sleep daemon not running.** Deferred until the document count
  makes dream pressure worthwhile. Low priority.
- ~~**Bonsai submodule dir rename.**~~ Done in a follow-up commit
  on 2026-04-23: `bonsai-llama.cpp/` → `llama.cpp/`. The rename was
  straightforward because the `$ORIGIN` RPATH means the built binary
  relocates without rebuild, and the ripple was smaller than feared
  (one path in `scripts/llama-services.sh`, two in
  `experiments/bench-moe-offload/`, a few current-setup references
  in README — archival docs left at their original path names).
- **Honcho upstream tool_choice patch.** Local fork still carries the
  `tool_choice: "any" → "required"` normalization. Not upstreamed by
  user decision (see `bottleneck.md` §12).
- **`#20003` qwen3 full-prompt re-prefill.** Still present; ~85 ms
  hit per 4k-token turn on this hardware. Not worth fighting now.
