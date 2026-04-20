# Memory consolidation — the nap daemon

This section is extracted from the hermes-stack top-level README to keep the deployment guide focused on the minimal bring-up path. The `sleep_daemon` is **optional**: Honcho already fires its own idle-triggered dream on the schedule configured by `[dream] IDLE_TIMEOUT_MINUTES`, so readers following the default path do not need to run this daemon at all. If you just want to stand the stack up and hand it to a user, skip this file and use the main README's "Running, stopping, restarting" section for the minimal (daemon-less) start / stop commands. Come back here when you want pressure-triggered naps, user-visible session notifications, or tighter control over when consolidation fires.

Dream (Honcho's memory-consolidation agent) is what resolves contradictions between old and new observations — the mechanism that lets the stack overwrite "Alice lives in Kyoto" with "Alice moved" rather than accumulating both forever. On the CUDA 12.9 GPU build a full dream cycle takes ~14 seconds (see `benchmark.md`). A small daemon watches Honcho state and fires dreams when they're needed.

## Why dream runs on Bonsai (not the chat model)

The scaffold wires dream to Bonsai and we keep it that way. Why not send dream to Ollama / the chat model to free Bonsai?

- **Bonsai-8B is trained for this task.** It's a Neuromancer-derivative fine-tune built specifically for observation-level memory reasoning; its tool-call discipline on `search_memory` / `create_observations` / `delete_observations` is how dream resolves contradictions cleanly.
- **General chat models hallucinate during consolidation.** We tried routing dream to `qwen3.5:9b` (the Ollama chat model) and dream started inventing facts during the tool loop — a fake employer, a made-up age field — rather than reconciling the observations actually in memory. Any general-purpose chat model does this to some degree; the consolidation task is too close to "fill in plausible details."
- **Separate process, shared VRAM.** Bonsai's `llama-server` and Ollama are independent processes. Both sit in the same 16 GiB VRAM on the RTX 5080 (Bonsai ~7.6 GiB + chat model ~8 GiB). No GPU-level coordination is needed — each process manages its own context and streams separately. If the user types while a dream is mid-flight, Ollama still answers promptly on its model and Bonsai continues on its own.
- **Idle-firing is still a nice-to-have.** Honcho cancels pending dreams whenever a user message arrives (`cancel_dreams_for_observed` in `src/deriver/enqueue.py`), so dreams naturally batch during quiet moments. On GPU this matters less (a dream ends in ~14 s so even an interruption is short), but the cancel behaviour is still useful to avoid redundant consolidation right before fresh observations land.

In short: dream runs on Bonsai because it's the right model for the task, and on GPU the cost is low enough that it can fire whenever Honcho's scheduler (or the sleep daemon) decides it's warranted.

## What the daemon does

`scripts/sleep_daemon.py` is a long-running process that belongs to the Hermes runtime (not a separate tmux pane — you don't have to watch it). Every 30 seconds it:

1. **Detects** the chat model Hermes is currently using (from `~/.hermes/config.yaml` → `model.default`, falling back to Ollama `/api/ps`). This is logged for operator visibility only. Dream itself stays wired to Bonsai per `config.toml`; the daemon does not rewrite that.
2. **Polls** Honcho's queue, messages, and active-session state (pending representation rows, pending token sum, seconds since last message, whether a dream is currently in `active_queue_sessions`).
3. **Decides** whether a nap is due: `pending > PENDING_THRESHOLD` or `pending_tokens > TOKEN_THRESHOLD` or `seconds_since_last_msg > IDLE_TIMEOUT_MINUTES × 60`. Honcho's own scheduler fires on the idle condition too, so the daemon's added value is the pressure triggers.
4. **Injects** an English system message into the current Hermes session (posted as the AI peer with a `[System]` prefix so Hermes sees it in context). Examples:
   - `[System] 💤 I've been idle for 12 minutes. Taking a nap to consolidate memories.`
   - `[System] 🥱 My pending queue is backing up (14 items). Getting drowsy — going to take a short nap to catch up.`
   - `[System] ☕ Awake and refreshed. Memory consolidated.`
5. **Enqueues** a dream via `enqueue_dream()` inside the deriver container.
6. **Waits** for `active_queue_sessions` and `queue(task_type='dream')` to clear, then posts the awake message.

## Production: run it from Hermes's runtime

Make it a systemd user service so it starts and stops with Hermes:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/hermes-sleep-daemon.service <<'UNIT'
[Unit]
Description=Hermes/Honcho memory-consolidation daemon
After=default.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 %h/hermes-stack/scripts/sleep_daemon.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now hermes-sleep-daemon
systemctl --user status hermes-sleep-daemon
```

Or, if you already use a process supervisor with the `hermes` CLI, add it there. There is no UX need to expose the daemon's stdout — notifications reach the user via the session-message injection.

## Debug: run it in foreground

When you want to see what the daemon is thinking (why it did or did not nap, which model it detected, how long the dream took), run it in a terminal instead of systemd and raise the log level:

```bash
# stop the production service first (if any)
systemctl --user stop hermes-sleep-daemon 2>/dev/null

# foreground, verbose
PYTHONUNBUFFERED=1 \
  PENDING_THRESHOLD=5 TOKEN_THRESHOLD=500 IDLE_TIMEOUT_MINUTES=5 \
  POLL_INTERVAL_SEC=10 \
  python3 ~/hermes-stack/scripts/sleep_daemon.py 2>&1 | tee ~/hermes-stack/sleep_daemon.log
```

The `PENDING_THRESHOLD=5` / `IDLE_TIMEOUT_MINUTES=5` overrides make naps happen aggressively so you can see the whole flow within a minute or two. Watch `docker compose logs -f deriver` in another pane to see the dream side fire.

To prove the whole loop works end-to-end, send a contradictory pair of messages via `hermes`:

```
I'm Alice. I live in Kyoto and work as a backend engineer.
```
…wait for the deriver to record that, then:
```
Actually Alice moved to Osaka and works as a designer now.
```

After the daemon's next trigger (a few minutes, or immediately in debug mode), ask Hermes "where does Alice live now?" — the response should say Osaka, not Kyoto. Before the daemon and dream were in place, this test would return Kyoto because accumulated observations drowned out the update.
