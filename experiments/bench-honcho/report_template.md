# Honcho × Hermes Agent benchmark

Compares six Honcho/Hermes backend permutations on a single fixed task, measuring wall-clock and Honcho-side activity. Run on 2026-04-21 from `/home/baba-y/nuncstans-hermes-stack` on the hermes-stack dev branch.

## Task

Each run invokes Hermes non-interactively with a fresh session:

```
hermes chat -Q --yolo --max-turns 60 --source tool -q "<TASK>"
```

where `<TASK>` (fixed across all runs, `{N}` substituted per experiment) is:

> Create a sample deck with 5 slides. This PowerPoint is for presenting "New Opportunities for Local LLMs." Although it should include forecasts with quantitative evaluation, you can place placeholders for those sections. The file name should be "llm_presentation_{N}.pptx". Save it in the current working directory.

## Configurations

| # | Honcho deriver/dialectic/summary/dream LLM | Hermes main inference | Bonsai flags | Ollama env |
|---|---|---|---|---|
| E1 | llama.cpp + Bonsai-8B (:8080) | Ollama qwen3.5:9b | `--parallel 1 -c 16384` | `OVERHEAD=unset KEEP_ALIVE=-1 KV_CACHE=q4_0` |
| E2 | llama.cpp + Bonsai-8B (:8080) | Ollama qwen3.6:35b | `--parallel 1 -c 16384` | same |
| E3 | Ollama qwen3.5:9b | Ollama qwen3.6:35b | — (bonsai stopped) | same; two models swap under VRAM pressure |
| E4 | Ollama qwen3.5:9b | Ollama qwen3.5:9b | — (bonsai stopped) | same; single loaded instance |
| E5 | Ollama qwen3.6:35b | Ollama qwen3.6:35b | — (bonsai stopped) | same; single loaded instance |
| E6 | llama.cpp + Bonsai-8B (:8080) | llama.cpp + Bonsai-8B (:8080) | `--parallel 2 -c 32768` | same |

### Notes on the matrix
- The original request had 8 cells. Two were duplicates (original #3 ≡ #5, original #7 ≡ #6) and were removed, collapsing to 6.
- E2's inference model (qwen3.6:35b, ~23 GiB) exceeds the 16 GiB VRAM available with Bonsai resident; Ollama partially offloads to CPU. This cell is labelled **"VRAM-oversubscribed control."**
- E3 runs with `OLLAMA_MAX_LOADED_MODELS` unset (effective default: 1); the deriver (9b) and chat (35b) models swap per-call. This is intentional — we measure the thrash behavior.
- E6 uses Bonsai-8B for general chat, which it is not tuned for. Quality may be poor; the run is included as a one-model upper bound.

## Hardware
- NVIDIA RTX 5080, 16 GiB VRAM, CUDA 12.9
- Intel i5-14600K, 108 GiB system RAM
- WSL2 Ubuntu 24.04

## Methodology
Each experiment runs as:

1. Stop prior llama-server (if any); unload all loaded Ollama models via `POST /api/generate {keep_alive:0}`.
2. Start Bonsai llama-server with the target flags (if this experiment needs it).
3. Swap `honcho/config.toml` to the target variant (bonsai / ollama-9b / ollama-35b) — same three files in `bench-honcho/configs/` for each run.
4. `docker compose up -d api deriver` + `docker compose restart deriver api`; wait for `/health`.
5. Delete `queue` rows scoped to `bench-e*` workspaces; `FLUSHDB` redis.
6. Swap `~/.hermes/config.yaml` to the inference target (`qwen3.5:9b` / `qwen3.6:35b` / `bonsai-8b`) via `swap_hermes.py`; rewrite `~/.hermes/honcho.json` to use workspace `bench-e{N}`.
7. Clear `~/.hermes/checkpoints/*` + audio cache + history for a clean Hermes session.
8. Warm up: send 32-token chat request to both the inference and the Honcho-deriver endpoints.
9. Snapshot `ollama ps` and VRAM free; start 1 Hz `nvidia-smi` recorder.
10. `cd /home/baba-y/nuncstans-hermes-stack && time hermes chat -Q --yolo --max-turns 60 --source tool -q "<TASK>"` (20 min hard timeout).
11. After Hermes returns, poll Honcho queue until `workspace_name='bench-e{N}' AND processed=false` count is zero — **drain_seconds** captures Honcho's async background work that happened after the chat call returned.
12. Validate the produced pptx via `python-pptx`.

## Metrics recorded
- `wall_seconds`: time from Hermes invocation to process exit (includes dialectic start-of-session call + the chat turn + tool execution).
- `drain_seconds`: additional time for the Honcho deriver queue to reach zero pending for the bench workspace. Captures observations extracted from the saved chat turn.
- `total_seconds` = wall + drain.
- `observations_created`: documents rows attributed to the bench workspace.
- `pptx.ok`: file exists with exactly 5 slides.
- `vram_peak_mib`: max VRAM used during the run, 1 Hz sampling.
- `deriver_stats`: `⚡ PERFORMANCE` log lines, LLM call count, observation count from the deriver container.
- `bonsai_stats`: `print_timing` + slot-release line count in the shared llama-server log.

## Results

<!-- RESULTS_TABLE -->

### Per-cell detail

<!-- PER_CELL -->

## Discussion

<!-- DISCUSSION -->

## Caveats

- **Single run per cell (n=1).** Noise from OS scheduling, KV-cache temperature, and Ollama model-swap timing can move wall-clock ±20%. Treat ordinal rankings as signal, but individual numbers as indicative rather than authoritative.
- **Task characteristic.** The task is overwhelmingly an *inference-side* workload: one Hermes chat turn that plans, writes python-pptx code, and saves a file. Honcho-side activity is bounded to one dialectic call at session start (empty representation in a fresh bench workspace, so it returns quickly) and one or two deriver firings triggered asynchronously after Hermes returns. The Honcho-backend dimension therefore shows up primarily in `drain_seconds`, not in the Hermes wall-clock.
- **Stale production queue.** At run time the Honcho database held ~71 stale `queue` rows from unrelated workspaces (oldest 26 h old, all `status='pending'` but not being actively consumed — likely blocked by `active_queue_sessions` locks on the source sessions). These did not contend with the bench during its wall-clock window; we verified this by comparing deriver container GPU activity between bench and idle periods. They do, however, remain in the queue afterwards.
- **VRAM contention in E2.** When Bonsai is resident on the GPU and Ollama is asked for qwen3.6:35b (~23 GiB), Ollama partially offloads 35b layers to system RAM. This cell's wall-clock is primarily CPU-bound and is not comparable to the pure-GPU cells.
- **E6 model-quality risk.** Bonsai-8B was fine-tuned for Honcho's specific tool-call loops and memory reasoning, not general coding / presentation tasks. If E6 fails to produce a valid pptx, this reflects model suitability, not system throughput.
- **No replication, no Latin-square ordering.** Sequential execution in listed order; the first run pays a cold filesystem cache cost that the others do not. Re-run in a different order if that is a concern.
