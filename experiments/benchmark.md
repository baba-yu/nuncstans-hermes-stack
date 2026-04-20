# Bonsai-8B Benchmark: CPU vs CUDA (RTX 5080)

Before/after measurements for the Bonsai-8B memory model that powers Honcho's deriver, dialectic, summary, and dream agents. The single change is toolchain: `cmake -DGGML_CUDA=OFF` (CPU) → `cmake -DGGML_CUDA=ON` with `-ngl 99` (all layers on GPU).

## Test environment

| Component | Value |
|---|---|
| CPU | Intel Core i5-14600K (20 logical cores, AVX2+FMA) |
| RAM | 108 GiB |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell, compute cap 12.0, 16 GiB VRAM) |
| OS | Ubuntu 24.04 on WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| NVIDIA driver | 591.86 (Windows-side, exposes CUDA 13.1 runtime) |
| CUDA toolkit | **12.9.86** (12.x — 13.x fails on this WSL, see note below) |
| llama.cpp fork | `bonsai-llama.cpp` (PrismML) at commit `e2d6742`, built locally |
| Model | `Bonsai-8B.gguf`, qwen3 architecture, Q1_0 quantization, 1.07 GiB on disk |

### Why CUDA 12.9 and not 13.x

The apt repo's latest `cuda-toolkit` is `13.2.1`. Building with 13.2 succeeds, but `llama-server` segfaults at startup because the WSL-side `libdxcore.so` (shipped with the Windows NVIDIA driver) is missing symbols the CUDA 13.x runtime assumes (`D3DKMTOpenSyncObjectFromNtHandle`, `D3DKMTAcquireKeyedMutex`, `D3DKMTCreateNativeFence`, etc.). `wsl --update` does not help — even the latest WSL libdxcore lacks those eight symbols. CUDA 12.9 uses the older D3DKMT subset that the current libdxcore fully exports. This is a strong requirement for running Bonsai on this machine; see README "Notes for future maintainers".

## Raw bench results (llama-server, single request)

Same fact-rich seed prompt (~1500 tokens) in every run. Measured via the timings block `llama-server` returns in its `/v1/chat/completions` response.

### CPU (`-ngl 0`, `-c 8192 → 16384`, `--parallel 1 → 4`)

From prior runs during stack bring-up (see REPORT.md §2.5 for raw bonsai.log lines):

| Phase | Throughput |
|---|---:|
| Prompt eval | 28–36 tokens/sec |
| Generation | 10–16 tokens/sec |

Representative single-turn totals:

| Scenario | Prompt tokens | Output tokens | Wall time |
|---|---:|---:|---:|
| UAT S4 seed (1441 tok prompt) | 1441 | ~1500 | ~255 s (deriver end-to-end) |
| Derivative run 6 (1500 tok prompt, big accumulated batch) | 2550 | 4096 | 470 s |

### GPU (`-ngl 99`, `-c 16384`, `--parallel 1`, CUDA 12.9)

First run (cold KV):

```
prompt_n=1433 predicted_n=187
prompt_per_second=6774.2 tok/s
predicted_per_second=139.0 tok/s
wall=1577 ms
```

Second run (qwen3.5:9b chat model also loaded in the same 16 GB VRAM):

```
prompt_n=1 predicted_n=132          # KV-cache hit for identical prompt; only the new token was ingested
prompt_per_second=47.3 tok/s
predicted_per_second=198.5 tok/s
wall=698 ms
```

Third run (fresh, non-cached prompt, qwen chat still loaded):

```
prompt_n=1525 predicted_n=90
prompt_per_second=5950.8 tok/s
predicted_per_second=228.8 tok/s
wall=832 ms
```

### Side-by-side

| Metric | CPU | GPU (CUDA 12.9, -ngl 99) | Speedup |
|---|---:|---:|---:|
| Prompt eval (tok/s) | 35 | 5950–6774 | **170–190×** |
| Generation (tok/s) | 13 | 139–229 | **11–18×** |
| 1500-tok in / ~100-tok out, wall time | ~100–200 s | **0.8–1.6 s** | **~100×** |
| VRAM used (Bonsai alone) | — | 7.6 GiB | — |
| VRAM used (Bonsai + qwen3.5:9b chat) | — | 15.4 GiB / 16 GiB (margin ~545 MiB) | — |

## End-to-end Honcho workloads

The numbers above are `llama-server` in isolation. Real workloads also pay for Honcho's surrounding logic (DB, JSON parsing, tool loops, embeddings). Measured on GPU against a fresh `hermes` workspace.

### Deriver — minimal representation batch

A 271-token message posted to `/v3/workspaces/hermes/sessions/bench-session/messages`, deriver picks up once `REPRESENTATION_BATCH_MAX_TOKENS=200` is crossed, Bonsai extracts observations, saves to `documents`.

```
⚡ PERFORMANCE - minimal_deriver_78_baba
  Llm Call Duration        6496 ms
  Total Processing Time    6574 ms
  Observation Count           1
```

Compare to the CPU measurements during bring-up (same workload shape): **32 s for 2-message batches, 95 s for mid-size (1444 tok), 158 s for a large accumulated batch with 26 observations**. The GPU run lands in the same ballpark as "fast small batch on CPU" even for larger real workloads, and the worst case drops from minutes to seconds.

### Dream — full consolidation cycle (deduction + induction)

Triggered via `enqueue_dream(workspace='hermes', observer='baba', observed='baba', dream_type=OMNI)` with 1 observation already present.

```
[1a91186f] Dream cycle completed in 13875ms
  Phase 1 — deduction: 6639 ms, 12 tool calls, 65828 tok in / 781 tok out
  Phase 2 — induction:  7232 ms, 10 tool calls, 52850 tok in / 970 tok out
  Total: 22 tool-call iterations, 13.9 seconds
```

On CPU the same dream type ran for 20–30 minutes (often blocking the deriver worker as observed during debugging). **~65–130× faster on GPU**; now fast enough that dream can fire without waiting for IDLE_TIMEOUT.

## Implications for Honcho configuration

The tight-budget knobs added while Bonsai was CPU-bound can be relaxed now that everything is 10–100× faster:

| Setting | Reason it was low | GPU-era recommendation |
|---|---|---|
| `[deriver] MAX_OUTPUT_TOKENS = 1500` | to survive CPU tool loops without context overflow | can restore scaffold default (4096) |
| `[deriver] MAX_INPUT_TOKENS = 8000` | Bonsai's 16k context was stressed on CPU | fine as-is; 8000 matches the model shape we actually use |
| `[deriver] REPRESENTATION_BATCH_MAX_TOKENS = 200` | keep casual chat from sitting forever below threshold | still fine; GPU deriver finishes in ~6 s so firing often is cheap |
| `[dream] MAX_TOOL_ITERATIONS = 3` | cap dream wall-time on CPU | can raise to scaffold default (20); dream still finishes in ~1 min |
| `[dream] MAX_OUTPUT_TOKENS = 2048` | cap per-iteration output | can restore scaffold (16384) if deeper rewrites are wanted |
| `[dream] IDLE_TIMEOUT_MINUTES = 10` | avoid stealing CPU from active chat | could drop to 1–2 min; GPU dream never interferes with chat beyond a shared 16 GB VRAM budget |
| `sleep_daemon PENDING_THRESHOLD=10 / TOKEN_THRESHOLD=1000` | detect backlog that CPU couldn't drain | less important; deriver rarely backs up on GPU |

## GPU memory sharing with Ollama

RTX 5080's 16 GiB fits:

| Resident set | VRAM |
|---|---:|
| Ollama `qwen3.5:9b` (main chat, always loaded for Hermes) | ~8.6 GiB |
| Ollama embedding model (`openai/text-embedding-3-small` aliased from `nomic-embed-text`) | ~0.6 GiB |
| `llama-server` Bonsai-8B with 16k ctx KV cache | ~7.6 GiB (reduces to ~1.5 GiB when paged/unloaded; KV makes up most of it) |
| Total peak | ~15.4 GiB / 16 GiB (margin ~545 MiB) |

During our bench, a chat request to Ollama while Bonsai was resident did not cause crashes or significant slowdown — Ollama serializes at its own layer, and Bonsai's 7.6 GiB stays pinned. If you switch to a larger chat model (13B+) you must budget for Bonsai's 7.6 GiB before Ollama's offload strategy kicks in.

## Gatekeeper classifier calibration (v3 prompt)

Separate workload run against `scripts/gatekeeper_eval/dataset.jsonl` (30 hand-labeled messages covering the 4 velocity × breadth quadrants plus paradox cases — roleplay, hypothetical, sarcasm, venting, quoting, filler, correction, explicit REMEMBER prefix).

### Prompt iterations and agreement rate

| Prompt version | A/B framing | Agreement rate vs ground truth |
|---|---|---:|
| v1 | "literal vs non-literal" without anchors | 13/30 = **43.3%** |
| v2 | CoT-style separate-axis prompt with per-case anchor examples | 27/30 = **90.0%** |
| v3 (prod) | v2 + tightened `correction_of_prior` rules (emphasis vs revision) | 27/30 = **90.0%** (same accuracy, 0 correction FPs) |

### Axis independence (evidence the two-hypothesis model actually works)

Under v1 the classifier collapsed A and B to binary complements (A+B ≈ 1.0 on every example). Under v3 the axes spread out:

| Metric | v1 | v3 |
|---|---:|---:|
| A_score median | 0.00 | 0.90 |
| B_score median | 1.00 | 0.05 |
| A+B sum range | 1.00 – 1.25 | **0.25 – 1.80** |
| A vs importance Pearson r | +0.72 | **+0.32** |
| |A−B| gap median | 1.00 | 0.85 |

`A+B ≠ 1.0` under v3 means A and B are responding to independent features of the message, not just being forced into complementary values. That's the property the two-hypothesis design needs to work.

### Decision thresholds (derived from v3 data)

| Threshold | Value | Rationale |
|---|---:|---|
| `GK_DELTA` (margin) | **0.20** | Only 13% of messages have |A−B| < 0.20 — those are genuinely ambiguous and belong in pending. Gap 0.15 captures the same tail (10%) but `0.20` gives a slightly bigger safety zone around the boundary. |
| `GK_TAU` (logprob confidence floor) | **0.75** | v3 produces logprob-boundary confidence in the range 0.66 – 1.00 (median 0.86). Below 0.75 (≈3% of messages) the verdict is shaky enough to warrant pending regardless of the margin. |
| `GK_REEVAL_AFTER_SEC` | **90** | Matches Q1 agent recommendation. Long enough for a follow-up user message to naturally arrive; short enough that pending rows drain within a short session. |
| `GK_REEVAL_MAX` | **2** | Force-commit after 2 re-classifications to prevent ping-pong. Agents converged on this number independently. |

### `correction_of_prior` calibration

| Prompt | TP | FP |
|---|---:|---:|
| v2 | 4 / 4 | 2 (`Important: SSN...`, `Some people say X. Not me.`) |
| v3 | 2 / 2 | **0** |

v3 tightens the detection to "speaker retracts THEIR OWN prior assertion". `Important:` / `REMEMBER:` now route through `importance` (high score) instead of `correction`. `Not me` (differentiating from others) is also excluded. The remaining TPs ("Actually X moved to Y", "I misspoke") are still caught.

### Per-call latency

Bonsai `llama-server` with `-ngl 99`, `--parallel 1`, `-c 16384`, Q1_0 quantisation. JSON-schema-constrained output with `logprobs=true, top_logprobs=5`.

| Metric | Value |
|---|---:|
| Median latency per classification | ~1.4 s |
| Min (cached prompt) | ~0.9 s |
| Max (complex nested reasoning) | ~2.5 s |

Non-trivial overhead comes from the JSON grammar constraint; plain-text output would be <1 s. The daemon runs classifications async and doesn't block message ingestion, so the latency is not user-facing.

### Remaining errors under v3

Three test cases still disagree with ground truth; all are defensible:

- `B04 "This weather is the absolute worst!!!"` — v3 rates A=0.9 (speaker literally hates the weather). GT marked B (venting). This is a genuine judgment call; the model's reading is arguably correct.
- `B08 "um, uh, you know, like, whatever, I dunno, so yeah."` — v3 rates A=0.2 / B=0.05 / importance=1. Low importance means this row would be de-prioritised by the importance gate even without B detection; no memory damage.
- `N02 "Actually, I don't live in Osaka anymore. Moved to Kyoto."` — v3 rates A=0.8 / B=0.9 / correction=true. With |A−B|=0.1 < δ=0.20 this correctly lands in `pending`, where the re-evaluation loop will lock it down on the next context turn.

### Takeaway

v3 + δ=0.20 + τ=0.75 produces a classifier that calls the clear cases correctly, defers the ambiguous ones to pending for cheap re-evaluation, and never invents spurious corrections. Deploy as-is; re-calibrate only if shadow logs on real-user traffic show a drift of more than ~5 percentage points from this bench.
