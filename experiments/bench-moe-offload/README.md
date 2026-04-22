# bench-moe-offload — qwen3.6:35b placement × reasoning benchmark

## Objective

Measure throughput and resource consumption of `qwen3.6:35b` (qwen35moe 36B, Q4_K_M)
under six combinations of (layer placement × reasoning mode) on a single-GPU
16 GiB RTX 5080 + 20-thread CPU + 108 GiB DDR5 host. Observe competition behaviour
when Hermes and Honcho share the single endpoint.

This is a benchmark, not a judgement. No pass/fail thresholds — the outputs are
raw numbers plus a ranked summary, and the reader decides which configuration to
adopt.

## Runtime topology during the experiment

```
[stopped]  Bonsai llama-server
[stopped]  ollama serve                    ← removed from the stack entirely
[persist]  embedding llama-server  :8081   ← nomic-embed-text (aliased to
                                              openai/text-embedding-3-small)
[rotating] test llama-server       :8080   ← qwen3.6:35b, restarted per cell
                                              with cell-specific flags
[running]  Honcho api / deriver             ← config.toml points at
                                              chat→:8080, embedding→:8081
[standby]  Hermes                           ← CLI not launched during bench
                                              runs (bench talks to :8080
                                              directly via curl to avoid
                                              client-side noise)
```

## Engine

`llama-server` from `../../bonsai-llama.cpp/build/bin/`. This is the
PrismML-Eng/llama.cpp fork (ggml commit `e2d6742`), built with CUDA 12.9
and `$ORIGIN` rpath — see `experiments/bottleneck.md` §2 for why.

Embedding is served by the same binary with `--embeddings` flag so the stack
contains exactly one chat-capable server process at any time during a cell,
plus one long-lived embedding server.

## Model / GGUF blob

Both chat and embedding use GGUFs already present in ollama's blob store
(content-addressed, no duplication):

| Purpose | Blob | Size | Architecture |
|---|---|---:|---|
| Chat | `sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` | 22 GiB | qwen35moe, Q4_K_M |
| Embedding | `sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6` | 261 MiB | nomic-bert, F16 |

Chat model has 65 layers, native context 262 144, 768-dim hidden is not its
output — qwen35moe outputs via typical head. Embedding model is native
768-dim output, 2048-token native context.

## Fixed flags (chat server, common to all cells)

```
--host 127.0.0.1
--port 8080
-c 65536
-fa on
-ctk q8_0
-ctv q8_0
--jinja
--alias qwen3.6-test
```

- `-c 65536` matches the `OLLAMA_CONTEXT_LENGTH` we set elsewhere; large enough
  to hold Hermes's realistic 4–13 k prompts with 5× headroom. KV at this ctx
  with `q8_0` quantization is roughly 2 GiB total.
- `-fa on` enables Flash Attention — required for efficient KV q8_0 handling
  on modern GPUs.
- `-ctk q8_0 -ctv q8_0` halves KV memory vs f16 with negligible quality
  cost; keeps L5/L6 VRAM budget tractable.
- `--jinja` enables the GGUF's chat template, which qwen35moe uses for
  tool-call schema rendering.

## Variable axes (8 cells)

| Cell | Placement | Reasoning | Cell-specific flags |
|:---:|---|---|---|
| **L1** | CPU-only | vanilla | `-ngl 0` |
| **L2** | CPU-only | nothink | `-ngl 0 --reasoning off` |
| **L3a** | partial N=30 | vanilla | `-ngl 30` |
| **L3b** | partial N=35 | vanilla | `-ngl 35` (conditional, see below) |
| **L4a** | partial N=30 | nothink | `-ngl 30 --reasoning off` |
| **L4b** | partial N=35 | nothink | `-ngl 35 --reasoning off` (conditional) |
| **L5** | expert offload | vanilla | `-ngl 99 -ot "ffn_(up\|down\|gate)_exps=CPU"` |
| **L6** | expert offload | nothink | L5 flags + `--reasoning off` |

### Conditional L3b / L4b

Run L3b / L4b **only if the matching L3a / L4a cell measured
`VRAM_peak ≤ 13.5 GiB` on its single-load pass**.

Reasoning:
- Total usable VRAM on the card: ~15.9 GiB (16 384 − CUDA runtime overhead).
- Moving from N=30 to N=35 adds 5 layers × 0.34 GiB ≈ **+1.7 GiB of weights**
  plus +0.2 GiB of KV (more layers keep their KV on GPU).
- Total increase: **+1.9 GiB**.
- 13.5 GiB + 1.9 GiB = 15.4 GiB, leaving a ~0.5 GiB margin for
  transient allocations and WSL2/CUDA overhead (observed 0.5–0.7 GiB
  on this host via `nvidia-smi` idle).
- If L3a exceeds 13.5, L3b would cross the physical ceiling.

## Load patterns

### Single load — Hermes shape
- Prompt size: **4 143 tokens** (reused from `/tmp/big_real.json` which was
  captured earlier in this session as a realistic Hermes turn).
- `max_tokens`: **800**.
- `stream: true` (required for TTFT measurement).
- 3 consecutive runs, metrics reported as median.

### Contention load — Hermes + Honcho deriver concurrent
- Hermes side: same Hermes-shape request as single load.
- Honcho side: a real deriver LLM request captured from the live pipeline
  during prep phase (see `prompts/honcho_deriver.json`).
- Both fired simultaneously with a brief stagger (~100 ms) so Hermes is
  first into the queue.
- Only run on **L4a and L6** — the two placement families expected to be
  practically usable. Other cells skip contention.
- 3 runs, median.

## Metrics captured per run

| Metric | Source | Single | Contention |
|---|---|:-:|:-:|
| TTFT (first-token time) | delta from request send to first non-empty chunk | ✓ | ✓ |
| prompt eval tok/s | `prompt_eval_count / prompt_eval_duration` | ✓ | ✓ |
| decode tok/s | `eval_count / eval_duration` | ✓ | ✓ |
| total duration | `curl -w %{time_total}` | ✓ | ✓ |
| content non-empty | `len(message.content) > 0` | ✓ | ✓ |
| VRAM peak | 1 Hz `nvidia-smi memory.used` during run | ✓ | ✓ |
| RAM peak | 1 Hz `free -b` during run | ✓ | ✓ |
| Hermes Δ (contention) | `contention_hermes_total − single_hermes_total` | — | ✓ |
| Honcho deriver total | wall-clock from Honcho-side request | — | ✓ |

## Execution order

Order chosen to see the expected winner first and fail fast if MoE offload
doesn't behave as hypothesised:

1. **L6** — expert offload + nothink, the best-case candidate
2. **L5** — expert offload + vanilla (isolates the "reasoning tax" by holding
   placement constant)
3. **L2** — CPU-only + nothink (floor case that's still usable)
4. **L1** — CPU-only + vanilla (absolute floor; already measured roughly
   during bottleneck investigation, repeated here under uniform conditions)
5. **L4a** — partial + nothink
6. **L3a** — partial + vanilla
7. **L4b / L3b** — conditional (see threshold rule)
8. **Contention** — re-run L6 and L4a with concurrent Honcho deriver

## File layout

```
experiments/bench-moe-offload/
├── README.md                   # this document
├── report.md                   # generated by run_all.sh after cells finish
├── run_all.sh                  # top-level orchestration
├── prep.sh                     # one-time: stop ollama, start embedding server,
│                                 update honcho + hermes configs, capture real
│                                 deriver prompt, verify stack
├── bench_cell.sh               # per-cell: start test server → warmup →
│                                 single load × 3 → optional contention ×3 →
│                                 stop server → write results
├── prompts/
│   ├── hermes_4k.json          # single-load payload
│   └── honcho_deriver.json     # contention-load payload (captured during prep)
├── scripts/
│   ├── capture_proxy.py        # Flask proxy used once in prep to capture real
│   │                             deriver request body
│   ├── sample_resources.sh     # 1 Hz VRAM + RAM sampler, tsv output
│   └── parse_stream.py         # extract TTFT and token rates from ndjson
└── results/
    ├── L1/ L2/ L3a/ L4a/ L5/ L6/   # always written
    │   ├── single_run{1,2,3}.json  # raw server response per run
    │   ├── single_metrics.json     # parsed median metrics
    │   ├── contention_run{1,2,3}.json  # L4a, L6 only
    │   ├── resources.tsv           # 1 Hz VRAM/RAM trace over the cell
    │   └── server.log              # llama-server stdout/stderr for the cell
    └── L3b/ L4b/                   # only if threshold allowed
```

## Prerequisites

Before running `prep.sh`:

- The earlier RPATH fix is committed and the build binary supports
  `-ot` / `-hf` / `--reasoning` / `--embeddings` (verified in the
  bottleneck investigation).
- The current user has permission to read
  `/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a...` and
  `sha256-970aa74c...`.
- Docker daemon up, `honcho-api-1` and `honcho-deriver-1` reachable by name.
- `~/.hermes/config.yaml` exists and is writable.
- Port 8080 and 8081 free at start.
- At least 14 GiB of GPU VRAM free (Bonsai must be stopped).

`prep.sh` will verify these and abort with a clear error if any fail.

## Interpreting the output

After `run_all.sh` completes, read `report.md`. It contains:

1. One-row-per-cell summary table (TTFT / prompt-eval / decode / total /
   content-non-empty / VRAM peak / RAM peak).
2. Contention comparison for L6 and L4a (single vs contended Hermes Δ,
   Honcho deriver wall-clock).
3. Raw run-level tables per cell for variance inspection.
4. No verdict. The numbers speak.

The investigation context (why each cell exists, what was expected,
what was already observed during bottleneck exploration) lives in
`experiments/bottleneck.md`. This README is the experimental spec only.
