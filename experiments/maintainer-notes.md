# Notes for future maintainers

This file is extracted from the hermes-stack top-level README to keep the setup walkthrough focused. These are design-rationale notes about choices made during bring-up, not setup instructions — if you are standing the stack up for the first time, use the main README.

Things the next person (you, six months from now, or someone new coming to this repo) should know. These are not bugs — they are hard requirements or design decisions that got made during bring-up, documented here so nobody re-learns them painfully.

## Bonsai + CPU is not a usable combination for this stack

Bonsai-8B ships as a 1-bit quantised GGUF that *looks* like it should run fine on CPU — and it does, in the sense that `llama-cli` will produce text. But with Honcho's deriver and dream on top, CPU Bonsai is **too slow to keep up with real chat**:

- Per-call latency on CPU: 30–160 s for a deriver turn, 20–30 min for a single dream cycle (see `benchmark.md`).
- Honcho's dream agent tool-loops 20+ times; on CPU a dream blocks the single deriver worker (`WORKERS=1`) for the entire duration.
- Observations pile up as `pending` in `queue` while dream runs, and contradictory updates ("Alice died" ≠ "Alice lives in Kyoto") stay unresolved because dream never gets a chance to consolidate them before the next fire.
- On GPU (RTX 5080, CUDA 12.9, `-ngl 99`), the same workloads complete in 1–15 s. That is the only configuration this repo is tuned for.

**If you don't have an NVIDIA GPU**, do not try to make this recipe work by sitting through the CPU times. Instead, drop Bonsai entirely and point Honcho's `[deriver]` / `[dialectic]` / `[summary]` / `[dream]` at a different local memory-adjacent model you can run fast. Candidates:

- A cloud API via the `custom` / `openai` provider slots (OpenRouter, Venice, Together). Defeats the data-sovereignty goal but lets Honcho function.
- A smaller local model via Ollama — `qwen3.5:4b` or `gemma3:4b` via `PROVIDER = "custom"` and `MODEL = "qwen3.5:4b"`. Lower quality memory reasoning but acceptable for toy scale.
- vLLM or another GPU inference stack on a LAN machine, pointed to via `LLM_VLLM_BASE_URL`.

Whichever you pick, the hyperparameter section still applies — the scaffold defaults assume a fast backend and will look the same kind of broken if you pick a slow one.

## CUDA 12.9 is the hard upper bound on current WSL2

The NVIDIA apt repo will happily offer `cuda-toolkit-13-2`. It builds clean. Then `llama-server` segfaults at startup:

```
/usr/lib/wsl/lib/libdxcore.so: undefined symbol: D3DKMTOpenSyncObjectFromNtHandle
/usr/lib/wsl/lib/libdxcore.so: undefined symbol: D3DKMTAcquireKeyedMutex
/usr/lib/wsl/lib/libdxcore.so: undefined symbol: D3DKMTCreateNativeFence
(+5 more)
```

CUDA 13.x's runtime assumes newer DirectX Kernel Mode Thunk APIs than the current WSL `libdxcore.so` exports. `wsl --update` does not help — even on the latest WSL kernel those eight symbols are physically absent from `libdxcore.so` (we grepped). CUDA 12.9 uses the older D3DKMT subset that libdxcore fully has.

This will eventually fix itself when Microsoft ships a libdxcore with the missing symbols, at which point the `cuda-toolkit` package you install can be 13.x. Until then, `cuda-toolkit-12-9` (or any `12-x` below it) is the compatible choice.

## This build is not redistributable

The Bonsai `llama-server` binary we compile here links against NVIDIA's cuBLAS / cuBLASLt / cuFFT / cuSOLVER libraries from the CUDA toolkit. NVIDIA's CUDA redistribution license (see the `EULA` shipped with the toolkit) limits redistributing those libraries as static or dynamically-linked copies. The practical implication:

- Don't publish a prebuilt `bonsai-llama.cpp` binary as part of a release artifact. Each operator needs to install CUDA 12.x and compile locally (which is what this README walks through).
- If you want distributable prebuilts, either use the upstream `llama.cpp` release's Vulkan variant (lower perf, no CUDA license, but no Blackwell-specific optimisations either) or ship Docker images that layer on `nvidia/cuda` base images — those are licensed for that redistribution path.
- The local CUDA paths (`/usr/local/cuda-12.9`) and `LD_LIBRARY_PATH=/usr/local/cuda/lib64` expectations assume the operator did Step 1's "Prerequisites" block — skipping that will produce confusing linker errors at `llama-server` startup.

## Why dream still lives on Bonsai, not the Ollama chat model

We tried routing dream to Ollama (`qwen3.5:9b`) because Ollama already owns the GPU for chat. That produced hallucinated observations during the consolidation tool-loop (a fake employer, a made-up age field) because general chat models weren't trained to discipline themselves inside `search_memory` / `create_observations` / `delete_observations` loops. Bonsai was fine-tuned for exactly that. Keeping dream on Bonsai (separate `llama-server` process) means the two workloads coexist on VRAM via separate process address spaces, which both Ollama and `llama-server` handle without special coordination.

## pgvector schema width is pinned to the embedding model

The scaffold's `Vector(1536)` is hardcoded across three migrations (`a1b2c3d4e5f6_initial_schema.py`, `917195d9b5e9_add_messageembedding_table.py`, `119a52b73c60_support_external_embeddings.py`) and `src/models.py`. Step 3's `sed` rewrites them to `Vector(768)` for `nomic-embed-text`. If you swap the embedding model to anything that outputs a different dimension, you must redo the sed and `docker compose down -v` before `up -d --build` — pgvector does not auto-migrate column width and the embedding insert will silently keep rolling back until somebody traces the `expected N dimensions, not M` error in `api` logs.

## Honcho's `config.toml` integer fields are strict

Recent Honcho uses Pydantic v2 with strict int parsing. Fields like `[dream] MIN_HOURS_BETWEEN_DREAMS` reject `0.5` with:
```
pydantic_core._pydantic_core.ValidationError: MIN_HOURS_BETWEEN_DREAMS
  Input should be a valid integer, got a number with a fractional part
```
and the entire `deriver` container refuses to start. Keep those fields integer. Sub-hour gating should be done from `sleep_daemon.py` (its `MIN_MINUTES_BETWEEN_NAPS` is a Python int which we treat as minutes, so fractional hours aren't needed).
