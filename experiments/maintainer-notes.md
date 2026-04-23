# Notes for future maintainers

This file is extracted from the hermes-stack top-level README to keep the setup walkthrough focused. These are design-rationale notes about choices made during bring-up, not setup instructions — if you are standing the stack up for the first time, use the main README.

Things the next person (you, six months from now, or someone new coming to this repo) should know. These are not bugs — they are hard requirements or design decisions that got made during bring-up, documented here so nobody re-learns them painfully.

> **Note on archived Bonsai-era sections.** Two sections that used to live in this file — "Bonsai + CPU is not a usable combination for this stack" and "Why dream still lives on Bonsai, not the Ollama chat model" — assumed the old two-endpoint Bonsai-8B + Ollama topology. They have been moved verbatim into `experiments/bonsai-archive.md` (see the "Archived sections pulled in from other experiments docs" tail) so the design-rationale record stays intact for anyone rebuilding that design on different hardware.

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

The `llama-server` binary we compile here links against NVIDIA's cuBLAS / cuBLASLt / cuFFT / cuSOLVER libraries from the CUDA toolkit. NVIDIA's CUDA redistribution license (see the `EULA` shipped with the toolkit) limits redistributing those libraries as static or dynamically-linked copies. The practical implication:

- Don't publish a prebuilt `llama-server` binary from the `llama.cpp/build/bin/` tree as part of a release artifact. Each operator needs to install CUDA 12.x and compile locally (which is what the README walks through).
- If you want distributable prebuilts, either use upstream `llama.cpp`'s release Vulkan variant (lower perf, no CUDA license, but no Blackwell-specific optimisations either) or ship Docker images that layer on `nvidia/cuda` base images — those are licensed for that redistribution path.
- The local CUDA paths (`/usr/local/cuda-12.9`) and `LD_LIBRARY_PATH=/usr/local/cuda/lib64` expectations assume the operator did Step 1's "Prerequisites" block — skipping that will produce confusing linker errors at `llama-server` startup.

## pgvector schema width is pinned to the embedding model

Upstream's `Vector(1536)` (matching OpenAI's `text-embedding-3-small`) is hardcoded in the initial schema and carried through subsequent migrations. For this stack, where `nomic-embed-text` produces 768-dim vectors, the fork ships a follow-up migration `h8i9j0k1l2m3_alter_vector_dim_to_768.py` that alters `documents.embedding` and `message_embeddings.embedding` to `Vector(768)` after upstream's schema lands. `[vector_store] MIGRATED = true` in `config.toml` tells `src/config.py`'s relaxed validator to accept non-1536 dims once that migration has run.

If you ever swap the embedding model to anything that outputs a different dimension (other than 1536 or 768), you need a new migration that alters the column to the new width and `docker compose down -v && up -d --build` to rebuild the volume from scratch — pgvector does not auto-migrate column width, and any prior schema survives a plain `up -d` because the data volume is preserved. The embedding insert will silently keep rolling back with `expected N dimensions, not M` in the `api` logs until the migration lands.

(An earlier version of this note described a `sed`-based rewrite of the scaffold schema. That was the procedure before the fork's migration was in place; it is superseded by the migration-based approach above.)

## Honcho's `config.toml` integer fields are strict

Recent Honcho uses Pydantic v2 with strict int parsing. Fields like `[dream] MIN_HOURS_BETWEEN_DREAMS` reject `0.5` with:
```
pydantic_core._pydantic_core.ValidationError: MIN_HOURS_BETWEEN_DREAMS
  Input should be a valid integer, got a number with a fractional part
```
and the entire `deriver` container refuses to start. Keep those fields integer. Sub-hour gating should be done from `sleep_daemon.py` (its `MIN_MINUTES_BETWEEN_NAPS` is a Python int which we treat as minutes, so fractional hours aren't needed).
