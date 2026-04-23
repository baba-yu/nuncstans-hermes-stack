# Switching the llama.cpp submodule from PrismML fork to upstream

Session date: 2026-04-23

Companion to `experiments/bonsai-archive.md` (the Bonsai+Ollama retirement)
and `experiments/save-point-pivot.md` (the recall-mode flip). This one
closes the circle on the Bonsai-era residue in the build chain.

## Context

The `llama.cpp/` submodule (previously checked out at
`bonsai-llama.cpp/` — see that rename's commit message) was pinned to
[PrismML-Eng/llama.cpp@e2d67422](https://github.com/PrismML-Eng/llama.cpp/commit/e2d67422)
on the `prism` branch. PrismML's fork exists to carry a set of patches
on top of upstream `ggml-org/llama.cpp` — and the reason it was chosen
originally was that those patches add **Q1_0 quantization support**,
which is the quantization format Bonsai-8B shipped in.

Today's stack does not use Bonsai-8B and does not use Q1_0. Current
chat model is `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL` (Q4_K_XL
quant); current embed model is `nomic-embed-text` (standard quant
variants). Neither touches the Q1_0 code path. Staying on the fork
meant staying on a less-maintained, less-reviewed source tree for no
active reason.

## Phase 0 — What the fork actually carries

After adding `ggml-org/llama.cpp` as a second remote in the submodule
and unshallowing, `git log upstream/master..e2d67422` lists 14 commits
that PrismML has on top of upstream's `e97492369` (which is the most
recent upstream commit on their branch's ancestry):

| Commit | Category |
|---|---|
| `195593bc4 Implemented optimized q1_0 dot for x86 and generic` | Q1_0 CPU |
| `e29cd486d Removed redundant helper definition` | Q1_0 CPU |
| `7c3501a55 [cuda] initial Q1_0 backend` | Q1_0 CUDA |
| `84ab75f5e remove unused code, fix AMD MMA guard` | Q1_0 CUDA |
| `bca0c0b89 attempt to support dp4a` | Q1_0 CUDA |
| `8587b5cc5 Removed two redundant instructions from AVX q1_0 dot` | Q1_0 CPU |
| `05b0c84e6 Apply suggestions from code review` | Q1_0 |
| `0c4fb41fc Fixed inconsistency with fp16 conversion for generic q1_0 dot` | Q1_0 CPU |
| `7f82cf0cf Style cleanup around AVX q1_0 dot` | Q1_0 CPU |
| `a2504b360 Merge branch 'pr-21629'` | upstream PR merge |
| `6df15a6f8 Merge branch 'pr-21636'` | upstream PR merge |
| `3b6d41b33 Add release-prism workflow from prism-launch` | CI only |
| `092bee7d9 Update release-prism.yml: add CPU/Vulkan/ROCm/HIP backends, CUDA 12.4/12.8` | CI only |
| `e2d67422c Remove Windows CUDA 12.8 (not supported by setup action)` | CI only |

Conclusion: **9 of 14 commits are Q1_0-only**; 3 are CI-workflow changes
that don't affect the built binary; 2 are upstream-PR merges that would
be redundant against the current upstream master. None of this affects
the Qwen3.6 / nomic-embed paths we exercise today.

## What Blackwell looks like on upstream CMake

One concern going in was that the `sm_120a` (compute capability 12.0,
RTX 5080) auto-detection might have been a PrismML fork feature. Diff
of `ggml/src/ggml-cuda/CMakeLists.txt`:

```
diff PrismML prism branch (e2d67422)  vs  upstream master (c78fb909b)
# identical
```

Upstream's CMake already has the `120a-real` / `121a-real` append logic
when CUDA ≥ 12.8. No extra `-DCMAKE_CUDA_ARCHITECTURES` flag needed in
the README Step 1.

The configure output confirms it picks the right arch:

```
-- Replacing 120-real in CMAKE_CUDA_ARCHITECTURES_NATIVE with 120a-real
-- Using CMAKE_CUDA_ARCHITECTURES=120a-real CMAKE_CUDA_ARCHITECTURES_NATIVE=120a-real
```

## Target commit

`c78fb909b23758f5e418cf98a69bc8a0ef142fb8` — upstream master tip at the
time of migration. Chosen reasons:

1. Latest stable-master (not a branch mid-feature).
2. Includes `server: fix heap-buffer-overflow from negative n_discard
   (CVE-2026-21869) (#22267)` — real server-side security fix worth
   having even on a local-only stack.
3. ggml version 0.10.0, `gguf-v0.18.0-734-gc78fb909b` tag lineage.

## Migration sequence (as executed)

1. Stopped llama-services (chat / embed / gatekeeper). All three
   processes were already down from an earlier shutdown; pidfiles were
   stale and cleaned up manually.
2. Renamed `llama.cpp/build/` → `llama.cpp/build.prismml-backup/`
   (231 MB) as the rollback escape hatch.
3. `.gitmodules` URL: `PrismML-Eng/llama.cpp` → `ggml-org/llama.cpp`.
   `git submodule sync llama.cpp` propagated to `.git/config`.
4. Inside `llama.cpp/`: `git remote set-url origin`, reset refspec to
   the standard `+refs/heads/*:refs/remotes/origin/*` (the shallow
   clone from PrismML had a tag-only refspec), fetched all branches,
   checked out `master` tracking `origin/master`, deleted the old
   `prism` local branch.
5. Parent repo's submodule pointer updated automatically to
   `c78fb909b`.
6. Clean rebuild from upstream source with the same three $ORIGIN RPATH
   flags from README Step 1.
7. `readelf -d build/bin/llama-server | grep RUNPATH` → `[$ORIGIN]`.
8. `./scripts/llama-services.sh start` → all three services up.
9. Smoke test: `/v1/models`, `/v1/chat/completions` ping, a single
   Hermes turn, cross-session recall.

(Each of these steps has its own Phase N marker in the planning note;
see the commit that landed this writeup.)

## What rolled back cleanly mid-migration

Nothing needed rolling back, but the escape hatch was:

- Revert `.gitmodules` + `.git/config` to PrismML URL
- `cd llama.cpp && git remote set-url origin ...PrismML... && git fetch && git checkout e2d67422 -B prism`
- `rm -rf build && mv build.prismml-backup build`
- Services restart from the restored binary (same inode as before, no
  rebuild needed because `$ORIGIN` rpath means the built PrismML binary
  doesn't care which dir it's in).

## What changed in the repo

- `.gitmodules` — submodule URL flipped to `ggml-org/llama.cpp`.
- Parent repo's submodule commit pointer — `e2d67422c` → `c78fb909b`.
- `README.md` — submodule table's "PrismML fork" row rewritten to just
  reference `ggml-org/llama.cpp`. The "unmodified" annotation is
  retained (it was true against PrismML; it's still true against
  upstream). Step 1 build block unchanged (same flags, same command,
  same expected output). The bonsai-archive note already points back
  here for operators who need Q1_0.
- `experiments/llamacpp-upstream-migration.md` — this file.

## What did NOT change

- `scripts/llama-services.sh` — binary lives at the same path
  (`llama.cpp/build/bin/llama-server`), started with the same flags.
- `scripts/gatekeeper_daemon.py` / `scripts/sleep_daemon.py` — no
  dependency on the fork choice.
- Honcho side — `honcho/config.toml` only cares about the OpenAI-compat
  endpoint, not which llama.cpp built the server.
- CUDA 12.x pin — still required (WSL libdxcore D3DKMT symbol issue
  is independent of llama.cpp fork choice; see
  `experiments/maintainer-notes.md`).
- $ORIGIN RPATH bit — upstream CMake honors all three flags the same
  way PrismML did.

## Open items

- `llama.cpp/build.prismml-backup/` is 231 MiB on disk. Safe to
  `rm -rf` once confident upstream build is stable through a few
  user-facing sessions. Left in place for now as a week-or-two safety
  net.
- If someone ever wants to run Bonsai-8B again (the whole archived
  two-endpoint design in `experiments/bonsai-archive.md`), they need
  to point the submodule at a fork that carries Q1_0 — either back to
  PrismML or to whichever fork is actively upstreaming the Q1_0
  patches. Document if this happens.
