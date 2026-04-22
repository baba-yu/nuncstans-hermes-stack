# Hermes-stack "stuck" investigation — bottleneck analysis

Session date: 2026-04-20 – 2026-04-21

Symptom the user reported: **every Hermes turn feels hung — sometimes minutes with no response, sometimes empty response, sometimes the whole stack looks like it's deadlocked.**

Final verdict: **no single bottleneck — a stacked failure of six distinct layers**, each masking the one below. Four of the six are fixed, one is hardware-limited, and one remains unfixed at the time of writing (qwen3 thinking-token leak, fix path identified).

---

## TL;DR

| # | Layer | Root cause | Status | Fix |
|---|---|---|---|---|
| 1 | Bonsai `llama-server` silently not running | `libmtmd.so.0` ENOENT from stale absolute `RUNPATH` after the repo dir was renamed `hermes-stack` → `nuncstans-hermes-stack` | **Fixed** | Rebuild with `CMAKE_BUILD_RPATH_USE_ORIGIN=ON` + `CMAKE_INSTALL_RPATH='$ORIGIN'`. Committed `bc5f36f`. |
| 2 | Ollama truncates 13k-token prompts to 4096 | Ollama default `num_ctx=4096` ignores GGUF-declared 262144; OpenAI-compat API has no per-request `num_ctx` field | **Fixed** | `OLLAMA_CONTEXT_LENGTH=65536` in systemd drop-in. |
| 3 | Ollama loads qwen3.6:35b on top of Bonsai, thrashing VRAM | Ollama's scheduler can't see Bonsai's VRAM reservation unless told | **Fixed** | `OLLAMA_GPU_OVERHEAD=7516192768` (7 GiB). |
| 4 | Honcho dialectic / deriver / summary / dream all 500 when hitting Bonsai | Honcho's OpenAI backend forwards `tool_choice: "any"` unchanged; llama.cpp only accepts `"auto"`/`"none"`/`"required"` | **Fixed** | Patched `honcho/src/llm/backends/openai.py` to normalize `"any"` → `"required"`. Committed `e875f63` in submodule, bumped in parent `e5e0cab`. |
| 5 | CPU-only decode of qwen3.6:35b tops out at ~9 tok/s | DDR5 bandwidth limit; MoE-A3B doesn't help as much as theory suggests because llama.cpp reads more than just active experts | **Inherent (no fix)** | Either downgrade to a model that fits in the ~8 GiB VRAM pocket left by Bonsai, or accept minutes-per-turn. |
| 6 | qwen3.6:35b returns empty content — every token goes into invisible reasoning | llama.cpp issue [#20099](https://github.com/ggml-org/llama.cpp/issues/20099); `completion_tokens` counts thinking tokens that are not surfaced in `message.content` | **Fix identified, not applied** | Pass `"think": false` in the API request. Measured: 26 s empty → 0.8 s with "one two three", **~32× faster AND actually produces output**. Implementation: ollama Modelfile `PARAMETER think false`, re-point `~/.hermes/config.yaml:model.default`. |

Additional unfixed issue, secondary priority:
- llama.cpp issue [#20003](https://github.com/ggml-org/llama.cpp/issues/20003) — same model re-prefills the full prompt every turn instead of reusing KV-cache prefix. Verified on our box (see §5). Adds ~7 s of redundant prefill per turn on a 4 k prompt, ~17 s on a 10 k prompt. Workarounds: ik_llama.cpp fork, or different chat template.

---

## 1. Timeline of what looked like what

The investigation kept bouncing between "we found the root cause" and "no wait, there's another one below it." Recorded here so the next person doesn't re-walk the tree.

| Observed | What we thought | What it actually was |
|---|---|---|
| Hermes's log says `Honcho dialectic query failed: Request timed out after 60.0s` | Honcho is broken | Honcho → Bonsai (:8080) — but Bonsai wasn't running |
| Bonsai's log says `libmtmd.so.0: cannot open shared object file`, but the `.so` is right next to the binary | Missing symlink? Bad install? | CMake baked the old absolute build-tree path as RUNPATH; the dir was renamed after build |
| Hermes hangs ~2 minutes per turn even after Bonsai is back | Bonsai talks slow? | Ollama is silently truncating 13 k-token prompts to 4096, feeding qwen a mid-sentence prompt which then loops and emits `<\|endoftext\|>` |
| Hermes still slow after `OLLAMA_CONTEXT_LENGTH=65536` | Hardware limit | Bonsai is now taking VRAM Ollama thought it had; Ollama thrashes between loading qwen and evicting embedding. `OLLAMA_GPU_OVERHEAD` fixes the arithmetic. |
| Hermes still slow after the overhead fix | Hardware limit (second attempt) | Honcho is returning 500 with `Invalid tool_choice: any` — patched that |
| Hermes still slow after all four fixes | **Genuinely hardware-limited** on CPU decode… | …plus two upstream llama.cpp bugs (#20003, #20099) compounding the symptom |

Lesson: **do not declare victory after the first green test.** Every "fix" here only removed the outermost failure, exposing the next one.

---

## 2. Layer 1 — Bonsai `llama-server` silently not running

### Symptom
```
$ cat bonsai.log
./build/bin/llama-server: error while loading shared libraries:
  libmtmd.so.0: cannot open shared object file: No such file or directory
```
but
```
$ ls bonsai-llama.cpp/build/bin/libmtmd.so.0
lrwxrwxrwx … libmtmd.so.0 -> libmtmd.so.0.0.1
```

### Root cause
```
$ readelf -d build/bin/llama-server | grep RUNPATH
 Library runpath: [/home/baba-y/hermes-stack/bonsai-llama.cpp/build/bin:]
```
The repo had been renamed from `hermes-stack` to `nuncstans-hermes-stack` some time after the CUDA build. CMake by default bakes the absolute build-tree path as `DT_RUNPATH`; the rename invalidated it, and `ld.so` gave up without trying `$ORIGIN`.

### Upstream status
Known, not cleanly fixed in mainline as of this investigation:
- [ggml-org/llama.cpp#17193](https://github.com/ggml-org/llama.cpp/issues/17193), [#17190](https://github.com/ggml-org/llama.cpp/issues/17190), [#17950](https://github.com/ggml-org/llama.cpp/issues/17950) — same-symptom reports
- [PR#17214](https://github.com/ggml-org/llama.cpp/pull/17214) is merged but addresses only the Docker-symlink side; the absolute-RUNPATH side has no upstream default fix

### Fix applied
Rebuild with:
```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```
Result: all `lib*.so.0` and `llama-server` carry `RUNPATH=[$ORIGIN]`. Move-and-rename relocation test passed (copy to `/tmp/tmp.XXX/`, rename mid-run, still works).

Committed as `bc5f36f docs(readme): pin bonsai-llama.cpp build to $ORIGIN RPATH` (parent repo, README Step 1).

### Downstream impact while it was broken
Honcho's `dialectic`, `deriver`, `summary`, `dream` modules all have `overrides.base_url = http://host.docker.internal:8080/v1` in `config.toml.bonsai-example`. With Bonsai dead, every call got `ECONNREFUSED` → tenacity retried with exponential backoff → failed at ~60 s. Hermes's memory-sync path waits on this synchronously, which is why the whole agent loop felt hung rather than just one subsystem.

---

## 3. Layer 2 — Ollama silent prompt truncation

### Symptom
- qwen3.6:35b GGUF declares `qwen35moe.context_length = 262144`
- Hermes caches this in `~/.hermes/context_length_cache.yaml` and sends 10–13 k-token prompts (memory context + tool schemas + dialectic result)
- Ollama's `/api/ps` reports the model loaded with `context_length: 4096`
- Ollama log: `truncating input prompt limit=4096 prompt=13272 keep=4 new=4096`
- The model receives a mid-sentence-cut prompt, emits repeated paragraphs, leaks `<|endoftext|>` / `<|im_start|>` tokens
- Honcho's message sync later chokes on these tokens during tiktoken validation

### Root cause
Ollama's default `num_ctx` is 4096. The OpenAI-compatible `/v1/chat/completions` endpoint has no `num_ctx` field in its spec, so no OpenAI-style client (including Hermes) can pass it per-request. The only knob is the service-wide env var `OLLAMA_CONTEXT_LENGTH`.

This affects **every** client that doesn't explicitly pass `options.num_ctx` (an Ollama-specific extension), not just Hermes.

### Fix applied
```ini
Environment="OLLAMA_CONTEXT_LENGTH=65536"
```
in `/etc/systemd/system/ollama.service.d/override.conf`.

Chose 65536 instead of the model-native 262144 because KV scales linearly with context and the 108 GiB RAM budget does not stretch to 262 k. See §4 for the VRAM/RAM arithmetic.

Verified: `/api/ps` now reports `context_length: 65536`. Reran the 6015-token test — Ollama accepted it with no truncation warning and the response's `prompt_tokens` reported the full 6015.

Committed as part of `e5e0cab docs(readme): ollama env vars + honcho tool_choice patch; bump honcho`.

---

## 4. Layer 3 — VRAM reservation for Bonsai

### Problem
- RTX 5080 = 15.9 GiB usable VRAM
- Bonsai `llama-server` (fully offloaded at `-ngl 99 -c 16384`) actually uses **6.3 GiB** (not the 7.6 GiB the README originally cited — see follow-ups)
- Ollama default scheduler calls NVML, sees whatever is free right then, and allocates against that
- If Ollama starts before Bonsai, it takes ~15.9 GiB for qwen; Bonsai then fails to start
- If Bonsai is started first, Ollama sees ~9.7 GiB free and plans accordingly — but any subsequent reshuffle (keep-alive expiry, model switch) can blow past that if Ollama forgets the Bonsai presence

### Fix applied
```ini
Environment="OLLAMA_GPU_OVERHEAD=7516192768"   # 7 GiB
```
Tells Ollama to subtract 7 GiB from whatever it sees free, before deciding what fits. Survives reshuffles.

### Semantic caveat discovered during tuning
`OLLAMA_GPU_OVERHEAD` is **subtracted from free-VRAM-at-allocation-time**, not from total. So when Bonsai already holds 6.3 GiB:
```
free VRAM as NVML reports it     = 15.9 − 6.3 = 9.6 GiB
ollama's effective budget        = 9.6 − overhead
  overhead=8 GiB                 = 1.6 GiB (qwen9b=8.8 → 0 GPU layers)
  overhead=7 GiB                 = 2.6 GiB (qwen9b=8.8 → 5/33 GPU layers — still mostly CPU)
  overhead=0 GiB                 = 9.6 GiB (qwen9b=8.8 → full GPU)
```
So **overhead ≥ Bonsai's real footprint costs ~the same VRAM twice**, because Bonsai already owns it and ollama still deducts. For qwen3.5:9b to fit fully next to Bonsai at ctx=64k, `OLLAMA_GPU_OVERHEAD` would need to be ~0; at that point the reservation semantics no longer really protect Bonsai. Net: on a 16 GiB card, the overhead knob protects Bonsai OR lets qwen9b fit on GPU, but **not both at ctx=64k**.

(With qwen3.6:35b on CPU-only — see §5 — this becomes moot; qwen isn't touching VRAM regardless.)

---

## 5. Layer 4 — Honcho `tool_choice: "any"` rejected by llama.cpp

### Symptom
```
# bonsai.log
srv    operator(): got exception: {"error":{"code":400,
  "message":"Invalid tool_choice: any","type":"invalid_request_error"}}
```
On every Honcho tool-using call (dialectic / deriver / summary / dream). Three tenacity retries, then `HTTP 500` to Hermes after ~12 s.

### Root cause
`honcho/src/llm/backends/openai.py:_build_params` passes `tool_choice` straight through. Honcho's internal tool_choice values are `None | "auto" | "any" | "required" | {…specific function}` — the Anthropic and Gemini backends normalize `"any"` (Anthropic-style synonym for "force tool call"), but the OpenAI backend does not.

Why upstream hasn't caught this: `"any"` is accepted by:
- vLLM (as a non-standard extension)
- OpenAI's legacy endpoints (pre-2024 before `"required"` replaced `"any"` in the spec)

but rejected by:
- **llama.cpp's OpenAI-compatible server** (strict spec: `auto | none | required | function-name`)

GitHub issue search returned zero hits for this exact combo — because `llama.cpp + Honcho` is an uncommon deployment shape. This stack is in that shape specifically because it runs Bonsai-8B via llama.cpp.

### Verification — I checked my own claim
The user correctly pushed back with "if this were a Honcho bug, someone would have filed an issue by now." I reverted the patch, re-copied the unpatched file to the containers, restarted, re-probed — **HTTP 500 re-appeared with exactly the same 400 message from Bonsai**. Re-applied patch → 200 / 1.2 s full dialectic with coherent response. Not a misread.

### Fix applied
```python
# honcho/src/llm/backends/openai.py:311
if tool_choice is not None:
    # OpenAI's spec accepts "none"|"auto"|"required"|{type:"function",...}.
    # Honcho normalizes to "any"/"required" internally (Anthropic-style),
    # but passing "any" to OpenAI-compatible servers like llama.cpp returns
    # 400 invalid_request_error. Map to the spec-compliant value here so
    # self-hosted backends accept the request.
    if tool_choice == "any":
        tool_choice = "required"
    params["tool_choice"] = tool_choice
```
Committed as `e875f63` in `baba-yu/nuncstans-honcho` (submodule), bumped in parent `e5e0cab` with README callout.

Equivalent behaviour for OpenAI proper (`"required"` has been the canonical since 2024) and for vLLM (also accepts `"required"`), so no regression for users pointing honcho at those backends. Upstream PR deliberately not filed for now — the user's explicit call ("option B makes me nervous").

---

## 6. Layer 5 — CPU decode hardware ceiling

### Setup
After layers 1–4 were fixed, Hermes was still slow. Measured one realistic turn:

```
prompt_tokens:      4,143
completion_tokens:  800 (configured cap)
total time:         116 s   (from curl -w %{time_total})
```

Working hypothesis at that point: Hermes was doing N extra LLM calls per turn (title-gen, vision-detect, auxiliary auto-detect, main, memory-sync). **This turned out to be wrong.** Reading `auxiliary_client.py` showed the "Vision auto-detect" log is emitted by a client-resolution path that doesn't actually call the LLM. In the 30-minute window where the user was experiencing stalls, ollama received:

```
chat/completions from 127.0.0.1: 5 calls — all for qwen3.6:35b
  5m23s   17.66s   1m7s   4.45s (title)   7.85s (my probe)
embeddings   from 172.21.0.4: 14 calls — all < 1.3 s
```

The long calls were the Hermes main chat, not hidden probes. Hermes' CPU inference *is* the wall.

### Thread-count sweep (llama-bench on the same GGUF, `-ngl 0`)
| NumThreads | prompt eval (tok/s) | decode (tok/s) | verdict |
|---:|---:|---:|---|
| 8   | 599 | **9.55** | marginal best |
| 10 (ollama default) | 588 | 9.06 | near-optimal |
| 16  | 583 | 9.32 | same ±noise |
| 20  | **336** | **4.49** | **regression** |

T20 halved performance — classic SMT/E-core contention on a hybrid Intel part. Ollama's instinct to pick "half of physical cores" is actually right on this hardware. **Threading tuning gives ≤ ~5 %.**

### Budget breakdown of the 116 s run
```
prompt eval  :  4143 tok / 588 tok·s⁻¹  =   7.0 s
decode       :   800 tok /   9.1 tok·s⁻¹ =  87.9 s
model load   : 99 ms on the 2nd call, ~21 s cold
                                     ≈ 116 s total
```
**Decode at ~9 tok/s is the ceiling.** For a Hermes turn with 2000 tokens of reply, that's 220 s. For 2500 tokens (Honcho's `DEFAULT_MAX_TOKENS`), 275 s. Matches the 5m23s observed in the logs — not an outlier, exactly the expected value.

### Cross-referenced against public benchmarks
From r/LocalLLaMA, hardware-corner.net, arsturn.com, llama.cpp issues:
- Qwen3-30B-A3B Q8 on Ryzen DDR5-6000: pe ≈ 160 / decode ≈ 22 tok/s
- Qwen3-30B-A3B Q4 on Ryzen DDR5-5600: decode 18–20 tok/s
- Qwen3-30B-A3B Q4 on Ryzen 7950X3D DDR5: decode 12–15 tok/s
- Qwen3-Next-80B-A3B on Ryzen AI 9 HX PRO 370: decode 7.74 tok/s ("3–4× slower than theory predicts")

Our 9 tok/s on WSL2 / 20-thread Intel / DDR5 is firmly in the envelope. Not anomalous, not fixable by threading.

### Cache invalidation (#20003) verified

Hypothesis from the research literature: llama.cpp is re-prefilling the full context every turn instead of reusing the prefix cache for this model family. Test: three sequential calls through the same warm runner.

| Turn | messages sent | expected `prompt_eval_count` (healthy) | observed |
|---|---|---:|---:|
| 1 (cold) | `[user_A]` (20 tok) | 20 | 20 |
| 2 (same) | `[user_A]` again | ~0 (full cache hit) | **20** — redone |
| 3 (extended) | `[user_A, assistant_A, user_B]` (37 tok) | ~17 (only delta) | **37** — redone |

**Every turn re-prefills from scratch.** For Hermes's realistic 4–13 k-token prompts, that's ~7–17 s per turn of prefill work that a healthy KV-cache would skip. Not the dominant cost (decode still dwarfs it) but real.

Known upstream bug for this model family: [llama.cpp#20003](https://github.com/ggml-org/llama.cpp/issues/20003). Workaround possibilities not tested yet: ik_llama.cpp fork, alternative chat templates.

---

## 7. Layer 6 — qwen3 thinking-token leak (the one that explains "minutes of empty response")

### Trigger
After the CPU speed was explained, one residual weirdness: my probes all showed
```
"usage": {"completion_tokens": 2299}
"message": {"content": ""}
```
The model was *consuming* budget but producing nothing visible.

### Root cause
[llama.cpp#20099](https://github.com/ggml-org/llama.cpp/issues/20099) — qwen3-series reasoning ("thinking") models return their chain-of-thought as `message.thinking` and the final answer as `message.content`. On this build / template, decode is exhausted mid-thinking before content is ever generated. With the default reasoning budget, `completion_tokens` is entirely thinking tokens.

### Direct verification — four variants, same tiny prompt
```
prompt: "Say exactly: one two three."
```

| Variant | `content` | `thinking` | time |
|---|---|---:|---:|
| baseline (`num_predict=64`) | `""` | 64 tok of "thinking process…" (unfinished) | **26.4 s** |
| `/no_think` prefix in user msg | `""` | 64 tok (prefix not respected by template) | 6.7 s |
| **`"think": false` in request options** | **`"one two three"`** | absent from response | **0.82 s** |
| `num_predict=400` (bigger budget) | `"one two three."` | 231 tok | 23.3 s |

**`think: false` is a 32× latency reduction AND actually produces output.** Without it, Hermes waits minutes for a model that never emits content.

### Fix path (identified, not yet applied in repo or live config)
The OpenAI-compatible API has no `think` field — it's an Ollama-specific extension. Hermes's `provider: custom` OpenAI client likely doesn't know about it. Two implementation options:

1. **Modelfile override (preferred):**
   ```
   FROM qwen3.6:35b
   PARAMETER think false
   ```
   ```
   ollama create qwen3.6:35b-nothink -f Modelfile
   ```
   Then point `~/.hermes/config.yaml:model.default` at `qwen3.6:35b-nothink`. Works for any OpenAI-compat client downstream.

2. **Hermes-side patch**: thread `think: false` into the custom provider's request-builder. Cleaner long-term, requires reading Hermes's OpenAI client code to find the extension-passthrough point.

Option 1 is tested-in-principle (the `"think": false` API call worked) but the Modelfile `PARAMETER think false` has not been verified — need to confirm Ollama's Modelfile parser accepts this parameter name. If not, an `OPTIONS` / `TEMPLATE` override achieves the same.

### Expected post-fix performance
With `think: false` and the other fixes in place, an 800-token Hermes reply becomes:
```
prompt eval : 4143 / 588 tok·s⁻¹ =  7 s
decode      :  800 /   9 tok·s⁻¹ = 89 s
total       :                    ~96 s
```
Still CPU-slow, but the 96 s is spent on actual content rather than invisible reasoning. For shorter replies (100–500 tok, typical casual response), 11–55 s — usable.

On top of that, if [#20003](https://github.com/ggml-org/llama.cpp/issues/20003) were also fixed (via ik_llama.cpp or template switch), each subsequent turn in a session drops another 7–17 s of redundant prefill.

---

## 8. Session evolution — how the diagnosis shifted

This investigation is instructive to revisit because the *first* diagnosis was wrong in a way that sounded plausible. Each user intervention below **corrected a category error I had committed** and moved the investigation forward. Recorded in order.

### Pivot 1 — "those are symptoms, not root causes"
Initial write-up listed 5 symptoms (Honcho dialectic failing, libmtmd missing, CLOSE_WAIT sockets, `<|endoftext|>` tiktoken errors, port 8080 unlistened). User correctly pointed out I had collected evidence but not yet isolated the root cause.

Corrected by `strace`-level introspection (proc wchan, lsof-equivalent via `ss -tnp`), container-to-host probing, and careful reading of Honcho logs. Reached first real root cause: Honcho routes everything to `:8080` which isn't listening, hence all the 60 s stalls — i.e. one cause, many downstream symptoms.

### Pivot 2 — "ChatGPT says this is also a known llama.cpp issue"
User brought context from a parallel conversation with ChatGPT: the `libmtmd.so.0` issue is known upstream. This forced me to stop framing my fix as a novel discovery and instead locate it in the upstream issue graph (#17193 / #17190 / #17950 / PR #17214).

This pivoted the fix from a one-off patch to a **documented known-issue mitigation**. Also led to noticing that PR #17214's merged fix covers only the Docker-symlink case, not the absolute-RUNPATH-after-rename case we were in. Without this pivot I would have closed the ticket on a false "upstream already fixed it" assumption.

### Pivot 3 — "doesn't this mean ollama is also truncating native calls?"
I had framed the `OLLAMA_CONTEXT_LENGTH=4096` issue as "this breaks Hermes." User correctly sharpened: it breaks *everything* that doesn't pass `options.num_ctx` — `ollama run`, Open WebUI, LangChain, etc. Not a Hermes problem, a host-wide silent truncation.

This sharpened the README copy (Step 2 callout now reads "prevents Ollama from silently truncating prompts at 4096 tokens" rather than mentioning Hermes), and reframed the fix as a general infrastructure setting rather than a Hermes-specific workaround.

### Pivot 4 — "why does Bonsai take VRAM? I don't get it"
I had written "taken almost entirely by Bonsai" for VRAM. User flagged the overstatement. I measured: Bonsai at `-c 16384` uses **6.3 GiB** of 16 GiB — not "almost entirely." The README had said ~7.6 GiB, also an over-estimate. Corrected both the reasoning (qwen still has ~8 GiB to work with in principle) and the doc target (README bump pending).

### Pivot 5 — "was this patch really necessary?" (about the `tool_choice` patch)
User pasted a sharp counter-analysis: "Honcho has zero open issues for this — maybe the bug is our config, not Honcho's code." This was the most useful intervention of the session because it demanded **evidence-grade verification** of a patch I had already committed.

Test: `git stash` the patch, `docker cp` the original into running containers, restart, re-probe. **Same HTTP 500 / `Invalid tool_choice: any` 400 recurred**. Re-applied patch → 200 / 1.2 s clean dialectic. Binary reproduction confirmed the patch is load-bearing, not speculative.

The counter-analysis's premise ("if it were a real Honcho bug, someone would file an issue") was wrong because `llama.cpp + Honcho` is a rare deployment. vLLM accepts `"any"` as an extension, OpenAI-proper's `TOOL_CHOICE=None` default sidesteps it, so most users never hit the bug. The counter-analysis correctly identified a *pattern* (zero issues) but drew the wrong inference.

This pivot produced the most confident fix of the session: **both the claim and the test protocol are now reproducible**.

### Pivot 6 — "qwen3.6:35b on RAM works fine for me"
After switching to qwen3.5:9b I declared the only path to interactive latency was model downgrade. User objected: they run qwen3.6:35b via plain `ollama run` on RAM and it's usable. Why was Hermes+qwen3.6:35b different?

My initial answer was "Hermes makes 5–10 extra LLM calls per turn." User asked me to prove it. I pulled 30 minutes of ollama access logs — and **my claim collapsed**. The real per-turn traffic was 1 main chat + few embeddings, not 5+ probes. The `Vision auto-detect` log entries I had cited were from client-resolution code that doesn't call the LLM.

Revised answer: "Hermes's perceived slowness is prompt-size selection bias — `ollama run` with tiny prompts *is* fast, but 10k-token prompts on the same model are slow regardless of who sends them."

### Pivot 7 — "hard to believe; have the Agent Team look for prior work and test"
User wouldn't accept "CPU is just this slow" without external corroboration. Dispatched two agents in parallel:
- Research agent → found qwen3.5-35b-a3b CPU benchmarks, identified llama.cpp#20003 and #20099 as active bugs for this model
- Benchmark agent → ran threaded sweep (llama-bench, 8/10/16/20 threads) and confirmed decode ceiling

The research agent's find of **#20099 (thinking-token leak)** was the session's biggest deliverable. Without this pivot I would have settled for "CPU is slow" as the answer, missing the real reason Hermes produced empty responses: qwen3.6:35b spends all its tokens on invisible reasoning and never emits content.

### Takeaway
Every pivot came from the user demanding one more layer of evidence or pointing out a framing error. This investigation is a strong argument for **treating "the bug is fixed, tests pass" as a hypothesis rather than a conclusion** until the claim has survived a revert-and-reproduce check.

---

## 9. Operational side-effects of this investigation

State changes not covered by any commit, worth knowing about:

- **`/home/baba-y/hermes-stack/`** — a root-owned empty directory stub exists at this path. Docker's daemon created it when the original honcho containers tried to bind-mount `./docker/entrypoint.sh` (relative to compose file at the old path `hermes-stack/honcho/docker-compose.yml`) and failed after the rename. Cosmetic; `sudo rmdir /home/baba-y/hermes-stack` removes it.

- **Honcho containers were recreated mid-investigation.** The original containers had bind mounts resolved to the pre-rename absolute path (`/home/baba-y/hermes-stack/...`), so `docker restart` against them failed with "not a directory" after the directory rename. Resolution: `cd honcho && docker compose up -d --force-recreate --no-deps api deriver`, which re-resolved the mount to the new path. Database and Redis containers were unaffected (they use docker volumes, not host binds).

- **`honcho/config.toml.bonsai-example`** has an uncommitted local edit (~11 lines of comment refactor re `MAX_EMBEDDING_TOKENS` moving between sections in upstream Honcho). Not touched in this session; left for the user to handle separately.

- **Old `bonsai-llama.cpp/build.old/`** preserved (218 MiB) — the pre-$ORIGIN build, kept for rollback. Safe to `rm -rf build.old` when confident.

- **Ollama systemd drop-in** now exists at `/etc/systemd/system/ollama.service.d/override.conf` with three env vars (`OLLAMA_HOST`, `OLLAMA_GPU_OVERHEAD`, `OLLAMA_CONTEXT_LENGTH`). Check with `systemctl show ollama -p Environment`.

- **Scratch files in `/tmp/`** from bench runs: `/tmp/qwen-bench/`, `/tmp/big_real*.json`, `/tmp/bonsai_*.log`, `/tmp/stream_bench.sh`. Harmless; will clear on next reboot (tmpfs on most distros).

---

## 10. Commits produced by this investigation

**Parent repo (`baba-yu/nuncstans-hermes-stack`, branch `dev`)** — 2 ahead of origin, unpushed:

```
e5e0cab docs(readme): ollama env vars + honcho tool_choice patch; bump honcho
bc5f36f docs(readme): pin bonsai-llama.cpp build to $ORIGIN RPATH
```

Both authored as `baba-yu <baba@zipteam.com>`, consistent with the `baba-yu/*` remote. No `Co-Authored-By:` or AI-attribution trailers (per the user's global preference in `~/.claude/CLAUDE.md`).

**Honcho submodule (`baba-yu/nuncstans-honcho`, branch `dev`)** — 1 ahead of origin, unpushed:

```
e875f63 fix(llm/openai): normalize tool_choice "any" to "required"
```

The parent repo's `e5e0cab` bumps the submodule pointer to this commit.

**Push ordering reminder** — when pushing, do the submodule first, then the parent, otherwise `origin/dev` on the parent will reference a commit the submodule remote doesn't yet have.

---

## 11. Things investigated and ruled out

Recording these so they don't get re-investigated:

| Hypothesis | Why we thought so | Outcome |
|---|---|---|
| Hermes makes 5–10 LLM calls per turn (title, vision-probe, aux, main, memory, …) | agent.log shows 4 "Vision auto-detect" logs at startup + aux + title | **Wrong.** auxiliary_client.py's "Vision auto-detect" log is emitted by client-resolution, not a real LLM call. Real per-turn traffic is ~1 main chat + embeddings via Honcho. |
| Honcho/Hermes config mixes `transport=anthropic` with an OpenAI URL, sending `{"type":"any"}` to a text/chat endpoint | Would explain `Invalid tool_choice: any` | **Wrong.** `config.toml` has `transport = "openai"` for every module; the bug was literal string `"any"` from Honcho's openai backend. |
| `llama-server` needs `--jinja` for tool calling | Known requirement on some llama.cpp versions | **Not the active cause.** Bonsai's GGUF has a Jinja chat template with `{% if tools %}` baked in; tools work (first chat request returns 200). The 400 only fires on the `tool_choice` value. |
| `OLLAMA_NUM_THREADS` env var will fix `NumThreads=10` | Obvious knob | **Partially wrong.** (a) env var is inconsistently respected under systemd; (b) the default of 10 is already near-optimal for this hybrid CPU. Raising to 20 *regresses*. |
| qwen3.5:9b on GPU will be dramatically faster in the Bonsai+Ollama coexistence | True in isolation | **Partially wrong in practice.** `OLLAMA_GPU_OVERHEAD=7 GiB` + Bonsai's 6.3 GiB real usage leaves only ~2.6 GiB effective for ollama's allocator; qwen9b at ctx=64k needs 8.8 GiB → falls back to 5/33 layers on GPU, still mostly CPU. Fitting qwen9b fully requires reducing overhead to ~0 or ctx to ~16 k. |
| WSL2 imposes a large CPU-inference penalty | Common folklore | **Not supported by data.** No authoritative recent benchmark shows more than single-digit % penalty on compute-bound workloads. Not our bottleneck. |

---

## 12. Current status and recommended next actions

| Item | State |
|---|---|
| Bonsai RPATH fix committed + README updated | ✅ `bc5f36f` |
| Ollama systemd env vars live + README updated | ✅ `e5e0cab` |
| Honcho `tool_choice` patch + submodule bump committed | ✅ `e875f63` (submodule) + parent bump in `e5e0cab` |
| Working qwen3.6:35b nothink Modelfile + Hermes config flip | ⬜ **not applied** — highest-impact remaining item |
| Verify ollama Modelfile `PARAMETER think false` syntax | ⬜ need to confirm; may need TEMPLATE override instead |
| Try ik_llama.cpp fork for cache-invalidation (#20003) | ⬜ optional follow-up, expected modest win |
| README ollama section currently states Bonsai uses ~7.6 GiB; the real number measured here is 6.3 GiB | ⬜ minor doc fix, non-urgent |
| Upstream PR for Honcho openai backend `"any"` normalization | ⬜ deferred by user decision; local patch stands |

### If the goal is interactive latency with `qwen3.6:35b` as the main model

1. **Apply `think: false` fix** (Modelfile or Hermes patch). ~32× faster on short replies, and responses actually exist.
2. **Live with CPU decode ceiling.** 9 tok/s is the hardware wall. Short replies (100–500 tok) are 11–55 s; long replies (2 k+) are 3–5 min. Not a bug, physics.

### If the goal is sub-10-second turns with real content

The 16 GiB GPU does not fit `qwen3.6:35b`. Realistic options on this box:

1. Run `qwen3.5:9b` or similar ≤ 8 GiB model fully on GPU, drop `OLLAMA_GPU_OVERHEAD` to near-0, accept Bonsai having no safety margin.
2. Keep Bonsai, drop Hermes's main model to `gpt-oss:20b` at short ctx (barely fits alongside Bonsai).
3. Accept hybrid quality / speed: Bonsai for Honcho memory (fast, on GPU), qwen3.6:35b for Hermes main chat (slow, on CPU, now with `think: false` so it at least produces output).

### Sources cited

- [ggml-org/llama.cpp#17193](https://github.com/ggml-org/llama.cpp/issues/17193), [#17190](https://github.com/ggml-org/llama.cpp/issues/17190), [#17950](https://github.com/ggml-org/llama.cpp/issues/17950) — libmtmd.so.0 RPATH issue
- [ggml-org/llama.cpp#17214](https://github.com/ggml-org/llama.cpp/pull/17214) — Docker-symlink fix (does not cover our case)
- [ggml-org/llama.cpp#20003](https://github.com/ggml-org/llama.cpp/issues/20003) — qwen3.5-35b-a3b full-prompt re-prefill
- [ggml-org/llama.cpp#20099](https://github.com/ggml-org/llama.cpp/issues/20099) — qwen3.5-35b-a3b thinking-budget / content leak
- [ollama/ollama#2496](https://github.com/ollama/ollama/issues/2496), [#2929](https://github.com/ollama/ollama/issues/2929), [#4477](https://github.com/ollama/ollama/issues/4477) — NumThreads default behaviour
- [glukhov.org — Ollama Intel P/E core testing](https://www.glukhov.org/post/2025/05/ollama-cpu-cores-usage/) — empirical P/E thread behaviour
- [hardware-corner.net — Qwen3 hardware requirements](https://www.hardware-corner.net/guides/qwen3-hardware-requirements/) — comparable decode numbers
- [arsturn.com — Qwen3-Coder 30B hardware guide](https://www.arsturn.com/blog/running-qwen3-coder-30b-at-full-context-memory-requirements-performance-tips) — Ryzen 7950X3D numbers
- [ubergarm — Qwen3 quant roundup](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38) — ik_llama.cpp notes
- [CMake BUILD_RPATH_USE_ORIGIN](https://cmake.org/cmake/help/latest/prop_tgt/BUILD_RPATH_USE_ORIGIN.html)
- [llama.cpp build docs — $ORIGIN pattern](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
