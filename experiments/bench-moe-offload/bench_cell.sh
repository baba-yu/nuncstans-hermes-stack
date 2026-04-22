#!/bin/bash
# bench_cell.sh — run one cell of the bench-moe-offload experiment.
#
# Usage: bench_cell.sh <cell_id>  where cell_id ∈ {L1,L2,L3a,L3b,L4a,L4b,L5,L6}
#
# For each cell:
#   1. Map cell id to llama-server flags.
#   2. Start test llama-server on :8080, log to results/<cell>/server.log.
#   3. Start 1 Hz resource sampler, log to results/<cell>/resources.tsv.
#   4. Wait for /health, then warmup (small request, discarded).
#   5. Single load: 3 consecutive non-stream requests with the Hermes-shape
#      payload, save response + parsed metrics.
#   6. If cell ∈ {L4a, L6}: contention load, 3 runs of (Hermes || Honcho)
#      in parallel, both non-stream.
#   7. Stop server and sampler.
#
# Safe to re-run a single cell (overwrites prior results).

set -euo pipefail

CELL="${1:?cell id required: L1|L2|L3a|L3b|L4a|L4b|L5|L6}"

# ---------- paths ----------
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT/experiments/bench-moe-offload"
LLAMA_BUILD="$ROOT/bonsai-llama.cpp/build/bin"
# Use unsloth's Q4_K_XL GGUF via -hf rather than ollama's stored blob.
# Ollama ships the qwen3.6:35b blob with a 3-element
# qwen35moe.rope.dimension_sections, which our llama.cpp fork rejects
# (expects 4). Unsloth's UD-Q4_K_XL uses the 4-element layout the build
# understands. llama.cpp caches the download at ~/.cache/llama.cpp so
# subsequent cell invocations reuse the same file.
HF_CHAT_SPEC="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"
RESULTS="$BENCH_DIR/results/$CELL"
HERMES_PROMPT="$BENCH_DIR/prompts/hermes_4k.json"
HONCHO_PROMPT="$BENCH_DIR/prompts/honcho_deriver.json"

mkdir -p "$RESULTS"

# ---------- common flags ----------
COMMON_FLAGS=(
    -hf "$HF_CHAT_SPEC"
    --host 127.0.0.1 --port 8080
    -c 65536
    -fa on
    -ctk q8_0 -ctv q8_0
    --jinja
    --alias qwen3.6-test
)

# ---------- per-cell flags ----------
case "$CELL" in
    L1)  CELL_FLAGS=(-ngl 0) ;;
    L2)  CELL_FLAGS=(-ngl 0 --reasoning off) ;;
    L3a) CELL_FLAGS=(-ngl 30) ;;
    L3b) CELL_FLAGS=(-ngl 35) ;;
    L4a) CELL_FLAGS=(-ngl 30 --reasoning off) ;;
    L4b) CELL_FLAGS=(-ngl 35 --reasoning off) ;;
    L5)  CELL_FLAGS=(-ngl 99 -ot "ffn_(up|down|gate)_exps=CPU") ;;
    L6)  CELL_FLAGS=(-ngl 99 -ot "ffn_(up|down|gate)_exps=CPU" --reasoning off) ;;
    *) echo "unknown cell: $CELL" >&2; exit 1 ;;
esac

info()  { echo "[$CELL] $*"; }
die()   { echo "[$CELL] ERROR: $*" >&2; exit 1; }

# ---------- preflight ----------
info "preflight"
[[ -x "$LLAMA_BUILD/llama-server" ]] || die "llama-server binary missing"
[[ -f "$HERMES_PROMPT" ]] || die "prompts/hermes_4k.json missing (run prep.sh)"
if [[ "$CELL" == "L4a" || "$CELL" == "L6" ]]; then
    [[ -f "$HONCHO_PROMPT" ]] || die "prompts/honcho_deriver.json missing (run prep.sh)"
fi
# port 8080 must be free
if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ':8080$'; then
    die "port 8080 already bound; stop bonsai/ollama and rerun"
fi

# ---------- start llama-server ----------
info "launching llama-server with cell-specific flags"
info "  flags: ${CELL_FLAGS[*]}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"$LLAMA_BUILD/llama-server" "${COMMON_FLAGS[@]}" "${CELL_FLAGS[@]}" \
    > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    info "cleanup"
    kill $SERVER_PID 2>/dev/null || true
    [[ -n "${SAMPLER_PID:-}" ]] && kill $SAMPLER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# ---------- start sampler ----------
"$BENCH_DIR/scripts/sample_resources.sh" "$RESULTS/resources.tsv" &
SAMPLER_PID=$!
sleep 1

# ---------- wait for health ----------
info "waiting for /health (up to 600s; first cell may download ~20 GiB)"
deadline=$(( $(date +%s) + 600 ))
while :; do
    if curl -sfS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
        info "server healthy"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        die "server died during startup; see $RESULTS/server.log"
    fi
    (( $(date +%s) >= deadline )) && die "server never became healthy"
    sleep 2
done

# ---------- warmup ----------
info "warmup (tiny request, discarded)"
curl -sS --max-time 300 http://127.0.0.1:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"qwen3.6-test","messages":[{"role":"user","content":"warmup"}],"max_tokens":4,"stream":false}' \
    -o "$RESULTS/warmup.json" >/dev/null || die "warmup request failed"
info "  warmup size: $(wc -c < "$RESULTS/warmup.json") bytes"

# ---------- single load: 3 runs ----------
fire_one() {
    local run_idx="$1" label="$2" prompt_file="$3" out_file="$4" time_file="$5"
    info "  $label run $run_idx"
    /usr/bin/time -f '%e' -o "$time_file" \
        curl -sS --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
            -H 'content-type: application/json' \
            --data-binary @"$prompt_file" \
            -o "$out_file"
}

info "single load (Hermes-shape × 3)"
for i in 1 2 3; do
    fire_one "$i" "single" "$HERMES_PROMPT" \
        "$RESULTS/single_run${i}.json" \
        "$RESULTS/single_run${i}.time"
done

# ---------- contention load (L4a, L6 only) ----------
if [[ "$CELL" == "L4a" || "$CELL" == "L6" ]]; then
    info "contention load (Hermes || Honcho × 3)"
    for i in 1 2 3; do
        info "  contention run $i"
        # fire Hermes first, ~100ms stagger, then Honcho
        (
            /usr/bin/time -f '%e' -o "$RESULTS/contention_hermes${i}.time" \
                curl -sS --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
                    -H 'content-type: application/json' \
                    --data-binary @"$HERMES_PROMPT" \
                    -o "$RESULTS/contention_hermes${i}.json"
        ) &
        H_PID=$!
        sleep 0.1
        (
            /usr/bin/time -f '%e' -o "$RESULTS/contention_honcho${i}.time" \
                curl -sS --max-time 900 http://127.0.0.1:8080/v1/chat/completions \
                    -H 'content-type: application/json' \
                    --data-binary @"$HONCHO_PROMPT" \
                    -o "$RESULTS/contention_honcho${i}.json"
        ) &
        D_PID=$!
        wait $H_PID $D_PID
    done
fi

# ---------- parse metrics ----------
info "parsing metrics"
python3 - "$RESULTS" "$CELL" "${CELL_FLAGS[*]}" <<'PY'
import json, os, statistics, sys, pathlib

results_dir = pathlib.Path(sys.argv[1])
cell = sys.argv[2]
flags = sys.argv[3]

def load_response(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

def extract(resp):
    if resp is None: return None
    try:
        content = resp["choices"][0]["message"].get("content", "")
    except Exception:
        content = ""
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    return {
        "content_len": len(content),
        "content_nonempty": bool(content.strip()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }

def read_time(path):
    try:
        return float(open(path).read().strip())
    except Exception:
        return None

def summarize(prefix, count):
    runs = []
    for i in range(1, count + 1):
        resp = load_response(results_dir / f"{prefix}_run{i}.json")
        wall = read_time(results_dir / f"{prefix}_run{i}.time")
        m = extract(resp) or {}
        m["wall_s"] = wall
        runs.append(m)
    # medians where numeric
    agg = {"runs": runs}
    for key in ("prompt_ms","prompt_per_second","predicted_ms","predicted_per_second","wall_s",
                "prompt_tokens","completion_tokens"):
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        if vals:
            try:
                agg[f"median_{key}"] = statistics.median(vals)
            except Exception:
                pass
    agg["content_nonempty_all"] = all(r.get("content_nonempty") for r in runs)
    return agg

# VRAM / RAM peaks from resources.tsv
peaks = {"vram_peak_mib": 0, "ram_peak_mib": 0}
rsv = results_dir / "resources.tsv"
if rsv.exists():
    for i, line in enumerate(open(rsv)):
        if i == 0: continue
        parts = line.strip().split("\t")
        if len(parts) < 5: continue
        try:
            vu = int(parts[1]); ru = int(parts[3])
            if vu > peaks["vram_peak_mib"]: peaks["vram_peak_mib"] = vu
            if ru > peaks["ram_peak_mib"]: peaks["ram_peak_mib"] = ru
        except Exception:
            pass

report = {
    "cell": cell,
    "flags": flags,
    "single": summarize("single", 3),
    "resources": peaks,
}

contention_hermes = results_dir / "contention_hermes1.json"
if contention_hermes.exists():
    # contention: parallel hermes+honcho × 3
    def summarize_side(side, count):
        runs = []
        for i in range(1, count + 1):
            resp = load_response(results_dir / f"contention_{side}{i}.json")
            wall = read_time(results_dir / f"contention_{side}{i}.time")
            m = extract(resp) or {}
            m["wall_s"] = wall
            runs.append(m)
        agg = {"runs": runs}
        for key in ("prompt_ms","prompt_per_second","predicted_ms","predicted_per_second","wall_s"):
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            if vals:
                try:
                    agg[f"median_{key}"] = statistics.median(vals)
                except Exception:
                    pass
        agg["content_nonempty_all"] = all(r.get("content_nonempty") for r in runs)
        return agg
    report["contention"] = {
        "hermes": summarize_side("hermes", 3),
        "honcho": summarize_side("honcho", 3),
    }
    hsum = report["single"].get("median_wall_s")
    csum = report["contention"]["hermes"].get("median_wall_s")
    if hsum is not None and csum is not None:
        report["contention"]["hermes_delta_s"] = csum - hsum

# Always emit single_metrics.json (single-load + resources only) so
# run_all.sh has a stable path for the L3b/L4b gate. If contention ran,
# also emit metrics.json as the complete record (single + contention).
single_only = {k: v for k, v in report.items() if k != "contention"}
open(results_dir / "single_metrics.json", "w").write(
    json.dumps(single_only, indent=2, ensure_ascii=False)
)
print(f"[py] wrote {results_dir / 'single_metrics.json'}", file=sys.stderr)
if "contention" in report:
    open(results_dir / "metrics.json", "w").write(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(f"[py] wrote {results_dir / 'metrics.json'}", file=sys.stderr)
PY

info "done. results in $RESULTS/"
