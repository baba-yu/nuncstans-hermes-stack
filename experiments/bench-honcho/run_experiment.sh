#!/usr/bin/env bash
# run_experiment.sh N [TAG] [TIMEOUT]
# N ∈ {1..6}, reconfigures stack, runs Hermes task, records metrics.
# Optional TAG appends "-TAG" to the run dir (e.g., "2" -> runs/E1-2/).
# Optional TIMEOUT in seconds (default 1800).
set -euo pipefail
N="$1"
TAG="${2:-}"
HERMES_TIMEOUT="${3:-1800}"
BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
source "$BENCH_ROOT/lib.sh"

# Per-experiment config
case "$N" in
  1) HONCHO_VARIANT=bonsai;      INFER_MODEL="qwen3.5:9b";   OLLAMA_OVH=8589934592; OLLAMA_MAXL=1; NEED_BONSAI=1; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  2) HONCHO_VARIANT=bonsai;      INFER_MODEL="qwen3.6:35b";  OLLAMA_OVH=8589934592; OLLAMA_MAXL=1; NEED_BONSAI=1; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  3) HONCHO_VARIANT=ollama-9b;   INFER_MODEL="qwen3.6:35b";  OLLAMA_OVH=0;          OLLAMA_MAXL=2; NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  4) HONCHO_VARIANT=ollama-9b;   INFER_MODEL="qwen3.5:9b";   OLLAMA_OVH=0;          OLLAMA_MAXL=1; NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  5) HONCHO_VARIANT=ollama-35b;  INFER_MODEL="qwen3.6:35b";  OLLAMA_OVH=0;          OLLAMA_MAXL=1; NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  6) HONCHO_VARIANT=bonsai;      INFER_MODEL="bonsai-8b";    OLLAMA_OVH=8589934592; OLLAMA_MAXL=1; NEED_BONSAI=1; BONSAI_PARALLEL=1; BONSAI_CTX=65536 ;;
  *) echo "bad experiment number: $N"; exit 2 ;;
esac

RUN_DIR="$BENCH_ROOT/runs/E${N}${TAG:+-$TAG}"
mkdir -p "$RUN_DIR"
log "=== BEGIN E${N}${TAG:+-$TAG}: honcho=$HONCHO_VARIANT inference=$INFER_MODEL timeout=${HERMES_TIMEOUT}s ==="

# --- Teardown prior stack state ---
stop_bonsai
# Unload all Ollama models
ollama_unload_all || true

# --- Ollama env reconfig (sudo-needed) ---
set_ollama_env "$OLLAMA_OVH" "$OLLAMA_MAXL"

# --- Bonsai startup (if needed) ---
if [ "$NEED_BONSAI" = "1" ]; then
  start_bonsai "$BONSAI_PARALLEL" "$BONSAI_CTX"
fi

# --- Honcho config swap + restart ---
write_honcho_config "$HONCHO_VARIANT"
honcho_restart_workers
honcho_cleanup_bench

# --- Hermes config swap ---
python3 "$BENCH_ROOT/swap_hermes.py" backup || true
python3 "$BENCH_ROOT/swap_hermes.py" "$INFER_MODEL"
update_hermes_workspace "bench-e${N}${TAG:+-$TAG}"
clean_hermes_session_state

# --- Warmup ---
if [ "$INFER_MODEL" = "bonsai-8b" ]; then
  bonsai_warmup
else
  ollama_warmup "$INFER_MODEL"
fi
# Also warmup the honcho deriver endpoint so first call isn't cold
if [ "$HONCHO_VARIANT" = "bonsai" ]; then
  bonsai_warmup
elif [ "$HONCHO_VARIANT" = "ollama-9b" ]; then
  ollama_warmup "qwen3.5:9b"
elif [ "$HONCHO_VARIANT" = "ollama-35b" ]; then
  ollama_warmup "qwen3.6:35b"
fi

# --- Snapshot pre-run state ---
snapshot_ollama_ps > "$RUN_DIR/ollama_ps_pre.json"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits > "$RUN_DIR/vram_pre.csv"

# --- Build task prompt ---
TASK_PROMPT=$(sed "s/{N}/${N}/" "$BENCH_ROOT/task_prompt.txt")
echo "$TASK_PROMPT" > "$RUN_DIR/task_prompt.txt"

# --- Launch VRAM recorder ---
vram_recorder_start "$RUN_DIR/vram_trace.csv"

# Truncate bonsai-current.log reference for per-run isolation
echo "--- E${N} start at $(date -Iseconds) ---" >> "$BENCH_ROOT/logs/bonsai-current.log" 2>/dev/null || true
RUN_START_ISO=$(date -Iseconds)
echo "$RUN_START_ISO" > "$RUN_DIR/start_iso.txt"

# --- Run Hermes (from STACK_ROOT so pptx lands there) ---
log "invoking hermes chat (this may take a while)"
# Ensure any prior pptx from this N is moved aside so we measure a fresh create
[ -f "$STACK_ROOT/llm_presentation_${N}.pptx" ] && \
  mv "$STACK_ROOT/llm_presentation_${N}.pptx" "$RUN_DIR/llm_presentation_${N}.prior.pptx"
START_NS=$(date +%s%N)
set +e
# --source 'tool' keeps it out of user session lists. -Q quiet. --yolo bypass approvals.
(cd "$STACK_ROOT" && timeout "$HERMES_TIMEOUT" hermes chat -Q --yolo --max-turns 60 --source tool -q "$TASK_PROMPT") \
    > "$RUN_DIR/hermes_stdout.txt" 2> "$RUN_DIR/hermes_stderr.txt"
EXIT_CODE=$?
set -e
END_NS=$(date +%s%N)
WALL_SEC=$(python3 -c "print(($END_NS-$START_NS)/1e9)")
log "hermes finished exit=$EXIT_CODE wall=${WALL_SEC}s"

# --- Wait for Honcho deriver queue to drain (captures async background work) ---
DRAIN_START_NS=$(date +%s%N)
for i in $(seq 1 180); do
  pending=$(cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -tA -c \
    "SELECT count(*) FROM queue WHERE workspace_name='bench-e${N}${TAG:+-$TAG}' AND processed=false;" 2>/dev/null || echo "0")
  [ "${pending:-0}" -eq 0 ] && break
  sleep 1
done
DRAIN_END_NS=$(date +%s%N)
DRAIN_SEC=$(python3 -c "print(($DRAIN_END_NS-$DRAIN_START_NS)/1e9)")
log "deriver queue drained after ${DRAIN_SEC}s (final pending=${pending:-?})"

# Also count deriver processing activity
DERIVER_ROWS=$(cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -tA -c \
  "SELECT count(*) FROM queue WHERE workspace_name='bench-e${N}${TAG:+-$TAG}';" 2>/dev/null || echo "0")
OBS_COUNT=$(cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -tA -c \
  "SELECT count(*) FROM documents d JOIN peers p ON d.peer_name=p.name AND d.workspace_name=p.workspace_name WHERE d.workspace_name='bench-e${N}${TAG:+-$TAG}';" 2>/dev/null || echo "0")

# --- Stop VRAM recorder ---
vram_recorder_stop

# --- Snapshot post-run state ---
snapshot_ollama_ps > "$RUN_DIR/ollama_ps_post.json"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits > "$RUN_DIR/vram_post.csv"

# --- Copy bonsai log slice ---
cp "$BENCH_ROOT/logs/bonsai-current.log" "$RUN_DIR/bonsai.log" 2>/dev/null || true
# Deriver logs for the run window
(cd "$HONCHO_DIR" && docker compose logs --since="$RUN_START_ISO" deriver > "$RUN_DIR/deriver.log" 2>&1) || true
(cd "$HONCHO_DIR" && docker compose logs --since="$RUN_START_ISO" api > "$RUN_DIR/api.log" 2>&1) || true

# --- Locate and validate pptx ---
# Hermes's tools write relative to $HOME, not the cd'd CWD. Check both.
PPTX_PATH=""
for cand in "$STACK_ROOT/llm_presentation_${N}.pptx" \
            "$HOME/llm_presentation_${N}.pptx" \
            "$HOME/.hermes/workspace/llm_presentation_${N}.pptx" \
            "/tmp/llm_presentation_${N}.pptx"; do
  if [ -f "$cand" ]; then PPTX_PATH="$cand"; break; fi
done
if [ -z "$PPTX_PATH" ]; then PPTX_PATH="$STACK_ROOT/llm_presentation_${N}.pptx"; fi
PPTX_RESULT=$(python3 "$BENCH_ROOT/validate_pptx.py" "$PPTX_PATH" || echo '{"ok":false,"error":"validator failed"}')
echo "$PPTX_RESULT" > "$RUN_DIR/pptx_result.json"
log "pptx result: $PPTX_RESULT"

# Move pptx into run dir for preservation
if [ -f "$PPTX_PATH" ]; then
  cp "$PPTX_PATH" "$RUN_DIR/" || true
  # Remove from candidate locations so they don't shadow next experiment's detection
  rm -f "$HOME/llm_presentation_${N}.pptx" \
        "$STACK_ROOT/llm_presentation_${N}.pptx" \
        "$HOME/.hermes/workspace/llm_presentation_${N}.pptx" 2>/dev/null || true
fi

# --- Save result summary ---
python3 - <<PY
import json, os
rd = "$RUN_DIR"
out = {
    "experiment": $N,
    "honcho_variant": "$HONCHO_VARIANT",
    "inference_model": "$INFER_MODEL",
    "bonsai": {"parallel": $BONSAI_PARALLEL, "ctx": $BONSAI_CTX, "needed": bool($NEED_BONSAI)},
    "ollama": {"overhead": $OLLAMA_OVH, "max_loaded": $OLLAMA_MAXL},
    "wall_seconds": $WALL_SEC,
    "drain_seconds": $DRAIN_SEC,
    "total_seconds": $WALL_SEC + $DRAIN_SEC,
    "deriver_queue_rows": int("${DERIVER_ROWS:-0}" or 0),
    "observations_created": int("${OBS_COUNT:-0}" or 0),
    "exit_code": $EXIT_CODE,
    "pptx": json.load(open(os.path.join(rd, "pptx_result.json"))),
    "start_iso": open(os.path.join(rd, "start_iso.txt")).read().strip(),
}
json.dump(out, open(os.path.join(rd, "summary.json"), "w"), indent=2)
PY

log "=== END E${N} ==="
cat "$RUN_DIR/summary.json"
