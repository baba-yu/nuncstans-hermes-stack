#!/usr/bin/env bash
# run_experiment_mt.sh N TAG
# Multi-turn version: 5 turns in a single Hermes session.
# - Turn 1 (self-intro, 600s)
# - Turns 2-4 (info, 1200s each)
# - Turn 5 (pptx, 3600s)
# If ANY turn times out, mark experiment failed and exit (retry handled by caller).
# Setup/teardown wall time is excluded from wall_seconds (only hermes chat invocations are summed).
set -uo pipefail
N="$1"
TAG="$2"
BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
source "$BENCH_ROOT/lib.sh"

# Per-experiment config (same matrix as single-prompt version)
case "$N" in
  1) HONCHO_VARIANT=bonsai;      INFER_MODEL="qwen3.5:9b";   NEED_BONSAI=1; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  2) HONCHO_VARIANT=bonsai;      INFER_MODEL="qwen3.6:35b";  NEED_BONSAI=1; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  3) HONCHO_VARIANT=ollama-9b;   INFER_MODEL="qwen3.6:35b";  NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  4) HONCHO_VARIANT=ollama-9b;   INFER_MODEL="qwen3.5:9b";   NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  5) HONCHO_VARIANT=ollama-35b;  INFER_MODEL="qwen3.6:35b";  NEED_BONSAI=0; BONSAI_PARALLEL=1; BONSAI_CTX=16384 ;;
  *) echo "bad experiment number: $N"; exit 2 ;;
esac

RUN_DIR="$BENCH_ROOT/runs/E${N}-${TAG}"
mkdir -p "$RUN_DIR"
log "=== BEGIN E${N}-${TAG}: honcho=$HONCHO_VARIANT inference=$INFER_MODEL (multi-turn) ==="

# --- Teardown prior stack state ---
stop_bonsai
ollama_unload_all || true

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
update_hermes_workspace "bench-e${N}-${TAG}"
clean_hermes_session_state

# --- Warmup ---
if [ "$INFER_MODEL" = "bonsai-8b" ]; then
  bonsai_warmup
else
  ollama_warmup "$INFER_MODEL"
fi
if [ "$HONCHO_VARIANT" = "bonsai" ]; then
  bonsai_warmup
elif [ "$HONCHO_VARIANT" = "ollama-9b" ]; then
  ollama_warmup "qwen3.5:9b"
elif [ "$HONCHO_VARIANT" = "ollama-35b" ]; then
  ollama_warmup "qwen3.6:35b"
fi

# Pre-run snapshots
snapshot_ollama_ps > "$RUN_DIR/ollama_ps_pre.json"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits > "$RUN_DIR/vram_pre.csv"

# Remove any stale pptx with same N
rm -f "$STACK_ROOT/llm_presentation_${N}.pptx" \
      "$HOME/llm_presentation_${N}.pptx" \
      "$HOME/.hermes/workspace/llm_presentation_${N}.pptx" \
      "/tmp/llm_presentation_${N}.pptx" 2>/dev/null || true

# Start VRAM recorder for the entire run
vram_recorder_start "$RUN_DIR/vram_trace.csv"

RUN_START_ISO=$(date -Iseconds)
echo "$RUN_START_ISO" > "$RUN_DIR/start_iso.txt"

# --- Read turn definitions via Python ---
parse_turn() {
  local idx="$1" field="$2"
  python3 -c "
import json, sys
turns = json.load(open('$BENCH_ROOT/turns.json'))
t = turns[$idx]
if '$field' == 'text':
    if 'text_template' in t:
        print(t['text_template'].replace('{N}', '$N'))
    else:
        print(t['text'])
else:
    print(t.get('$field',''))
"
}

run_turn() {
  local idx="$1" session_flag="$2"
  local timeout_s=$(parse_turn "$idx" "timeout_s")
  local label=$(parse_turn "$idx" "label")
  local text=$(parse_turn "$idx" "text")
  local turn_id=$(parse_turn "$idx" "id")
  local out="$RUN_DIR/t${turn_id}_${label}.stdout"
  local err="$RUN_DIR/t${turn_id}_${label}.stderr"

  log "  T${turn_id} (${label}) timeout=${timeout_s}s"
  local start_ns=$(date +%s%N)
  set +e
  (cd "$STACK_ROOT" && timeout "$timeout_s" hermes chat -Q --yolo --max-turns 60 --source tool $session_flag -q "$text") \
      > "$out" 2> "$err"
  local exit_code=$?
  set -e
  local end_ns=$(date +%s%N)
  local wall=$(python3 -c "print(($end_ns-$start_ns)/1e9)")

  # Capture session_id (printed on stdout or stderr)
  local sid=$(grep -oE 'session_id: [A-Za-z0-9_]+' "$out" "$err" 2>/dev/null | head -1 | awk -F': ' '{print $NF}')

  # Emit JSON summary line for this turn
  python3 -c "
import json
print(json.dumps({
    'id': $turn_id,
    'label': '$label',
    'timeout_s': $timeout_s,
    'wall_seconds': $wall,
    'exit_code': $exit_code,
    'session_id': '$sid',
    'stdout_bytes': __import__('os').path.getsize('$out'),
    'stderr_bytes': __import__('os').path.getsize('$err'),
}))
" >> "$RUN_DIR/turns.jsonl"

  log "    exit=$exit_code wall=${wall}s sid=$sid"
  echo "$exit_code:$wall:$sid"
}

# Clear turns log
: > "$RUN_DIR/turns.jsonl"

# --- Turn 1: new session, no --resume ---
RES1=$(run_turn 0 "")  # index 0 = turn id 1
EXIT_T1=${RES1%%:*}
WALL_T1=$(echo "$RES1" | awk -F: '{print $2}')
SID=$(echo "$RES1" | awk -F: '{print $3}')
FAIL=0
if [ "$EXIT_T1" != "0" ]; then FAIL=1; FAIL_AT="T1"; fi
if [ -z "$SID" ]; then
  log "  WARN: no session_id from T1; resume flag will be empty"
fi

# --- Turns 2-4: resume session ---
if [ "$FAIL" = "0" ]; then
  for IDX in 1 2 3; do  # array indices 1,2,3 = turn ids 2,3,4
    RES=$(run_turn "$IDX" "--resume $SID")
    EXIT_T=${RES%%:*}
    if [ "$EXIT_T" != "0" ]; then FAIL=1; FAIL_AT="T$((IDX+1))"; break; fi
  done
fi

# --- Turn 5: resume session, create pptx ---
if [ "$FAIL" = "0" ]; then
  RES5=$(run_turn 4 "--resume $SID")
  EXIT_T5=${RES5%%:*}
  if [ "$EXIT_T5" != "0" ]; then FAIL=1; FAIL_AT="T5"; fi
fi

# --- Post-run ---
vram_recorder_stop
snapshot_ollama_ps > "$RUN_DIR/ollama_ps_post.json"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits > "$RUN_DIR/vram_post.csv"

# Drain Honcho queue
DRAIN_START_NS=$(date +%s%N)
pending=0
for i in $(seq 1 180); do
  pending=$(cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -tA -c \
    "SELECT count(*) FROM queue WHERE workspace_name='bench-e${N}-${TAG}' AND processed=false;" 2>/dev/null || echo "0")
  [ "${pending:-0}" -eq 0 ] && break
  sleep 1
done
DRAIN_END_NS=$(date +%s%N)
DRAIN_SEC=$(python3 -c "print(($DRAIN_END_NS-$DRAIN_START_NS)/1e9)")

# Copy bonsai log slice + deriver logs
cp "$BENCH_ROOT/logs/bonsai-current.log" "$RUN_DIR/bonsai.log" 2>/dev/null || true
(cd "$HONCHO_DIR" && docker compose logs --since="$RUN_START_ISO" deriver > "$RUN_DIR/deriver.log" 2>&1) || true

# --- Locate and validate pptx ---
PPTX_PATH=""
for cand in "$STACK_ROOT/llm_presentation_${N}.pptx" \
            "$HOME/llm_presentation_${N}.pptx" \
            "$HOME/.hermes/workspace/llm_presentation_${N}.pptx" \
            "/tmp/llm_presentation_${N}.pptx"; do
  if [ -f "$cand" ]; then PPTX_PATH="$cand"; break; fi
done
if [ -z "$PPTX_PATH" ]; then PPTX_PATH="$STACK_ROOT/llm_presentation_${N}.pptx"; fi
PPTX_RESULT=$(python3 "$BENCH_ROOT/validate_pptx.py" "$PPTX_PATH" 2>/dev/null || echo '{"ok":false,"error":"validator failed"}')
echo "$PPTX_RESULT" > "$RUN_DIR/pptx_result.json"

if [ -f "$PPTX_PATH" ]; then
  cp "$PPTX_PATH" "$RUN_DIR/" || true
  rm -f "$HOME/llm_presentation_${N}.pptx" \
        "$STACK_ROOT/llm_presentation_${N}.pptx" \
        "$HOME/.hermes/workspace/llm_presentation_${N}.pptx" 2>/dev/null || true
fi

# --- Write summary ---
python3 - <<PY
import json, os
rd = "$RUN_DIR"
turns = []
with open(os.path.join(rd, "turns.jsonl")) as f:
    for line in f:
        line = line.strip()
        if line: turns.append(json.loads(line))

total_wall = sum(t["wall_seconds"] for t in turns)
any_timeout = any(t["exit_code"] == 124 for t in turns)
any_fail = any(t["exit_code"] != 0 for t in turns)
completed_all = len(turns) == 5 and not any_fail

pptx = json.load(open(os.path.join(rd, "pptx_result.json")))

out = {
    "experiment": $N,
    "tag": "$TAG",
    "honcho_variant": "$HONCHO_VARIANT",
    "inference_model": "$INFER_MODEL",
    "bonsai": {"parallel": $BONSAI_PARALLEL, "ctx": $BONSAI_CTX, "needed": bool($NEED_BONSAI)},
    "turns": turns,
    "wall_seconds": total_wall,
    "drain_seconds": $DRAIN_SEC,
    "total_seconds": total_wall + $DRAIN_SEC,
    "completed_all_turns": completed_all,
    "any_turn_timeout": any_timeout,
    "fail_at": "${FAIL_AT:-}" or None,
    "session_id": "$SID",
    "pptx": pptx,
    "start_iso": open(os.path.join(rd, "start_iso.txt")).read().strip(),
}
json.dump(out, open(os.path.join(rd, "summary.json"), "w"), indent=2, default=str)
PY

log "=== END E${N}-${TAG} (fail=$FAIL, fail_at=${FAIL_AT:-none}) ==="
cat "$RUN_DIR/summary.json"
# Exit 0 regardless so the caller can decide on retry based on summary content
exit 0
