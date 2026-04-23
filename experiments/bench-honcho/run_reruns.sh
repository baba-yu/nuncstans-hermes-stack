#!/usr/bin/env bash
# Run E1-2 .. E5-2 with timeout=3600s. Retry once on failure (pptx_ok=false).
set -u
BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
TIMEOUT=3600
TAG="2"

pptx_ok_for() {
  local rd="$1"
  [ -f "$rd/summary.json" ] || return 1
  python3 -c "import json,sys;d=json.load(open('$rd/summary.json'));sys.exit(0 if d['pptx'].get('ok') else 1)"
}

for N in 1 2 3 4 5; do
  RUN_DIR_A="$BENCH_ROOT/runs/E${N}-${TAG}"
  RUN_DIR_B="$BENCH_ROOT/runs/E${N}-${TAG}-retry"
  echo "=== RERUN E${N}-${TAG} (attempt 1) ===" >> "$BENCH_ROOT/logs/reruns_master.log"
  date +%Y-%m-%dT%H:%M:%S >> "$BENCH_ROOT/logs/reruns_master.log"
  bash "$BENCH_ROOT/run_experiment.sh" "$N" "$TAG" "$TIMEOUT" >> "$BENCH_ROOT/logs/E${N}-${TAG}.log" 2>&1 || true
  if ! pptx_ok_for "$RUN_DIR_A"; then
    echo "=== RERUN E${N}-${TAG} FAILED, retrying as E${N}-${TAG}-retry ===" >> "$BENCH_ROOT/logs/reruns_master.log"
    bash "$BENCH_ROOT/run_experiment.sh" "$N" "${TAG}-retry" "$TIMEOUT" >> "$BENCH_ROOT/logs/E${N}-${TAG}-retry.log" 2>&1 || true
  fi
  echo "=== E${N}-${TAG} summary ===" >> "$BENCH_ROOT/logs/reruns_master.log"
  cat "$RUN_DIR_A/summary.json" >> "$BENCH_ROOT/logs/reruns_master.log" 2>/dev/null || echo "no summary" >> "$BENCH_ROOT/logs/reruns_master.log"
  if [ -d "$RUN_DIR_B" ]; then
    echo "=== E${N}-${TAG}-retry summary ===" >> "$BENCH_ROOT/logs/reruns_master.log"
    cat "$RUN_DIR_B/summary.json" >> "$BENCH_ROOT/logs/reruns_master.log" 2>/dev/null || echo "no summary" >> "$BENCH_ROOT/logs/reruns_master.log"
  fi
  echo "" >> "$BENCH_ROOT/logs/reruns_master.log"
done
echo "ALL RERUNS DONE" >> "$BENCH_ROOT/logs/reruns_master.log"
date +%Y-%m-%dT%H:%M:%S >> "$BENCH_ROOT/logs/reruns_master.log"
