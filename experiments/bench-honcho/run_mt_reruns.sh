#!/usr/bin/env bash
# Sequential E1..E5 multi-turn reruns with 1 retry on failure.
# Failure definition: any turn timed out OR final pptx is not ok.
set -u
BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
TAG="2"

passed() {
  local rd="$1"
  [ -f "$rd/summary.json" ] || return 1
  python3 -c "
import json, sys
d = json.load(open('$rd/summary.json'))
ok = d.get('completed_all_turns') and d.get('pptx',{}).get('ok')
sys.exit(0 if ok else 1)
"
}

for N in 1 2 3 4 5; do
  RUN_A="$BENCH_ROOT/runs/E${N}-${TAG}"
  RUN_B="$BENCH_ROOT/runs/E${N}-${TAG}-retry"
  echo "=== RERUN E${N}-${TAG} attempt 1 @ $(date +%H:%M:%S) ===" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
  bash "$BENCH_ROOT/run_experiment_mt.sh" "$N" "$TAG" >> "$BENCH_ROOT/logs/E${N}-${TAG}_mt.log" 2>&1
  if ! passed "$RUN_A"; then
    echo "=== E${N}-${TAG} FAILED — retrying as E${N}-${TAG}-retry @ $(date +%H:%M:%S) ===" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
    bash "$BENCH_ROOT/run_experiment_mt.sh" "$N" "${TAG}-retry" >> "$BENCH_ROOT/logs/E${N}-${TAG}-retry_mt.log" 2>&1
  fi
  # Append summaries
  echo "--- E${N}-${TAG} summary ---" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
  cat "$RUN_A/summary.json" >> "$BENCH_ROOT/logs/mt_reruns_master.log" 2>/dev/null || echo "no summary" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
  if [ -d "$RUN_B" ]; then
    echo "--- E${N}-${TAG}-retry summary ---" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
    cat "$RUN_B/summary.json" >> "$BENCH_ROOT/logs/mt_reruns_master.log" 2>/dev/null || echo "no summary" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
  fi
done
echo "ALL MT RERUNS DONE @ $(date +%H:%M:%S)" >> "$BENCH_ROOT/logs/mt_reruns_master.log"
