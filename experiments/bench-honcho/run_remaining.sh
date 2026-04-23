#!/usr/bin/env bash
# Run E2..E6 sequentially, tolerating timeouts.
set -u
BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
for N in 2 3 4 5 6; do
  bash "$BENCH_ROOT/run_experiment.sh" "$N" >> "$BENCH_ROOT/logs/E${N}-run.log" 2>&1 || true
  echo "=== E$N complete, summary ===" >> "$BENCH_ROOT/logs/master.log"
  cat "$BENCH_ROOT/runs/E${N}/summary.json" >> "$BENCH_ROOT/logs/master.log" 2>/dev/null || echo "no summary" >> "$BENCH_ROOT/logs/master.log"
  echo "" >> "$BENCH_ROOT/logs/master.log"
done
echo "ALL DONE" >> "$BENCH_ROOT/logs/master.log"
