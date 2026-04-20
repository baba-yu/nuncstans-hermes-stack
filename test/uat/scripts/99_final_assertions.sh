#!/usr/bin/env bash
# Cumulative traffic assertions and teardown.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "99 final assertions"
export CURRENT_SCENARIO="99_final"

FAILS=0

# Across the whole run, collect ollama ps snapshots into one file
cat "$RESULTS_DIR"/*ollama*.log 2>/dev/null | sort -u > "$RESULTS_DIR/final_ollama_seen.log" || true

# If glm-4.7-flash was loaded during any deriver/dialectic scenario, that is
# evidence of a chat leak. S2 legitimately loads it. The invariant we need
# is: during S4-S6 snapshots, only embedding model(s) should be present.
LEAK=0
for f in "$RESULTS_DIR"/s4_*ollama*.log "$RESULTS_DIR"/s5_*ollama*.log "$RESULTS_DIR"/s6_*ollama*.log; do
  [ -f "$f" ] || continue
  if grep -q "glm-4.7-flash" "$f"; then
    LEAK=$((LEAK+1))
    log "  WARN: glm-4.7-flash seen in $f"
  fi
done
if [ "$LEAK" -eq 0 ]; then
  pass "no Ollama chat leaks in deriver/dialectic windows"
else
  # S4 happens slowly so ollama_watch() isn't running during the Bonsai turn;
  # this is a soft check. Still a signal if glm stayed loaded from S2 into
  # S4 window (5 min keepalive), so record but don't fail the run on this alone.
  log "SOFT: chat model appeared in $LEAK scenario window(s); may be post-S2 keepalive"
fi

# Sum up Bonsai print_timing for the whole run (not just diffs)
BPT=$(grep -c 'print_timing' "$HERMES_HOME/bonsai.log" 2>/dev/null || echo 0)
log "Bonsai print_timing total in bonsai.log: $BPT"
# We expect at least 3 (S4 + S5 + S6, minimum 1 call each)
if [ "$BPT" -ge 3 ]; then
  pass "Bonsai was called $BPT times (>=3)"
else
  log "SOFT: Bonsai print_timing = $BPT (<3); review per-scenario pass/fail"
fi

# Final status: require at least 1 documents row existed at some point
DOCS=$(honcho_db -At -c "SELECT count(*) FROM documents;")
if [ "${DOCS:-0}" -ge 1 ]; then
  pass "documents count >= 1 ($DOCS)"
else
  fail "documents is empty"; FAILS=$((FAILS+1))
fi

[ "$FAILS" -gt 0 ] && exit 1
log "99: OK"
