#!/usr/bin/env bash
# S4 — deriver -> Bonsai observation extraction (UAT anchor 1).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S4 deriver -> Bonsai observation"
export CURRENT_SCENARIO="s4_deriver"

WS="$UAT_RUN_ID-s4"; PEER="alice"; SES="derive"
FAILS=0
setup_ws "$WS" "$PEER" "$SES"

BONSAI_MARK=$(bonsai_mark)
TS_START=$(date +%s)

# Inject a dense, fact-heavy prompt that crosses the 1024-token batch gate
SEED='My name is Alice. I love matcha lattes every morning. I go rock climbing at the Gravity Gym every Sunday. I ride a red road bike to the gym. I collect bonsai trees, especially junipers. I live in Kyoto. I work as a backend engineer. I prefer PostgreSQL over MySQL. My cat is named Miso.'
PROMPT=$(python3 -c "import sys; s=sys.argv[1]; print((s+' ')*20)" "$SEED")
TOKENS=$(post_message "$WS" "$PEER" "$SES" "$PROMPT")
log "posted token_count=$TOKENS"

# Expect deriver to pick up within 60s
for i in $(seq 1 20); do
  sleep 3
  STATUS=$(curl -s --max-time 3 "$HONCHO_URL/v3/workspaces/$WS/queue/status")
  if echo "$STATUS" | jq -e '(.pending_work_units + .in_progress_work_units) > 0' >/dev/null; then
    log "queue claimed at i=$i"
    break
  fi
done

# Wait for drain; budget 6 min
if wait_queue_drain "$WS" 360; then
  pass "queue drained"
else
  fail "queue did not drain within 6 min"; FAILS=$((FAILS+1))
fi

DUR=$(( $(date +%s) - TS_START ))
log "S4 total elapsed: ${DUR}s"

# Documents table has workspace_name directly — no join needed.
OBS=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS';")
log "documents obs count for $WS: $OBS"
if [ "${OBS:-0}" -ge 1 ]; then
  pass "observations saved: $OBS"
else
  fail "no observations saved for $WS"; FAILS=$((FAILS+1))
fi

# Representation endpoint must contain a seed keyword
REP=$(curl -sf --max-time 10 -X POST "$HONCHO_URL/v3/workspaces/$WS/peers/$PEER/representation" \
  -H 'Content-Type: application/json' -d '{}' | jq -r '.representation // ""')
echo "$REP" > "$RESULTS_DIR/${CURRENT_SCENARIO}_representation.txt"
REPLOW=$(echo "$REP" | tr '[:upper:]' '[:lower:]')
HITS=0
for kw in matcha climb bonsai bike kyoto postgres miso alice engineer; do
  if echo "$REPLOW" | grep -q "$kw"; then
    HITS=$((HITS+1))
    log "  keyword match: $kw"
  fi
done
if [ "$HITS" -ge 1 ]; then
  pass "representation contains $HITS seed keyword(s)"
else
  fail "representation contains no seed keywords"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -ge 1 ]; then
  pass "Bonsai hit $BDIFF times"
else
  fail "Bonsai print_timing = $BDIFF (expected >=1)"; FAILS=$((FAILS+1))
fi

[ "$FAILS" -gt 0 ] && exit 1
log "S4: OK"
