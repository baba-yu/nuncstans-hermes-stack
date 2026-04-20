#!/usr/bin/env bash
# S8 — full gatekeeper → deriver → observation path with real token volume.
# Uses a long fact-dense prompt so the deriver's REPRESENTATION_BATCH_MAX_TOKENS
# threshold actually fires. Verifies:
#   * gatekeeper marks the message ready
#   * deriver processes it
#   * documents row appears with the expected content
# Then sends a clearly non-literal message and asserts NO document is created.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S8 gatekeeper → deriver → observation"
export CURRENT_SCENARIO="s8_gk_deriver"

WS="$UAT_RUN_ID-s8"
PEER="alice"
SES="e2e"
FAILS=0

setup_ws "$WS" "$PEER" "$SES"
log "created ws=$WS"

# Long literal seed (>= 200 tokens)
SEED_A="I'm Alice. I love matcha lattes every morning. I go rock climbing at the Gravity Gym every Sunday. I ride a red road bike to the gym. I collect bonsai trees, especially junipers. I live in Kyoto. I work as a backend engineer. I prefer PostgreSQL over MySQL. My cat is named Miso. My emergency contact is my sister."
LONG_A=$(python3 -c "import sys; s=sys.argv[1]; print((s+' ')*3)" "$SEED_A")

# Long hypothetical (should be demoted)
LONG_B="If I were Napoleon Bonaparte, I would have led my troops through a Russian winter with much better preparation. Imagine what would have happened if Napoleon had taken ten thousand more horses, warmer coats, and a longer supply line. Suppose he had asked the Tsar for passage through St Petersburg. Hypothetically, Europe would have looked completely different today."

log "=== phase 1: literal A message (expect ready + observation) ==="
TOKENS_A=$(post_message "$WS" "$PEER" "$SES" "$LONG_A")
log "A tokens=$TOKENS_A"

log "waiting up to 60s for gatekeeper verdict..."
for i in $(seq 1 20); do
  STATUS_A=$(honcho_db -At -c "SELECT status FROM queue WHERE workspace_name='$WS' AND task_type='representation' ORDER BY id LIMIT 1;")
  if [ "$STATUS_A" = "ready" ] || [ "$STATUS_A" = "demoted" ]; then break; fi
  sleep 3
done
log "A status=$STATUS_A"

if [ "$STATUS_A" = "ready" ]; then
  pass "A gatekeeper verdict: ready"
else
  fail "A gatekeeper verdict: $STATUS_A (expected ready)"; FAILS=$((FAILS+1))
fi

log "waiting up to 5 min for deriver to drain the ready batch..."
if wait_queue_drain "$WS" 300; then
  pass "queue drained"
else
  log "queue didn't drain within 5 min"
fi

sleep 3
DOCS_A=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS';")
log "documents in $WS after A: $DOCS_A"
if [ "${DOCS_A:-0}" -ge 1 ]; then
  pass "observations created: $DOCS_A"
else
  fail "no observations created for A"; FAILS=$((FAILS+1))
fi

REP_A=$(curl -sf --max-time 15 -X POST "$HONCHO_URL/v3/workspaces/$WS/peers/$PEER/representation" \
  -H 'Content-Type: application/json' -d '{}' | jq -r '.representation // ""' | tr '[:upper:]' '[:lower:]')
HITS=0
for kw in alice matcha climb bonsai kyoto engineer postgres miso; do
  echo "$REP_A" | grep -q "$kw" && HITS=$((HITS+1))
done
if [ "$HITS" -ge 3 ]; then
  pass "representation contains $HITS seed keywords"
else
  fail "representation has only $HITS seed keywords"; FAILS=$((FAILS+1))
fi

log "=== phase 2: hypothetical B message (expect demoted, no new observation) ==="
DOCS_BEFORE_B=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS';")
TOKENS_B=$(post_message "$WS" "$PEER" "$SES" "$LONG_B")
log "B tokens=$TOKENS_B"

log "waiting up to 60s for gatekeeper verdict..."
for i in $(seq 1 20); do
  STATUS_B=$(honcho_db -At -c "SELECT status FROM queue WHERE workspace_name='$WS' AND task_type='representation' ORDER BY id DESC LIMIT 1;")
  if [ "$STATUS_B" = "ready" ] || [ "$STATUS_B" = "demoted" ]; then break; fi
  sleep 3
done
log "B status=$STATUS_B"

if [ "$STATUS_B" = "demoted" ]; then
  pass "B gatekeeper verdict: demoted"
else
  fail "B gatekeeper verdict: $STATUS_B (expected demoted)"; FAILS=$((FAILS+1))
fi

log "waiting 30s to give deriver a chance (it shouldn't process demoted row)..."
sleep 30
DOCS_AFTER_B=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS';")
log "documents in $WS after B: $DOCS_AFTER_B (was $DOCS_BEFORE_B)"
if [ "$DOCS_AFTER_B" = "$DOCS_BEFORE_B" ]; then
  pass "demoted message produced no new observations"
else
  fail "demoted message unexpectedly produced new observations ($DOCS_BEFORE_B -> $DOCS_AFTER_B)"; FAILS=$((FAILS+1))
fi

NAPOLEON_DOCS=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND content ILIKE '%napoleon%';")
if [ "${NAPOLEON_DOCS:-0}" -eq 0 ]; then
  pass "no document mentions Napoleon"
else
  fail "unexpected Napoleon documents ($NAPOLEON_DOCS)"; FAILS=$((FAILS+1))
fi

honcho_db -c "SELECT substring(content for 60) AS msg, status FROM queue q JOIN messages m ON q.message_id=m.id WHERE m.workspace_name='$WS' ORDER BY q.id;" >> "$RESULTS_DIR/run.log"
honcho_db -c "SELECT substring(content for 80) FROM documents WHERE workspace_name='$WS' ORDER BY created_at LIMIT 5;" >> "$RESULTS_DIR/run.log"

cleanup_workspace "$WS"

[ "$FAILS" -gt 0 ] && exit 1
log "S8: OK"
