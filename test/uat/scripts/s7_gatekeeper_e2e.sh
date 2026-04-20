#!/usr/bin/env bash
# S7 — end-to-end test of the gatekeeper + queue status pipeline.
#
# Posts four synthetic messages to a dedicated workspace and asserts the
# verdict the gatekeeper assigns to each. Then waits for deriver to process
# the 'ready' rows and checks the documents table.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S7 gatekeeper end-to-end"
export CURRENT_SCENARIO="s7_gatekeeper"

WS="$UAT_RUN_ID-s7"
PEER="yuki"
SES="gk"
FAILS=0

setup_ws "$WS" "$PEER" "$SES"
log "created ws=$WS peer=$PEER ses=$SES"

post_and_record() {
  local label=$1 content=$2
  local resp mid tokens
  resp=$(curl -sf -X POST "$HONCHO_URL/v3/workspaces/$WS/sessions/$SES/messages" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$PEER" --arg c "$content" '{messages:[{peer_id:$p, content:$c}]}')")
  mid=$(echo "$resp" | jq -r '.[0].id')
  tokens=$(echo "$resp" | jq -r '.[0].token_count')
  log "posted $label mid=$mid tokens=$tokens"
  echo "$mid"
}

MID_A=$(post_and_record "A (literal)"       "My name is Yuki and I'm a backend engineer in California.")
MID_B=$(post_and_record "B (hypothetical)"  "If I were Napoleon, I would have conquered Russia in winter.")
MID_C=$(post_and_record "C (ambiguous)"     "I might be allergic to shellfish, but not sure yet.")
sleep 8
MID_D=$(post_and_record "D (correction)"    "Actually, I misspoke — my name is not Yuki, it's Daiki.")

wait_for_verdict() {
  local mid=$1 max_sec=${2:-60}
  local start end has_verdict status
  start=$(date +%s)
  while :; do
    has_verdict=$(honcho_db -At -c "SELECT gate_verdict IS NOT NULL FROM queue WHERE message_id = (SELECT id FROM messages WHERE public_id = '$mid');")
    status=$(honcho_db -At -c "SELECT status FROM queue WHERE message_id = (SELECT id FROM messages WHERE public_id = '$mid');")
    if [ "$has_verdict" = "t" ] || [ "$status" = "ready" ] || [ "$status" = "demoted" ]; then
      echo "$status"
      return 0
    fi
    end=$(date +%s)
    if [ $((end - start)) -gt "$max_sec" ]; then
      echo "$status"
      return 1
    fi
    sleep 3
  done
}

assert_eq() {
  local name=$1 got=$2 want=$3
  if [ "$got" = "$want" ]; then
    pass "$name: got '$got' (expected)"
  else
    fail "$name: got '$got', expected '$want'"
    FAILS=$((FAILS+1))
  fi
}

sleep 5
log "waiting for gatekeeper to classify each message (up to 90s each)..."
STATUS_A=$(wait_for_verdict "$MID_A" 90)
STATUS_B=$(wait_for_verdict "$MID_B" 90)
STATUS_C=$(wait_for_verdict "$MID_C" 90)
STATUS_D=$(wait_for_verdict "$MID_D" 90)

assert_eq "A verdict"        "$STATUS_A" "ready"
assert_eq "B verdict"        "$STATUS_B" "demoted"

if [ "$STATUS_C" = "pending" ] || [ "$STATUS_C" = "ready" ]; then
  pass "C verdict ('$STATUS_C' acceptable — ambiguous)"
else
  fail "C verdict: got '$STATUS_C', expected pending or ready"
  FAILS=$((FAILS+1))
fi

assert_eq "D verdict"        "$STATUS_D" "ready"

CORR_D=$(honcho_db -At -c "SELECT gate_verdict->>'correction_of_prior' FROM queue WHERE message_id = (SELECT id FROM messages WHERE public_id = '$MID_D');")
assert_eq "D correction_of_prior" "$CORR_D" "true"

IMPORTANCE_A=$(honcho_db -At -c "SELECT (gate_verdict->>'importance')::int FROM queue WHERE message_id = (SELECT id FROM messages WHERE public_id = '$MID_A');")
if [ -n "$IMPORTANCE_A" ] && [ "$IMPORTANCE_A" -ge 7 ]; then
  pass "A importance=$IMPORTANCE_A (>=7)"
else
  fail "A importance=$IMPORTANCE_A (expected >= 7)"
  FAILS=$((FAILS+1))
fi

log "waiting for deriver to consume 'ready' rows (up to 5 min)..."
if wait_queue_drain "$WS" 360; then
  pass "queue drained"
else
  log "queue didn't fully drain within 6 min (pending rows may still be sitting)"
fi

DOCS_A=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND content ILIKE '%yuki%';")
DOCS_B=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND content ILIKE '%napoleon%';")

log "documents mentioning 'yuki' in $WS: $DOCS_A"
if [ "${DOCS_B:-0}" -eq 0 ]; then
  pass "no 'napoleon' documents (B demoted as expected)"
else
  fail "unexpected Napoleon document found (count=$DOCS_B)"
  FAILS=$((FAILS+1))
fi

log "verdict summary: A=$STATUS_A  B=$STATUS_B  C=$STATUS_C  D=$STATUS_D"

honcho_db -c "SELECT substring(content for 60) AS msg, status, (gate_verdict->>'A_score')::float AS a, (gate_verdict->>'B_score')::float AS b, (gate_verdict->>'importance')::int AS imp, gate_verdict->>'correction_of_prior' AS corr FROM queue q JOIN messages m ON q.message_id=m.id WHERE m.workspace_name='$WS' ORDER BY q.id;" >> "$RESULTS_DIR/run.log"

cleanup_workspace "$WS"

[ "$FAILS" -gt 0 ] && exit 1
log "S7: OK"
