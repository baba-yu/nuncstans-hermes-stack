#!/usr/bin/env bash
# S6 — cross-session recall. Same peer in new session must surface S4 facts.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S6 cross-session recall"
export CURRENT_SCENARIO="s6_cross_session"

WS="$UAT_RUN_ID-s4"; PEER="alice"; NEW_SES="recall"
FAILS=0

# Create the new session under the existing workspace, with the same peer
curl -sf -X POST "$HONCHO_URL/v3/workspaces/$WS/sessions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg id "$NEW_SES" --arg p "$PEER" '{id:$id, peers:{($p):{}}}')" >/dev/null

# Post a neutral line (doesn't mention any of the S4 facts)
post_message "$WS" "$PEER" "$NEW_SES" "Good morning, what would you like to do today?" > /dev/null
log "created new session $NEW_SES under $WS with a neutral message"

BONSAI_MARK=$(bonsai_mark)

# (a) Dialectic scoped to new session
RESP=$(curl -sf --max-time 300 -X POST "$HONCHO_URL/v3/workspaces/$WS/peers/$PEER/chat" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg sid "$NEW_SES" '{query:"Tell me what you remember about Alice hobbies and preferences.", reasoning_level:"minimal", session_id:$sid, stream:false}')")
echo "$RESP" > "$RESULTS_DIR/${CURRENT_SCENARIO}_chat_response.json"
CONTENT=$(echo "$RESP" | jq -r '.content // ""')
echo "$CONTENT" > "$RESULTS_DIR/${CURRENT_SCENARIO}_chat_content.txt"
log "cross-session chat length: ${#CONTENT} chars"

# (b) Representation scoped to new session
REP=$(curl -sf --max-time 20 -X POST "$HONCHO_URL/v3/workspaces/$WS/peers/$PEER/representation" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg sid "$NEW_SES" '{session_id:$sid, search_query:"hobbies", search_top_k:10}')" | jq -r '.representation // ""')
echo "$REP" > "$RESULTS_DIR/${CURRENT_SCENARIO}_representation.txt"

COMBINED=$(echo "$CONTENT $REP" | tr '[:upper:]' '[:lower:]')
HITS=0
for kw in matcha climb bonsai bike kyoto postgres miso engineer; do
  if echo "$COMBINED" | grep -q "$kw"; then
    HITS=$((HITS+1))
    log "  keyword match: $kw"
  fi
done
if [ "$HITS" -ge 1 ]; then
  pass "cross-session recall surfaced $HITS keyword(s)"
else
  fail "cross-session recall surfaced no keywords"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -ge 1 ]; then
  pass "Bonsai hit $BDIFF times"
else
  fail "Bonsai print_timing = $BDIFF (expected >=1)"; FAILS=$((FAILS+1))
fi

[ "$FAILS" -gt 0 ] && exit 1
log "S6: OK"
