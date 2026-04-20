#!/usr/bin/env bash
# S5 — dialectic (peer.chat) -> Bonsai. Requires S4 to have left observations.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S5 dialectic -> Bonsai"
export CURRENT_SCENARIO="s5_dialectic"

WS="$UAT_RUN_ID-s4"
PEER="alice"
FAILS=0

BONSAI_MARK=$(bonsai_mark)

# Use reasoning_level=minimal to keep the Bonsai turn budget small
RESP=$(curl -sf --max-time 300 -X POST "$HONCHO_URL/v3/workspaces/$WS/peers/$PEER/chat" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What hobbies does Alice have?","reasoning_level":"minimal","stream":false}')
echo "$RESP" > "$RESULTS_DIR/${CURRENT_SCENARIO}_response.json"
CONTENT=$(echo "$RESP" | jq -r '.content // ""')
echo "$CONTENT" > "$RESULTS_DIR/${CURRENT_SCENARIO}_content.txt"
log "dialectic response length: ${#CONTENT} chars"
log "content preview: $(echo "$CONTENT" | head -c 200)"

if [ -n "$CONTENT" ]; then
  pass "dialectic returned content"
else
  fail "dialectic returned empty content"; FAILS=$((FAILS+1))
fi

CLOW=$(echo "$CONTENT" | tr '[:upper:]' '[:lower:]')
HITS=0
for kw in matcha climb bonsai bike kyoto postgres miso engineer; do
  if echo "$CLOW" | grep -q "$kw"; then
    HITS=$((HITS+1))
    log "  keyword match: $kw"
  fi
done
if [ "$HITS" -ge 1 ]; then
  pass "dialectic response contains $HITS seed keyword(s)"
else
  fail "dialectic response contains no seed keywords"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -ge 1 ]; then
  pass "Bonsai hit $BDIFF times"
else
  fail "Bonsai print_timing = $BDIFF (expected >=1)"; FAILS=$((FAILS+1))
fi

[ "$FAILS" -gt 0 ] && exit 1
log "S5: OK"
