#!/usr/bin/env bash
# S2 — direct Ollama chat. Must hit glm-4.7-flash, must NOT hit Bonsai.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S2 Ollama direct chat"
export CURRENT_SCENARIO="s2_ollama_chat"
FAILS=0

BONSAI_MARK=$(bonsai_mark)

# Kick a chat call. glm-4.7-flash is a thinking model — reasoning tokens
# consume max_tokens before content, so allow 128.
RESP=$(curl -sf --max-time 120 -X POST "$OLLAMA_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"Reply with a single word: ping"}],"max_tokens":128}')
echo "$RESP" > "$RESULTS_DIR/${CURRENT_SCENARIO}_response.json"
CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // ""')
REASONING=$(echo "$RESP" | jq -r '.choices[0].message.reasoning // ""')
log "response content: $CONTENT"
log "response reasoning len: ${#REASONING}"

# Pass if either content or reasoning is non-empty — either proves generation
if [ -n "$CONTENT" ] || [ -n "$REASONING" ]; then
  pass "Ollama generated output (content=${#CONTENT}, reasoning=${#REASONING} chars)"
else
  fail "Ollama returned empty response"; FAILS=$((FAILS+1))
fi

# /api/ps right after should show glm-4.7-flash loaded
sleep 1
PS_JSON=$(curl -s --max-time 3 "$OLLAMA_URL/api/ps")
echo "$PS_JSON" > "$RESULTS_DIR/${CURRENT_SCENARIO}_ps_after.json"
if echo "$PS_JSON" | jq -e '.models | map(.name) | any(contains("glm-4.7-flash"))' >/dev/null; then
  pass "glm-4.7-flash loaded"
else
  fail "glm-4.7-flash not loaded after chat"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -eq 0 ]; then
  pass "Bonsai not hit ($BDIFF print_timing)"
else
  fail "Bonsai hit $BDIFF times"; FAILS=$((FAILS+1))
fi

[ "$FAILS" -gt 0 ] && exit 1
log "S2: OK"
