#!/usr/bin/env bash
# S1 — Pass-through smoke: short message -> embedding only; deriver NOT triggered.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S1 pass-through smoke"
export CURRENT_SCENARIO="s1_smoke"

WS="$UAT_RUN_ID-s1"
PEER="alice"
SES="smoke"
FAILS=0

setup_ws "$WS" "$PEER" "$SES"
log "created ws=$WS peer=$PEER ses=$SES"

BONSAI_MARK=$(bonsai_mark)
EMB_BEFORE=$(honcho_db -At -c "SELECT count(*) FROM message_embeddings WHERE workspace_name='$WS';")

# Ollama /api/ps baseline
ollama_loaded > "$RESULTS_DIR/${CURRENT_SCENARIO}_ps_before.log"

TOKENS=$(post_message "$WS" "$PEER" "$SES" "hello from UAT $UAT_RUN_ID")
log "posted message, token_count=$TOKENS"

# Watch ollama for 10 s (embedding should happen within this window)
ollama_watch 10 > "$RESULTS_DIR/${CURRENT_SCENARIO}_ollama_seen.log"

EMB_AFTER=$(honcho_db -At -c "SELECT count(*) FROM message_embeddings WHERE workspace_name='$WS';")
log "message_embeddings $WS: $EMB_BEFORE -> $EMB_AFTER"
if [ "$EMB_AFTER" -gt "$EMB_BEFORE" ]; then
  pass "embedding row written"
else
  fail "embedding row NOT written"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -eq 0 ]; then
  pass "Bonsai not hit ($BDIFF print_timing)"
else
  fail "Bonsai hit $BDIFF times (expected 0)"; FAILS=$((FAILS+1))
fi

if ollama_chat_was_loaded; then
  fail "glm-4.7-flash loaded during S1 (unexpected chat traffic)"; FAILS=$((FAILS+1))
else
  pass "Ollama chat model not loaded"
fi

cleanup_workspace "$WS"

[ "$FAILS" -gt 0 ] && exit 1
log "S1: OK"
