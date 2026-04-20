#!/usr/bin/env bash
# S3 — 1500 tokens message -> embedding still works (regression for MAX_EMBEDDING_TOKENS fix).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S3 embedding regression (middle-sized)"
export CURRENT_SCENARIO="s3_embed_regression"

WS="$UAT_RUN_ID-s3"; PEER="alice"; SES="s3"
FAILS=0
setup_ws "$WS" "$PEER" "$SES"

BONSAI_MARK=$(bonsai_mark)

TXT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 160)")
TOKENS=$(post_message "$WS" "$PEER" "$SES" "$TXT")
log "posted token_count=$TOKENS"

# Wait 20s for embedding to finish
sleep 20

# Scrape api log for embedding errors over that window
ERRORS=$(docker compose -f "$HONCHO_DIR/docker-compose.yml" logs --since 1m api 2>&1 \
  | grep -cE '401|AuthenticationError|input length exceeds|expected .* dimensions' || true)
log "api error log hits: $ERRORS"

EMB=$(honcho_db -At -c "SELECT count(*) FROM message_embeddings WHERE workspace_name='$WS';")
log "message_embeddings in $WS: $EMB"

if [ "$ERRORS" -eq 0 ]; then
  pass "no embedding errors"
else
  fail "embedding errors: $ERRORS"; FAILS=$((FAILS+1))
fi

if [ "$EMB" -ge 1 ]; then
  pass "embedding row written ($EMB)"
else
  fail "no embedding rows"; FAILS=$((FAILS+1))
fi

BDIFF=$(bonsai_diff_count "$BONSAI_MARK")
if [ "$BDIFF" -eq 0 ]; then
  pass "Bonsai not hit (expected embedding-only)"
else
  fail "Bonsai hit $BDIFF times"; FAILS=$((FAILS+1))
fi

cleanup_workspace "$WS"
[ "$FAILS" -gt 0 ] && exit 1
log "S3: OK"
