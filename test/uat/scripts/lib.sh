#!/usr/bin/env bash
# Shared UAT helpers. Source this from every scenario script.

set -eu
set -o pipefail

export HERMES_HOME="${HERMES_HOME:-$HOME/hermes-stack}"
export HONCHO_DIR="${HONCHO_DIR:-$HERMES_HOME/honcho}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export BONSAI_URL="${BONSAI_URL:-http://localhost:8080}"
export HONCHO_URL="${HONCHO_URL:-http://localhost:8000}"
export UAT_RUN_ID="${UAT_RUN_ID:-uat-$(date +%Y%m%d-%H%M%S)}"
export RESULTS_DIR="$HERMES_HOME/test/uat/results/$UAT_RUN_ID"

mkdir -p "$RESULTS_DIR"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$RESULTS_DIR/run.log"
}

step_header() {
  local name="$1"
  printf '\n========================================\n%s (UAT_RUN_ID=%s)\n========================================\n' "$name" "$UAT_RUN_ID" | tee -a "$RESULTS_DIR/run.log"
}

fail() {
  log "FAIL: $*"
  echo "FAIL: $*" >> "$RESULTS_DIR/failures.log"
  return 1
}

pass() {
  log "PASS: $*"
  echo "PASS: $*" >> "$RESULTS_DIR/passes.log"
}

# Traffic assertion helpers
# ------------------------------------------------------------------
# Bonsai:   count `print_timing` lines added to bonsai.log
# Ollama:   poll /api/ps to see which models are loaded

bonsai_mark() {
  wc -l < "$HERMES_HOME/bonsai.log"
}

bonsai_diff_count() {
  local mark=$1
  tail -n +"$((mark+1))" "$HERMES_HOME/bonsai.log" | grep -cE 'print_timing' || true
}

# Poll /api/ps once and return loaded model names as newline-separated list
ollama_loaded() {
  curl -s --max-time 3 "$OLLAMA_URL/api/ps" 2>/dev/null \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin)
    for m in d.get('models',[]): print(m.get('name',''))
except Exception: pass" || true
}

# Watch /api/ps repeatedly for DURATION sec and accumulate any model names seen
ollama_watch() {
  local duration=$1
  local out="$RESULTS_DIR/${CURRENT_SCENARIO:-unknown}_ollama_ps.log"
  local end=$(( $(date +%s) + duration ))
  : > "$out"
  while [ "$(date +%s)" -lt "$end" ]; do
    ollama_loaded >> "$out"
    sleep 2
  done
  sort -u "$out"
}

# Returns 0 if glm-4.7-flash (chat model) was ever loaded during the window
ollama_chat_was_loaded() {
  local out="$RESULTS_DIR/${CURRENT_SCENARIO:-unknown}_ollama_ps.log"
  [ -f "$out" ] && grep -q "glm-4.7-flash" "$out"
}

# Returns 0 if only embedding model(s) were loaded during the window
ollama_embedding_only() {
  local out="$RESULTS_DIR/${CURRENT_SCENARIO:-unknown}_ollama_ps.log"
  [ -f "$out" ] || return 0
  if grep -q "glm-4.7-flash\|qwen3\|llama3\|gpt-oss" "$out"; then
    return 1
  fi
  return 0
}

# GPU util sampled once (MiB, %)
gpu_sample() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1
}

honcho_db() {
  docker compose -f "$HONCHO_DIR/docker-compose.yml" exec -T database \
    psql -U honcho -d honcho "$@"
}

honcho_exec_api() {
  docker compose -f "$HONCHO_DIR/docker-compose.yml" exec -T api "$@"
}

# Create a standard ws/peer/session triplet
setup_ws() {
  local ws=$1 peer=$2 ses=$3
  curl -sf -X POST "$HONCHO_URL/v3/workspaces" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg id "$ws" '{id:$id, metadata:{}}')" > /dev/null
  curl -sf -X POST "$HONCHO_URL/v3/workspaces/$ws/peers" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg id "$peer" '{id:$id, metadata:{}}')" > /dev/null
  curl -sf -X POST "$HONCHO_URL/v3/workspaces/$ws/sessions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg id "$ses" --arg p "$peer" '{id:$id, peers:{($p):{}}}')" > /dev/null
}

# Post a user message; echo the token_count
post_message() {
  local ws=$1 peer=$2 ses=$3 content=$4
  curl -sf -X POST "$HONCHO_URL/v3/workspaces/$ws/sessions/$ses/messages" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$peer" --arg t "$content" '{messages:[{peer_id:$p, content:$t}]}')" \
    | jq -r '.[0].token_count'
}

# Wait up to max_sec for the queue to drain (pending + in_progress == 0)
wait_queue_drain() {
  local ws=$1 max_sec=$2
  local start=$(date +%s)
  while :; do
    local s
    s=$(curl -s --max-time 3 "$HONCHO_URL/v3/workspaces/$ws/queue/status" | jq -r '.pending_work_units + .in_progress_work_units')
    if [ "$s" = "0" ]; then return 0; fi
    if [ $(( $(date +%s) - start )) -gt "$max_sec" ]; then
      log "queue drain TIMEOUT ($max_sec s); last status: $s pending+in_progress"
      return 1
    fi
    sleep 5
  done
}

# Delete workspace + cascade (API first, SQL fallback)
cleanup_workspace() {
  local ws=$1
  curl -s -X DELETE "$HONCHO_URL/v3/workspaces/$ws" -o /dev/null || true
  honcho_db <<SQL >/dev/null 2>&1 || true
DELETE FROM documents WHERE workspace_name='$ws';
DELETE FROM message_embeddings WHERE workspace_name='$ws';
DELETE FROM session_peers WHERE workspace_name='$ws';
DELETE FROM active_queue_sessions
  WHERE work_unit_key LIKE 'representation:$ws:%'
     OR work_unit_key LIKE 'summary:$ws:%'
     OR work_unit_key LIKE 'dream:$ws:%';
DELETE FROM queue WHERE workspace_name='$ws';
DELETE FROM sessions WHERE workspace_name='$ws';
DELETE FROM peers WHERE workspace_name='$ws';
DELETE FROM workspaces WHERE name='$ws';
SQL
}
