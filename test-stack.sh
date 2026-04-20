#!/usr/bin/env bash
# End-to-end smoke test for the hermes-bonsai-combo stack.
#
# Passes when:
#   - Bonsai llama-server responds with the bonsai-8b alias
#   - Ollama responds with a 768-dim embedding for the aliased model name
#   - Honcho api /health is ok
#   - POSTing a message does NOT produce a 401 / AuthenticationError in api logs
#   - Within WAIT_SECS, alice's representation contains at least one of the
#     facts we planted (case-insensitive keyword match)
#
# Run from anywhere. Requires bash, curl, jq, docker compose, and the Honcho
# stack at HONCHO_DIR (default: ~/hermes-stack/honcho).

set -u
set -o pipefail

HONCHO_DIR="${HONCHO_DIR:-$HOME/hermes-stack/honcho}"
BONSAI_URL="${BONSAI_URL:-http://localhost:8080}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
HONCHO_URL="${HONCHO_URL:-http://localhost:8000}"
EMBED_MODEL_NAME="${EMBED_MODEL_NAME:-openai/text-embedding-3-small}"
WAIT_SECS="${WAIT_SECS:-300}"   # 5 minutes — Bonsai on CPU needs this

WORKSPACE="test-$(date +%s)"
SESSION="s1"
PEER="alice"
FACTS=("bonsai trees" "rock climbing" "matcha" "road bike")
KEYWORDS=("bonsai" "climb" "matcha" "bike")

pass=0
fail=0

ok()   { printf "  [PASS] %s\n" "$*"; pass=$((pass+1)); }
ko()   { printf "  [FAIL] %s\n" "$*"; fail=$((fail+1)); }
info() { printf "==> %s\n" "$*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 2; }
}
for t in curl jq docker; do need "$t"; done

info "1. Bonsai llama-server"
models=$(curl -s -m 5 "$BONSAI_URL/v1/models" || true)
if echo "$models" | jq -e '.models[0].name == "bonsai-8b"' >/dev/null 2>&1; then
  ok "bonsai-8b alias served at $BONSAI_URL"
else
  ko "bonsai-8b not served at $BONSAI_URL"
fi

info "2. Ollama embeddings (aliased model name)"
embed=$(curl -s -m 10 -X POST "$OLLAMA_URL/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer ollama' \
  -d "$(jq -n --arg m "$EMBED_MODEL_NAME" '{model:$m,input:"hi"}')" || true)
dims=$(echo "$embed" | jq -r '.data[0].embedding | length' 2>/dev/null || echo 0)
if [ "${dims:-0}" -gt 0 ]; then
  ok "$EMBED_MODEL_NAME returned $dims-dim vector"
else
  ko "no embedding from $EMBED_MODEL_NAME at $OLLAMA_URL (did you run \`ollama cp nomic-embed-text $EMBED_MODEL_NAME\`?)"
fi

info "3. Honcho api /health"
if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HONCHO_URL/health")" = "200" ]; then
  ok "/health returns 200"
else
  ko "/health not reachable at $HONCHO_URL"
fi

info "4. Seed workspace / peer / session / messages"
post() { curl -s -m 10 -X POST -H 'Content-Type: application/json' "$@"; }
post "$HONCHO_URL/v3/workspaces" -d "$(jq -n --arg id "$WORKSPACE" '{id:$id}')" >/dev/null
post "$HONCHO_URL/v3/workspaces/$WORKSPACE/peers" -d "$(jq -n --arg id "$PEER" '{id:$id}')" >/dev/null
post "$HONCHO_URL/v3/workspaces/$WORKSPACE/sessions" \
  -d "$(jq -n --arg sid "$SESSION" --arg pid "$PEER" '{id:$sid,peers:{($pid):{}}}')" >/dev/null
msgs_json=$(jq -n --argjson arr "$(printf '%s\n' "${FACTS[@]}" | jq -R . | jq -s .)" \
  --arg pid "$PEER" \
  '{messages: ($arr | map({peer_id:$pid, content:(.)}))}')
post "$HONCHO_URL/v3/workspaces/$WORKSPACE/sessions/$SESSION/messages" -d "$msgs_json" >/dev/null
ok "seeded workspace $WORKSPACE with ${#FACTS[@]} messages"

info "5. No 401 / AuthenticationError in api logs since seed"
sleep 5
errs=$(docker compose -f "$HONCHO_DIR/docker-compose.yml" -f "$HONCHO_DIR/docker-compose.override.yml" \
  logs --since 30s api 2>/dev/null | grep -cE '401|AuthenticationError' || true)
if [ "${errs:-0}" -eq 0 ]; then
  ok "api log has 0 auth errors since seed"
else
  ko "api log has $errs auth errors since seed — embedding path still broken"
fi

info "6. Representation picks up at least one planted keyword (wait up to ${WAIT_SECS}s)"
found=""
deadline=$(( $(date +%s) + WAIT_SECS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  rep=$(post "$HONCHO_URL/v3/workspaces/$WORKSPACE/peers/$PEER/representation" -d '{}' || true)
  body=$(echo "$rep" | jq -r '.. | strings? // empty' 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr '\n' ' ')
  for kw in "${KEYWORDS[@]}"; do
    if echo "$body" | grep -q -F -- "$kw"; then
      found="$kw"
      break 2
    fi
  done
  sleep 10
done
if [ -n "$found" ]; then
  ok "representation contains \"$found\""
else
  ko "representation did not surface any of: ${KEYWORDS[*]} within ${WAIT_SECS}s"
fi

echo
echo "---"
echo "pass: $pass  fail: $fail"
[ "$fail" -eq 0 ]
