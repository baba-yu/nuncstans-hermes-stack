#!/usr/bin/env bash
# UAT preflight: fails fast if the stack is not in the expected shape.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

step_header "PREFLIGHT"
export CURRENT_SCENARIO="00_preflight"
FAILS=0

# 2.2 services alive -------------------------------------------------
log "2.2 Bonsai /v1/models ..."
if curl -sf --max-time 5 "$BONSAI_URL/v1/models" | jq -e '.data[0].id == "bonsai-8b"' >/dev/null; then
  pass "Bonsai alive with alias bonsai-8b"
else
  fail "Bonsai /v1/models did not return bonsai-8b"; FAILS=$((FAILS+1))
fi

log "2.2 Ollama /api/tags ..."
if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | jq -e '.models | map(.name) | any(. == "glm-4.7-flash:latest")' >/dev/null; then
  pass "Ollama serves glm-4.7-flash"
else
  fail "Ollama missing glm-4.7-flash"; FAILS=$((FAILS+1))
fi

if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | jq -e '.models | map(.name) | any(. == "openai/text-embedding-3-small:latest")' >/dev/null; then
  pass "Ollama serves openai/text-embedding-3-small alias"
else
  fail "Ollama missing openai/text-embedding-3-small alias (run: ollama cp nomic-embed-text openai/text-embedding-3-small)"; FAILS=$((FAILS+1))
fi

log "2.2 Honcho /openapi.json ..."
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$HONCHO_URL/openapi.json")" = "200" ]; then
  pass "Honcho API responds 200"
else
  fail "Honcho API not 200"; FAILS=$((FAILS+1))
fi

# 2.3 container settings ----------------------------------------------
log "2.3 Honcho container settings ..."
CFG=$(honcho_exec_api python3 -c "
from src.config import settings
import json
print(json.dumps({
  'EMBEDDING_PROVIDER': settings.LLM.EMBEDDING_PROVIDER,
  'OPENAI_COMPATIBLE_BASE_URL': settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
  'VLLM_BASE_URL': settings.LLM.VLLM_BASE_URL,
  'DERIVER_PROVIDER': settings.DERIVER.PROVIDER,
  'DERIVER_MODEL': settings.DERIVER.MODEL,
  'DERIVER_MAX_OUTPUT_TOKENS': settings.DERIVER.MAX_OUTPUT_TOKENS,
  'DERIVER_MAX_INPUT_TOKENS': settings.DERIVER.MAX_INPUT_TOKENS,
  'DERIVER_REPR_BATCH': settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS,
  'BACKUP_PROVIDER': settings.DERIVER.BACKUP_PROVIDER,
  'APP_MAX_EMBEDDING_TOKENS': settings.MAX_EMBEDDING_TOKENS,
  'VS_DIMENSIONS': settings.VECTOR_STORE.DIMENSIONS,
}))
" 2>/dev/null | tail -1)
echo "$CFG" | jq . | tee "$RESULTS_DIR/config_inside_container.json" >/dev/null
want_json() {
  local key=$1 expected=$2
  local got=$(echo "$CFG" | jq -r ".$key")
  if [ "$got" = "$expected" ]; then
    pass "$key = $got"
  else
    fail "$key = $got (expected $expected)"; FAILS=$((FAILS+1))
  fi
}
want_json "EMBEDDING_PROVIDER" "openrouter"
want_json "OPENAI_COMPATIBLE_BASE_URL" "http://host.docker.internal:11434/v1"
want_json "VLLM_BASE_URL" "http://host.docker.internal:8080/v1"
want_json "DERIVER_PROVIDER" "vllm"
want_json "DERIVER_MODEL" "bonsai-8b"
want_json "DERIVER_REPR_BATCH" "1024"
want_json "APP_MAX_EMBEDDING_TOKENS" "2048"
want_json "VS_DIMENSIONS" "768"
want_json "BACKUP_PROVIDER" "null"

# 2.4 backup lines ----------------------------------------------------
log "2.4 No BACKUP_* lines in config.toml ..."
if grep -E "^BACKUP_(PROVIDER|MODEL)" "$HONCHO_DIR/config.toml" >/dev/null; then
  fail "BACKUP_* lines still present in config.toml"; FAILS=$((FAILS+1))
else
  pass "no BACKUP_* lines"
fi

# 2.5 pgvector dim ----------------------------------------------------
log "2.5 pgvector message_embeddings dim ..."
DIM=$(honcho_db -At -c "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'message_embeddings'::regclass AND attname = 'embedding';")
if [ "$DIM" = "768" ]; then
  pass "message_embeddings.embedding = Vector($DIM)"
else
  fail "pgvector dim = $DIM (expected 768)"; FAILS=$((FAILS+1))
fi

# 2.6 hermes cli optional ---------------------------------------------
if command -v hermes >/dev/null 2>&1; then
  pass "hermes CLI installed: $(hermes --version 2>&1 | head -1)"
else
  log "SKIP: hermes CLI not installed"
fi

echo ""
if [ "$FAILS" -gt 0 ]; then
  log "PREFLIGHT: $FAILS failures"
  exit 1
fi
log "PREFLIGHT: OK"
