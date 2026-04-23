#!/usr/bin/env bash
# Shared helpers for the Honcho/Hermes benchmark.
set -euo pipefail

BENCH_ROOT="/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho"
STACK_ROOT="/home/baba-y/nuncstans-hermes-stack"
HONCHO_DIR="$STACK_ROOT/honcho"
BONSAI_DIR="$STACK_ROOT/bonsai-llama.cpp"
BONSAI_GGUF="$STACK_ROOT/models/Bonsai-8B.gguf"
HERMES_HOME="$HOME/.hermes"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$BENCH_ROOT/logs/orchestrator.log" >&2; }

# Wait until a URL returns HTTP 200
wait_for_url() {
  local url="$1" tries="${2:-60}"
  for ((i=0;i<tries;i++)); do
    if curl -sf -o /dev/null "$url"; then return 0; fi
    sleep 1
  done
  log "timeout waiting for $url"; return 1
}

stop_bonsai() {
  if pgrep -f 'llama-server -m .*Bonsai-8B' >/dev/null 2>&1; then
    log "killing llama-server"
    pkill -9 -f 'llama-server -m .*Bonsai-8B' || true
    sleep 2
  fi
}

start_bonsai() {
  local parallel="${1:-1}" ctx="${2:-16384}"
  stop_bonsai
  # For large contexts (>=32768), use quantized KV cache to stay in VRAM budget.
  local cache_flags=""
  if [ "$ctx" -ge 32768 ]; then
    cache_flags="-ctk q4_0 -ctv q4_0"
  fi
  log "starting llama-server --parallel $parallel -c $ctx $cache_flags"
  (cd "$BONSAI_DIR" && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
    nohup ./build/bin/llama-server \
      -m "$BONSAI_GGUF" \
      --host 0.0.0.0 --port 8080 -ngl 99 -c "$ctx" --parallel "$parallel" \
      $cache_flags \
      --alias bonsai-8b > "$BENCH_ROOT/logs/bonsai-current.log" 2>&1 &
    disown)
  wait_for_url http://localhost:8080/v1/models 120
  log "bonsai ready"
}

set_ollama_env() {
  # No-op: sudo is unavailable in this run. We accept whatever the existing
  # systemd drop-in configured and document the effective env in the report.
  # Values passed here are recorded for post-hoc interpretation only.
  local overhead="$1" max_loaded="$2"
  log "ollama env: using systemd drop-in as-is (would-set OVERHEAD=$overhead MAX_LOADED=$max_loaded)"
  # Snapshot effective env for the record
  systemctl show ollama -p Environment --no-pager 2>&1 | head -1 | tee -a "$BENCH_ROOT/logs/orchestrator.log" >&2
}

ollama_unload_all() {
  local loaded
  loaded=$(curl -s http://localhost:11434/api/ps | python3 -c 'import json,sys;d=json.load(sys.stdin);print(" ".join(m["name"] for m in d.get("models",[])))')
  for m in $loaded; do
    log "unloading ollama model $m"
    curl -s -X POST http://localhost:11434/api/generate -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
  done
  sleep 2
}

ollama_warmup() {
  local model="$1"
  log "warming up $model (32 tokens)"
  curl -s http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32,\"stream\":false}" \
    >/dev/null
}

bonsai_warmup() {
  curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-8b","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stream":false}' \
    >/dev/null
}

write_honcho_config() {
  local variant="$1"  # bonsai|ollama-9b|ollama-35b
  cp "$BENCH_ROOT/configs/honcho-$variant.toml" "$HONCHO_DIR/config.toml"
  log "honcho config.toml -> $variant"
}

honcho_restart_workers() {
  (cd "$HONCHO_DIR" && docker compose up -d api deriver >/dev/null 2>&1)
  # hot config reload: recreate workers to re-read config.toml
  (cd "$HONCHO_DIR" && docker compose restart deriver api >/dev/null 2>&1)
  wait_for_url http://localhost:8000/health 60
  log "honcho api+deriver restarted"
}

honcho_cleanup_bench() {
  # Remove any queue entries scoped to bench workspaces from prior runs.
  # Production 'hermes' workspace queue rows are preserved.
  (cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -c \
    "DELETE FROM queue WHERE workspace_name LIKE 'bench-e%';" 2>&1 | tail -1) || true
  # Wait briefly for any in-flight production deriver run to conclude
  for i in $(seq 1 10); do
    pending=$(cd "$HONCHO_DIR" && docker compose exec -T database psql -U honcho -d honcho -tA -c \
      "SELECT count(*) FROM queue WHERE processed=false;" 2>/dev/null || echo "0")
    [ "$pending" -lt 5 ] 2>/dev/null && break
    sleep 2
  done
  (cd "$HONCHO_DIR" && docker compose exec -T redis redis-cli FLUSHDB >/dev/null 2>&1) || true
  log "honcho cleanup done (queue pending=${pending:-?})"
}

update_hermes_workspace() {
  local ws="$1"
  python3 -c "
import json
p='$HERMES_HOME/honcho.json'
d=json.load(open(p))
d['hosts']['hermes']['workspace']='$ws'
json.dump(d,open(p,'w'),indent=2)
"
  log "hermes honcho.json workspace -> $ws"
}

clean_hermes_session_state() {
  # Clear ephemeral session caches; preserve config.yaml / honcho.json
  rm -rf "$HERMES_HOME/checkpoints"/* 2>/dev/null || true
  rm -rf "$HERMES_HOME/audio_cache"/* 2>/dev/null || true
  rm -f "$HERMES_HOME/.hermes_history" 2>/dev/null || true
  log "hermes session state cleaned"
}

vram_recorder_start() {
  local out="$1"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "$out" 2>/dev/null &
  echo $! > "$BENCH_ROOT/logs/vram_pid"
}
vram_recorder_stop() {
  if [ -f "$BENCH_ROOT/logs/vram_pid" ]; then
    kill -9 "$(cat $BENCH_ROOT/logs/vram_pid)" 2>/dev/null || true
    rm -f "$BENCH_ROOT/logs/vram_pid"
  fi
}

snapshot_ollama_ps() {
  curl -s http://localhost:11434/api/ps
}
