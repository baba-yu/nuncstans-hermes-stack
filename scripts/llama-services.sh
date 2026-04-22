#!/bin/bash
# llama-services.sh — manage the two llama-server processes that serve
# hermes-stack's local LLM endpoints.
#
# Subcommands:
#   start   launch chat + embedding servers (idempotent; skips what's up)
#   stop    kill both servers
#   status  report PIDs, ports, healthy/not-healthy
#   restart stop then start
#   logs    tail the log for a specific service
#
# The chat server runs qwen3.6:35b (unsloth UD-Q4_K_XL) with the L6
# config validated by experiments/bench-moe-offload (see its report.md):
# expert-tensor CPU offload + nothink, giving ~5.5 s per 4k-prompt/800-token
# Hermes turn at ~36 tok/s decode and ~7.5 GiB VRAM.
#
# The embedding server runs nomic-embed-text (GGUF blob originally pulled
# by ollama), aliased as `openai/text-embedding-3-small` so Honcho's
# hardcoded embedding client finds it by name.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_BUILD="$ROOT/bonsai-llama.cpp/build/bin"
EMBED_BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
HF_CHAT_SPEC="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"
CHAT_ALIAS="qwen3.6-test"
LOG_DIR="$HOME/.local/state/hermes-stack"
mkdir -p "$LOG_DIR"

CHAT_LOG="$LOG_DIR/chat-server.log"
EMBED_LOG="$LOG_DIR/embed-server.log"
CHAT_PID_FILE="$LOG_DIR/chat-server.pid"
EMBED_PID_FILE="$LOG_DIR/embed-server.pid"

info()  { echo "[llama-services] $*"; }
die()   { echo "[llama-services] ERROR: $*" >&2; exit 1; }

# reads pid from file; echoes nothing if file absent or pid dead
get_live_pid() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] || return 0
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

port_open() {
    local port="$1"
    ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"
}

wait_health() {
    local url="$1" label="$2" timeout="${3:-600}"
    local deadline=$(( $(date +%s) + timeout ))
    while :; do
        curl -sfS --max-time 2 "$url" >/dev/null 2>&1 && return 0
        (( $(date +%s) >= deadline )) && die "$label did not become healthy in ${timeout}s"
        sleep 2
    done
}

start_embed() {
    if [[ -n "$(get_live_pid "$EMBED_PID_FILE")" ]]; then
        info "embedding server already running (pid $(cat "$EMBED_PID_FILE"))"
        return 0
    fi
    if port_open 8081; then
        die "port 8081 is bound but no tracked pid — investigate before starting"
    fi
    info "starting embedding server on :8081"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    nohup "$LLAMA_BUILD/llama-server" \
        -m "$EMBED_BLOB" \
        --host 0.0.0.0 --port 8081 \
        --embeddings \
        --alias openai/text-embedding-3-small \
        -ngl 99 \
        > "$EMBED_LOG" 2>&1 &
    echo $! > "$EMBED_PID_FILE"
    disown
    wait_health "http://127.0.0.1:8081/health" "embedding server" 60
    info "  embedding server ready (pid $(cat "$EMBED_PID_FILE"))"
}

start_chat() {
    if [[ -n "$(get_live_pid "$CHAT_PID_FILE")" ]]; then
        info "chat server already running (pid $(cat "$CHAT_PID_FILE"))"
        return 0
    fi
    if port_open 8080; then
        die "port 8080 is bound but no tracked pid — investigate before starting"
    fi
    info "starting chat server on :8080 (L6 config: expert offload + nothink)"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    nohup "$LLAMA_BUILD/llama-server" \
        -hf "$HF_CHAT_SPEC" \
        --host 0.0.0.0 --port 8080 \
        -c 65536 \
        -fa on \
        -ctk q8_0 -ctv q8_0 \
        --jinja \
        -ngl 99 \
        -ot "ffn_(up|down|gate)_exps=CPU" \
        --reasoning off \
        --alias "$CHAT_ALIAS" \
        > "$CHAT_LOG" 2>&1 &
    echo $! > "$CHAT_PID_FILE"
    disown
    # first start may need download; later starts are ~60 s cold load
    wait_health "http://127.0.0.1:8080/health" "chat server" 900
    info "  chat server ready (pid $(cat "$CHAT_PID_FILE"))"
}

stop_one() {
    local label="$1" pidfile="$2"
    local pid
    pid="$(get_live_pid "$pidfile")"
    if [[ -z "$pid" ]]; then
        info "$label not running"
        rm -f "$pidfile"
        return 0
    fi
    info "stopping $label (pid $pid)"
    kill "$pid"
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        info "  $label did not exit cleanly, SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
}

cmd_start() {
    # embedding first: honcho deriver may fire during hermes turn warmups,
    # and the chat server start triggers embedding-model ensure elsewhere.
    start_embed
    start_chat
}

cmd_stop() {
    stop_one "chat server" "$CHAT_PID_FILE"
    stop_one "embedding server" "$EMBED_PID_FILE"
}

cmd_status() {
    for spec in "chat:$CHAT_PID_FILE:8080:$CHAT_LOG" "embed:$EMBED_PID_FILE:8081:$EMBED_LOG"; do
        IFS=: read -r label pidfile port log <<<"$spec"
        pid="$(get_live_pid "$pidfile" || true)"
        if [[ -n "$pid" ]]; then
            health="unknown"
            if curl -sfS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
                health="healthy"
            else
                health="unhealthy"
            fi
            printf "  %-5s  pid %s  port %s  %s  log %s\n" "$label" "$pid" "$port" "$health" "$log"
        else
            printf "  %-5s  stopped  (no tracked pid)\n" "$label"
        fi
    done
}

cmd_restart() { cmd_stop; cmd_start; }

cmd_logs() {
    local which="${1:-chat}"
    case "$which" in
        chat)  tail -f "$CHAT_LOG" ;;
        embed) tail -f "$EMBED_LOG" ;;
        *) die "unknown log target: $which (chat|embed)" ;;
    esac
}

sub="${1:-status}"
shift || true
case "$sub" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    logs)    cmd_logs "$@" ;;
    *) echo "usage: $0 {start|stop|status|restart|logs [chat|embed]}" >&2; exit 2 ;;
esac
