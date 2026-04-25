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
LLAMA_BUILD="$ROOT/llama.cpp/build/bin"

# Defaults. scripts/llama-services.conf overrides any of these when
# present; missing keys fall back to these values. scripts/switch-endpoints.py
# writes to the .conf file, never to this script.
CHAT_HF_SPEC="unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"
CHAT_ALIAS="qwen3.6-test"
CHAT_CTX=131072
CHAT_NGL=99
CHAT_IS_MOE=1
CHAT_REASONING_OFF=1
CHAT_PARALLEL=2
EMBED_BLOB="/usr/share/ollama/.ollama/models/blobs/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
EMBED_ALIAS="openai/text-embedding-3-small"
EMBED_NGL=99
# Gatekeeper classifier endpoint (chat-engine follower by default).
# switch-endpoints.py keeps these in sync with the Honcho chat endpoint
# so the classifier uses the same engine as the rest of the stack.
GK_LLM_URL="http://localhost:8080"
GK_LLM_MODEL="$CHAT_ALIAS"

LLAMA_SERVICES_CONF="$ROOT/scripts/llama-services.conf"
if [[ -f "$LLAMA_SERVICES_CONF" ]]; then
    # shellcheck source=/dev/null
    source "$LLAMA_SERVICES_CONF"
fi

# State dir holds pid files, log files, and endpoint-snapshots. Override
# with HERMES_STATE_DIR to run a second instance on the same host. The
# default assumes single-instance-per-user (see docs/specs/scripts/
# llama-services.md for the multi-instance / containerization guidance).
LOG_DIR="${HERMES_STATE_DIR:-$HOME/.local/state/nuncstans-hermes-stack}"
mkdir -p "$LOG_DIR"

CHAT_LOG="$LOG_DIR/chat-server.log"
EMBED_LOG="$LOG_DIR/embed-server.log"
GK_LOG="$LOG_DIR/gatekeeper.log"
CHAT_PID_FILE="$LOG_DIR/chat-server.pid"
EMBED_PID_FILE="$LOG_DIR/embed-server.pid"
GK_PID_FILE="$LOG_DIR/gatekeeper.pid"

info()  { echo "[llama-services] $*"; }
die()   { echo "[llama-services] ERROR: $*" >&2; exit 1; }

# reads pid from file; echoes nothing if file absent or pid dead
get_live_pid() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] || return 0
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && echo "$pid"
    # The && chain above returns non-zero when the pid is dead. Under
    # `set -euo pipefail` that propagates through `pid="$(get_live_pid …)"`
    # and silently kills the script before stop_one / start_chat can
    # handle the empty-pid case. Force a clean rc=0 here so the caller
    # always reaches its own logic.
    return 0
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
        --alias "$EMBED_ALIAS" \
        -ngl "$EMBED_NGL" \
        > "$EMBED_LOG" 2>&1 &
    echo $! > "$EMBED_PID_FILE"
    disown
    wait_health "http://127.0.0.1:8081/health" "embedding server" 60
    info "  embedding server ready (pid $(cat "$EMBED_PID_FILE"))"
}

start_gatekeeper() {
    # Gatekeeper daemon classifies pending representation queue rows and
    # promotes them to ready (or demotes non-literal / low-importance ones).
    # Uses the chat server on :8080 as the classifier LLM (GK_LLM_URL /
    # GK_LLM_MODEL). See scripts/gatekeeper_daemon.py for the decision rules.
    if [[ -n "$(get_live_pid "$GK_PID_FILE")" ]]; then
        info "gatekeeper already running (pid $(cat "$GK_PID_FILE"))"
        return 0
    fi
    info "starting gatekeeper daemon (classifier=$GK_LLM_URL model=$GK_LLM_MODEL)"
    GK_LLM_URL="$GK_LLM_URL" \
    GK_LLM_MODEL="$GK_LLM_MODEL" \
    HERMES_HOME="$ROOT" \
    HONCHO_DIR="$ROOT/honcho" \
    nohup python3 "$ROOT/scripts/gatekeeper_daemon.py" \
        > "$GK_LOG" 2>&1 &
    echo $! > "$GK_PID_FILE"
    disown
    sleep 2
    if ! kill -0 "$(cat "$GK_PID_FILE")" 2>/dev/null; then
        die "gatekeeper died immediately; see $GK_LOG"
    fi
    info "  gatekeeper running (pid $(cat "$GK_PID_FILE"))"
}

start_chat() {
    if [[ -n "$(get_live_pid "$CHAT_PID_FILE")" ]]; then
        info "chat server already running (pid $(cat "$CHAT_PID_FILE"))"
        return 0
    fi
    if port_open 8080; then
        die "port 8080 is bound but no tracked pid — investigate before starting"
    fi
    info "starting chat server on :8080 (alias=$CHAT_ALIAS, ctx=$CHAT_CTX, moe=$CHAT_IS_MOE, reasoning_off=$CHAT_REASONING_OFF)"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

    # Build optional flags from the conf. -ot (MoE expert offload) and
    # --reasoning off are model-specific, so we gate them rather than
    # hardcoding. If CHAT_HF_SPEC starts with '/' assume it's a local
    # path and use -m; otherwise it's an HF repo spec, use -hf.
    local -a chat_extra=()
    (( CHAT_IS_MOE )) && chat_extra+=(-ot "ffn_(up|down|gate)_exps=CPU")
    (( CHAT_REASONING_OFF )) && chat_extra+=(--reasoning off)

    local -a model_flag
    if [[ "$CHAT_HF_SPEC" == /* ]]; then
        model_flag=(-m "$CHAT_HF_SPEC")
    else
        model_flag=(-hf "$CHAT_HF_SPEC")
    fi

    nohup "$LLAMA_BUILD/llama-server" \
        "${model_flag[@]}" \
        --host 0.0.0.0 --port 8080 \
        -c "$CHAT_CTX" \
        -fa on \
        -ctk q8_0 -ctv q8_0 \
        --jinja \
        -ngl "$CHAT_NGL" \
        "${chat_extra[@]}" \
        --parallel "$CHAT_PARALLEL" \
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
    local target="${1:-all}"
    case "$target" in
        all)
            # embedding first: honcho deriver may fire during hermes turn warmups,
            # and the chat server start triggers embedding-model ensure elsewhere.
            start_embed
            start_chat
            # gatekeeper last: it calls the chat server, so chat must be healthy
            # before the daemon's first poll fires.
            start_gatekeeper
            ;;
        chat)   start_chat ;;
        embed)  start_embed ;;
        gk|gatekeeper) start_gatekeeper ;;
        *) die "unknown start target: $target (all|chat|embed|gk)" ;;
    esac
}

cmd_stop() {
    local target="${1:-all}"
    case "$target" in
        all)
            # reverse order: stop the daemon first so it doesn't see a missing
            # chat server mid-classification, then the servers themselves
            stop_one "gatekeeper" "$GK_PID_FILE"
            stop_one "chat server" "$CHAT_PID_FILE"
            stop_one "embedding server" "$EMBED_PID_FILE"
            ;;
        chat)   stop_one "chat server" "$CHAT_PID_FILE" ;;
        embed)  stop_one "embedding server" "$EMBED_PID_FILE" ;;
        gk|gatekeeper) stop_one "gatekeeper" "$GK_PID_FILE" ;;
        *) die "unknown stop target: $target (all|chat|embed|gk)" ;;
    esac
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
    # gatekeeper has no HTTP health endpoint; report process status only
    gk_pid="$(get_live_pid "$GK_PID_FILE" || true)"
    if [[ -n "$gk_pid" ]]; then
        printf "  %-5s  pid %s  daemon    running   log %s\n" "gk" "$gk_pid" "$GK_LOG"
    else
        printf "  %-5s  stopped  (no tracked pid)\n" "gk"
    fi
}

cmd_restart() { cmd_stop "$@"; cmd_start "$@"; }

cmd_logs() {
    local which="${1:-chat}"
    case "$which" in
        chat)  tail -f "$CHAT_LOG" ;;
        embed) tail -f "$EMBED_LOG" ;;
        gk)    tail -f "$GK_LOG" ;;
        *) die "unknown log target: $which (chat|embed|gk)" ;;
    esac
}

sub="${1:-status}"
shift || true
case "$sub" in
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    status)  cmd_status ;;
    restart) cmd_restart "$@" ;;
    logs)    cmd_logs "$@" ;;
    *)
        echo "usage: $0 {start|stop|restart} [all|chat|embed|gk]" >&2
        echo "       $0 status" >&2
        echo "       $0 logs {chat|embed|gk}" >&2
        exit 2
        ;;
esac
