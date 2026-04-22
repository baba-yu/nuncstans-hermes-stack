#!/bin/bash
# prep.sh — one-time preparation for the bench-moe-offload experiment.
#
# 1. Sanity-check the environment (binaries, blobs, ports, VRAM).
# 2. Stop Bonsai and ollama.
# 3. Launch the embedding llama-server on :8081.
# 4. Rewrite Honcho config.toml to point at :8080 (chat) and :8081 (embedding).
# 5. Rewrite ~/.hermes/config.yaml to point at :8080.
# 6. Recreate honcho-api / honcho-deriver with the new config.
# 7. Capture a real deriver LLM request body into prompts/honcho_deriver.json
#    by proxying it transparently during a triggered Hermes-style message post.
# 8. Verify the stack is healthy at the end.
#
# Idempotent where feasible. Stops on first failure with a clear message.
# Never runs sudo — if something requires sudo (service file edits), exit with
# instructions for the human operator.

set -euo pipefail

# ---------- paths ----------
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT/experiments/bench-moe-offload"
LLAMA_BUILD="$ROOT/bonsai-llama.cpp/build/bin"
BLOB_DIR="/usr/share/ollama/.ollama/models/blobs"
CHAT_BLOB="$BLOB_DIR/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
EMBED_BLOB="$BLOB_DIR/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
HONCHO_CONFIG="$ROOT/honcho/config.toml"
HERMES_CONFIG="$HOME/.hermes/config.yaml"
HERMES_CTX_CACHE="$HOME/.hermes/context_length_cache.yaml"

LOG_DIR="/tmp/bench-moe-offload"
mkdir -p "$LOG_DIR"

# ---------- helpers ----------
die() { echo "[prep] ERROR: $*" >&2; exit 1; }
info() { echo "[prep] $*"; }
wait_healthy() {
    local url="$1" label="$2" deadline=$(( $(date +%s) + 60 ))
    while :; do
        if curl -sfS --max-time 2 "$url" >/dev/null 2>&1; then
            info "$label ready"
            return 0
        fi
        (( $(date +%s) >= deadline )) && die "$label did not become healthy in 60s"
        sleep 1
    done
}

# ---------- 1. sanity checks ----------
info "1/8 sanity-checking environment"
[[ -x "$LLAMA_BUILD/llama-server" ]] || die "llama-server binary missing at $LLAMA_BUILD"
[[ -r "$CHAT_BLOB" ]]  || die "chat blob not readable: $CHAT_BLOB"
[[ -r "$EMBED_BLOB" ]] || die "embedding blob not readable: $EMBED_BLOB"
[[ -f "$HONCHO_CONFIG" ]] || die "honcho config.toml missing: $HONCHO_CONFIG"
[[ -f "$HERMES_CONFIG" ]] || die "hermes config.yaml missing: $HERMES_CONFIG"
command -v docker >/dev/null || die "docker not found in PATH"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found in PATH"
command -v python3 >/dev/null || die "python3 not found in PATH"

# ports 8080, 8081, 8090 should be free by the end of the script;
# check them now too because leftovers cause confusing failures later
for port in 8080 8081 8090; do
    if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
        info "port $port currently in use — will attempt to clean up"
    fi
done

# ---------- 2. stop bonsai, stop ollama ----------
info "2/8 stopping Bonsai llama-server"
pkill -f 'build/bin/llama-server.*--port 8080' 2>/dev/null || true
# give it a beat to release the port
for _ in 1 2 3 4 5; do
    ss -tln 2>/dev/null | awk '{print $4}' | grep -q ':8080$' || break
    sleep 1
done
ss -tln 2>/dev/null | awk '{print $4}' | grep -q ':8080$' && die "port 8080 still bound after kill"

info "2/8 stopping ollama (systemd)"
if systemctl is-active --quiet ollama 2>/dev/null; then
    info "NOTE: ollama is running under systemd; stopping it requires sudo."
    info "Run: sudo systemctl stop ollama"
    info "Then re-run prep.sh."
    die "ollama stop requires manual sudo action"
fi

# ---------- 3. launch embedding llama-server on :8081 ----------
info "3/8 launching embedding llama-server on :8081"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
nohup "$LLAMA_BUILD/llama-server" \
    -m "$EMBED_BLOB" \
    --host 127.0.0.1 --port 8081 \
    --embeddings \
    --alias openai/text-embedding-3-small \
    -ngl 99 \
    > "$LOG_DIR/embedding_server.log" 2>&1 &
disown
wait_healthy "http://127.0.0.1:8081/health" "embedding server"

# sanity probe
info "3/8 probing /v1/embeddings"
probe_result=$(curl -sS --max-time 10 http://127.0.0.1:8081/v1/embeddings \
    -H 'content-type: application/json' \
    -d '{"model":"openai/text-embedding-3-small","input":"hello"}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); v=d["data"][0]["embedding"]; print(f"dim={len(v)}")' 2>/dev/null || echo "FAIL")
[[ "$probe_result" == "dim=768" ]] || die "embedding probe failed or wrong dim: $probe_result"
info "embedding server OK ($probe_result)"

# ---------- 4. rewrite honcho config.toml ----------
info "4/8 rewriting Honcho config.toml"
cp "$HONCHO_CONFIG" "$HONCHO_CONFIG.pre-bench"

python3 - "$HONCHO_CONFIG" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()

# Chat base_url: any host.docker.internal:8080/v1 stays (bonsai), any :XXXX/v1
# under [deriver|dialectic.levels.*|summary|dream.*] → keep at :8080 (our test
# chat server). This file already has :8080 under those sections, so this is a
# no-op — but enforce the model name.
text = re.sub(
    r'(model\s*=\s*")bonsai-8b(")',
    r'\1qwen3.6-test\2',
    text,
)

# Embedding base_url: :11434 → :8081
text = re.sub(
    r'(\[embedding\.model_config\.overrides\][^\[]*?base_url\s*=\s*")http://[^"]*:11434/v1(")',
    r'\1http://host.docker.internal:8081/v1\2',
    text,
    flags=re.DOTALL,
)

open(path, 'w').write(text)
print(f"[py] wrote {path}", file=sys.stderr)
PY

grep -q 'qwen3.6-test' "$HONCHO_CONFIG" || die "honcho config did not get model rename"
grep -q '8081/v1' "$HONCHO_CONFIG" || die "honcho config did not get embedding url change"

# ---------- 5. rewrite hermes config ----------
info "5/8 rewriting Hermes config.yaml"
cp "$HERMES_CONFIG" "$HERMES_CONFIG.pre-bench"
sed -i 's|^  default: .*|  default: qwen3.6-test|' "$HERMES_CONFIG"
sed -i 's|^  base_url: http://[^ ]*|  base_url: http://localhost:8080/v1|' "$HERMES_CONFIG"
: > "$HERMES_CTX_CACHE"

grep -q 'qwen3.6-test' "$HERMES_CONFIG" || die "hermes config did not get model rename"
grep -q ':8080/v1' "$HERMES_CONFIG" || die "hermes config did not get base_url change"

# ---------- 6. recreate honcho containers ----------
info "6/8 recreating honcho-api and honcho-deriver with new config"
cd "$ROOT/honcho"
docker compose up -d --force-recreate --no-deps api deriver >/dev/null
cd - >/dev/null
wait_healthy "http://127.0.0.1:8000/v3/workspaces" "honcho-api"

# ---------- 7. capture real deriver request ----------
info "7/8 capturing real Honcho deriver request via transparent proxy"
CAPTURE_OUT="/tmp/honcho_captured.jsonl"
: > "$CAPTURE_OUT"

# proxy: listen :8090, forward to :8080 (no test server yet — this only captures
# the request to be sent; upstream 503 is fine since we drop the response)
# we need SOMETHING at :8080 for the proxy's forward to not error. Start a
# placeholder: a minimal llama-server with the smallest context just to complete
# forwarding, OR use a simpler "always 200" stub. Simpler: run a stub that
# returns a mocked OpenAI response so deriver doesn't hang.
python3 - <<'STUB' &
import http.server, json, socketserver, threading
class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        _ = self.rfile.read(length)
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.end_headers()
        resp = {"id":"stub","object":"chat.completion","choices":[{"index":0,
                "message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
        self.wfile.write(json.dumps(resp).encode())
    def log_message(self, *a, **k): pass
with socketserver.ThreadingTCPServer(("127.0.0.1", 8080), Stub) as httpd:
    httpd.serve_forever()
STUB
STUB_PID=$!
sleep 1

# start proxy in background
python3 "$BENCH_DIR/scripts/capture_proxy.py" > "$LOG_DIR/capture_proxy.log" 2>&1 &
PROXY_PID=$!
disown
sleep 2

# temporarily point honcho deriver at proxy (:8090) for just this capture
python3 - <<'SWITCH'
import re
path = "/home/baba-y/nuncstans-hermes-stack/honcho/config.toml"
text = open(path).read()
# switch deriver base_url to :8090 for this capture
text = re.sub(
    r'(\[deriver\.model_config\.overrides\][^\[]*?base_url\s*=\s*")http://[^"]*:8080/v1(")',
    r'\1http://host.docker.internal:8090/v1\2',
    text, flags=re.DOTALL,
)
open(path, 'w').write(text)
SWITCH

cd "$ROOT/honcho"
docker compose up -d --force-recreate --no-deps deriver >/dev/null
cd - >/dev/null
sleep 3

# trigger a message so deriver actually fires
info "7/8 triggering deriver by posting messages to Honcho"
SESSION_ID="bench-prep-$(date +%s)"
for content in "Bonsaiビルドに$ORIGINを焼き込んだ、動作確認済。" \
               "Ollamaのcontext lengthを65kに上げた、OLLAMA_GPU_OVERHEADも設定済。"; do
    curl -sS --max-time 10 "http://127.0.0.1:8000/v3/workspaces/octoball/sessions/$SESSION_ID/messages" \
        -H 'content-type: application/json' \
        -d "$(python3 -c "import json,sys; print(json.dumps({'messages':[{'peer_id':'Yuki','content':sys.argv[1]}]}))" "$content")" \
        >/dev/null || info "(warning) message post failed, continuing"
done

# wait up to 60s for deriver to emit at least one captured request
deadline=$(( $(date +%s) + 60 ))
while :; do
    if [[ -s "$CAPTURE_OUT" ]]; then
        count=$(wc -l < "$CAPTURE_OUT")
        info "captured $count request(s)"
        if [[ $count -ge 1 ]]; then break; fi
    fi
    (( $(date +%s) >= deadline )) && break
    sleep 2
done

# pick the largest captured body (most representative of deriver's real shape)
if [[ -s "$CAPTURE_OUT" ]]; then
    python3 - "$CAPTURE_OUT" "$BENCH_DIR/prompts/honcho_deriver.json" <<'PICK'
import json, sys
in_path, out_path = sys.argv[1], sys.argv[2]
best = None; best_size = -1
for line in open(in_path):
    rec = json.loads(line)
    if rec.get("kind") != "chat_completions": continue
    body = rec.get("body") or {}
    size = len(json.dumps(body))
    if size > best_size:
        best_size = size; best = body
if best is None:
    sys.exit("no chat_completions captured")
# rename model to bench target
best["model"] = "qwen3.6-test"
best["stream"] = False
# keep max_tokens as deriver sent it; bench will reuse as-is
open(out_path, "w").write(json.dumps(best, ensure_ascii=False))
print(f"wrote {out_path}, prompt_size_chars={best_size}", file=sys.stderr)
PICK
else
    info "WARN: no deriver request captured; contention bench will use a fabricated fallback"
    # write a minimal fallback
    cat > "$BENCH_DIR/prompts/honcho_deriver.json" <<'FB'
{"model":"qwen3.6-test","messages":[{"role":"system","content":"You are a helpful observation-extraction agent. Extract key facts from the conversation as structured observations."},{"role":"user","content":"Conversation pair:\nUser: Hermesの遅延調査を進めた、bonsaiのRPATHとtool_choiceの問題を修正した。\nAssistant: 修正内容を記録しました。"}],"stream":false,"max_tokens":400}
FB
fi

# ---------- restore honcho config, kill stub+proxy, recreate deriver ----------
kill $PROXY_PID 2>/dev/null || true
kill $STUB_PID 2>/dev/null || true
sleep 1

python3 - <<'RESTORE'
import re
path = "/home/baba-y/nuncstans-hermes-stack/honcho/config.toml"
text = open(path).read()
text = re.sub(
    r'(\[deriver\.model_config\.overrides\][^\[]*?base_url\s*=\s*")http://[^"]*:8090/v1(")',
    r'\1http://host.docker.internal:8080/v1\2',
    text, flags=re.DOTALL,
)
open(path, 'w').write(text)
RESTORE

cd "$ROOT/honcho"
docker compose up -d --force-recreate --no-deps deriver >/dev/null
cd - >/dev/null

# ---------- 8. final verification ----------
info "8/8 final verification"
# embedding server still up
curl -sfS --max-time 3 http://127.0.0.1:8081/health >/dev/null || die "embedding server died during prep"
# ports 8080 free (we killed the stub, test server not yet started)
if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ':8080$'; then
    die "port 8080 still bound — something is squatting, investigate before bench_cell.sh"
fi
# honcho containers up
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E '^honcho-(api|deriver)-1' | \
    awk '{if ($2 !~ /^Up/) exit 1}' || die "honcho containers not healthy"

info "prep complete. Next: ./run_all.sh"
