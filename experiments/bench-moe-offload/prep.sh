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
# Preserve a known-good baseline on first run; on re-runs restore from it
# before rewriting so the regex transforms are deterministic regardless of
# any half-finished state left by a prior aborted prep.
if [[ -f "$HONCHO_CONFIG.pre-bench" ]]; then
    cp "$HONCHO_CONFIG.pre-bench" "$HONCHO_CONFIG"
    info "  restored from pre-bench backup"
else
    cp "$HONCHO_CONFIG" "$HONCHO_CONFIG.pre-bench"
    info "  saved pre-bench backup"
fi

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

# ---------- 6. rebuild and recreate honcho containers ----------
info "6/8 rebuilding honcho image (ensures local tool_choice patch is baked in)"
cd "$ROOT/honcho"
# build only api image (deriver shares it in this compose file) with a clean
# cache miss on the src copy layer so our committed patch takes effect
docker compose build --quiet api deriver 2>&1 | tail -5 || die "image build failed"
info "6/8 recreating honcho-api and honcho-deriver with new image"
docker compose up -d --force-recreate --no-deps api deriver >/dev/null
cd - >/dev/null
wait_healthy "http://127.0.0.1:8000/docs" "honcho-api"

# verify patch is present in the running containers
for svc in honcho-api-1 honcho-deriver-1; do
    if ! docker exec "$svc" grep -q 'if tool_choice == "any"' /app/src/llm/backends/openai.py; then
        die "$svc does not have the tool_choice patch; image rebuild failed silently"
    fi
done
info "  tool_choice patch verified in both containers"

# ---------- 7. verify committed deriver prompt ----------
# The representative Honcho-deriver-shape request used by the contention
# bench is committed at prompts/honcho_deriver.json. We tried live-capturing
# a real deriver request via a transparent proxy earlier, but the dance
# (stub + proxy + config swap + docker recreate + wait-for-fire) was
# brittle enough that it repeatedly dominated prep time without adding
# measurement value. The committed payload mirrors src/deriver/prompts.py
# minimal_deriver_prompt + the create_observations tool schema, sized to
# ~1-2k tokens — representative of what a real deriver turn sends.
info "7/8 verifying prompts/honcho_deriver.json"
[[ -s "$BENCH_DIR/prompts/honcho_deriver.json" ]] \
    || die "prompts/honcho_deriver.json missing or empty"
python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('model')=='qwen3.6-test', 'model not set to qwen3.6-test'
assert isinstance(d.get('messages'), list) and len(d['messages'])>=2, 'messages array malformed'
assert isinstance(d.get('tools'), list) and len(d['tools'])>=1, 'tools array missing'
print(f\"  honcho_deriver.json OK: {len(d['messages'])} msgs, {len(d['tools'])} tool(s), max_tokens={d.get('max_tokens')}\")
" "$BENCH_DIR/prompts/honcho_deriver.json" || die "honcho_deriver.json validation failed"

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
