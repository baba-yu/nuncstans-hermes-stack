#!/bin/bash
# 1 Hz sampler of GPU VRAM and system RAM usage.
# Usage: sample_resources.sh <output_tsv>
# Runs until killed (send SIGTERM from the caller).

set -u
OUT="${1:?output tsv path required}"
INTERVAL="${SAMPLE_INTERVAL:-1}"

printf 'ts_unix\tvram_used_mib\tvram_free_mib\tram_used_mib\tram_avail_mib\n' > "$OUT"

while :; do
    ts=$(date +%s.%N)
    vram=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | tr ',' '\t' | tr -d ' ')
    ram=$(free -m | awk 'NR==2{printf "%s\t%s", $3, $7}')
    printf '%s\t%s\t%s\n' "$ts" "${vram:-0	0}" "${ram:-0	0}" >> "$OUT"
    sleep "$INTERVAL"
done
