#!/usr/bin/env bash
# Top-level UAT runner.
# usage: run_all.sh

set -u

export UAT_RUN_ID="uat-$(date +%Y%m%d-%H%M%S)"
export HERMES_HOME="$HOME/hermes-stack"
export HONCHO_DIR="$HERMES_HOME/honcho"
export RESULTS_DIR="$HERMES_HOME/test/uat/results/$UAT_RUN_ID"
mkdir -p "$RESULTS_DIR"
echo "UAT_RUN_ID=$UAT_RUN_ID"
echo "RESULTS_DIR=$RESULTS_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

STATUS_FILE="$RESULTS_DIR/scenarios.tsv"
printf "scenario\tstatus\telapsed_s\n" > "$STATUS_FILE"

run_scenario() {
  local name=$1 script=$2
  local start=$(date +%s)
  if bash "$script"; then
    local elapsed=$(( $(date +%s) - start ))
    printf "%s\tPASS\t%d\n" "$name" "$elapsed" >> "$STATUS_FILE"
    return 0
  else
    local elapsed=$(( $(date +%s) - start ))
    printf "%s\tFAIL\t%d\n" "$name" "$elapsed" >> "$STATUS_FILE"
    return 1
  fi
}

set -e
run_scenario "00_preflight" "$SCRIPT_DIR/00_preflight.sh"
run_scenario "s1_smoke"     "$SCRIPT_DIR/s1_smoke.sh"
run_scenario "s2_ollama"    "$SCRIPT_DIR/s2_ollama_chat.sh"
run_scenario "s3_embed"     "$SCRIPT_DIR/s3_embed_regression.sh"
run_scenario "s4_deriver"   "$SCRIPT_DIR/s4_deriver.sh"
run_scenario "s5_dialectic" "$SCRIPT_DIR/s5_dialectic.sh"
run_scenario "s6_cross_ses" "$SCRIPT_DIR/s6_cross_session.sh"
run_scenario "99_final"     "$SCRIPT_DIR/99_final_assertions.sh"
set +e

echo ""
echo "========================================"
echo "UAT RESULTS ($UAT_RUN_ID)"
echo "========================================"
column -t -s $'\t' < "$STATUS_FILE"

if grep -qP '\tFAIL\t' "$STATUS_FILE"; then
  echo "OVERALL: FAIL"
  exit 1
fi
echo "OVERALL: PASS"
