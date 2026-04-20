#!/usr/bin/env bash
# S9 — correction → supersede round trip.
#
# Posts a long literal introduction (phase 1), lets the deriver create
# observations, then posts a CORRECTION (phase 2) that explicitly retracts
# one of the original facts. Verifies:
#   * B gets ready + gate_verdict.correction_of_prior = true
#   * deriver processes B
#   * the secondary supersede pass fires: at least one original observation is
#     soft-deleted with deleted_reason='superseded' and a non-empty
#     supersede_reason recorded in internal_metadata.
#
# This test exercises the Q2-follow-up wiring: gate_verdict → deriver → tool-mode
# LLM call → supersede_observations handler → CRUD soft-delete. The "happy path"
# here is that the Kyoto claim specifically gets retracted; we assert that as a
# strong check but fall back to a weaker "something was superseded" check if the
# seed happened not to produce a Kyoto observation (deriver output is not 100%
# deterministic).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
step_header "S9 correction → supersede"
export CURRENT_SCENARIO="s9_supersede"

WS="$UAT_RUN_ID-s9"
PEER="emma"
SES="corr"
FAILS=0

setup_ws "$WS" "$PEER" "$SES"
log "created ws=$WS peer=$PEER ses=$SES"

# --- phase 1: post a fact-dense literal seed -------------------------------
SEED_A="I'm Emma. I live in Kyoto, and I work as a frontend engineer. I have a rescue cat named Pepper. I do pottery classes on Saturdays. I'm vegetarian. My partner's name is Jun. I grew up in Hokkaido but moved to Kyoto five years ago. I use a TypeScript stack at work. I'm allergic to peanuts."
LONG_A=$(python3 -c "import sys; s=sys.argv[1]; print((s+' ')*3)" "$SEED_A")

log "=== phase 1: post literal A (expect ready + observation) ==="
TOKENS_A=$(post_message "$WS" "$PEER" "$SES" "$LONG_A")
log "A tokens=$TOKENS_A"

log "waiting up to 90s for A gatekeeper verdict..."
for i in $(seq 1 30); do
  STATUS_A=$(honcho_db -At -c "SELECT status FROM queue WHERE workspace_name='$WS' AND task_type='representation' ORDER BY id LIMIT 1;")
  if [ "$STATUS_A" = "ready" ] || [ "$STATUS_A" = "demoted" ]; then break; fi
  sleep 3
done
log "A status=$STATUS_A"
if [ "$STATUS_A" = "ready" ]; then
  pass "A verdict=ready"
else
  fail "A verdict=$STATUS_A (expected ready)"; FAILS=$((FAILS+1))
fi

log "waiting up to 5 min for deriver to drain A..."
if wait_queue_drain "$WS" 300; then
  pass "A queue drained"
else
  log "A queue didn't drain within 5 min — proceeding anyway"
fi
sleep 3

DOCS_A=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND deleted_at IS NULL;")
log "live documents after A: $DOCS_A"
if [ "${DOCS_A:-0}" -ge 1 ]; then
  pass "observations created for A: $DOCS_A"
else
  fail "no observations created for A"; FAILS=$((FAILS+1))
fi

KYOTO_BEFORE=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND deleted_at IS NULL AND content ILIKE '%kyoto%';")
log "live Kyoto observations before correction: $KYOTO_BEFORE"

# --- phase 2: post the correction ------------------------------------------
SEED_B="Actually I need to correct something important. I do NOT live in Kyoto — I moved to Osaka last month. Please retract anything you remembered about me living in Kyoto. I live in Osaka now. The rest is still correct: I'm Emma, a frontend engineer, vegetarian, partner is Jun, cat is Pepper."
# Must exceed REPRESENTATION_BATCH_MAX_TOKENS (200) so the deriver actually
# picks up the correction message in a new batch.
LONG_B=$(python3 -c "import sys; s=sys.argv[1]; print((s+' ')*4)" "$SEED_B")

log "=== phase 2: post correction B (expect ready + correction_of_prior=true) ==="
TOKENS_B=$(post_message "$WS" "$PEER" "$SES" "$LONG_B")
log "B tokens=$TOKENS_B"

log "waiting up to 90s for B gatekeeper verdict..."
for i in $(seq 1 30); do
  STATUS_B=$(honcho_db -At -c "SELECT status FROM queue WHERE workspace_name='$WS' AND task_type='representation' ORDER BY id DESC LIMIT 1;")
  if [ "$STATUS_B" = "ready" ] || [ "$STATUS_B" = "demoted" ]; then break; fi
  sleep 3
done
log "B status=$STATUS_B"
if [ "$STATUS_B" = "ready" ]; then
  pass "B verdict=ready"
else
  fail "B verdict=$STATUS_B (expected ready)"; FAILS=$((FAILS+1))
fi

CORR_B=$(honcho_db -At -c "SELECT gate_verdict->>'correction_of_prior' FROM queue WHERE workspace_name='$WS' AND task_type='representation' ORDER BY id DESC LIMIT 1;")
log "B correction_of_prior=$CORR_B"
if [ "$CORR_B" = "true" ]; then
  pass "B correction_of_prior=true"
else
  fail "B correction_of_prior=$CORR_B (expected true)"; FAILS=$((FAILS+1))
fi

log "waiting up to 5 min for deriver + supersede pass to finish..."
if wait_queue_drain "$WS" 300; then
  pass "B queue drained"
else
  log "B queue didn't drain within 5 min — proceeding anyway"
fi
# The supersede pass is a secondary tool-mode LLM call after queue drain.
# Give it some breathing room.
sleep 8

# --- phase 3: verify supersede side effects --------------------------------
SUPERSEDED=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND deleted_at IS NOT NULL AND internal_metadata->>'deleted_reason'='superseded';")
log "superseded documents: $SUPERSEDED"
if [ "${SUPERSEDED:-0}" -ge 1 ]; then
  pass "at least one observation was superseded"
else
  fail "no observations superseded after correction"; FAILS=$((FAILS+1))
fi

# Strong check: the specific Kyoto claim was retracted (the correction's target).
KYOTO_SUPERSEDED=$(honcho_db -At -c "SELECT count(*) FROM documents WHERE workspace_name='$WS' AND deleted_at IS NOT NULL AND content ILIKE '%kyoto%';")
log "Kyoto observations now soft-deleted: $KYOTO_SUPERSEDED"
if [ "${KYOTO_BEFORE:-0}" -ge 1 ]; then
  if [ "${KYOTO_SUPERSEDED:-0}" -ge 1 ]; then
    pass "Kyoto claim specifically superseded"
  else
    fail "Kyoto observation survived the correction"; FAILS=$((FAILS+1))
  fi
else
  log "no Kyoto claim was produced by A — skipping strong check"
fi

REASON_SAMPLE=$(honcho_db -At -c "SELECT internal_metadata->>'supersede_reason' FROM documents WHERE workspace_name='$WS' AND deleted_at IS NOT NULL AND internal_metadata->>'deleted_reason'='superseded' LIMIT 1;")
log "sample supersede_reason: $REASON_SAMPLE"
if [ -n "$REASON_SAMPLE" ] && [ "$REASON_SAMPLE" != "NULL" ]; then
  pass "supersede_reason recorded"
else
  fail "supersede_reason missing or empty"; FAILS=$((FAILS+1))
fi

# Dump post-state for run.log
honcho_db -c "SELECT substring(content for 80) AS snippet, (deleted_at IS NOT NULL) AS dead, internal_metadata->>'deleted_reason' AS reason FROM documents WHERE workspace_name='$WS' ORDER BY created_at;" >> "$RESULTS_DIR/run.log"

cleanup_workspace "$WS"

[ "$FAILS" -gt 0 ] && exit 1
log "S9: OK"
