#!/usr/bin/env python3
"""gatekeeper_daemon: async classifier that moves queue rows out of 'pending'.

Runs alongside the Hermes runtime. Polls the queue for new representation
rows and calls Bonsai to classify each one against the A (literal
self-reference) / B (non-literal framing) hypotheses, plus importance and
correction_of_prior. Uses JSON-schema-constrained output with per-token
logprobs to derive a separate confidence signal.

Decision rules (calibrated from scripts/gatekeeper_eval/):
  A_score − B_score ≥ DELTA    → status='ready'    (deriver picks it up)
  B_score − A_score ≥ DELTA    → status='demoted'  (never promoted to observation)
  |A − B| < DELTA               → stay 'pending' until re-evaluation
  logprob_conf < TAU            → stay 'pending' regardless of margin

Re-evaluation:
  - pending rows whose last verdict is older than REEVAL_AFTER_SEC get
    re-classified (with `reclassify_count` incremented).
  - after REEVAL_MAX attempts (default 2), force a commit: ready if last
    A≥B else demoted. Decision is stamped with a `forced: true` flag in
    gate_verdict.

The daemon never touches queue rows that don't have `status='pending'`;
it also ignores any task_type other than 'representation'.

ENV:
  HERMES_HOME            default ~/hermes-stack
  HONCHO_DIR             default $HERMES_HOME/honcho
  BONSAI_URL             default http://localhost:8080
  BONSAI_MODEL           default bonsai-8b
  GK_POLL_INTERVAL_SEC   default 5
  GK_DELTA               default 0.20   (margin threshold on A−B)
  GK_TAU                 default 0.75   (logprob confidence floor)
  GK_REEVAL_AFTER_SEC    default 90
  GK_REEVAL_MAX          default 2
  GK_BATCH_LIMIT         default 10
  GK_MAX_OUTPUT_TOKENS   default 400
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / "hermes-stack"))
HONCHO_DIR = Path(os.environ.get("HONCHO_DIR", HERMES_HOME / "honcho"))
BONSAI_URL = os.environ.get("BONSAI_URL", "http://localhost:8080")
BONSAI_MODEL = os.environ.get("BONSAI_MODEL", "bonsai-8b")
POLL_INTERVAL_SEC = int(os.environ.get("GK_POLL_INTERVAL_SEC", "5"))
DELTA = float(os.environ.get("GK_DELTA", "0.20"))
TAU = float(os.environ.get("GK_TAU", "0.75"))
REEVAL_AFTER_SEC = int(os.environ.get("GK_REEVAL_AFTER_SEC", "90"))
REEVAL_MAX = int(os.environ.get("GK_REEVAL_MAX", "2"))
BATCH_LIMIT = int(os.environ.get("GK_BATCH_LIMIT", "10"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GK_MAX_OUTPUT_TOKENS", "400"))

CLASSIFIER_VERSION = "gatekeeper-v3"

logging.basicConfig(
    level=logging.INFO,
    format="[gatekeeper] %(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Classifier prompt (v3, frozen after shadow calibration — 90% agreement,
# A/B axes independent, importance r=0.32 vs A_score).

SYSTEM_PROMPT = (
    "You are a gatekeeper classifier. For each chat message, output strict "
    "JSON with four scores. These scores are INDEPENDENT — do NOT treat them "
    "as mutually exclusive.\n\n"

    "A_score (0.0-1.0): LITERALNESS. Does the speaker MEAN this message "
    "literally? A first-person statement is literal if the speaker is making "
    "a sincere assertion about their own world or self. Literal does NOT "
    "require the content to be IMPORTANT or NOTABLE — trivia, greetings, "
    "preferences, grocery lists, casual mood reports, and even boring "
    "everyday facts are all LITERAL. Set A high (>= 0.8) by DEFAULT for any "
    "first-person declarative. Only lower A when there are explicit markers "
    "of non-literalness (see B_score below). Anchor examples:\n"
    "    'My name is Yuki'                    -> A=1.0\n"
    "    'I had coffee this morning'           -> A=1.0 (literal, just trivial)\n"
    "    'I prefer Python for ML'              -> A=1.0 (literal preference)\n"
    "    'hi' / 'thanks'                       -> A=0.9 (literal micro-utterance)\n"
    "    'I moved to Tokyo last month'         -> A=1.0\n"
    "    'I misspoke, my cat is Tofu not Miso' -> A=1.0 (still literal assertion)\n"
    "    'If I were Napoleon'                  -> A=0.05 (explicit hypothetical)\n"
    "    'Oh I LOVE Mondays /s'                -> A=0.1 (sarcasm inverts meaning)\n\n"

    "B_score (0.0-1.0): NON-LITERAL FRAMING STRENGTH. Are there explicit "
    "signals that the content should NOT be taken at face value? Only raise "
    "B when one of these is present in the message itself:\n"
    "  - Explicit hypothetical markers: 'if I were', 'imagine', 'suppose', "
    "'hypothetically', 'what if'\n"
    "  - Sarcasm markers: '/s', 'obviously', emphatic caps with opposite "
    "connotation\n"
    "  - Quoting others: 'she said...', 'they claim...'\n"
    "  - Explicit fiction/creative context: 'my character', 'in my novel', "
    "'writing a story where'\n"
    "  - Explicit negation OF SELF-IDENTITY (NOT general negation): "
    "'I'm not from Mars' (B stays LOW because still literal)\n"
    "Anchor examples:\n"
    "    'If I moved to Osaka, would it be crazy?'  -> B=0.9\n"
    "    'I'm writing a novel where I'm a pirate'   -> B=0.9\n"
    "    'Oh yeah, I LOVE Mondays /s'               -> B=0.9\n"
    "    'She said \"I'm from Mars\"'                -> B=0.9\n"
    "    'I had coffee this morning'                -> B=0.05 (NO non-literal markers)\n"
    "    'I work as a backend engineer'             -> B=0.05\n"
    "    'hi'                                       -> B=0.05\n"
    "    'My name is NOT Alice, it's Yuki'          -> B=0.05 (literal self-identification, negation is part of the assertion)\n\n"

    "IMPORTANT: A_score and B_score are NOT complements. A message can have "
    "A=1.0 and B=0.05 (most literal messages). A message can have A=0.1 and "
    "B=0.9 (clearly hypothetical). Messages with A=0.5/B=0.5 are GENUINELY "
    "AMBIGUOUS — use that only when you truly cannot tell.\n\n"

    "importance (integer 0-10): INDEPENDENT of A/B, how memorable is the "
    "CONTENT of what was asserted? Ignore how literal the speaker was; "
    "imagine the content were true and rate its memory value.\n"
    "  0-1: trivia / greetings / single-word acknowledgments\n"
    "  2-3: routine daily life (meals, weather, moods)\n"
    "  4-6: preferences, short-term plans, mild opinions\n"
    "  7-8: durable biographical facts (job, location, relationships, pets)\n"
    "  9-10: critical identity / safety info (name, address, allergies, "
    "emergency contacts, medical, financial — or explicit REMEMBER prefix)\n"
    "Anchor examples:\n"
    "    'hi'                          -> importance=1\n"
    "    'I had coffee this morning'   -> importance=2\n"
    "    'I prefer Python for ML'      -> importance=5\n"
    "    'My name is Yuki'             -> importance=9\n"
    "    'I'm allergic to peanuts'     -> importance=9\n"
    "    'REMEMBER: my number is 555'  -> importance=10\n\n"

    "correction_of_prior (boolean): true ONLY if the speaker is explicitly "
    "retracting, revising, or overwriting a prior assertion that THEY "
    "THEMSELVES made earlier in this conversation. The message must refer "
    "to the speaker's OWN prior statement, not to someone else's claim, "
    "not to general beliefs, and not merely to something the speaker wants "
    "to emphasize.\n"
    "  true examples:\n"
    "    'Actually, I don't live in Osaka anymore. Moved to Kyoto.'\n"
    "    'Wait, I misspoke earlier — my cat is Tofu, not Miso.'\n"
    "    'Correction: I'm allergic to shellfish, not peanuts.'\n"
    "    'Earlier I said X, but that was wrong.'\n"
    "  false examples (do NOT flag these):\n"
    "    'Important: my SSN is ...'  -- emphasis marker, not a revision\n"
    "    'REMEMBER: my number is ...' -- emphasis marker\n"
    "    'Some people say X. Not me.' -- differentiating self from others, not revising past self\n"
    "    'I'm not from Osaka.' -- literal negation, no prior claim to retract\n"
    "Do NOT set correction_of_prior=true merely because the message contains "
    "'important', 'REMEMBER', negation words, or differentiation from others.\n\n"

    "confidence (0.0-1.0): your own subjective certainty that the four "
    "fields above are correct. Be calibrated — if the message is ambiguous, "
    "lower confidence.\n\n"

    "Emit the JSON object ONLY — no surrounding prose."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "A_score":             {"type": "number", "minimum": 0, "maximum": 1},
        "B_score":             {"type": "number", "minimum": 0, "maximum": 1},
        "importance":          {"type": "integer", "minimum": 0, "maximum": 10},
        "correction_of_prior": {"type": "boolean"},
        "A_reason":            {"type": "string"},
        "B_reason":            {"type": "string"},
        "importance_reason":   {"type": "string"},
        "confidence":          {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "A_score", "B_score", "importance",
        "correction_of_prior", "A_reason", "B_reason",
        "importance_reason", "confidence",
    ],
}


# -----------------------------------------------------------------------------
# Helpers

def _psql(sql: str) -> str:
    result = subprocess.run(
        [
            "docker", "compose", "-f", str(HONCHO_DIR / "docker-compose.yml"),
            "exec", "-T", "database",
            "psql", "-U", "honcho", "-d", "honcho", "-At", "-F", "\t",
            "-c", sql,
        ],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


def _psql_json(sql: str) -> list[dict]:
    """Run a SQL that returns rows as JSON (use json_agg)."""
    raw = _psql(sql)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _call_bonsai(msg: str) -> tuple[dict, list[dict] | None, int]:
    body = {
        "model": BONSAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Message to evaluate:\n{msg}"},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "gatekeeper_verdict", "schema": JSON_SCHEMA},
        },
    }
    req = urllib.request.Request(
        f"{BONSAI_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.load(r)
    dt_ms = int((time.time() - t0) * 1000)
    choice = raw["choices"][0]
    content = choice["message"].get("content", "").strip()
    parsed = json.loads(content) if content else {}
    lp = (choice.get("logprobs") or {}).get("content")
    return parsed, lp, dt_ms


def _decision_boundary_confidence(logprobs: list[dict] | None) -> float | None:
    """Geometric mean of top-1 probs across JSON field value tokens
    (digits, booleans). This proxies the decisive-token confidence."""
    if not logprobs:
        return None
    interesting: list[float] = []
    for tok in logprobs:
        t = (tok.get("token") or "").strip()
        if t in {"true", "false"} or t.replace(".", "").isdigit():
            interesting.append(math.exp(tok.get("logprob", 0)))
    if not interesting:
        return None
    return math.exp(sum(math.log(p) for p in interesting) / len(interesting))


# -----------------------------------------------------------------------------
# Decision logic

def _decide(verdict: dict, conf_logprob: float | None) -> tuple[str, dict]:
    """Apply δ / τ rules. Return (new_status, audit_annotations)."""
    a = float(verdict.get("A_score", 0.0))
    b = float(verdict.get("B_score", 0.0))
    corr = bool(verdict.get("correction_of_prior", False))
    gap = a - b

    audit = {
        "classifier_version": CLASSIFIER_VERSION,
        "delta": DELTA,
        "tau": TAU,
        "decided_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "confidence_logprob": conf_logprob,
        "gap": gap,
    }

    if conf_logprob is not None and conf_logprob < TAU:
        audit["reason"] = f"low_logprob_conf ({conf_logprob:.2f} < τ={TAU})"
        return "pending", audit

    # Correction short-circuit: when the classifier is confident a message is
    # the speaker retracting their own prior assertion, and A (literalness) is
    # at least moderate, land as 'ready' even if B is also high. A correction
    # phrase naturally raises B (framing marker) but the new fact is still
    # literal and we want it memorised so the deriver can run
    # supersede_observations.
    if corr and a >= 0.7:
        audit["reason"] = f"correction override (A={a:.2f}, B={b:.2f}, corr=True)"
        audit["correction_override"] = True
        return "ready", audit

    if gap >= DELTA:
        audit["reason"] = f"A wins by {gap:.2f}"
        return "ready", audit
    if -gap >= DELTA:
        audit["reason"] = f"B wins by {-gap:.2f}"
        return "demoted", audit

    audit["reason"] = f"|A−B|={abs(gap):.2f} < δ={DELTA}"
    return "pending", audit


def _forced_decide(verdict: dict) -> tuple[str, dict]:
    """After REEVAL_MAX attempts, force a commit. Corrections always go ready
    (losing a correction silently is worse than storing it provisionally).
    Otherwise lean A."""
    a = float(verdict.get("A_score", 0.0))
    b = float(verdict.get("B_score", 0.0))
    corr = bool(verdict.get("correction_of_prior", False))
    audit = {
        "classifier_version": CLASSIFIER_VERSION,
        "decided_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "forced": True,
    }
    if corr:
        audit["reason"] = "forced commit: correction_of_prior=true always lands ready"
        return "ready", audit
    if a >= b:
        audit["reason"] = f"forced commit after REEVAL_MAX={REEVAL_MAX}: A>=B"
        return "ready", audit
    audit["reason"] = f"forced commit after REEVAL_MAX={REEVAL_MAX}: B>A"
    return "demoted", audit


# -----------------------------------------------------------------------------
# DB queries

def _fetch_pending_unverdict(limit: int) -> list[dict]:
    """New pending rows with no gate_verdict yet."""
    sql = (
        "SELECT row_to_json(t) FROM ("
        " SELECT q.id, q.work_unit_key, q.reclassify_count, m.content "
        " FROM queue q LEFT JOIN messages m ON q.message_id = m.id "
        " WHERE q.status = 'pending' "
        "   AND q.task_type = 'representation' "
        "   AND q.processed = false "
        "   AND q.gate_verdict IS NULL "
        "   AND q.message_id IS NOT NULL "
        f" ORDER BY q.created_at LIMIT {limit}"
        ") t;"
    )
    rows = []
    raw = _psql(sql)
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fetch_pending_for_reeval(limit: int) -> list[dict]:
    """pending rows older than REEVAL_AFTER_SEC that still haven't been
    re-evaluated the max number of times."""
    sql = (
        "SELECT row_to_json(t) FROM ("
        " SELECT q.id, q.work_unit_key, q.reclassify_count, m.content "
        " FROM queue q LEFT JOIN messages m ON q.message_id = m.id "
        " WHERE q.status = 'pending' "
        "   AND q.task_type = 'representation' "
        "   AND q.processed = false "
        "   AND q.gate_verdict IS NOT NULL "
        f"   AND q.gate_decided_at < now() - interval '{REEVAL_AFTER_SEC} seconds' "
        f"   AND q.reclassify_count < {REEVAL_MAX} "
        "   AND q.message_id IS NOT NULL "
        f" ORDER BY q.gate_decided_at LIMIT {limit}"
        ") t;"
    )
    rows: list[dict] = []
    for line in _psql(sql).splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fetch_reeval_exhausted(limit: int) -> list[dict]:
    """pending rows that have already burned REEVAL_MAX attempts — time to
    force-commit them."""
    sql = (
        "SELECT row_to_json(t) FROM ("
        " SELECT q.id, q.work_unit_key, q.reclassify_count, q.gate_verdict::text AS verdict_json, m.content "
        " FROM queue q LEFT JOIN messages m ON q.message_id = m.id "
        " WHERE q.status = 'pending' "
        "   AND q.task_type = 'representation' "
        "   AND q.processed = false "
        f"   AND q.reclassify_count >= {REEVAL_MAX} "
        "   AND q.gate_verdict IS NOT NULL "
        f" ORDER BY q.gate_decided_at LIMIT {limit}"
        ") t;"
    )
    rows: list[dict] = []
    for line in _psql(sql).splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _apply_verdict(
    queue_id: int,
    new_status: str,
    verdict: dict,
    audit: dict,
    bump_reclassify: bool,
) -> None:
    full = {**verdict, "audit": audit}
    verdict_json = json.dumps(full).replace("'", "''")
    bump = "reclassify_count + 1" if bump_reclassify else "reclassify_count"
    sql = (
        "UPDATE queue SET "
        f" status = '{new_status}', "
        f" gate_verdict = '{verdict_json}'::jsonb, "
        " gate_decided_at = now(), "
        f" reclassify_count = {bump} "
        f"WHERE id = {queue_id};"
    )
    _psql(sql)


# -----------------------------------------------------------------------------
# Per-row processing

def _classify_and_apply(row: dict, *, reeval: bool) -> None:
    content = row.get("content") or ""
    qid = int(row["id"])
    if not content.strip():
        log.warning("queue %s has empty content; marking demoted", qid)
        _apply_verdict(
            qid, "demoted",
            {"A_score": 0, "B_score": 1},
            {"reason": "empty_content", "classifier_version": CLASSIFIER_VERSION},
            bump_reclassify=False,
        )
        return

    try:
        verdict, lp, call_ms = _call_bonsai(content)
    except urllib.error.URLError as e:
        log.warning("bonsai call failed for queue %s: %s", qid, e)
        return
    except json.JSONDecodeError as e:
        log.warning("bonsai produced invalid JSON for queue %s: %s", qid, e)
        return

    conf_lp = _decision_boundary_confidence(lp)
    new_status, audit = _decide(verdict, conf_lp)
    audit["bonsai_call_ms"] = call_ms

    _apply_verdict(qid, new_status, verdict, audit, bump_reclassify=reeval)
    mode = "reeval" if reeval else "init"
    log.info(
        "q=%s %s a=%.2f b=%.2f imp=%s corr=%s conf=%s -> %s (%s)",
        qid, mode,
        verdict.get("A_score", 0), verdict.get("B_score", 0),
        verdict.get("importance"), verdict.get("correction_of_prior"),
        f"{conf_lp:.2f}" if conf_lp is not None else "n/a",
        new_status, audit["reason"],
    )


def _force_commit(row: dict) -> None:
    qid = int(row["id"])
    try:
        verdict = json.loads(row.get("verdict_json") or "{}")
    except json.JSONDecodeError:
        verdict = {}
    new_status, audit = _forced_decide(verdict)
    _apply_verdict(qid, new_status, verdict, audit, bump_reclassify=False)
    log.info("q=%s forced -> %s", qid, new_status)


# -----------------------------------------------------------------------------
# Main loop

def loop() -> None:
    log.info(
        "starting gatekeeper_daemon δ=%s τ=%s reeval_after=%ss reeval_max=%s poll=%ss",
        DELTA, TAU, REEVAL_AFTER_SEC, REEVAL_MAX, POLL_INTERVAL_SEC,
    )
    while True:
        try:
            # 1) classify brand-new pending rows
            for row in _fetch_pending_unverdict(BATCH_LIMIT):
                _classify_and_apply(row, reeval=False)

            # 2) re-evaluate stale-pending rows
            for row in _fetch_pending_for_reeval(BATCH_LIMIT):
                _classify_and_apply(row, reeval=True)

            # 3) force-commit rows that exhausted re-evaluations
            for row in _fetch_reeval_exhausted(BATCH_LIMIT):
                _force_commit(row)
        except Exception:
            log.exception("error in gatekeeper loop")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        loop()
    except KeyboardInterrupt:
        sys.exit(0)
