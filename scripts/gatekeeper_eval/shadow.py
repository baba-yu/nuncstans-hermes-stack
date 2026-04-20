#!/usr/bin/env python3
"""Gatekeeper shadow runner — evaluate each message in the eval set against
Bonsai and record raw + logprob-derived scores.

Outputs JSONL to ./results.jsonl.

Design (per Q3 / Q4 commits):
  - Single Bonsai call per message, JSON-schema-constrained output.
  - CoT style prompt that separates "A vs B literal-ness" from "importance".
  - Request logprobs to compute a confidence independent of the self-reported one.
  - No side effects on the real Honcho queue — this runs against the model in
    isolation and is safe to repeat.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

BONSAI_URL = os.environ.get("BONSAI_URL", "http://localhost:8080")
MODEL = os.environ.get("BONSAI_MODEL", "bonsai-8b")
DATA_PATH = Path(__file__).parent / "dataset.jsonl"
OUT_PATH = Path(__file__).parent / "results.jsonl"

SYSTEM = (
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
    "    'My name is NOT Alice. It's Yuki.' -- self-identification with embedded negation, no prior self-claim being retracted (unless the speaker had previously said 'My name is Alice')\n"
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
        "A_score":              {"type": "number", "minimum": 0, "maximum": 1},
        "B_score":              {"type": "number", "minimum": 0, "maximum": 1},
        "importance":           {"type": "integer", "minimum": 0, "maximum": 10},
        "correction_of_prior":  {"type": "boolean"},
        "A_reason":             {"type": "string"},
        "B_reason":             {"type": "string"},
        "importance_reason":    {"type": "string"},
        "confidence":           {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "A_score", "B_score", "importance",
        "correction_of_prior", "A_reason", "B_reason",
        "importance_reason", "confidence",
    ],
}


def call_bonsai(msg: str) -> tuple[dict, list[dict] | None, int]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"Message to evaluate:\n{msg}"},
        ],
        "max_tokens": 400,
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


def geometric_mean_prob(logprobs: list[dict], tokens_of_interest: set[str]) -> float | None:
    """Geometric mean of top-1 token probabilities, limited to tokens whose
    text contains any string in tokens_of_interest (so we focus on the
    field values, not JSON syntax)."""
    if not logprobs:
        return None
    ps = []
    for tok in logprobs:
        text = tok.get("token", "")
        if any(k in text for k in tokens_of_interest):
            ps.append(math.exp(tok.get("logprob", 0)))
    if not ps:
        return None
    return math.exp(sum(math.log(p) for p in ps) / len(ps))


def mean_top1_prob(logprobs: list[dict] | None) -> float | None:
    if not logprobs:
        return None
    ps = [math.exp(tok.get("logprob", 0)) for tok in logprobs]
    if not ps:
        return None
    return sum(ps) / len(ps)


def decision_boundary_confidence(logprobs: list[dict] | None) -> float | None:
    """Look at tokens that sit around numerical JSON field values (digits or
    true/false). Return geometric mean of their top-1 probabilities."""
    if not logprobs:
        return None
    interesting = []
    for tok in logprobs:
        t = tok.get("token", "").strip()
        if t in {"true", "false"} or t.replace(".", "").isdigit():
            interesting.append(math.exp(tok.get("logprob", 0)))
    if not interesting:
        return None
    return math.exp(sum(math.log(p) for p in interesting) / len(interesting))


def main() -> int:
    # Health check
    try:
        with urllib.request.urlopen(f"{BONSAI_URL}/v1/models", timeout=5) as r:
            assert "bonsai-8b" in r.read().decode()
    except Exception as e:
        print(f"bonsai not responding at {BONSAI_URL}: {e}", file=sys.stderr)
        return 1

    items = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]
    print(f"evaluating {len(items)} messages", file=sys.stderr)

    with OUT_PATH.open("w") as out:
        for i, it in enumerate(items, 1):
            try:
                parsed, lp, dt = call_bonsai(it["msg"])
            except Exception as e:
                print(f"  [{i}/{len(items)}] {it['id']} ERROR: {e}", file=sys.stderr)
                out.write(json.dumps({"id": it["id"], "error": str(e), "msg": it["msg"]}) + "\n")
                out.flush()
                continue
            conf_logprob = decision_boundary_confidence(lp)
            mean_top1 = mean_top1_prob(lp)
            rec = {
                "id":       it["id"],
                "msg":      it["msg"],
                "gt":       it["gt"],
                "note":     it["note"],
                "verdict":  parsed,
                "confidence_logprob_boundary": conf_logprob,
                "confidence_logprob_mean_top1": mean_top1,
                "dt_ms":    dt,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            a = parsed.get("A_score"); b = parsed.get("B_score"); im = parsed.get("importance"); cp = parsed.get("correction_of_prior")
            print(f"  [{i:2d}/{len(items)}] {it['id']:5s}  A={a}  B={b}  imp={im}  corr={cp}  c_lp={conf_logprob}  {dt}ms  gt={it['gt']}", file=sys.stderr)
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
