#!/usr/bin/env python3
"""Analyze gatekeeper_eval/results.jsonl vs ground truth from dataset.jsonl.

Reports:
  - agreement rate (did A>B match the A-family ground truths, B>A match the B ground truths)
  - A/B score + importance + correction distributions
  - A vs importance correlation (checks if axes are redundant)
  - confidence distribution
  - per-category error list (which msgs the gatekeeper got wrong)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median, stdev

HERE = Path(__file__).parent
results = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines() if l.strip()]


def expected_a_beats_b(gt: str) -> bool | None:
    if gt.startswith("A"):
        return True
    if gt == "B":
        return False
    return None


def gap(v: dict) -> float:
    return (v.get("A_score") or 0) - (v.get("B_score") or 0)


hits = 0
total = 0
errors = []
for r in results:
    if "error" in r:
        continue
    v = r["verdict"]
    exp = expected_a_beats_b(r["gt"])
    if exp is None:
        continue
    total += 1
    got_a_beats = gap(v) > 0
    if got_a_beats == exp:
        hits += 1
    else:
        errors.append(r)

print(f"=== Agreement rate (A>B vs B>A match GT) ===")
print(f"  {hits}/{total} = {hits/total:.1%}\n")

print("=== Errors (where gatekeeper disagreed with ground truth) ===")
for r in errors:
    v = r["verdict"]
    print(f"  {r['id']} gt={r['gt']:20s}  A={v.get('A_score'):.2f}  B={v.get('B_score'):.2f}  "
          f"imp={v.get('importance')}  corr={v.get('correction_of_prior')}")
    print(f"       msg={r['msg'][:80]}")
print()

A = [r["verdict"]["A_score"] for r in results if "verdict" in r]
B = [r["verdict"]["B_score"] for r in results if "verdict" in r]
I = [r["verdict"]["importance"] for r in results if "verdict" in r]
CLP = [r.get("confidence_logprob_boundary") for r in results]
CLP = [x for x in CLP if x is not None]
GAPS = [abs(gap(r["verdict"])) for r in results if "verdict" in r]

print("=== Score distributions ===")
print(f"  A_score: min={min(A):.2f} median={median(A):.2f} mean={mean(A):.2f} max={max(A):.2f}")
print(f"  B_score: min={min(B):.2f} median={median(B):.2f} mean={mean(B):.2f} max={max(B):.2f}")
print(f"  |A-B| gap: min={min(GAPS):.2f} median={median(GAPS):.2f} mean={mean(GAPS):.2f} max={max(GAPS):.2f}")
print(f"  importance: min={min(I)} median={median(I)} mean={mean(I):.1f} max={max(I)}")
print(f"  confidence (logprob boundary): min={min(CLP):.2f} median={median(CLP):.2f} mean={mean(CLP):.2f} max={max(CLP):.2f}")

import statistics
ab_sum = [a + b for a, b in zip(A, B)]
print(f"\n=== A/B independence check ===")
print(f"  A+B sum: min={min(ab_sum):.2f} mean={mean(ab_sum):.2f} max={max(ab_sum):.2f}")
print(f"  (if A and B are binary-complementary, sum ≈ 1.0 every time; higher variance means independent axes)")

# Pearson correlation between A and importance
if len(A) > 2:
    ma, mi = mean(A), mean(I)
    num = sum((a - ma) * (i - mi) for a, i in zip(A, I))
    den = math.sqrt(sum((a - ma) ** 2 for a in A) * sum((i - mi) ** 2 for i in I))
    r = num / den if den else 0.0
    print(f"\n=== A_score vs importance correlation ===")
    print(f"  Pearson r = {r:+.2f}  (|r| > 0.9 means redundant, consider separate pass)")

# Agreement on correction_of_prior
print(f"\n=== correction_of_prior detection (ground-truth labels contain 'correction' or 'remember') ===")
for r in results:
    if "verdict" not in r: continue
    gt_is_correction = any(k in r["note"] for k in ("correction", "remember"))
    got = r["verdict"].get("correction_of_prior")
    if gt_is_correction and not got:
        print(f"  FN (missed correction): {r['id']} {r['note']}: {r['msg'][:60]}")
    elif not gt_is_correction and got:
        print(f"  FP (spurious correction): {r['id']} {r['note']}: {r['msg'][:60]}")

# margin gap distribution: how many fall below common δ thresholds?
print(f"\n=== Hypothetical pending rate for different δ ===")
for d in (0.10, 0.15, 0.20, 0.30, 0.50):
    pend = sum(1 for g in GAPS if g < d)
    print(f"  δ={d:.2f} → pending = {pend}/{len(GAPS)} ({pend/len(GAPS):.0%})")
