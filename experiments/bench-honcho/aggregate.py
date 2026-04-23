#!/usr/bin/env python3
"""Aggregate per-experiment summary.json files into a single results table + analysis."""
import json, os, pathlib, re, sys
from statistics import mean

BENCH = pathlib.Path("/home/baba-y/nuncstans-hermes-stack/experiments/bench-honcho")

def parse_vram_trace(path):
    if not path.exists(): return {}
    used = []
    util = []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4: continue
            try:
                used.append(int(parts[1]))
                util.append(int(parts[3]))
            except ValueError:
                continue
    if not used: return {}
    return {
        "vram_peak_mib": max(used),
        "vram_mean_mib": round(mean(used)),
        "gpu_util_mean_pct": round(mean(util), 1),
        "samples": len(used),
    }

def parse_deriver_log(path):
    if not path.exists(): return {}
    text = path.read_text(errors="ignore")
    perf_lines = re.findall(r"⚡ PERFORMANCE[^\n]*", text)
    llm_calls = len(re.findall(r"Llm Call Duration", text))
    obs_count = 0
    for m in re.finditer(r"Observation Count\s+(\d+)", text):
        obs_count += int(m.group(1))
    return {"deriver_perf_lines": len(perf_lines), "deriver_llm_calls": llm_calls, "observations_logged": obs_count}

def parse_bonsai_log(path):
    if not path.exists(): return {}
    text = path.read_text(errors="ignore")
    timings = re.findall(r"print_timing", text)
    slots = re.findall(r"slot\s+release", text)
    return {"bonsai_timings": len(timings), "bonsai_slots_released": len(slots)}

def collect():
    runs = []
    # Only real experiment dirs E1..E6 — ignore archived failed attempts (E1-timeout-1200, E6-ctxfail, etc.)
    for n in range(1, 7):
        d = BENCH / "runs" / f"E{n}"
        s_path = d / "summary.json"
        if not s_path.exists(): continue
        s = json.load(open(s_path))
        s["vram"] = parse_vram_trace(d / "vram_trace.csv")
        s["deriver_stats"] = parse_deriver_log(d / "deriver.log")
        s["bonsai_stats"] = parse_bonsai_log(d / "bonsai.log")
        runs.append(s)
    return runs

def render_md(runs):
    lines = []
    lines.append("| # | Honcho backend | Inference | Wall (s) | Drain (s) | Total (s) | Obs | pptx OK | Slides | VRAM peak (MiB) | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        pptx = r.get("pptx", {})
        notes = []
        if r.get("exit_code") != 0:
            notes.append(f"exit={r['exit_code']}")
        if not pptx.get("ok"):
            notes.append("pptx-fail")
        vp = r.get("vram", {}).get("vram_peak_mib", "—")
        lines.append(f"| E{r['experiment']} | {r['honcho_variant']} | {r['inference_model']} | "
                     f"{r['wall_seconds']:.1f} | {r.get('drain_seconds',0):.1f} | {r.get('total_seconds',0):.1f} | "
                     f"{r.get('observations_created',0)} | {'✓' if pptx.get('ok') else '✗'} | "
                     f"{pptx.get('slides','?')} | {vp} | {'; '.join(notes) or '—'} |")
    return "\n".join(lines)

def main():
    runs = collect()
    print(json.dumps(runs, indent=2, default=str))
    md = render_md(runs)
    print("\n\n", md, sep="")

if __name__ == "__main__":
    main()
