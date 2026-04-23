#!/usr/bin/env python3
"""Swap Hermes config.yaml model+provider block in-place. Idempotent."""
import sys, yaml, os, shutil, pathlib

HERMES_HOME = pathlib.Path(os.path.expanduser("~/.hermes"))
CFG = HERMES_HOME / "config.yaml"
BAK = HERMES_HOME / "config.yaml.bench-backup"

TARGETS = {
    "qwen3.5:9b":   {"provider": "ollama-launch", "base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama"},
    "qwen3.6:35b":  {"provider": "ollama-launch", "base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama"},
    "bonsai-8b":    {"provider": "bonsai-llama",  "base_url": "http://127.0.0.1:8080/v1",  "api_key": "not-needed"},
}

def main():
    if len(sys.argv) != 2:
        print("usage: swap_hermes.py <model>|restore", file=sys.stderr); sys.exit(2)
    action = sys.argv[1]
    if action == "backup" and not BAK.exists():
        shutil.copy2(CFG, BAK); print("backed up"); return
    if action == "restore":
        if BAK.exists():
            shutil.copy2(BAK, CFG); print("restored")
        return
    if action not in TARGETS:
        print(f"unknown model: {action}", file=sys.stderr); sys.exit(2)
    t = TARGETS[action]
    with open(CFG) as f: cfg = yaml.safe_load(f)
    # Preserve provider map, inject bonsai-llama if needed
    if t["provider"] == "bonsai-llama":
        cfg.setdefault("providers", {})["bonsai-llama"] = {
            "api": t["base_url"],
            "default_model": "bonsai-8b",
            "models": ["bonsai-8b"],
            "name": "Bonsai (llama.cpp)",
        }
    cfg["model"] = {
        "api_key": t["api_key"],
        "base_url": t["base_url"],
        "default": action,
        "provider": t["provider"],
    }
    with open(CFG, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"set model.default={action} provider={t['provider']}")

if __name__ == "__main__":
    main()
