#!/usr/bin/env python3
"""Transparent HTTP proxy for capturing Honcho deriver request bodies.

Used once during prep phase to snapshot a real deriver LLM request that we
can then replay during contention benchmarks. Forwards everything to the
configured target, appends each seen body to a jsonl capture file.

Env vars:
  CAPTURE_TARGET  forward target (default: http://localhost:8080)
  CAPTURE_OUT     output jsonl path  (default: /tmp/honcho_captured.jsonl)
  CAPTURE_PORT    port to listen on   (default: 8090)

Dependencies: flask, requests (install with `pip install flask requests`
inside a venv if needed).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

try:
    from flask import Flask, Response, request
    import requests
except ImportError as exc:
    print(f"[capture_proxy] missing dep: {exc}", file=sys.stderr)
    print("install: pip install flask requests", file=sys.stderr)
    sys.exit(1)


TARGET = os.environ.get("CAPTURE_TARGET", "http://localhost:8080")
OUT_PATH = os.environ.get("CAPTURE_OUT", "/tmp/honcho_captured.jsonl")
PORT = int(os.environ.get("CAPTURE_PORT", "8090"))

app = Flask(__name__)


def _forward_headers() -> dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() != "host"}


def _record(kind: str, body: Any) -> None:
    entry = {
        "ts": time.time(),
        "kind": kind,
        "path": request.path,
        "headers": dict(request.headers),
        "body": body,
    }
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Response:
    body = request.get_json(silent=True)
    _record("chat_completions", body)
    upstream = requests.post(
        f"{TARGET}/v1/chat/completions",
        json=body,
        headers=_forward_headers(),
        stream=True,
        timeout=600,
    )
    return Response(
        upstream.iter_content(chunk_size=1024),
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type"),
    )


@app.route("/v1/embeddings", methods=["POST"])
def embeddings() -> Response:
    body = request.get_json(silent=True)
    _record("embeddings", body)
    upstream = requests.post(
        f"{TARGET}/v1/embeddings", json=body, headers=_forward_headers(), timeout=120
    )
    return Response(
        upstream.content,
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type"),
    )


@app.route("/<path:rest>", methods=["GET", "POST"])
def passthrough(rest: str) -> Response:
    fn = requests.post if request.method == "POST" else requests.get
    upstream = fn(
        f"{TARGET}/{rest}",
        data=request.get_data() if request.method == "POST" else None,
        headers=_forward_headers(),
        timeout=120,
    )
    return Response(
        upstream.content,
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(
        f"[capture_proxy] listening on :{PORT}, forwarding to {TARGET}, "
        f"logging to {OUT_PATH}",
        file=sys.stderr,
    )
    app.run(host="0.0.0.0", port=PORT, threaded=True)
