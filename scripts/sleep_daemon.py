#!/usr/bin/env python3
"""sleep_daemon: Honcho memory-consolidation pressure monitor

Runs alongside the Hermes runtime. Polls Honcho state every POLL_INTERVAL
seconds and triggers a "nap" (dream / memory consolidation) when:
  - the workspace has been idle for IDLE_TIMEOUT_MINUTES (default 10), OR
  - pending queue rows exceed PENDING_THRESHOLD (default 10), OR
  - pending message tokens exceed TOKEN_THRESHOLD (default 1000).

When a nap starts, the daemon:
  1. Detects the chat model Hermes is currently using (from
     ~/.hermes/config.yaml, falling back to Ollama /api/ps). This is logged
     for operator visibility only — the actual dream model is whatever
     honcho/config.toml's [dream.deduction_model_config] / [dream.induction_model_config]
     point at. In the current single-endpoint stack that's the shared
     qwen3.6 chat llama-server on :8080. Earlier two-endpoint deployments
     pointed dream at a separate memory-specialised model (Bonsai-8B) on
     a different port; that split is archived in experiments/bonsai-archive.md.
  2. Injects an English "system" message into the active Honcho session so
     the user sees the notification the next time they look at Hermes.
  3. Enqueues the dream via the deriver container.
  4. Waits for completion and injects an "awake" message.

This daemon is intended to be part of the Hermes runtime (systemd user
service, process-supervisor child, or a small wrapper around `hermes`).
It is NOT meant to be tailed interactively; see README
"Memory consolidation — the nap daemon" / "Debug" for foreground usage.

Environment overrides:
  HERMES_HOME            default ~/hermes-stack
  HONCHO_URL             default http://localhost:8000
  HONCHO_WS              default hermes (Honcho workspace)
  AI_PEER                default hermes (the AI peer that posts notifications)
  POLL_INTERVAL_SEC      default 30
  IDLE_TIMEOUT_MINUTES   default 10
  PENDING_THRESHOLD      default 10
  TOKEN_THRESHOLD        default 1000
  MIN_MINUTES_BETWEEN_NAPS  default 30
  OLLAMA_URL             default http://localhost:11434 (used for the
                         "detect currently loaded model" fallback only)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Config

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / "hermes-stack"))
HONCHO_DIR = HERMES_HOME / "honcho"
HONCHO_URL = os.environ.get("HONCHO_URL", "http://localhost:8000")
HONCHO_WS = os.environ.get("HONCHO_WS", "hermes")
AI_PEER = os.environ.get("AI_PEER", "hermes")
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "30"))
IDLE_TIMEOUT_MIN = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "10"))
PENDING_THRESHOLD = int(os.environ.get("PENDING_THRESHOLD", "10"))
TOKEN_THRESHOLD = int(os.environ.get("TOKEN_THRESHOLD", "1000"))
MIN_MINUTES_BETWEEN_NAPS = int(os.environ.get("MIN_MINUTES_BETWEEN_NAPS", "30"))
HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CONFIG_TOML = HONCHO_DIR / "config.toml"
STATE_FILE = HERMES_HOME / ".sleep_daemon_state.json"

logging.basicConfig(
    format="[sleep_daemon] %(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers

def _run(cmd: list[str], check: bool = True, input_str: str | None = None) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, input=input_str, check=check
    )
    return result.stdout


def _psql(sql: str) -> str:
    return _run(
        [
            "docker", "compose", "-f", str(HONCHO_DIR / "docker-compose.yml"),
            "exec", "-T", "database",
            "psql", "-U", "honcho", "-d", "honcho", "-At", "-c", sql,
        ],
        check=False,
    ).strip()


def _http_get(url: str, timeout: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None


def _http_post(url: str, body: dict, timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# -----------------------------------------------------------------------------
# Model detection: Hermes chat model + base URL
# Strategy:
#   1. Read ~/.hermes/config.yaml `model:` block
#   2. Fallback: query Ollama /api/ps for a loaded non-embedding model

@dataclass
class InferenceTarget:
    model: str
    base_url: str
    provider: str  # honcho PROVIDER slot: "custom" (Ollama/OpenAI-compatible) or "vllm"

    def key(self) -> str:
        return f"{self.provider}:{self.model}@{self.base_url}"


def detect_inference_target() -> InferenceTarget:
    # Try Hermes config first
    if HERMES_CONFIG.exists():
        try:
            text = HERMES_CONFIG.read_text()
            default = _yaml_scalar(text, ["model", "default"])
            base_url = _yaml_scalar(text, ["model", "base_url"])
            provider = _yaml_scalar(text, ["model", "provider"])
            if default and base_url:
                slot = "vllm" if provider == "vllm" else "custom"
                return InferenceTarget(default, base_url, slot)
        except Exception as e:
            log.warning("hermes config parse failed: %s", e)

    # Fallback: Ollama /api/ps
    raw = _http_get(f"{OLLAMA_URL}/api/ps")
    if raw:
        try:
            models = json.loads(raw).get("models") or []
            for m in models:
                name = (m.get("name") or "").strip()
                if not name or "embed" in name.lower():
                    continue
                return InferenceTarget(name, f"{OLLAMA_URL}/v1", "custom")
        except Exception as e:
            log.warning("ollama /api/ps parse failed: %s", e)

    raise RuntimeError("could not detect Hermes inference target")


def _yaml_scalar(text: str, path: list[str]) -> str | None:
    """Tiny, schema-specific YAML reader that can walk a 2-deep flat path.

    Good enough for Hermes's config structure:
      model:
        default: qwen3.5:9b
        base_url: http://localhost:11434/v1
        provider: custom
    Not a full YAML parser; avoids adding a dependency.
    """
    current_indent = -1
    target_depth = 0
    target = path[target_depth]
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if target_depth == 0:
            if stripped.startswith(f"{target}:"):
                rest = stripped.split(":", 1)[1].strip()
                if rest and len(path) == 1:
                    return _strip_scalar(rest)
                target_depth = 1
                target = path[target_depth]
                current_indent = indent
            continue
        if indent <= current_indent:
            return None  # left the block
        bare = stripped.lstrip()
        if bare.startswith(f"{target}:"):
            rest = bare.split(":", 1)[1].strip()
            return _strip_scalar(rest)
    return None


def _strip_scalar(v: str) -> str:
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (
        v.startswith("'") and v.endswith("'")
    ):
        v = v[1:-1]
    return v


# -----------------------------------------------------------------------------
# Honcho config.toml alignment
# Make sure [dream] PROVIDER / MODEL / BASE_URL match the detected target.

def align_dream_config(target: InferenceTarget) -> bool:
    """Return True if config was changed (deriver rebuild needed)."""
    if not CONFIG_TOML.exists():
        log.warning("config.toml not found at %s", CONFIG_TOML)
        return False
    text = CONFIG_TOML.read_text()

    want = {
        "PROVIDER": target.provider,
        "MODEL": target.model,
        "DEDUCTION_MODEL": target.model,
        "INDUCTION_MODEL": target.model,
    }

    new_text = text
    for key, want_value in want.items():
        new_text = _replace_in_section(new_text, "dream", key, want_value)

    if new_text == text:
        return False
    CONFIG_TOML.write_text(new_text)
    return True


def _replace_in_section(text: str, section: str, key: str, value: str) -> str:
    """Replace/insert `KEY = "value"` inside the [section] block of a TOML file.

    Only handles string-quoted values — sufficient for PROVIDER/MODEL fields.
    """
    lines = text.splitlines()
    in_section = False
    section_start = section_end = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1].strip()
            if in_section:
                section_end = i
                break
            if current == section:
                in_section = True
                section_start = i
    if in_section and section_end == -1:
        section_end = len(lines)
    if not in_section:
        return text

    replaced = False
    for i in range(section_start + 1, section_end):
        m = re.match(rf"^\s*{re.escape(key)}\s*=", lines[i])
        if m:
            lines[i] = f'{key} = "{value}"'
            replaced = True
            break
    if not replaced:
        lines.insert(section_end, f'{key} = "{value}"')
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def rebuild_deriver() -> None:
    log.info("rebuilding deriver to apply dream config changes...")
    _run(
        [
            "docker", "compose", "-f", str(HONCHO_DIR / "docker-compose.yml"),
            "up", "-d", "--build", "deriver",
        ],
        check=False,
    )


# -----------------------------------------------------------------------------
# Queue / activity inspection

@dataclass
class PressureSnapshot:
    pending: int
    pending_tokens: int
    seconds_since_last_msg: int
    dream_running: bool
    last_session: str | None
    doc_count: int

    def idle(self) -> bool:
        return self.seconds_since_last_msg >= IDLE_TIMEOUT_MIN * 60

    def pending_high(self) -> bool:
        return self.pending > PENDING_THRESHOLD

    def tokens_high(self) -> bool:
        return self.pending_tokens > TOKEN_THRESHOLD

    def should_trigger(self) -> str | None:
        if self.dream_running:
            return None
        if self.pending_high():
            return "pending"
        if self.tokens_high():
            return "tokens"
        if self.idle() and self.doc_count > 0:
            return "idle"
        return None


def snapshot() -> PressureSnapshot:
    pending = int(_psql(
        f"SELECT count(*) FROM queue WHERE workspace_name='{HONCHO_WS}' "
        "AND NOT processed AND task_type='representation';"
    ) or 0)
    tokens = int(_psql(
        f"SELECT COALESCE(SUM(m.token_count), 0) FROM queue q "
        f"JOIN messages m ON q.message_id = m.id "
        f"WHERE q.workspace_name='{HONCHO_WS}' AND NOT q.processed "
        f"AND q.task_type='representation';"
    ) or 0)
    last_msg_age = _psql(
        f"SELECT EXTRACT(EPOCH FROM (now() - max(created_at)))::int "
        f"FROM messages WHERE workspace_name='{HONCHO_WS}';"
    )
    seconds = int(last_msg_age) if last_msg_age and last_msg_age != "" else 10**6
    dream_running = bool(_psql(
        "SELECT 1 FROM active_queue_sessions WHERE work_unit_key LIKE 'dream:%' LIMIT 1;"
    ))
    last_session = _psql(
        f"SELECT name FROM sessions WHERE workspace_name='{HONCHO_WS}' "
        "ORDER BY created_at DESC LIMIT 1;"
    ) or None
    doc_count = int(_psql(
        f"SELECT count(*) FROM documents WHERE workspace_name='{HONCHO_WS}';"
    ) or 0)
    return PressureSnapshot(
        pending, tokens, seconds, dream_running, last_session, doc_count,
    )


# -----------------------------------------------------------------------------
# System-message injection into Hermes session

def inject_system_message(session_name: str, text: str) -> None:
    url = f"{HONCHO_URL}/v3/workspaces/{HONCHO_WS}/sessions/{session_name}/messages"
    body = {
        "messages": [{
            "peer_id": AI_PEER,
            "content": f"[System] {text}",
            "metadata": {"system_notification": True, "source": "sleep_daemon"},
        }]
    }
    code, resp = _http_post(url, body)
    if 200 <= code < 300:
        log.info("injected: %s", text)
    else:
        log.warning("inject failed (%s): %s", code, resp[:200])


# -----------------------------------------------------------------------------
# Dream triggering & wait loop

def trigger_dream(observer: str = "baba", observed: str = "baba") -> bool:
    """Enqueue an omni dream via the deriver container.

    Returns True if enqueue succeeded, False otherwise.
    """
    dream_py = (
        "import asyncio\n"
        "from src.deriver.enqueue import enqueue_dream\n"
        "from src import schemas\n"
        "asyncio.run(enqueue_dream(\n"
        f"    workspace_name={HONCHO_WS!r},\n"
        f"    observer={observer!r},\n"
        f"    observed={observed!r},\n"
        "    dream_type=schemas.DreamType.OMNI,\n"
        "    document_count=0,\n"
        "))\n"
    )
    try:
        out = _run(
            [
                "docker", "compose", "-f", str(HONCHO_DIR / "docker-compose.yml"),
                "exec", "-T", "deriver",
                "python3", "-c", dream_py,
            ],
            check=False,
        )
        log.info("dream enqueued (%s)", out.strip()[:120] or "ok")
        return True
    except Exception as e:
        log.error("dream enqueue failed: %s", e)
        return False


def wait_for_dream_to_clear(timeout_sec: int = 900) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        if not bool(_psql(
            "SELECT 1 FROM active_queue_sessions WHERE work_unit_key LIKE 'dream:%' LIMIT 1;"
        )):
            still_pending = int(_psql(
                "SELECT count(*) FROM queue WHERE task_type='dream' AND NOT processed;"
            ) or 0)
            if still_pending == 0:
                return True
        time.sleep(5)
    return False


# -----------------------------------------------------------------------------
# Main loop

def loop() -> None:
    state = _load_state()
    try:
        target = detect_inference_target()
        log.info(
            "detected Hermes chat target: %s (dream model is whatever "
            "honcho/config.toml [dream.*_model_config] points at)",
            target.key(),
        )
        state["last_chat_target"] = target.key()
    except Exception as e:
        log.warning("could not detect Hermes chat target: %s", e)
    _save_state(state)

    while True:
        try:
            snap = snapshot()
            log.debug(
                "snap: pending=%d tokens=%d idle=%ds dream_running=%s",
                snap.pending, snap.pending_tokens, snap.seconds_since_last_msg,
                snap.dream_running,
            )

            # Detect honcho's auto-scheduled dream firing and notify
            if snap.dream_running and not state.get("last_dream_notified"):
                if snap.last_session:
                    inject_system_message(
                        snap.last_session,
                        "💤 My memories piled up — taking a short nap to sort them out.",
                    )
                state["last_dream_notified"] = True
                _save_state(state)

            # Dream finished — send awake notification
            if not snap.dream_running and state.get("last_dream_notified"):
                if snap.last_session:
                    inject_system_message(
                        snap.last_session,
                        "☕ Awake and refreshed. Memory consolidated.",
                    )
                state["last_dream_notified"] = False
                state["last_nap_at"] = datetime.now(timezone.utc).isoformat()
                _save_state(state)

            # Our own pressure trigger (on top of honcho's auto-scheduler)
            reason = snap.should_trigger()
            if reason and _can_nap_now(state):
                if snap.last_session:
                    msg = _pressure_message(reason, snap)
                    inject_system_message(snap.last_session, msg)
                if trigger_dream():
                    state["last_dream_notified"] = True
                    _save_state(state)
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(POLL_INTERVAL_SEC)


def _can_nap_now(state: dict) -> bool:
    last = state.get("last_nap_at")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        return (
            datetime.now(timezone.utc) - last_dt
        ).total_seconds() / 60 >= MIN_MINUTES_BETWEEN_NAPS
    except Exception:
        return True


def _pressure_message(reason: str, snap: PressureSnapshot) -> str:
    if reason == "pending":
        return (
            f"🥱 My pending queue is backing up ({snap.pending} items). "
            "Getting drowsy — going to take a short nap to catch up."
        )
    if reason == "tokens":
        return (
            f"🥱 Lots of unprocessed content ({snap.pending_tokens} tokens). "
            "Getting drowsy — going to take a short nap to consolidate."
        )
    return (
        f"💤 Been idle for {snap.seconds_since_last_msg // 60} minutes. "
        "Taking a nap to consolidate memories."
    )


if __name__ == "__main__":
    try:
        loop()
    except KeyboardInterrupt:
        sys.exit(0)
