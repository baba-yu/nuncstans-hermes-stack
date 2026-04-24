#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tomlkit>=0.13",
#   "ruamel.yaml>=0.18",
#   "httpx>=0.27",
#   "questionary>=2.0",
# ]
# ///
"""switch-endpoints.py — interactive LLM endpoint/model switcher for the
hermes-stack. Rewrites honcho/config.toml, scripts/llama-services.conf, and
~/.hermes/config.yaml in a coherent snapshot/rollback envelope.

Subcommands:
  (default)            interactive switch flow
  --dry-run            compute + print diffs, no writes, no restarts
  --rollback           restore from the most recent snapshot
  --restore <id>       restore from a specific snapshot id
  --list-snapshots     show the 10 most recent snapshots with summaries

Safety: before the first write the script takes a coherent snapshot of all
affected files under ~/.local/state/nuncstans-hermes-stack/endpoint-snapshots/
(override via $HERMES_STATE_DIR). On any
write- or restart- error it promotes auto-rollback (atomic os.replace of each
file, status=rolled_back in manifest). LRU pruned to 10 entries at startup.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import httpx
import questionary
import tomlkit
from ruamel.yaml import YAML

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/home/baba-y/nuncstans-hermes-stack")
HONCHO_TOML = Path(os.environ.get("HONCHO_TOML_OVERRIDE", REPO_ROOT / "honcho" / "config.toml"))
LLAMA_CONF = Path(os.environ.get("LLAMA_CONF_OVERRIDE", REPO_ROOT / "scripts" / "llama-services.conf"))
LLAMA_RESTART_SH = REPO_ROOT / "scripts" / "llama-services.sh"
HONCHO_COMPOSE = REPO_ROOT / "honcho" / "docker-compose.yml"
HERMES_YAML = Path(os.environ.get("HERMES_YAML_OVERRIDE", Path.home() / ".hermes" / "config.yaml"))

_DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "nuncstans-hermes-stack"
SNAPSHOT_ROOT = Path(
    os.environ.get("HERMES_STATE_DIR") or _DEFAULT_STATE_DIR
) / "endpoint-snapshots"
SNAPSHOT_KEEP = 10

# The 9 chat model_config blocks inside honcho/config.toml.
CHAT_BLOCKS: tuple[tuple[str, ...], ...] = (
    ("deriver", "model_config"),
    ("dialectic", "levels", "minimal", "model_config"),
    ("dialectic", "levels", "low", "model_config"),
    ("dialectic", "levels", "medium", "model_config"),
    ("dialectic", "levels", "high", "model_config"),
    ("dialectic", "levels", "max", "model_config"),
    ("summary", "model_config"),
    ("dream", "deduction_model_config"),
    ("dream", "induction_model_config"),
)
EMBED_BLOCK: tuple[str, ...] = ("embedding", "model_config")

OLLAMA_PORT_HINT = 11434
CTX_HEADROOM = 20_000  # reserve output+system tokens when capping GET_CONTEXT_MAX_TOKENS
CHAT_CTX_HARD_CAP = 131_072

USE_COLOR = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Errors + small helpers
# ---------------------------------------------------------------------------


class FatalError(RuntimeError):
    """Unrecoverable error — main() catches and exits non-zero."""


def _c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if USE_COLOR else s


def cprint(level: str, msg: str) -> None:
    prefix = {
        "info": _c("36", "[info]"),
        "warn": _c("33", "[warn]"),
        "err": _c("31", "[err ]"),
        "ok": _c("32", "[ ok ]"),
        "step": _c("35", "==>"),
    }.get(level, "[info]")
    print(f"{prefix} {msg}")


def render_diff(old: str, new: str, path: str) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    if not lines:
        return ""
    if not USE_COLOR:
        return "".join(lines)
    out: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            out.append(_c("1", line))
        elif line.startswith("@@"):
            out.append(_c("36", line))
        elif line.startswith("+"):
            out.append(_c("32", line))
        elif line.startswith("-"):
            out.append(_c("31", line))
        else:
            out.append(line)
    return "".join(out)


def atomic_write(path: Path, content: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile.mkstemp creates files with mode 0600 by default — that bit
    # us hard when honcho/config.toml was bind-mounted into the api /
    # deriver containers (they run as UID 100 'app'; 0600 means only the
    # host owner UID 1000 can read, so the container got Permission
    # denied and silently fell back to defaults). Force 0644 so
    # containers running as arbitrary UIDs can still read the file.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelMeta:
    n_ctx_train: int
    n_embd: int
    n_params: int | None
    is_moe: bool | None
    arch: str | None


@dataclass(slots=True)
class ChatParams:
    hf_spec: str
    alias: str
    ctx: int
    ngl: int
    is_moe: bool
    reasoning_off: bool
    parallel: int


@dataclass(slots=True)
class EmbedParams:
    alias: str
    dim: int
    max_input_tokens: int
    n_ctx_train: int


@dataclass(slots=True)
class EndpointChoice:
    base_url_host: str    # user-visible form (may be localhost)
    base_url_docker: str  # rewritten for config.toml
    model: str
    meta: ModelMeta | None


@dataclass(slots=True)
class Snapshot:
    dir: Path
    manifest: dict[str, Any]

    @property
    def id(self) -> str:
        return self.dir.name


@dataclass(slots=True)
class PlannedChanges:
    honcho_chat: EndpointChoice | None = None
    honcho_embed: EndpointChoice | None = None
    hermes: EndpointChoice | None = None
    llama_chat_params: ChatParams | None = None
    llama_embed_params: EmbedParams | None = None
    caps: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def needs_honcho_restart(self) -> bool:
        return bool(self.honcho_chat or self.honcho_embed or self.caps)

    def needs_llama_restart(self) -> bool:
        # honcho_chat also triggers a restart because the gatekeeper daemon
        # (started by llama-services.sh) auto-syncs its classifier endpoint
        # to the new chat URL via GK_LLM_URL / GK_LLM_MODEL in the conf.
        return bool(
            self.llama_chat_params or self.llama_embed_params or self.honcho_chat
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def to_docker_url(url: str) -> str:
    """Rewrite localhost/127.0.0.1 to host.docker.internal. Keep everything else."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = parsed.hostname or ""
    if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
        return url
    port = parsed.port
    new_netloc = "host.docker.internal" + (f":{port}" if port else "")
    if parsed.username:
        userinfo = parsed.username + (f":{parsed.password}" if parsed.password else "")
        new_netloc = f"{userinfo}@{new_netloc}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def looks_like_ollama(url: str) -> bool:
    try:
        return (urlparse(url).port or 0) == OLLAMA_PORT_HINT
    except ValueError:
        return False


def normalize_base(url: str) -> str:
    """Strip trailing slash; ensure we have a full URL (scheme://host[:port][/path])."""
    if not url:
        return url
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = "http://" + url
    return url


# ---------------------------------------------------------------------------
# Probing — /v1/models and ollama /api/show
# ---------------------------------------------------------------------------


def probe_models(base_url: str, *, timeout: float = 5.0) -> list[dict[str, Any]] | None:
    """GET {base}/models. Returns list of model objects or None on failure."""
    url = normalize_base(base_url) + "/models"
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        cprint("warn", f"GET {url} failed: {e}")
        return None
    try:
        data = r.json().get("data") or []
        if not isinstance(data, list):
            return None
        return data
    except Exception as e:  # noqa: BLE001
        cprint("warn", f"parsing {url} response failed: {e}")
        return None


def probe_ollama_show(base_url: str, model: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """POST {host}:11434/api/show {"model": ...}. base_url is a /v1-style URL."""
    try:
        parsed = urlparse(normalize_base(base_url))
    except ValueError:
        return None
    host = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"
    show_url = f"{scheme}://{host}:{OLLAMA_PORT_HINT}/api/show"
    try:
        r = httpx.post(show_url, json={"model": model}, timeout=timeout)
        if r.status_code == 404:
            return {"__404__": True}
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        cprint("warn", f"POST {show_url} failed: {e}")
        return None


def probe_model_meta(base_url: str, model: str) -> ModelMeta | None:
    """Unified metadata probe across llama-server and ollama."""
    models = probe_models(base_url)
    meta_blob: dict[str, Any] | None = None
    if models:
        for m in models:
            if m.get("id") == model:
                meta_blob = m.get("meta") if isinstance(m.get("meta"), dict) else None
                break

    if meta_blob:  # llama-server path
        return ModelMeta(
            n_ctx_train=int(meta_blob.get("n_ctx_train") or 0),
            n_embd=int(meta_blob.get("n_embd") or 0),
            n_params=int(meta_blob["n_params"]) if meta_blob.get("n_params") else None,
            is_moe=None,
            arch=None,
        )

    # ollama path (models endpoint returned entries but no meta, or /api/show is available)
    if looks_like_ollama(base_url) or models is None:
        show = probe_ollama_show(base_url, model)
        if show and not show.get("__404__"):
            mi = show.get("model_info") or {}
            arch = mi.get("general.architecture") or show.get("details", {}).get("family")
            ctx = 0
            embd = 0
            expert_count = 0
            if arch:
                ctx = int(mi.get(f"{arch}.context_length") or 0)
                embd = int(mi.get(f"{arch}.embedding_length") or 0)
                expert_count = int(mi.get(f"{arch}.expert_count") or 0)
            details = show.get("details") or {}
            param_size = details.get("parameter_size") or ""
            n_params = _parse_param_size(param_size)
            is_moe = (expert_count or 0) > 0 or (arch and "moe" in str(arch).lower())
            return ModelMeta(
                n_ctx_train=ctx,
                n_embd=embd,
                n_params=n_params,
                is_moe=bool(is_moe),
                arch=str(arch) if arch else None,
            )
        elif show and show.get("__404__"):
            cprint(
                "warn",
                f"ollama does not know about '{model}'. Run: ollama pull {model}",
            )
            return None
    return None


def _parse_param_size(s: str) -> int | None:
    """'35B' -> 35_000_000_000. Returns None on parse failure."""
    if not s:
        return None
    m = re.match(r"\s*([\d.]+)\s*([BMK]?)", s, re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    suffix = (m.group(2) or "").upper()
    mult = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000, "": 1_000_000_000}[suffix]
    return int(val * mult)


# ---------------------------------------------------------------------------
# Derivation of chat / embed params
# ---------------------------------------------------------------------------


def derive_chat_params(
    meta: ModelMeta,
    *,
    current: ChatParams,
    hf_spec: str,
    alias: str,
) -> ChatParams:
    ctx = min(meta.n_ctx_train or current.ctx, CHAT_CTX_HARD_CAP) if meta.n_ctx_train else current.ctx
    params = meta.n_params
    # ngl default: 99 for <=15B, otherwise keep current / ask later.
    if params is not None and params <= 15_000_000_000:
        ngl = 99
    else:
        ngl = current.ngl  # caller will prompt if needed
    is_moe = meta.is_moe if meta.is_moe is not None else current.is_moe
    reasoning_off = bool(meta.arch and meta.arch.lower().startswith("qwen3"))
    # parallel: default 2; suggest 1 for dense models >= 35B
    parallel = 2
    if params is not None and params >= 35_000_000_000 and not is_moe:
        parallel = 1
    return ChatParams(
        hf_spec=hf_spec,
        alias=alias,
        ctx=ctx,
        ngl=ngl,
        is_moe=bool(is_moe),
        reasoning_off=reasoning_off,
        parallel=parallel,
    )


def derive_embed_params(meta: ModelMeta, *, alias: str) -> EmbedParams:
    return EmbedParams(
        alias=alias,
        dim=meta.n_embd or 0,
        max_input_tokens=meta.n_ctx_train or 0,
        n_ctx_train=meta.n_ctx_train or 0,
    )


# ---------------------------------------------------------------------------
# Read-side helpers for current state
# ---------------------------------------------------------------------------


def read_honcho_toml() -> tomlkit.TOMLDocument:
    return tomlkit.parse(HONCHO_TOML.read_text(encoding="utf-8"))


def toml_get(doc: Any, path: tuple[str, ...]) -> Any:
    cur = doc
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if hasattr(cur, "get") else None
    return cur


def current_honcho_chat(doc: tomlkit.TOMLDocument) -> tuple[str, str]:
    """Return (base_url, model) from the deriver block (representative)."""
    mc = toml_get(doc, ("deriver", "model_config"))
    if not mc:
        return ("", "")
    base = (toml_get(mc, ("overrides", "base_url")) or "")
    return (str(base), str(mc.get("model") or ""))


def current_honcho_embed(doc: tomlkit.TOMLDocument) -> tuple[str, str]:
    mc = toml_get(doc, EMBED_BLOCK)
    if not mc:
        return ("", "")
    base = (toml_get(mc, ("overrides", "base_url")) or "")
    return (str(base), str(mc.get("model") or ""))


def current_caps(doc: tomlkit.TOMLDocument) -> dict[str, int]:
    return {
        "app.GET_CONTEXT_MAX_TOKENS": int(toml_get(doc, ("app", "GET_CONTEXT_MAX_TOKENS")) or 0),
        "dialectic.MAX_INPUT_TOKENS": int(toml_get(doc, ("dialectic", "MAX_INPUT_TOKENS")) or 0),
        "embedding.VECTOR_DIMENSIONS": int(toml_get(doc, ("embedding", "VECTOR_DIMENSIONS")) or 0),
        "vector_store.DIMENSIONS": int(toml_get(doc, ("vector_store", "DIMENSIONS")) or 0),
        "embedding.MAX_INPUT_TOKENS": int(toml_get(doc, ("embedding", "MAX_INPUT_TOKENS")) or 0),
    }


def read_llama_conf() -> dict[str, str]:
    """Parse KEY=VALUE lines (values possibly double-quoted). Preserves only keys."""
    out: dict[str, str] = {}
    if not LLAMA_CONF.exists():
        return out
    for raw in LLAMA_CONF.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        out[key] = val
    return out


def current_chat_params_from_conf() -> ChatParams:
    c = read_llama_conf()
    return ChatParams(
        hf_spec=c.get("CHAT_HF_SPEC", ""),
        alias=c.get("CHAT_ALIAS", "qwen3.6-test"),
        ctx=int(c.get("CHAT_CTX") or CHAT_CTX_HARD_CAP),
        ngl=int(c.get("CHAT_NGL") or 99),
        is_moe=c.get("CHAT_IS_MOE", "0") == "1",
        reasoning_off=c.get("CHAT_REASONING_OFF", "0") == "1",
        parallel=int(c.get("CHAT_PARALLEL") or 2),
    )


def read_hermes_config() -> dict[str, Any] | None:
    if not HERMES_YAML.exists():
        return None
    yaml = YAML(typ="rt")
    with HERMES_YAML.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def current_hermes_summary() -> tuple[str, str]:
    data = read_hermes_config()
    if not data:
        return ("", "")
    model = data.get("model") or {}
    return (str(model.get("base_url", "")), str(model.get("default", "")))


def ollama_context_ceiling() -> int | None:
    try:
        r = subprocess.run(
            ["systemctl", "cat", "ollama.service"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        m = re.search(r'OLLAMA_CONTEXT_LENGTH=(\d+)', line)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Snapshot / rollback
# ---------------------------------------------------------------------------


SNAPSHOT_FILES: tuple[tuple[str, Path], ...] = (
    ("config.toml", HONCHO_TOML),
    ("llama-services.conf", LLAMA_CONF),
    ("hermes-config.yaml", HERMES_YAML),
)


def _snapshot_id() -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}.{os.getpid()}"


def prune_snapshots(keep: int = SNAPSHOT_KEEP) -> int:
    """Remove oldest directories when count > keep. Returns delete count."""
    if not SNAPSHOT_ROOT.exists():
        return 0
    dirs = sorted(
        (p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if len(dirs) <= keep:
        return 0
    to_delete = dirs[: len(dirs) - keep]
    n = 0
    for d in to_delete:
        try:
            shutil.rmtree(d)
            n += 1
        except OSError as e:
            cprint("warn", f"failed to prune {d}: {e}")
    return n


def list_snapshots() -> list[Snapshot]:
    if not SNAPSHOT_ROOT.exists():
        return []
    out: list[Snapshot] = []
    dirs = sorted(
        (p for p in SNAPSHOT_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in dirs:
        mf_path = d / "manifest.json"
        try:
            mf = json.loads(mf_path.read_text(encoding="utf-8")) if mf_path.exists() else {}
        except json.JSONDecodeError:
            mf = {"__corrupt__": True}
        out.append(Snapshot(dir=d, manifest=mf))
    return out


def create_snapshot(user_choices: dict[str, Any], planned_restarts: list[str]) -> Snapshot:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        st = shutil.disk_usage(SNAPSHOT_ROOT)
        if st.free < 5 * 1024 * 1024:
            raise FatalError("not enough free space under ~/.local/state for snapshot")
    except OSError as e:
        raise FatalError(f"cannot stat snapshot dir: {e}") from e

    snap_dir = SNAPSHOT_ROOT / _snapshot_id()
    snap_dir.mkdir(parents=True, exist_ok=False)

    snapshotted: list[str] = []
    for name, src in SNAPSHOT_FILES:
        if src.exists():
            shutil.copy2(src, snap_dir / name)
            snapshotted.append(name)

    prev = None
    existing = list_snapshots()
    for s in existing:
        if s.dir == snap_dir:
            continue
        prev = s.id
        break

    manifest = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "user_choices": user_choices,
        "files_snapshotted": snapshotted,
        "planned_restarts": planned_restarts,
        "status": "snapshot_only",
        "errors": [],
        "previous_snapshot": prev,
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Snapshot(dir=snap_dir, manifest=manifest)


def finalize_snapshot(snap: Snapshot, status: str, errors: list[str] | None = None) -> None:
    snap.manifest["status"] = status
    if errors:
        snap.manifest["errors"] = errors
    (snap.dir / "manifest.json").write_text(
        json.dumps(snap.manifest, indent=2) + "\n", encoding="utf-8"
    )


def restore_snapshot(snap: Snapshot, *, dry_run: bool = False) -> None:
    for name, dest in SNAPSHOT_FILES:
        src = snap.dir / name
        if not src.exists():
            continue
        cprint("info", f"restoring {dest} <- {src}")
        if dry_run:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # stage a tmp then os.replace for atomicity
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dest.name}.", suffix=".restore", dir=str(dest.parent)
        )
        os.close(fd)
        shutil.copy2(src, tmp_name)
        os.replace(tmp_name, dest)


def auto_rollback(snap: Snapshot, reason: str, errors: list[str]) -> None:
    cprint("err", f"auto-rollback: {reason}")
    try:
        restore_snapshot(snap)
        finalize_snapshot(snap, "rolled_back", errors + [f"trigger: {reason}"])
        cprint("ok", f"files restored from snapshot {snap.id}")
        cprint(
            "info",
            "you may need to manually re-run restarts (docker compose / llama-services.sh) "
            "to pick up the restored files.",
        )
    except Exception as e:  # noqa: BLE001
        finalize_snapshot(
            snap, "applied_with_errors",
            errors + [f"rollback failure: {e}", f"original trigger: {reason}"],
        )
        cprint("err", f"ROLLBACK FAILED: {e}. Snapshot at {snap.dir}")


# ---------------------------------------------------------------------------
# Config writers
# ---------------------------------------------------------------------------


def update_honcho_toml(
    plan: PlannedChanges, *, dry_run: bool
) -> tuple[str, str]:
    """Return (old_text, new_text). Writes atomically unless dry_run."""
    old_text = HONCHO_TOML.read_text(encoding="utf-8")
    doc = tomlkit.parse(old_text)

    if plan.honcho_chat:
        for path in CHAT_BLOCKS:
            block = toml_get(doc, path)
            if block is None:
                continue
            block["model"] = plan.honcho_chat.model
            overrides = block.get("overrides")
            if overrides is None:
                continue
            overrides["base_url"] = plan.honcho_chat.base_url_docker

    if plan.honcho_embed:
        block = toml_get(doc, EMBED_BLOCK)
        if block is not None:
            block["model"] = plan.honcho_embed.model
            overrides = block.get("overrides")
            if overrides is not None:
                overrides["base_url"] = plan.honcho_embed.base_url_docker

    # Apply caps (co-moving knobs)
    if "app.GET_CONTEXT_MAX_TOKENS" in plan.caps and "app" in doc:
        doc["app"]["GET_CONTEXT_MAX_TOKENS"] = plan.caps["app.GET_CONTEXT_MAX_TOKENS"]
    if "dialectic.MAX_INPUT_TOKENS" in plan.caps and "dialectic" in doc:
        doc["dialectic"]["MAX_INPUT_TOKENS"] = plan.caps["dialectic.MAX_INPUT_TOKENS"]
    if "embedding.VECTOR_DIMENSIONS" in plan.caps and "embedding" in doc:
        doc["embedding"]["VECTOR_DIMENSIONS"] = plan.caps["embedding.VECTOR_DIMENSIONS"]
    if "vector_store.DIMENSIONS" in plan.caps and "vector_store" in doc:
        doc["vector_store"]["DIMENSIONS"] = plan.caps["vector_store.DIMENSIONS"]
    if "embedding.MAX_INPUT_TOKENS" in plan.caps and "embedding" in doc:
        doc["embedding"]["MAX_INPUT_TOKENS"] = plan.caps["embedding.MAX_INPUT_TOKENS"]

    new_text = tomlkit.dumps(doc)
    if new_text != old_text:
        atomic_write(HONCHO_TOML, new_text, dry_run=dry_run)
    return old_text, new_text


_LLAMA_KEYS: tuple[str, ...] = (
    "CHAT_HF_SPEC", "CHAT_ALIAS", "CHAT_CTX", "CHAT_NGL", "CHAT_IS_MOE",
    "CHAT_REASONING_OFF", "CHAT_PARALLEL", "EMBED_BLOB", "EMBED_ALIAS", "EMBED_NGL",
    "GK_LLM_URL", "GK_LLM_MODEL",
)


def _gk_base_url(chat_base_url_host: str) -> str:
    """Strip the trailing '/v1' that Honcho/Hermes base_urls carry — the
    gatekeeper daemon appends '/v1/chat/completions' itself, so GK_LLM_URL
    is the plain '<scheme>://<host>:<port>' root."""
    u = chat_base_url_host.rstrip("/")
    return u[:-3].rstrip("/") if u.endswith("/v1") else u


def _engine_of_url(url: str) -> str:
    """Classify an endpoint URL into 'llama-server' / 'ollama' / 'custom'.
    Port-based: 8080 / 8081 => llama-server; OLLAMA_PORT_HINT (11434) => ollama.
    """
    try:
        port = urlparse(normalize_base(url)).port or 0
    except ValueError:
        return "custom"
    if port in (8080, 8081):
        return "llama-server"
    if port == OLLAMA_PORT_HINT:
        return "ollama"
    return "custom"


def _embed_endpoint_for_engine(engine: str) -> tuple[str, str] | None:
    """Canonical (host-form URL, 768-dim model id) for a given engine, or
    None when the engine is custom/unknown. nomic-embed-text is the same
    blob behind both aliases on ollama; we pick the openai/... name to
    match the existing Honcho config value and minimize the TOML diff."""
    if engine == "llama-server":
        return ("http://localhost:8081/v1", "openai/text-embedding-3-small")
    if engine == "ollama":
        return (
            f"http://localhost:{OLLAMA_PORT_HINT}/v1",
            "openai/text-embedding-3-small:latest",
        )
    return None


def _ollama_models_in_use(
    chat_url: str, chat_model: str,
    embed_url: str, embed_model: str,
    hermes_url: str, hermes_model: str,
) -> set[str]:
    """Set of ollama-resident model ids currently in use across the three
    consumer axes (Honcho chat, Honcho embed, Hermes). Axes pointed at
    non-ollama engines contribute nothing."""
    used: set[str] = set()
    if _engine_of_url(chat_url) == "ollama":
        used.add(chat_model)
    if _engine_of_url(embed_url) == "ollama":
        used.add(embed_model)
    if _engine_of_url(hermes_url) == "ollama":
        used.add(hermes_model)
    return used


def _ollama_unload_targets(
    plan: PlannedChanges,
    cur_chat: tuple[str, str],
    cur_embed: tuple[str, str],
    cur_hermes: tuple[str, str],
) -> list[str]:
    """Which ollama models were in use before the switch but will not be
    after it? Those are candidates for POST /api/generate keep_alive=0
    so ollama releases their VRAM without us needing to stop the systemd
    service (which would require sudo and affect unrelated clients)."""
    before = _ollama_models_in_use(*cur_chat, *cur_embed, *cur_hermes)

    def resolve(
        ep: EndpointChoice | None, fallback: tuple[str, str]
    ) -> tuple[str, str]:
        return (ep.base_url_host, ep.model) if ep is not None else fallback

    new_chat = resolve(plan.honcho_chat, cur_chat)
    new_embed = resolve(plan.honcho_embed, cur_embed)
    new_hermes = resolve(plan.hermes, cur_hermes)
    after = _ollama_models_in_use(*new_chat, *new_embed, *new_hermes)
    return sorted(before - after)


def ollama_unload_model(
    model_id: str, *, base_host: str = "http://localhost:11434", dry_run: bool
) -> bool:
    """POST /api/generate with keep_alive=0 — ollama unloads the model
    from VRAM and returns {done_reason:"unload"}. Returns True on success
    (non-fatal: caller may log and continue if False)."""
    url = f"{base_host.rstrip('/')}/api/generate"
    cprint("step", f"ollama unload {model_id} via {url}")
    if dry_run:
        return True
    try:
        r = httpx.post(
            url,
            json={"model": model_id, "keep_alive": 0, "prompt": "", "stream": False},
            timeout=15.0,
        )
        r.raise_for_status()
        done_reason = r.json().get("done_reason", "")
        if done_reason == "unload":
            cprint("ok", f"ollama: {model_id} unloaded")
            return True
        cprint(
            "warn",
            f"ollama unload {model_id}: unexpected done_reason={done_reason!r}",
        )
        return False
    except Exception as e:  # noqa: BLE001
        cprint("warn", f"ollama unload {model_id} failed: {e}")
        return False


def _format_conf_value(key: str, value: str) -> str:
    # Quote string-ish values; leave ints unquoted.
    if re.fullmatch(r"-?\d+", value):
        return value
    return f'"{value}"'


def rewrite_llama_conf_text(old_text: str, updates: dict[str, str]) -> str:
    """Line-based rewrite. Unknown lines preserved; matching keys get new values."""
    out_lines: list[str] = []
    seen: set[str] = set()
    for raw in old_text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
        if not m or line.lstrip().startswith("#"):
            out_lines.append(raw)
            continue
        key = m.group(1)
        if key in updates:
            new_val = _format_conf_value(key, updates[key])
            # preserve leading whitespace if any
            leading = re.match(r"^(\s*)", raw).group(1)
            end = "\n" if raw.endswith("\n") else ""
            out_lines.append(f"{leading}{key}={new_val}{end}")
            seen.add(key)
        else:
            out_lines.append(raw)
    # Keys not present in file but requested: append at end with a blank sentinel.
    missing = [k for k in updates if k not in seen]
    if missing:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines.append("\n")
        out_lines.append("# --- appended by switch-endpoints.py ---\n")
        for k in missing:
            out_lines.append(f"{k}={_format_conf_value(k, updates[k])}\n")
    return "".join(out_lines)


def update_llama_conf(
    plan: PlannedChanges, *, dry_run: bool
) -> tuple[str, str]:
    if not LLAMA_CONF.exists():
        return "", ""
    old_text = LLAMA_CONF.read_text(encoding="utf-8")
    updates: dict[str, str] = {}

    if plan.llama_chat_params is not None:
        p = plan.llama_chat_params
        updates.update({
            "CHAT_HF_SPEC": p.hf_spec,
            "CHAT_ALIAS": p.alias,
            "CHAT_CTX": str(p.ctx),
            "CHAT_NGL": str(p.ngl),
            "CHAT_IS_MOE": "1" if p.is_moe else "0",
            "CHAT_REASONING_OFF": "1" if p.reasoning_off else "0",
            "CHAT_PARALLEL": str(p.parallel),
        })
    if plan.llama_embed_params is not None:
        # Only alias is typically changed here; EMBED_BLOB requires manual spec.
        updates["EMBED_ALIAS"] = plan.llama_embed_params.alias

    # Auto-sync the gatekeeper classifier to the Honcho chat endpoint so
    # the gk daemon uses the same engine as the rest of the stack (one
    # source of truth). Users who want a dedicated small classifier can
    # hand-edit llama-services.conf — but the switcher will rewrite
    # these keys on the next Axis A run.
    if plan.honcho_chat is not None:
        updates["GK_LLM_URL"] = _gk_base_url(plan.honcho_chat.base_url_host)
        updates["GK_LLM_MODEL"] = plan.honcho_chat.model

    if not updates:
        return old_text, old_text

    new_text = rewrite_llama_conf_text(old_text, updates)
    if new_text != old_text:
        atomic_write(LLAMA_CONF, new_text, dry_run=dry_run)
    return old_text, new_text


def update_hermes_config(plan: PlannedChanges, *, dry_run: bool) -> list[str]:
    """Run hermes config set ... for base_url/default AND fully sync the
    matching providers.<model.provider> entry (api, default_model, and
    models[]). Hermes v0.10 uses the provider entry as the actual runtime
    endpoint while it still reads model.* for header display — any
    desync between the two layers causes a display-vs-runtime
    bifurcation. Returns log lines. No-op if hermes CLI is absent.
    Caller guarantees snapshot was taken first.
    """
    log: list[str] = []
    if plan.hermes is None:
        return log
    if shutil.which("hermes") is None:
        cprint("warn", "hermes CLI not on PATH; skipping Hermes config update")
        return log

    base = plan.hermes.base_url_host  # Hermes runs on host, keep user form
    model = plan.hermes.model
    cmds = [
        ["hermes", "config", "set", "model.base_url", base],
        ["hermes", "config", "set", "model.default", model],
    ]
    for cmd in cmds:
        cprint("info", f"$ {' '.join(cmd)}")
        if dry_run:
            log.append(" ".join(cmd) + "  [dry-run]")
            continue
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        log.append(f"{' '.join(cmd)} -> rc={r.returncode}")
        if r.returncode != 0:
            raise FatalError(f"hermes config set failed: {r.stderr.strip() or r.stdout.strip()}")

    # Force-sync the entire provider entry to match model.*. Without this
    # the top-level model.* (display) and provider.* (runtime) can diverge
    # when an earlier run set one but not the other, or when auto-rollback
    # restored the YAML partially. The models[] append is not strictly
    # runtime-load-bearing (api + default_model are what Hermes actually
    # calls) but keeping it in step avoids a third kind of drift where
    # `hermes model`'s picker does not know about the currently-active
    # model.
    if not dry_run:
        try:
            yaml = YAML(typ="rt")
            yaml.preserve_quotes = True
            with HERMES_YAML.open("r", encoding="utf-8") as f:
                data = yaml.load(f)
            providers = data.get("providers") or {}
            prov_name = (data.get("model") or {}).get("provider") or ""
            if prov_name and prov_name in providers:
                changed: list[str] = []
                if providers[prov_name].get("api") != base:
                    providers[prov_name]["api"] = base
                    changed.append("api")
                if providers[prov_name].get("default_model") != model:
                    providers[prov_name]["default_model"] = model
                    changed.append("default_model")
                models_list = providers[prov_name].get("models") or []
                if model not in models_list:
                    models_list.insert(0, model)
                    providers[prov_name]["models"] = models_list
                    changed.append("models[+]")
                if changed:
                    with HERMES_YAML.open("w", encoding="utf-8") as f:
                        yaml.dump(data, f)
                    log.append(
                        f"synced providers[{prov_name}].{{{','.join(changed)}}}"
                    )
        except Exception as e:  # noqa: BLE001
            cprint("warn", f"provider-sync failed: {e}")

    return log


# ---------------------------------------------------------------------------
# Restart orchestration
# ---------------------------------------------------------------------------


def restart_honcho_compose(*, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    # Use --no-deps so database/redis are not recreated (only api/deriver
    # need to re-read config.toml). Also pass the override file explicitly:
    # honcho/docker-compose.override.yml does `ports: !reset []` for
    # database/redis so they do not fight llm-postgres / llm-redis on the
    # host. Compose auto-merges override only when no -f is passed; once we
    # specify -f we have to list both files ourselves.
    override = HONCHO_COMPOSE.parent / "docker-compose.override.yml"
    f_args: list[str] = ["-f", str(HONCHO_COMPOSE)]
    if override.exists():
        f_args += ["-f", str(override)]
    cmd = ["docker", "compose", *f_args, "up", "-d",
           "--force-recreate", "--no-deps", "api", "deriver"]
    cprint("step", f"restart honcho: {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def restart_llama(*, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    """Restart all llama-services (chat+embed+gk). Kept for backwards-compat
    callers; cmd_switch now prefers the per-target helper below."""
    cmd = [str(LLAMA_RESTART_SH), "restart"]
    cprint("step", f"restart llama-services: {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def llama_services_sub(
    subcmd: str, target: str, *, dry_run: bool
) -> subprocess.CompletedProcess[str] | None:
    """Run `./scripts/llama-services.sh <subcmd> <target>` for one of
    start / stop / restart against one of all / chat / embed / gk."""
    cmd = [str(LLAMA_RESTART_SH), subcmd, target]
    cprint("step", " ".join(cmd))
    if dry_run:
        return None
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _llama_lifecycle_plan(plan: PlannedChanges) -> tuple[list[str], list[str], list[str]]:
    """Decide per-service actions based on the planned axis changes.

    Returns (start_targets, restart_targets, stop_targets). Rules:
    - chat llama-server (:8080):
        - llama_chat_params set          → restart chat (new model/flags to load)
        - else honcho_chat → llama-server → start chat (idempotent, ensures up)
        - else honcho_chat → ollama/other → stop chat (no caller left)
    - embed llama-server (:8081):
        - llama_embed_params set         → restart embed
        - else honcho_embed → llama-server → start embed (idempotent)
        - else honcho_embed → ollama/other → stop embed
    - gk daemon:
        - honcho_chat set                → restart gk (GK_LLM_URL/MODEL changed)
    """
    start: list[str] = []
    restart: list[str] = []
    stop: list[str] = []

    if plan.llama_chat_params is not None:
        restart.append("chat")
    elif plan.honcho_chat is not None:
        engine = _engine_of_url(plan.honcho_chat.base_url_host)
        if engine == "llama-server":
            start.append("chat")
        else:
            stop.append("chat")

    if plan.llama_embed_params is not None:
        restart.append("embed")
    elif plan.honcho_embed is not None:
        engine = _engine_of_url(plan.honcho_embed.base_url_host)
        if engine == "llama-server":
            start.append("embed")
        else:
            stop.append("embed")

    if plan.honcho_chat is not None:
        restart.append("gk")

    return start, restart, stop


# ---------------------------------------------------------------------------
# Interactive UI — switching flow
# ---------------------------------------------------------------------------


def _print_current_state() -> None:
    cprint("step", "Current state")
    try:
        doc = read_honcho_toml()
    except FileNotFoundError:
        cprint("err", f"honcho config.toml not found at {HONCHO_TOML}")
        raise FatalError("no config.toml")
    c_base, c_model = current_honcho_chat(doc)
    e_base, e_model = current_honcho_embed(doc)
    caps = current_caps(doc)
    h_base, h_model = current_hermes_summary()
    llama = current_chat_params_from_conf()

    print(f"  Honcho chat  : {c_base}  model={c_model}")
    print(f"  Honcho embed : {e_base}  model={e_model}  "
          f"(dim={caps['embedding.VECTOR_DIMENSIONS']})")
    print(f"  Hermes       : {h_base}  model={h_model}")
    print(f"  llama-server : {llama.hf_spec}  alias={llama.alias}  "
          f"ctx={llama.ctx}  ngl={llama.ngl}  "
          f"moe={'yes' if llama.is_moe else 'no'}  parallel={llama.parallel}")
    print(f"  caps         : app.GET_CONTEXT_MAX_TOKENS={caps['app.GET_CONTEXT_MAX_TOKENS']}, "
          f"dialectic.MAX_INPUT_TOKENS={caps['dialectic.MAX_INPUT_TOKENS']}")
    # Startup drift heads-up: if Hermes already points somewhere different
    # from Honcho chat, say so up front. This catches the case where a
    # prior run left the two axes out of sync.
    if (h_base, h_model) != (c_base, c_model):
        # Normalize host.docker.internal vs localhost — they can resolve to
        # the same place even though the strings differ.
        h_norm = h_base.replace("localhost", "host.docker.internal").replace(
            "127.0.0.1", "host.docker.internal"
        )
        if (h_norm, h_model) != (c_base, c_model):
            cprint(
                "warn",
                f"Hermes and Honcho chat are out of sync "
                f"(Hermes={h_model}@{h_base}, Honcho chat={c_model}@{c_base}). "
                "Consider syncing them on the C axis below.",
            )


def _prompt_endpoint(
    label: str, *, default_url: str, allow_llama_model_change: bool = False,
    for_embed: bool = False,
) -> tuple[str, bool]:
    """Return (chosen_host_url, wants_llama_model_change). Empty chosen_host_url = skip.

    for_embed=True: offer the :8081 embed server instead of :8080 chat, since
    the chat server does not serve /embeddings and its n_embd reflects the
    chat model's hidden size rather than an embedding dim.
    """
    llama_port = 8081 if for_embed else 8080
    llama_url = f"http://localhost:{llama_port}/v1"
    llama_label = "llama-server (embed, :8081)" if for_embed else "llama-server (chat,  :8080)"
    choices = [
        questionary.Choice(f"{llama_label} — {llama_url}", value="ll"),
        questionary.Choice(f"ollama                  — http://localhost:{OLLAMA_PORT_HINT}/v1", value="ol"),
        questionary.Choice("custom URL...", value="custom"),
        questionary.Choice(f"no change (keep {default_url or '<empty>'})", value="keep"),
    ]
    if allow_llama_model_change:
        choices.insert(
            3,
            questionary.Choice(
                "change the llama-server model itself (rewrites llama-services.conf)",
                value="llama_swap",
            ),
        )
    pick = questionary.select(f"{label} — target endpoint?", choices=choices).ask()
    if pick is None:
        raise KeyboardInterrupt
    if pick == "keep":
        return (default_url or "", False)
    if pick == "ll":
        return (llama_url, False)
    if pick == "ol":
        return (f"http://localhost:{OLLAMA_PORT_HINT}/v1", False)
    if pick == "llama_swap":
        return ("http://localhost:8080/v1", True)
    # custom
    url = questionary.text(
        f"{label} — base URL (include /v1)", default=default_url or llama_url
    ).ask()
    if url is None:
        raise KeyboardInterrupt
    return (normalize_base(url), False)


def _llama_alias_default(base_url: str, *, embed_filter: bool) -> str | None:
    """When the chosen endpoint is our local llama-server (:8080 chat or
    :8081 embed), read scripts/llama-services.conf and return the alias
    it serves. This gives a useful manual-input default when the server
    is currently stopped (probe fails) but the user is switching
    *toward* it — the lifecycle step will `start chat` / `start embed`
    and the alias is what the server will advertise once up."""
    try:
        port = urlparse(normalize_base(base_url)).port or 0
    except ValueError:
        return None
    try:
        conf = read_llama_conf()
    except Exception:  # noqa: BLE001
        return None
    if embed_filter and port == 8081:
        return conf.get("EMBED_ALIAS")
    if not embed_filter and port == 8080:
        return conf.get("CHAT_ALIAS")
    return None


def _pick_model(base_url: str, *, purpose: str, default_model: str,
                embed_filter: bool = False) -> str:
    models = probe_models(base_url)
    if not models:
        alias = _llama_alias_default(base_url, embed_filter=embed_filter)
        hint = default_model
        if alias and alias != default_model:
            cprint(
                "info",
                f"endpoint unreachable; using llama-services.conf alias "
                f"'{alias}' as the default (the lifecycle step will start "
                f"the server once you confirm)",
            )
            hint = alias
        else:
            cprint("warn", "could not list models; enter manually")
        m = questionary.text(f"{purpose} — model id", default=hint).ask()
        if m is None:
            raise KeyboardInterrupt
        return m.strip()
    ids = [m.get("id") for m in models if m.get("id")]
    if not ids:
        m = questionary.text(f"{purpose} — model id", default=default_model).ask()
        if m is None:
            raise KeyboardInterrupt
        return m.strip()
    # For the embed axis, most backends list chat models too. Prefer
    # name-based filtering (fast, no extra API calls) and fall back to the
    # full list if the heuristic produces nothing.
    if embed_filter:
        markers = ("embed", "embedding", "bge-", "e5-", "gte-", "bert")
        filtered = [x for x in ids if any(m in x.lower() for m in markers)]
        if filtered:
            ids = filtered
        else:
            cprint("warn", "no embedding-looking models found by name; showing all")
    choices = [questionary.Choice(x, value=x) for x in ids] + [
        questionary.Choice("<enter manually>", value="__manual__"),
    ]
    pick = questionary.select(
        f"{purpose} — which model at {base_url}?",
        choices=choices,
        default=default_model if default_model in ids else ids[0],
    ).ask()
    if pick is None:
        raise KeyboardInterrupt
    if pick == "__manual__":
        m = questionary.text("model id", default=default_model).ask()
        if m is None:
            raise KeyboardInterrupt
        return m.strip()
    return pick


def _warn_ollama_pitfalls(base_url: str, model: str, meta: ModelMeta | None) -> list[str]:
    warnings: list[str] = []
    if not looks_like_ollama(base_url):
        return warnings
    ceiling = ollama_context_ceiling()
    if ceiling:
        warnings.append(
            f"ollama OLLAMA_CONTEXT_LENGTH={ceiling}; "
            f"Honcho caps will be capped at min(ceiling-{CTX_HEADROOM}, requested)."
        )
    else:
        warnings.append(
            "ollama OLLAMA_CONTEXT_LENGTH not detected via systemctl; default is 4096. "
            "Prompts over that are silently truncated."
        )
    if meta and meta.arch and meta.arch.lower().startswith("qwen3"):
        warnings.append(
            f"qwen3-family model '{model}' on ollama: create a Modelfile with "
            "`PARAMETER think false` and use that alias, or message.content will be empty."
        )
    warnings.append("ollama OpenAI-compat tool-call support varies by model; "
                    "llama-server is preferred for tool-heavy workflows.")
    return warnings


def _pick_honcho_chat(plan: PlannedChanges, current_url: str, current_model: str) -> bool:
    """Returns True if the user asked to change the llama-server model."""
    url_host, swap_llama = _prompt_endpoint(
        "A: Honcho chat", default_url=current_url, allow_llama_model_change=True
    )
    if not url_host:
        return False
    if url_host == current_url and not swap_llama:
        # still allow a no-op endpoint, but they may want to pick a different model
        pass

    if swap_llama:
        # We'll do axis D later; mark chat endpoint as llama-server:8080 and defer model until we know alias
        plan.honcho_chat = EndpointChoice(
            base_url_host=url_host,
            base_url_docker=to_docker_url(url_host),
            model="",  # filled after D picks alias
            meta=None,
        )
        return True

    model = _pick_model(url_host, purpose="Honcho chat", default_model=current_model)
    meta = probe_model_meta(url_host, model)
    plan.honcho_chat = EndpointChoice(
        base_url_host=url_host,
        base_url_docker=to_docker_url(url_host),
        model=model,
        meta=meta,
    )
    plan.warnings.extend(_warn_ollama_pitfalls(url_host, model, meta))
    return False


def _pick_honcho_embed(plan: PlannedChanges, current_url: str, current_model: str,
                       current_dim: int) -> None:
    url_host, _ = _prompt_endpoint("B: Honcho embed", default_url=current_url, for_embed=True)
    if not url_host:
        return
    model = _pick_model(url_host, purpose="Honcho embed", default_model=current_model,
                       embed_filter=True)
    meta = probe_model_meta(url_host, model)
    if meta and meta.n_embd and current_dim and meta.n_embd != current_dim:
        cprint(
            "err",
            f"embedding DIM mismatch: selected model reports n_embd={meta.n_embd}, "
            f"current pgvector DIM is {current_dim}. DB migration required. "
            "Aborting the embed switch (chat/hermes axes unaffected).",
        )
        return
    plan.honcho_embed = EndpointChoice(
        base_url_host=url_host,
        base_url_docker=to_docker_url(url_host),
        model=model,
        meta=meta,
    )
    if meta:
        plan.caps["embedding.VECTOR_DIMENSIONS"] = meta.n_embd or current_dim
        plan.caps["vector_store.DIMENSIONS"] = meta.n_embd or current_dim
        if meta.n_ctx_train:
            plan.caps["embedding.MAX_INPUT_TOKENS"] = meta.n_ctx_train


def _pick_hermes(plan: PlannedChanges, current_url: str, current_model: str) -> None:
    if plan.honcho_chat and plan.honcho_chat.model:
        default_to_honcho = questionary.confirm(
            f"C: Hermes — use the same endpoint/model as Honcho chat? "
            f"(current Hermes: {current_model} @ {current_url})",
            default=True,
        ).ask()
        if default_to_honcho is None:
            raise KeyboardInterrupt
        if default_to_honcho:
            plan.hermes = EndpointChoice(
                base_url_host=plan.honcho_chat.base_url_host,
                base_url_docker=plan.honcho_chat.base_url_host,  # hermes runs on host
                model=plan.honcho_chat.model,
                meta=plan.honcho_chat.meta,
            )
            return
    url_host, _ = _prompt_endpoint("C: Hermes", default_url=current_url)
    if not url_host:
        # User picked "no change". If that leaves Hermes pointing somewhere
        # different from the new Honcho chat (or from what it used to track),
        # raise a visible warning — this is the drift that bit us when a
        # prior run set Hermes to ollama/qwen3.5 and a later run changed the
        # Honcho chat axis while leaving Hermes untouched.
        if plan.honcho_chat:
            new_chat_url = plan.honcho_chat.base_url_host
            new_chat_model = plan.honcho_chat.model
            if current_url != new_chat_url or current_model != new_chat_model:
                plan.warnings.append(
                    f"Hermes will keep pointing at {current_model} @ {current_url} "
                    f"while Honcho chat moves to {new_chat_model} @ {new_chat_url}. "
                    f"If this is unintended, re-run and pick 'same as Honcho chat' "
                    f"or 'different URL' on the Hermes axis."
                )
        return
    model = _pick_model(url_host, purpose="Hermes", default_model=current_model)
    plan.hermes = EndpointChoice(
        base_url_host=url_host,
        base_url_docker=url_host,
        model=model,
        meta=probe_model_meta(url_host, model),
    )


@dataclass(slots=True)
class _LlamaCandidate:
    spec: str          # what goes into CHAT_HF_SPEC (HF repo:quant or absolute /path)
    label: str         # human-readable ("qwen3.6:27b" or "unsloth/Qwen3.6-35B-A3B-GGUF")
    origin: str        # "ollama" | "hf-cache" | "current-conf"
    size_gib: float


_OLLAMA_MODELS_ROOT = Path("/usr/share/ollama/.ollama/models")
_HF_HUB_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def _list_ollama_blobs() -> list[_LlamaCandidate]:
    """Enumerate ollama-installed models via their manifests, resolve each to
    the largest blob (the model tensor layer), and return candidates that
    llama-server can load via `-m /abs/path`."""
    results: list[_LlamaCandidate] = []
    mf_root = _OLLAMA_MODELS_ROOT / "manifests"
    if not mf_root.is_dir():
        return results
    for mf_path in mf_root.rglob("*"):
        if not mf_path.is_file():
            continue
        try:
            doc = json.loads(mf_path.read_text())
        except (OSError, ValueError):
            continue
        layers = doc.get("layers") or []
        if not layers:
            continue
        biggest = max(layers, key=lambda l: l.get("size", 0))
        digest = (biggest.get("digest") or "").replace(":", "-")
        if not digest:
            continue
        blob = _OLLAMA_MODELS_ROOT / "blobs" / digest
        if not blob.exists():
            continue
        # manifest path shape: registry.ollama.ai/library/<name>/<tag>  or
        #                     registry.ollama.ai/<user>/<name>/<tag>
        rel = mf_path.relative_to(mf_root).parts
        if len(rel) >= 2:
            tag = rel[-1]
            name = rel[-2]
            label = f"{name}:{tag}"
        else:
            label = "/".join(rel)
        size_gib = biggest.get("size", 0) / 1e9
        results.append(_LlamaCandidate(str(blob), label, "ollama", size_gib))
    return sorted(results, key=lambda c: c.label)


def _list_hf_cached_ggufs(min_gib: float = 0.5) -> list[_LlamaCandidate]:
    """Enumerate GGUF files already present in the HuggingFace hub cache
    (usually downloaded by `llama-server -hf ...`). Skips small files
    (e.g. mmproj projectors that are <0.5 GiB)."""
    results: list[_LlamaCandidate] = []
    if not _HF_HUB_ROOT.is_dir():
        return results
    # projector files (vision / multimodal side-tensors) and embed
    # projectors are not chat-model GGUFs — filter them out so the
    # picker is not noisy with irrelevant entries.
    skip_patterns = ("mmproj", "projector", "vision-", "-vision")
    for gguf in _HF_HUB_ROOT.rglob("*.gguf"):
        try:
            if not gguf.is_file():
                continue
            size_gib = gguf.stat().st_size / 1e9
        except OSError:
            continue
        if size_gib < min_gib:
            continue
        name_lower = gguf.name.lower()
        if any(pat in name_lower for pat in skip_patterns):
            continue
        try:
            repo_dir = gguf.relative_to(_HF_HUB_ROOT).parts[0]
        except (IndexError, ValueError):
            continue
        if not repo_dir.startswith("models--"):
            continue
        repo = "/".join(repo_dir.removeprefix("models--").split("--"))
        label = f"{repo} ({gguf.name})"
        results.append(_LlamaCandidate(str(gguf), label, "hf-cache", size_gib))
    return sorted(results, key=lambda c: c.label)


def _derive_alias(spec: str, fallback: str) -> str:
    """Pick a reasonable advertise-alias from a spec (path or HF repo:quant)."""
    if not spec:
        return fallback
    if spec.startswith("/"):
        # local path: use the stem of the filename minus common quant suffixes
        stem = Path(spec).stem
        for suffix in ("-UD-Q4_K_XL", "-Q4_K_M", "-Q5_K_M", "-Q8_0", "-F16"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem.lower() or fallback
    # HF repo:quant -> repo tail, strip "-GGUF"
    tail = spec.split("/")[-1].split(":")[0]
    if tail.endswith("-GGUF"):
        tail = tail[:-5]
    return tail.lower() if tail else fallback


def _pick_llama_model(plan: PlannedChanges, current: ChatParams) -> None:
    # Enumerate local candidates so the user can pick from ollama blobs or
    # the HF cache without retyping the path. Always offer manual entry as
    # escape hatches for not-yet-pulled HF specs or arbitrary paths.
    candidates: list[_LlamaCandidate] = []
    candidates.append(
        _LlamaCandidate(current.hf_spec, f"{current.hf_spec}  [currently running]",
                        "current-conf", 0.0)
    )
    candidates.extend(_list_ollama_blobs())
    candidates.extend(_list_hf_cached_ggufs())

    # dedupe by spec, keeping the first occurrence (current-conf wins)
    seen: set[str] = set()
    uniq: list[_LlamaCandidate] = []
    for c in candidates:
        if c.spec in seen:
            continue
        seen.add(c.spec)
        uniq.append(c)
    candidates = uniq

    def _fmt(c: _LlamaCandidate) -> str:
        if c.origin == "current-conf":
            return c.label
        return f"{c.label}  [{c.origin}, {c.size_gib:.1f} GiB]"

    choices = [questionary.Choice(_fmt(c), value=c) for c in candidates]
    choices.extend([
        questionary.Choice("<enter HF spec manually (repo:quant)>", value="__hf__"),
        questionary.Choice("<enter local GGUF path manually>", value="__path__"),
    ])
    pick = questionary.select(
        "D: llama-server — which GGUF should chat-server load?",
        choices=choices,
        default=candidates[0] if candidates else None,
    ).ask()
    if pick is None:
        raise KeyboardInterrupt
    if pick == "__hf__":
        spec = (questionary.text(
            "HF spec (repo:quant)", default=current.hf_spec
        ).ask() or "").strip()
    elif pick == "__path__":
        spec = (questionary.text(
            "local GGUF path (absolute)", default=current.hf_spec
        ).ask() or "").strip()
    else:
        spec = pick.spec

    if not spec:
        raise FatalError("no spec provided; aborting Axis D")

    # Derive an alias: prefer the label from the picked candidate (e.g.
    # "qwen3.6:27b") so the /v1/models id matches ollama's conventional
    # name; fall back to deriving from the spec string.
    if pick not in ("__hf__", "__path__") and pick.origin != "current-conf":
        default_alias = pick.label
    else:
        default_alias = (
            current.alias if spec == current.hf_spec
            else _derive_alias(spec, current.alias)
        )
    alias = (questionary.text(
        "D: alias advertised in /v1/models", default=default_alias
    ).ask() or "").strip()
    if not alias:
        alias = default_alias

    # Try probing llama-server at its current endpoint to get meta — only useful if the user
    # just changed alias/spec but the server still holds the old model. Otherwise skip and ask.
    meta: ModelMeta | None = None
    probe = probe_model_meta("http://localhost:8080/v1", alias)
    if probe and probe.n_ctx_train:
        meta = probe

    # We often cannot probe the *new* spec without first pulling it. Fall back to asking.
    if meta is None:
        cprint("info", "could not probe meta for the new spec — asking interactively.")
        ctx = int(questionary.text(
            "  -c context length", default=str(current.ctx),
            validate=lambda s: s.isdigit() or "must be a positive integer",
        ).ask() or current.ctx)
        ngl = int(questionary.text(
            "  -ngl GPU layers (99 = all)", default=str(current.ngl),
            validate=lambda s: s.isdigit() or "must be a positive integer",
        ).ask() or current.ngl)
        is_moe = questionary.confirm(
            "  is this an MoE model (adds -ot 'ffn_(up|down|gate)_exps=CPU')?",
            default=current.is_moe,
        ).ask()
        reasoning_off = questionary.confirm(
            "  qwen3-family (add --reasoning off)?", default=current.reasoning_off
        ).ask()
        parallel = int(questionary.text(
            "  --parallel (concurrent slots; 70B dense → 1 recommended)",
            default=str(current.parallel),
            validate=lambda s: s.isdigit() or "must be a positive integer",
        ).ask() or current.parallel)
        plan.llama_chat_params = ChatParams(
            hf_spec=spec, alias=alias, ctx=ctx, ngl=ngl,
            is_moe=bool(is_moe), reasoning_off=bool(reasoning_off), parallel=parallel,
        )
    else:
        proposed = derive_chat_params(meta, current=current, hf_spec=spec, alias=alias)
        cprint(
            "info",
            f"proposed: ctx={proposed.ctx} ngl={proposed.ngl} "
            f"moe={'yes' if proposed.is_moe else 'no'} "
            f"reasoning_off={proposed.reasoning_off} parallel={proposed.parallel}",
        )
        accept = questionary.confirm("accept proposed values?", default=True).ask()
        if not accept:
            ctx = int(questionary.text("  -c", default=str(proposed.ctx)).ask() or proposed.ctx)
            ngl = int(questionary.text("  -ngl", default=str(proposed.ngl)).ask() or proposed.ngl)
            is_moe = questionary.confirm("  MoE?", default=proposed.is_moe).ask()
            reasoning_off = questionary.confirm(
                "  --reasoning off?", default=proposed.reasoning_off
            ).ask()
            parallel = int(questionary.text(
                "  --parallel", default=str(proposed.parallel)
            ).ask() or proposed.parallel)
            proposed = ChatParams(
                hf_spec=spec, alias=alias, ctx=ctx, ngl=ngl,
                is_moe=bool(is_moe), reasoning_off=bool(reasoning_off), parallel=parallel,
            )
        plan.llama_chat_params = proposed

    # Chat endpoint model points to the new alias
    if plan.honcho_chat:
        plan.honcho_chat = dataclasses.replace(plan.honcho_chat, model=alias)
    else:
        plan.honcho_chat = EndpointChoice(
            base_url_host="http://localhost:8080/v1",
            base_url_docker=to_docker_url("http://localhost:8080/v1"),
            model=alias,
            meta=None,
        )


def _compute_honcho_caps(plan: PlannedChanges, current: dict[str, int]) -> None:
    """Co-move app.GET_CONTEXT_MAX_TOKENS and dialectic.MAX_INPUT_TOKENS with chat ctx."""
    ctx: int | None = None
    if plan.llama_chat_params is not None:
        ctx = plan.llama_chat_params.ctx
    elif plan.honcho_chat and plan.honcho_chat.meta and plan.honcho_chat.meta.n_ctx_train:
        ctx = min(plan.honcho_chat.meta.n_ctx_train, CHAT_CTX_HARD_CAP)

    if ctx is None:
        return
    # If we're going to ollama, cap by OLLAMA_CONTEXT_LENGTH as well
    if plan.honcho_chat and looks_like_ollama(plan.honcho_chat.base_url_host):
        ceiling = ollama_context_ceiling() or 4096
        ctx = min(ctx, ceiling)
    target = max(ctx - CTX_HEADROOM, 1024)
    if current.get("app.GET_CONTEXT_MAX_TOKENS", 0) != target:
        plan.caps["app.GET_CONTEXT_MAX_TOKENS"] = target
    if current.get("dialectic.MAX_INPUT_TOKENS", 0) != target:
        plan.caps["dialectic.MAX_INPUT_TOKENS"] = target


# ---------------------------------------------------------------------------
# Top-level flow
# ---------------------------------------------------------------------------


def _diff_and_confirm(
    plan: PlannedChanges, *, dry_run: bool
) -> tuple[str, str, str, str]:
    """Render both file diffs (without actually writing). Returns (toml_old, toml_new,
    conf_old, conf_new)."""
    # Use tempfile-free dry run: call update_* with dry_run=True to get texts
    toml_old, toml_new = update_honcho_toml(plan, dry_run=True)
    conf_old, conf_new = update_llama_conf(plan, dry_run=True)

    if toml_new != toml_old:
        cprint("step", "honcho/config.toml diff")
        print(render_diff(toml_old, toml_new, "honcho/config.toml"))
    if conf_new != conf_old:
        cprint("step", "scripts/llama-services.conf diff")
        print(render_diff(conf_old, conf_new, "scripts/llama-services.conf"))
    if plan.hermes:
        cprint(
            "step",
            f"hermes: set model.base_url={plan.hermes.base_url_host} "
            f"model.default={plan.hermes.model}",
        )
    else:
        cprint("step", "hermes: unchanged")
    if plan.warnings:
        cprint("step", "warnings")
        for w in plan.warnings:
            cprint("warn", w)
    return toml_old, toml_new, conf_old, conf_new


def _user_choice_summary(plan: PlannedChanges) -> dict[str, Any]:
    def ec(e: EndpointChoice | None) -> dict[str, str] | None:
        if e is None:
            return None
        return {"base_url": e.base_url_host, "model": e.model}

    return {
        "honcho_chat": ec(plan.honcho_chat),
        "honcho_embed": ec(plan.honcho_embed),
        "hermes": ec(plan.hermes),
        "llama_chat_params": dataclasses.asdict(plan.llama_chat_params)
            if plan.llama_chat_params else None,
        "llama_embed_params": dataclasses.asdict(plan.llama_embed_params)
            if plan.llama_embed_params else None,
        "caps": plan.caps,
    }


def cmd_switch(
    *, dry_run: bool, with_embed: bool = False, unload_ollama: bool = False
) -> None:
    _print_current_state()

    doc = read_honcho_toml()
    c_url, c_model = current_honcho_chat(doc)
    e_url, e_model = current_honcho_embed(doc)
    caps_now = current_caps(doc)
    h_url, h_model = current_hermes_summary()
    llama_now = current_chat_params_from_conf()

    plan = PlannedChanges()

    try:
        swap_llama = _pick_honcho_chat(plan, c_url, c_model)
        if swap_llama:
            _pick_llama_model(plan, llama_now)

        if with_embed:
            _pick_honcho_embed(plan, e_url, e_model, caps_now["embedding.VECTOR_DIMENSIONS"])
        elif plan.honcho_chat and _engine_of_url(plan.honcho_chat.base_url_host) != _engine_of_url(e_url) \
                and _engine_of_url(plan.honcho_chat.base_url_host) in ("llama-server", "ollama"):
            # Lightweight "keep chat/embed engines in step" offer in the
            # default flow: same engine as the new chat axis, same
            # nomic-embed-text model (768 dim), no DB migration.
            new_engine = _engine_of_url(plan.honcho_chat.base_url_host)
            pair = _embed_endpoint_for_engine(new_engine)
            if pair is not None:
                new_url, new_model = pair
                resp = questionary.confirm(
                    f"also move Honcho embed to {new_engine} ({new_url}, "
                    f"model {new_model}, still 768-dim nomic-embed-text)? "
                    f"current: {e_model} @ {e_url}",
                    default=True,
                ).ask()
                if resp is None:
                    raise KeyboardInterrupt
                if resp:
                    plan.honcho_embed = EndpointChoice(
                        base_url_host=new_url,
                        base_url_docker=to_docker_url(new_url),
                        model=new_model,
                        meta=probe_model_meta(new_url, new_model),
                    )
                else:
                    cprint(
                        "info",
                        f"embed axis left at {e_model} @ {e_url} "
                        f"(use --with-embed for a full picker)",
                    )
            else:
                cprint(
                    "info",
                    "embed axis skipped (pass --with-embed to include). "
                    f"current: {e_model} @ {e_url}",
                )
        else:
            cprint(
                "info",
                "embed axis skipped (pass --with-embed to include). "
                f"current: {e_model} @ {e_url}",
            )
        _pick_hermes(plan, h_url, h_model)
    except KeyboardInterrupt:
        cprint("warn", "cancelled during prompts; no changes made")
        return

    _compute_honcho_caps(plan, caps_now)

    if not (plan.honcho_chat or plan.honcho_embed or plan.hermes
            or plan.llama_chat_params or plan.caps):
        cprint("info", "no changes selected; nothing to do")
        return

    # Pre-write: show diffs (dry)
    toml_old, toml_new, conf_old, conf_new = _diff_and_confirm(plan, dry_run=True)
    if toml_new == toml_old and conf_new == conf_old and not plan.hermes:
        cprint("info", "resolved diffs are empty; nothing to write")
        return

    if dry_run:
        cprint("info", "dry-run: no writes, no restarts, no snapshot")
        return

    confirm = questionary.confirm("apply these changes?", default=False).ask()
    if not confirm:
        cprint("warn", "aborted by user; no changes made")
        return

    # ---- snapshot, then writes, then restarts. auto-rollback on error. ----
    planned_restarts: list[str] = []
    if plan.needs_honcho_restart():
        planned_restarts.append(
            "docker compose -f honcho/docker-compose.yml up -d --force-recreate api deriver"
        )
    if plan.needs_llama_restart():
        planned_restarts.append("scripts/llama-services.sh restart")

    snap = create_snapshot(_user_choice_summary(plan), planned_restarts)
    cprint("ok", f"snapshot: {snap.dir}")

    errors: list[str] = []
    try:
        # 1. writes
        if toml_new != toml_old:
            update_honcho_toml(plan, dry_run=False)
            cprint("ok", f"wrote {HONCHO_TOML}")
        if conf_new != conf_old:
            update_llama_conf(plan, dry_run=False)
            cprint("ok", f"wrote {LLAMA_CONF}")
        if plan.hermes:
            update_hermes_config(plan, dry_run=False)
            cprint(
                "ok",
                "updated hermes config via `hermes config set` + full provider sync",
            )

        # 2. restarts — per-service scope so we don't needlessly relaunch
        # a llama-server that the new config no longer calls (e.g. chat
        # moved to ollama leaves :8080 with no caller). Unused services
        # get a 'stop' action instead of a 'restart'; still-needed ones
        # get 'start' (idempotent) when params did not change, or
        # 'restart' when they did. Also unload any ollama model that
        # was in use before the switch but is not any more, so VRAM is
        # released without having to stop the systemd ollama service.
        start_targets, restart_targets, stop_targets = _llama_lifecycle_plan(plan)

        # Ollama unload is opt-in (default off) — overriding the user's
        # OLLAMA_KEEP_ALIVE policy without asking was too aggressive:
        # unloading drops prompt/KV caches, a 30B-class model reload
        # from disk costs ~10-30s, and shared-ollama setups would lose
        # state for other clients. Ask explicitly unless --unload-ollama
        # was passed.
        unload_candidates = _ollama_unload_targets(
            plan, (c_url, c_model), (e_url, e_model), (h_url, h_model)
        )
        if not unload_candidates:
            unload_models: list[str] = []
        elif unload_ollama:
            unload_models = unload_candidates
            cprint(
                "info",
                f"--unload-ollama: will unload {', '.join(unload_candidates)}",
            )
        else:
            resp = questionary.confirm(
                f"refresh VRAM by unloading {len(unload_candidates)} ollama model(s) "
                f"({', '.join(unload_candidates)})? "
                f"Note: drops prompt/KV caches; 30B-class reload from disk costs "
                f"~10-30s; overrides your OLLAMA_KEEP_ALIVE setting. "
                f"Leave as N to keep models warm; manual unload is "
                f"`curl -X POST http://localhost:11434/api/generate "
                f"-d '{{\"model\":\"<id>\",\"keep_alive\":0,\"prompt\":\"\",\"stream\":false}}'`.",
                default=False,
            ).ask()
            if resp is None:
                raise KeyboardInterrupt
            unload_models = unload_candidates if resp else []
            if not resp:
                cprint(
                    "info",
                    f"keeping ollama models warm: {', '.join(unload_candidates)}",
                )
        if start_targets or restart_targets or stop_targets or unload_models:
            scope_msg_parts: list[str] = []
            if stop_targets:
                scope_msg_parts.append(f"stop {{{','.join(stop_targets)}}} (no longer in use)")
            if unload_models:
                scope_msg_parts.append(
                    f"ollama unload {{{','.join(unload_models)}}} (VRAM reclaim)"
                )
            if restart_targets:
                scope_msg_parts.append(f"restart {{{','.join(restart_targets)}}}")
            if start_targets:
                scope_msg_parts.append(f"start {{{','.join(start_targets)}}} (idempotent)")
            do_lifecycle = questionary.confirm(
                "apply llama-services lifecycle: " + "; ".join(scope_msg_parts) + "?",
                default=True,
            ).ask()
            if do_lifecycle:
                # Stops + unloads first so VRAM is released before any
                # 'start' / 'restart' tries to bind the same GPU arena.
                for t in stop_targets:
                    r = llama_services_sub("stop", t, dry_run=False)
                    if r is not None and r.returncode != 0:
                        err_text = (r.stderr or r.stdout).strip()
                        cprint("warn", f"stop {t} rc={r.returncode}: {err_text[:200]}")
                        # non-fatal: if stop fails we still proceed, but record it
                        errors.append(
                            f"llama-services stop {t} rc={r.returncode}: {err_text[:300]}"
                        )
                for model_id in unload_models:
                    ok = ollama_unload_model(model_id, dry_run=False)
                    if not ok:
                        errors.append(f"ollama unload {model_id}: non-OK (see warn above)")
                for t in restart_targets:
                    r = llama_services_sub("restart", t, dry_run=False)
                    if r is not None and r.returncode != 0:
                        err_text = (r.stderr or r.stdout).strip()
                        cprint("err", f"restart {t} rc={r.returncode}:")
                        for line in err_text.splitlines()[-20:]:
                            sys.stderr.write(f"    {line}\n")
                        errors.append(
                            f"llama-services restart {t} rc={r.returncode}: {err_text[:500]}"
                        )
                        raise FatalError(f"llama-services restart {t} failed")
                for t in start_targets:
                    r = llama_services_sub("start", t, dry_run=False)
                    if r is not None and r.returncode != 0:
                        err_text = (r.stderr or r.stdout).strip()
                        cprint("err", f"start {t} rc={r.returncode}:")
                        for line in err_text.splitlines()[-20:]:
                            sys.stderr.write(f"    {line}\n")
                        errors.append(
                            f"llama-services start {t} rc={r.returncode}: {err_text[:500]}"
                        )
                        raise FatalError(f"llama-services start {t} failed")
        if plan.needs_honcho_restart():
            do_restart = questionary.confirm(
                "restart Honcho compose (api + deriver) now?", default=True
            ).ask()
            if do_restart:
                r = restart_honcho_compose(dry_run=False)
                if r is not None and r.returncode != 0:
                    err_text = (r.stderr or r.stdout).strip()
                    cprint("err", f"compose up rc={r.returncode}:")
                    for line in err_text.splitlines()[-20:]:
                        sys.stderr.write(f"    {line}\n")
                    errors.append(f"compose up rc={r.returncode}: {err_text[:500]}")
                    raise FatalError("docker compose restart failed")

        finalize_snapshot(snap, "applied", errors)
        cprint("ok", "all writes and restarts succeeded")
        cprint("info", f"snapshot kept at {snap.dir} for audit")
    except KeyboardInterrupt:
        auto_rollback(snap, "interrupted by user (Ctrl-C)", errors)
        raise
    except FatalError as e:
        auto_rollback(snap, str(e), errors)
        raise
    except Exception as e:  # noqa: BLE001
        auto_rollback(snap, f"unexpected: {e}", errors + [repr(e)])
        raise


def cmd_list() -> None:
    snaps = list_snapshots()
    if not snaps:
        cprint("info", f"no snapshots in {SNAPSHOT_ROOT}")
        return
    for s in snaps[:SNAPSHOT_KEEP]:
        mf = s.manifest
        status = mf.get("status", "?")
        created = mf.get("created_at", "?")
        files = ",".join(mf.get("files_snapshotted", []))
        choices = mf.get("user_choices") or {}
        chat = (choices.get("honcho_chat") or {}).get("model", "-")
        embed = (choices.get("honcho_embed") or {}).get("model", "-")
        hermes = (choices.get("hermes") or {}).get("model", "-")
        print(f"  {s.id}  {status:<22}  {created}")
        print(f"      files=[{files}]  chat={chat}  embed={embed}  hermes={hermes}")


def cmd_rollback() -> None:
    snaps = list_snapshots()
    if not snaps:
        cprint("err", "no snapshots to roll back to")
        raise FatalError("empty snapshot dir")
    target = snaps[0]
    cprint("info", f"latest snapshot: {target.id} ({target.manifest.get('status', '?')})")
    yes = questionary.confirm(
        "restore all files from this snapshot (atomic)?", default=False
    ).ask()
    if not yes:
        cprint("warn", "aborted")
        return
    restore_snapshot(target)
    finalize_snapshot(target, "rolled_back", [])
    cprint("ok", "restored; manually re-run restarts if services are stuck")


def cmd_restore(snap_id: str) -> None:
    snaps = list_snapshots()
    target = next((s for s in snaps if s.id == snap_id), None)
    if target is None:
        cprint("err", f"no snapshot with id {snap_id}")
        raise FatalError("unknown snapshot id")
    cprint("info", f"restore {target.dir}")
    yes = questionary.confirm("proceed?", default=False).ask()
    if not yes:
        return
    restore_snapshot(target)
    finalize_snapshot(target, "rolled_back", [])
    cprint("ok", "restored")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="switch-endpoints.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Interactive LLM endpoint/model switcher for the hermes-stack.
            Writes are snapshot-protected under
            ~/.local/state/nuncstans-hermes-stack/endpoint-snapshots/
            (override: $HERMES_STATE_DIR). LRU of 10.
            """
        ).strip(),
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print diffs, skip all writes and restarts")
    ap.add_argument("--with-embed", action="store_true",
                    help="also prompt for the Honcho embedding axis (default: skipped — "
                         "changing embed dim requires a destructive pgvector migration, "
                         "so it is opt-in)")
    ap.add_argument("--unload-ollama", action="store_true",
                    help="force-unload ollama models whose axes moved off ollama "
                         "(skips the interactive prompt). Default is to ask "
                         "(default No) so your OLLAMA_KEEP_ALIVE setting and any "
                         "warm prompt/KV caches are preserved.")
    ap.add_argument("--rollback", action="store_true",
                    help="restore from the most recent snapshot")
    ap.add_argument("--restore", metavar="SNAPSHOT_ID",
                    help="restore from a specific snapshot id (e.g. 20260423-151230.12345)")
    ap.add_argument("--list-snapshots", action="store_true",
                    help="list the 10 most recent snapshots with manifest summaries")
    args = ap.parse_args()

    # Always prune at startup so the user's first mental model matches reality.
    try:
        n = prune_snapshots()
        if n:
            cprint("info", f"pruned {n} old snapshot(s)")
    except Exception as e:  # noqa: BLE001
        cprint("warn", f"prune failed: {e}")

    try:
        if args.list_snapshots:
            cmd_list()
        elif args.rollback:
            cmd_rollback()
        elif args.restore:
            cmd_restore(args.restore)
        else:
            cmd_switch(
                dry_run=args.dry_run,
                with_embed=args.with_embed,
                unload_ollama=args.unload_ollama,
            )
    except FatalError as e:
        cprint("err", str(e))
        sys.exit(2)
    except KeyboardInterrupt:
        cprint("warn", "interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
