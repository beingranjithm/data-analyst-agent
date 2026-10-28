"""Day 23a: settings for the PUBLIC hosted version (a free website host such as Streamlit Community Cloud).

Hosted mode is switched on by the setting DAA_HOSTED=1 (an environment variable or a Streamlit secret). It changes:
  * the visitor pastes THEIR OWN Gemini key; it lives only in that visitor's session (never in the environment, a
    file or the saved-replies cache), so two visitors can never use each other's key;
  * the "saved replies" cache that writes to the server's disk is off (it would store visitors' questions and data
    summaries on the server and share them); only "Replay only" and "Ask the AI" remain;
  * smaller upload limits (free hosts have little memory) and a smaller sandbox memory cap;
  * the Code sandbox tab (a box where anyone could type their own Python) is hidden;
  * the AI's code only runs when the sandbox's file-system lock works on the host (fail closed). One exception
    (Day 23c): code that is word for word in the shipped demo file (deploy/demo_cache.json) was recorded and
    checked on the owner's laptop, so it may run on a host without the lock. Everything else still needs the lock.
Nothing here calls a model. Plain, tested functions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HOSTED_FILE_MB = 15          # per file
HOSTED_TOTAL_MB = 30         # all files together
HOSTED_FILES = 15
HOSTED_SANDBOX_MB = 512
_TRUE = {"1", "true", "yes", "on"}


def _secret_flag() -> bool:
    try:
        import streamlit as st
        return str(st.secrets.get("DAA_HOSTED", "")).strip().lower() in _TRUE
    except Exception:                                    # no secrets file, no streamlit: not a secret-configured host
        return False


def is_hosted() -> bool:
    return os.getenv("DAA_HOSTED", "").strip().lower() in _TRUE or _secret_flag()


DEMO_CACHE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "demo_cache.json"
_RECORDED = {}


def recorded_code(path=None) -> frozenset:
    """The code blocks found in the shipped demo replies (stripped), read once. Empty when the file is missing or
    unreadable: then nothing counts as recorded and the file-system lock is demanded for every program."""
    path = Path(path) if path else DEMO_CACHE_FILE
    if str(path) not in _RECORDED:
        found = set()
        try:
            from agents.chart_agent import extract_code
            data = json.loads(path.read_text())
            for entry in (data.values() if isinstance(data, dict) else []):
                text = entry.get("text") if isinstance(entry, dict) else None
                code = extract_code(text) if isinstance(text, str) else None
                if code and code.strip():
                    found.add(code.strip())
        except Exception:                                # noqa: BLE001  (a bad file means: nothing is recorded)
            found = set()
        _RECORDED[str(path)] = frozenset(found)
    return _RECORDED[str(path)]


def is_recorded(code) -> bool:
    return isinstance(code, str) and code.strip() in recorded_code()


def sandbox_options(code=None) -> dict:
    """Extra arguments for tools.sandbox.run_code. `code` is the program about to run: when it is one of the recorded
    demo programs the file-system lock is not demanded (it is still applied when the host has it)."""
    if not is_hosted():
        return {}
    if is_recorded(code):
        return {"memory_mb": HOSTED_SANDBOX_MB}
    return {"require_isolation": True, "memory_mb": HOSTED_SANDBOX_MB}


def file_limit_mb(default: int) -> int:
    return min(default, HOSTED_FILE_MB) if is_hosted() else default


def upload_limits(max_files: int, max_total_mb: int) -> tuple:
    return (min(max_files, HOSTED_FILES), min(max_total_mb, HOSTED_TOTAL_MB)) if is_hosted() else (max_files, max_total_mb)


def redact(text, key) -> str:
    """Removes a visitor's key from a message before it is shown (errors can echo what was sent)."""
    text = str(text)
    return text.replace(key, "[your key]") if key and len(key) >= 8 else text


_PROBE = {}


def isolation_ok(run=None, refresh: bool = False) -> tuple:
    """(True, '') when the sandbox's file-system lock works on this machine, else (False, reason). Checked once per
    process by running a one-line program with require_isolation=True."""
    if "result" in _PROBE and not refresh:
        return _PROBE["result"]
    if run is None:
        from tools.sandbox import run_code as run
    import pandas as pd
    try:
        res = run("result = 1", pd.DataFrame({"a": [1]}), timeout=30, require_isolation=True)
        ok = bool(getattr(res, "ok", False))
        out = (True, "") if ok else (False, "the sandbox's file-system lock is not available on this host")
    except Exception as e:                               # noqa: BLE001  (any failure means: do not run code here)
        out = (False, f"the sandbox could not be started ({type(e).__name__})")
    _PROBE["result"] = out
    return out
