"""Runs AI-written Python code safely (as safely as a plain process allows).

Layers of protection:
  1. Static check     - tools/code_guard.py rejects dangerous code before it runs.
  2. Separate process - a crash or freeze cannot take down the app.
  3. Scrubbed secrets - the child process does not receive your API keys.
  4. Resource limits  - wall-clock timeout, CPU time and memory caps.
  5. Restricted world - no open(), no network, only allow-listed imports.
  6. File-system lock - Linux Landlock: the code can only read Python's own folders
                        and write inside its private temp folder (needs no admin rights).

Honest limit: still not a full container. Docker is added later for stronger isolation.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tools.code_guard import check_code

CHILD_SCRIPT = Path(__file__).with_name("sandbox_child.py")
DEFAULT_TIMEOUT = 30
DEFAULT_MEMORY_MB = 1024


@dataclass
class RunResult:
    status: str  # ok | error | blocked | timeout | killed
    stdout: str = ""
    error: dict | None = None
    problems: list = field(default_factory=list)
    table: pd.DataFrame | None = None
    table_total_rows: int | None = None
    value: Any = None
    figure_json: str | None = None
    seconds: float = 0.0
    debug: str = ""
    isolation: str = "unknown"  # "landlock-vN" when the file-system lock is active, "none" if not

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def figure(self):
        """The Plotly figure produced by the code (variable `fig`), or None."""
        if not self.figure_json:
            return None
        import plotly.io as pio
        return pio.from_json(self.figure_json)

    def message(self) -> str:
        """A short description of what happened (used later to help the AI fix its code)."""
        if self.status == "ok":
            return "Code ran successfully."
        if self.status == "blocked":
            return "Code was blocked before running: " + "; ".join(self.problems)
        if self.status == "timeout":
            return "Code took too long and was stopped (possible infinite loop or very heavy calculation)."
        if self.status == "killed":
            return "Code was stopped because it used too much memory or CPU time."
        e = self.error or {}
        where = f" (line {e['line']}: {e['code_line']})" if e.get("line") and e.get("code_line") else ""
        text = f"{e.get('type', 'Error')}: {e['message']}" if e.get("message") else str(e.get("type", "Error"))
        return text + where


def _clean_env(workdir: str) -> dict:
    """A minimal environment. Nothing from your own environment (API keys!) is passed on."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        # Use the normal system allocator so the memory limit works with pandas' Arrow library
        "ARROW_DEFAULT_MEMORY_POOL": "system",
    }


def run_code(code, tables, *, active=None, timeout=DEFAULT_TIMEOUT,
             memory_mb=DEFAULT_MEMORY_MB, check=True, require_isolation=False) -> RunResult:
    """Run `code` against one or more tables.

    Inside the code you can use:  df (the active table), tables (dict of all tables),
    pd, np, px, go. Put the answer in a variable called `result` and/or a Plotly
    figure in `fig`.
    `check=False` skips the static check. Only for testing the inner layers.
    `require_isolation=True` refuses to run if the file-system lock is unavailable.
    """
    if isinstance(tables, pd.DataFrame):
        tables = {"data": tables}
    if not tables:
        return RunResult("error", error={"type": "NoData", "message": "No table was provided.", "line": None, "code_line": None})
    active = active or next(iter(tables))

    if check:
        problems = check_code(code)
        if problems:
            return RunResult("blocked", problems=problems)

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="daa_sandbox_") as workdir:
        entries = []
        for i, (name, frame) in enumerate(tables.items()):
            filename = f"table_{i}.pkl"
            frame.to_pickle(os.path.join(workdir, filename))
            entries.append({"name": name, "file": filename})
        Path(workdir, "code.py").write_text(code, encoding="utf-8")
        Path(workdir, "config.json").write_text(json.dumps({
            "tables": entries, "active": active, "timeout": timeout, "memory_mb": memory_mb,
            "require_isolation": require_isolation,
        }))

        proc = subprocess.Popen(
            [sys.executable, "-I", str(CHILD_SCRIPT), workdir],
            cwd=workdir, env=_clean_env(workdir),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, stderr = proc.communicate()

        seconds = round(time.time() - started, 2)
        debug = (stderr or "")[-1500:]
        if timed_out:
            return RunResult("timeout", seconds=seconds, debug=debug)

        result_file = Path(workdir, "result.json")
        if proc.returncode != 0 or not result_file.exists():
            return RunResult("killed", seconds=seconds, debug=debug)

        payload = json.loads(result_file.read_text(encoding="utf-8"))

    table = None
    if "table" in payload:
        t = payload["table"]
        table = pd.DataFrame(t["data"], columns=t["columns"])
    return RunResult(
        status="error" if payload.get("error") else "ok",
        stdout=payload.get("stdout", ""),
        error=payload.get("error"),
        table=table,
        table_total_rows=payload.get("table", {}).get("total_rows"),
        value=payload.get("value"),
        figure_json=payload.get("figure_json"),
        seconds=seconds,
        debug=debug,
        isolation=payload.get("isolation", "none"),
    )
