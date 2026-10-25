"""Runs inside a separate, restricted process. Started by tools/sandbox.py only.

Layers applied here: scrubbed environment (set by the parent), memory and CPU
limits, no 'open', no network, an import allow-list, and a file-system lock (Landlock).
"""
import builtins
import contextlib
import importlib.util
import io
import json
import os
import resource
import sys
import traceback
import warnings

workdir = sys.argv[1]
warnings.simplefilter("ignore")

# Load the shared rules from the same folder (works in isolated mode)
_spec = importlib.util.spec_from_file_location("code_guard", os.path.join(os.path.dirname(__file__), "code_guard.py"))
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

cfg = json.load(open(os.path.join(workdir, "config.json")))
code = open(os.path.join(workdir, "code.py"), encoding="utf-8").read()
MAX_TABLE_ROWS, MAX_TABLE_COLS, MAX_STDOUT, MAX_FIG_BYTES = 500, 100, 20_000, 8_000_000

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

tables = {entry["name"]: pd.read_pickle(os.path.join(workdir, entry["file"])) for entry in cfg["tables"]}
active = cfg["active"]


def write_result(payload):
    with open(os.path.join(workdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str)


def current_data_bytes():
    """Memory the process has really claimed (VmData), not just reserved address space."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmData:"):
                return int(line.split()[1]) * 1024
    return 0


# ---- resource limits (memory is relative to what is already loaded) ----
# RLIMIT_DATA counts memory that is actually claimed. RLIMIT_AS would also count the
# large address ranges that pandas' Arrow library reserves, and would break pandas.
mem = int(cfg["memory_mb"]) * 1024 * 1024
base = current_data_bytes()
if base:
    resource.setrlimit(resource.RLIMIT_DATA, (base + mem, base + mem))
resource.setrlimit(resource.RLIMIT_CPU, (int(cfg["timeout"]) + 1, int(cfg["timeout"]) + 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# ---- block the network ----
import socket  # noqa: E402


def _no_network(*args, **kwargs):
    raise PermissionError("network access is blocked in the sandbox")


class _BlockedSocket:
    def __init__(self, *a, **k):
        _no_network()


socket.socket = _BlockedSocket
socket.create_connection = _no_network
socket.getaddrinfo = _no_network
socket.gethostbyname = _no_network

# ---- file-system lock (Linux Landlock; needs no admin rights) ----
def apply_landlock(read_dirs, rw_dir):
    """Restrict this process: read-only access to Python's own folders, read/write to rw_dir only.
    Returns the Landlock version used, or 0 if the kernel does not support it."""
    import ctypes
    import struct

    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, None, 0, 1)  # landlock_create_ruleset(version query)
    if abi < 1:
        return 0
    EXEC, WRITE, READ_F, READ_D = 1 << 0, 1 << 1, 1 << 2, 1 << 3
    handled = 0x1FFF  # all ABI 1 file rights
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    if abi >= 5:
        handled |= 1 << 15
    handled_net = 3 if abi >= 4 else 0  # block TCP bind and connect at kernel level
    attr = struct.pack("QQ", handled, handled_net) if abi >= 4 else struct.pack("Q", handled)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset_fd = libc.syscall(444, buf, len(attr), 0)
    if ruleset_fd < 0:
        return 0
    read_only = READ_F | READ_D
    file_rights = READ_F | WRITE
    rw_all = handled & ~EXEC

    def add(path, rights, is_dir):
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            rule = struct.pack("QiI", rights if is_dir else rights & (READ_F | WRITE | (1 << 14)), fd, 0)[:12]
            libc.syscall(445, ruleset_fd, 1, ctypes.create_string_buffer(rule, 12), 0)
        finally:
            os.close(fd)

    for d in read_dirs:
        add(d, read_only, True)
    add(rw_dir, rw_all, True)
    for f in ("/dev/null", "/dev/urandom", "/dev/zero"):
        add(f, file_rights, False)
    libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
    ok = libc.syscall(446, ruleset_fd, 0)  # landlock_restrict_self
    os.close(ruleset_fd)
    return abi if ok == 0 else 0


# ---- import allow-list (only applies to the analysis code itself) ----
_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # This guard exists only inside the analysis code's builtins. Libraries such as
    # pandas keep using the real import, so we can apply the rules to every call here.
    if level:
        raise ImportError("relative imports are not allowed")
    reason = guard.import_problem(name, fromlist or ())
    if reason:
        raise ImportError(reason)
    return _real_import(name, globals, locals, fromlist, level)


class _LazyModule:
    """Imports a module the first time it is used (keeps start-up fast)."""

    def __init__(self, name):
        self._name, self._mod = name, None

    def __getattr__(self, item):
        if self._mod is None:
            self._mod = _real_import(self._name, fromlist=["_"])
        return getattr(self._mod, item)


_SAFE = (
    "abs all any bool bytes callable chr dict divmod enumerate filter float format frozenset "
    "hasattr hash hex int isinstance issubclass iter len list map max min next object ord pow print "
    "range repr reversed round set slice sorted str sum tuple zip "
    "Exception ValueError KeyError TypeError IndexError ZeroDivisionError ArithmeticError "
    "AttributeError RuntimeError StopIteration NotImplementedError ImportError LookupError "
    "OverflowError AssertionError NotImplemented Ellipsis"
).split()
safe_builtins = {n: getattr(builtins, n) for n in _SAFE}
safe_builtins["__import__"] = _guarded_import

ns = {
    "__builtins__": safe_builtins,
    "__name__": "__sandbox__",
    "pd": pd,
    "np": np,
    "px": _LazyModule("plotly.express"),
    "go": _LazyModule("plotly.graph_objects"),
    "tables": tables,
    "df": tables[active],
}


class _CappedOut(io.TextIOBase):
    def __init__(self, limit):
        self.limit, self.parts, self.size = limit, [], 0

    def writable(self):
        return True

    def write(self, s):
        if self.size < self.limit:
            self.parts.append(s[: self.limit - self.size])
            self.size += len(s)
        return len(s)

    def text(self):
        out = "".join(self.parts)
        return out + ("\n... (output cut)" if self.size > self.limit else "")


def table_payload(obj):
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=str(obj.name) if obj.name is not None else "value")
    if not isinstance(obj.index, pd.RangeIndex):
        obj = obj.reset_index()
    obj = obj.copy()
    obj.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in obj.columns]
    total = len(obj)
    part = obj.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLS]
    data = json.loads(part.to_json(orient="split", date_format="iso", index=False))
    return {"columns": data["columns"], "data": data["data"], "total_rows": total}


def serialize(namespace):
    out = {}
    fig = namespace.get("fig")
    if fig is not None and hasattr(fig, "to_json") and hasattr(fig, "data"):
        text = fig.to_json()
        if len(text) > MAX_FIG_BYTES:
            out["error"] = {"type": "ResultTooLarge", "message": "The chart is too large to return.", "line": None, "code_line": None}
        else:
            out["figure_json"] = text
    result = namespace.get("result")
    if isinstance(result, (pd.DataFrame, pd.Series)):
        out["table"] = table_payload(result)
    elif isinstance(result, (bool, int, float, str, np.integer, np.floating, np.bool_)) or result is None:
        value = result.item() if isinstance(result, np.generic) else result
        out["value"] = value if not isinstance(value, str) else value[:5000]
    elif result is not None:
        try:
            out["value"] = json.loads(json.dumps(result, default=str))
        except Exception:
            out["value"] = repr(result)[:5000]
    return out


_read_dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/usr", "/lib", "/lib64", "/etc/localtime", "/usr/share/zoneinfo"}
isolation_abi = apply_landlock(sorted(d for d in _read_dirs if os.path.exists(d)), workdir)

capped = _CappedOut(MAX_STDOUT)
payload = {}
if cfg.get("require_isolation") and not isolation_abi:
    write_result({"error": {"type": "IsolationUnavailable", "line": None, "code_line": None,
                            "message": "The file-system lock (Landlock) is not available on this computer."},
                  "stdout": "", "isolation": "none"})
    sys.exit(0)
try:
    compiled = compile(code, "<analysis>", "exec")
    with contextlib.redirect_stdout(capped):
        exec(compiled, ns)
    payload = serialize(ns)
except BaseException as e:  # noqa: BLE001 - we must report everything
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<analysis>"]
    line = frames[-1].lineno if frames else getattr(e, "lineno", None)
    lines = code.splitlines()
    ns.clear()
    payload = {"error": {
        "type": type(e).__name__,
        "message": str(e)[:600],
        "line": line,
        "code_line": lines[line - 1].strip()[:200] if line and 0 < line <= len(lines) else None,
    }}
payload["stdout"] = capped.text()
payload["isolation"] = f"landlock-v{isolation_abi}" if isolation_abi else "none"
write_result(payload)
