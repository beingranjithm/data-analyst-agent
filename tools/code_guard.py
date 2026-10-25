"""Static checker for AI-written code. Uses only the standard library.

It reads the code WITHOUT running it and lists anything that is not allowed.
This is one layer of protection (see tools/sandbox.py for the others).
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "math", "statistics", "datetime", "re", "json",
    "collections", "itertools", "functools", "operator", "calendar", "decimal",
    "fractions", "string", "textwrap", "typing", "random",
}
PLOTLY_OK = {"express", "graph_objects", "subplots", "figure_factory", "colors"}

# Names that must never be called or used directly
BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit", "help",
}
# Attribute / module names that reach files, the system or the network
BLOCKED_ATTRS = {
    # system and plumbing
    "os", "sys", "io", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "builtins", "ctypes", "ctypeslib", "f2py", "distutils", "testing", "compat",
    "offline", "system", "popen", "spawn", "environ", "eval", "exec", "query",
    # writing files / showing windows
    "to_csv", "to_excel", "to_json", "to_pickle", "to_parquet", "to_sql", "to_html",
    "to_clipboard", "to_feather", "to_hdf", "to_latex", "to_xml", "to_orc", "to_stata",
    "to_gbq", "write_html", "write_image", "write_images", "write_json", "to_image",
    "show", "save", "savez", "savez_compressed", "savetxt", "tofile", "dump", "dumps",
    # reading files
    "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap",
    # file-like objects and helpers found by red-teaming
    "DataSource", "ExcelFile", "ExcelWriter", "HDFStore", "codecs", "urllib",
    "urlopen", "urlretrieve", "tempfile", "glob", "fileinput", "zipfile", "tarfile",
    "gzip", "bz2", "lzma", "pickle", "marshal", "shelve", "sqlite3", "webbrowser",
    "signal", "threading", "multiprocessing", "asyncio", "runpy", "pkgutil",
}
BLOCKED_PREFIXES = ("read_", "open_", "write_")

MAX_CODE_CHARS = 20_000
MAX_CODE_LINES = 400


def attr_problem(name: str):
    """Why this attribute/module name is not allowed, or None if it is fine."""
    if name.startswith("_"):
        return f"access to private attribute '{name}' is not allowed"
    if name in BLOCKED_ATTRS or name.startswith(BLOCKED_PREFIXES):
        return f"'{name}' is not allowed (files, system and network access are blocked)"
    return None


def import_problem(module: str, names=()):
    """Why this import is not allowed, or None if it is fine."""
    parts = module.split(".")
    if parts[0] not in ALLOWED_IMPORTS:
        return f"import of '{module}' is not allowed"
    for p in parts[1:]:
        reason = attr_problem(p)
        if reason:
            return f"import of '{module}': {reason}"
    if parts[0] == "plotly":
        if len(parts) > 1 and parts[1] not in PLOTLY_OK:
            return f"import of '{module}' is not allowed (use plotly.express or plotly.graph_objects)"
        if len(parts) == 1:
            for n in names:
                if n != "*" and n not in PLOTLY_OK:
                    return f"'from plotly import {n}' is not allowed"
    for n in names:
        reason = attr_problem(n) if n != "*" else "star imports are not allowed"
        if reason:
            return f"import of '{n}' from '{module}': {reason}"
    return None


def check_code(code: str):
    """Return a list of problems (empty list means the code passed the check)."""
    if not isinstance(code, str) or not code.strip():
        return ["no code was provided"]
    if len(code) > MAX_CODE_CHARS or code.count("\n") > MAX_CODE_LINES:
        return [f"code is too long (limit {MAX_CODE_LINES} lines)"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error on line {e.lineno}: {e.msg}"]

    problems = []

    def add(node, text):
        problems.append(f"line {getattr(node, 'lineno', '?')}: {text}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                reason = import_problem(alias.name)
                if reason:
                    add(node, reason)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                add(node, "relative imports are not allowed")
            else:
                reason = import_problem(node.module or "", [a.name for a in node.names])
                if reason:
                    add(node, reason)
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                add(node, f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            reason = attr_problem(node.attr)
            if reason:
                add(node, reason)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                add(node, "text containing a double underscore is not allowed")
        elif isinstance(node, ast.ClassDef):
            add(node, "defining classes is not allowed (use functions)")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            add(node, "async code is not allowed")
    # remove exact duplicates but keep order
    return list(dict.fromkeys(problems))
