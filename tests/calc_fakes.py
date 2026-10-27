"""A stand-in for the sandbox in the chat tests: runs the code with plain exec (NO isolation, tests only)."""
import numpy as np
import pandas as pd


class FakeRun:
    """The few RunResult attributes the chat agent uses."""

    def __init__(self, status="ok", table=None, value=None, stdout="", error=None, problems=None, total=None):
        self.status, self.table, self.value, self.stdout = status, table, value, stdout
        self.error, self.problems, self.table_total_rows = error, problems or [], total

    @property
    def ok(self):
        return self.status == "ok"

    def message(self):
        if self.status == "blocked":
            return "Code was blocked before running: " + "; ".join(self.problems)
        e = self.error or {}
        return f"{e.get('type', 'Error')}: {e.get('message', '')}"


def local_runner(code, tables):
    df = tables if hasattr(tables, "columns") else next(iter(tables.values()))
    ns = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, ns)
    except Exception as e:
        return FakeRun("error", error={"type": type(e).__name__, "message": str(e)})
    result = ns.get("result")
    if isinstance(result, pd.Series):
        result = result.reset_index()
    if isinstance(result, pd.DataFrame):
        return FakeRun(table=result.head(500), total=len(result))
    return FakeRun(value=result)
