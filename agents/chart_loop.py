"""Self-healing chart loop (Day 10).

Runs the chart agent; if the code fails, sends the model its own code plus a SCRUBBED
description of the problem and asks for a corrected version. Stops after max_attempts,
when the model repeats identical code, or when the model call itself fails.

PRIVACY: sandbox error text can contain data values (e.g. "could not convert string to
float: 'North'"). scrub_message() hides them before anything goes back to the model.
It is a best-effort filter, not a guarantee: see its docstring.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from agents.chart_agent import DEFAULT_MODEL, ChartResult, make_chart_async
from agents.resilience import DEFAULT_DELAYS
from tools.plan_schema import ChartSpec

MAX_ATTEMPTS = 3
_QUOTED = re.compile(r"""(?<![A-Za-z0-9_])(['"])(.{0,200}?)\1""")


def _known_values(df: pd.DataFrame, per_column: int = 5000) -> set[str]:
    """Text values found in non-numeric, non-date columns (3 characters or fewer are ignored)."""
    out: set[str] = set()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        out.update(str(v) for v in s.dropna().unique()[:per_column])
    return {v for v in out if len(v) >= 4}


def scrub_message(message: str, code: Optional[str], profile: dict,
                  df: Optional[pd.DataFrame] = None, limit: int = 500) -> str:
    """Hide data values in an error message before it is shown to the model.

    1. Any text value that occurs in the dataframe is replaced by <value>, unless it is a
       column name or appears in the model's own code (the model already knows those).
    2. Any other quoted text is replaced by <value> under the same exception.
    3. The result is cut to `limit` characters.
    Limits: a value shorter than 4 characters, a value beyond the first 5000 distinct ones
    per column, and unquoted numbers in a message can slip through.
    """
    code = code or ""
    names = {c["name"] for c in profile.get("columns", [])}
    if df is not None:
        for v in sorted(_known_values(df), key=len, reverse=True):
            if v in message and v not in code and v not in names:
                message = message.replace(v, "<value>")

    def keep_or_hide(m: re.Match) -> str:
        inner = m.group(2)
        return m.group(0) if (inner in names or inner in code) and len(inner) <= 60 else "<value>"

    return _QUOTED.sub(keep_or_hide, message)[:limit]


@dataclass
class Attempt:
    number: int
    ok: bool
    code: Optional[str]
    message: str          # unscrubbed: for your logs and UI, never sent to the model as is


@dataclass
class HealResult:
    ok: bool
    message: str
    final: Optional[ChartResult] = None
    history: list = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.history)

    @property
    def figure(self):
        return self.final.figure if self.final is not None else None

    @property
    def code(self):
        return self.final.code if self.final is not None else None


def build_feedback(history: list[Attempt], profile: dict, df: Optional[pd.DataFrame]) -> str:
    last = history[-1]
    lines = ["", "YOUR PREVIOUS ATTEMPT FAILED. Fix it."]
    if last.code:
        lines += ["Your previous code:", "```python", last.code, "```"]
    else:
        lines.append("Your previous reply contained no code block.")
    lines.append("What went wrong: " + scrub_message(last.message, last.code, profile, df))
    earlier = [scrub_message(a.message, a.code, profile, df, limit=200) for a in history[:-1]]
    if earlier:
        lines.append("Earlier failed attempts (do not repeat them): " + " | ".join(earlier))
    lines.append("Reply with ONE complete corrected python code block and nothing else.")
    return "\n".join(lines)


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and " ".join(a.split()) == " ".join(b.split())


async def make_chart_healing_async(df: pd.DataFrame, profile: dict, chart: ChartSpec,
                                   model=DEFAULT_MODEL, max_attempts: int = MAX_ATTEMPTS,
                                   delays=DEFAULT_DELAYS, sleep=asyncio.sleep, on_wait=None) -> HealResult:
    history: list[Attempt] = []
    feedback = ""
    res: Optional[ChartResult] = None
    for n in range(1, max_attempts + 1):
        res = await make_chart_async(df, profile, chart, model, feedback=feedback)
        waits = 0
        while res.transient and waits < len(delays):
            if on_wait:
                on_wait(waits + 1, delays[waits], res.message[:200])
            await sleep(delays[waits])
            waits += 1
            res = await make_chart_async(df, profile, chart, model, feedback=feedback)
        history.append(Attempt(n, res.ok, res.code, res.message))
        if res.ok:
            return HealResult(True, f"Chart built on attempt {n}.", res, history)
        if res.model_error:   # API/rate-limit problem: asking again immediately will not fix the code
            return HealResult(False, res.message, res, history)
        if n > 1 and _same(res.code, history[-2].code):
            return HealResult(False, "The model repeated the same broken code. " + res.message, res, history)
        feedback = build_feedback(history, profile, df)
    return HealResult(False, f"Gave up after {max_attempts} attempts. Last problem: {res.message}", res, history)


def make_chart_healing(df: pd.DataFrame, profile: dict, chart: ChartSpec,
                       model=DEFAULT_MODEL, max_attempts: int = MAX_ATTEMPTS,
                       delays=DEFAULT_DELAYS, sleep=asyncio.sleep, on_wait=None) -> HealResult:
    return asyncio.run(make_chart_healing_async(df, profile, chart, model, max_attempts,
                                                delays=delays, sleep=sleep, on_wait=on_wait))
