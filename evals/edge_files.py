"""Day 19: odd CSV files through the WHOLE no-AI pipeline.

`make_files()` builds about 20 awkward files (empty, one row, duplicate column names, dates in mixed formats, money
written as text, semicolons, accents in the wrong encoding, hostile cell text ...). `run_sweep()` pushes each one
through the same steps the app takes: load, clean, profile, the summary the chat reads, starter questions, filters,
the 30 evaluation questions, the rule-based charts and KPIs, and one real calculation in the sandbox.

A file may be refused at the very first step with a clear message (a plain ValueError raised by our own code:
"this file has no rows"); that is a good outcome. A ValueError that comes straight from a library (pandas' "No columns to
parse from file", a UnicodeDecodeError) is UGLY: the app would not crash, but the person would read a raw technical message.
Anything else that raises is a CRASH: a bug to fix. It uses the real tools, so run it on your own
computer: python scripts/edge_files.py
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

OK, REFUSED, UGLY, CRASH = "OK", "REFUSED", "UGLY", "CRASH"
STAGES = ("load", "clean", "profile", "summary", "starters", "filters", "questions", "gallery", "sandbox")

from evals.edge_live import HOSTILE  # noqa: E402  (one copy of the hostile text, shared with the live script)


def _csv(rows, sep=",") -> bytes:
    return ("\n".join(sep.join(map(str, r)) for r in rows) + "\n").encode()


def make_files() -> dict:
    """name -> file bytes."""
    base = [["order_date", "region", "units", "revenue"]] + \
           [[f"2025-01-{d:02d}", ["East", "West", "North"][d % 3], d, d * 10.5] for d in range(1, 21)]
    files = {
        "empty file (0 bytes)": b"",
        "only blank lines": b"\n\n\n",
        "header only, no rows": _csv([["a", "b", "c"]]),
        "one data row": _csv([["order_date", "region", "revenue"], ["2025-01-01", "East", 10]]),
        "one column": _csv([["revenue"]] + [[i] for i in range(10)]),
        "all text columns": _csv([["name", "city", "note"]] + [[f"n{i}", ["Pune", "Delhi"][i % 2], "ok"] for i in range(10)]),
        "all numeric columns": _csv([["a", "b", "c"]] + [[i, i * 2, i * 3.5] for i in range(30)]),
        "duplicate column names": _csv([["revenue", "revenue", "region"]] + [[i, i + 1, "East"] for i in range(10)]),
        "blank and unnamed columns": _csv([["revenue", "", "region", ""]] + [[i, "x", "East", i] for i in range(10)]),
        "spaces and symbols in names": _csv([["Order Date", "Sales $", "Region (name)", "% off"]] +
                                           [[f"2025-02-{d:02d}", d * 3.2, ["East", "West"][d % 2], d % 7] for d in range(1, 21)]),
        "mixed date formats": _csv([["when", "revenue"], ["01/02/2025", 5], ["2025-02-03", 6], ["Feb 4 2025", 7],
                                    ["5 March 2025", 8], ["not a date", 9], ["", 10]]),
        "money written as text": _csv([["item", "price"]] + [[f"i{i}", f'"${i * 1000 + 0.5:,.2f}"'] for i in range(1, 12)]),
        "semicolon separated": _csv(base, sep=";"),
        "accents in latin-1 encoding": "name,city,revenue\nJosé,São Paulo,10\nZoë,Zürich,20\nRenée,Genève,30\n".encode("latin-1"),
        "starts with a BOM": b"\xef\xbb\xbf" + _csv(base),
        "a completely empty column": _csv([["a", "empty", "revenue"]] + [[i, "", i * 2] for i in range(15)]),
        "60 columns": _csv([[f"c{i}" for i in range(60)]] + [[(r + i) % 9 for i in range(60)] for r in range(25)]),
        "very long text with quotes and newlines": ('note,revenue\n"' + ("a long note with ""quotes"" and\nnewlines " * 80)
                                                    + '",10\n"short",20\n').encode(),
        "hostile text in a category": _csv([["region", "units", "revenue"]] + [[HOSTILE if i % 3 == 0 else ["East", "West"][i % 2], i, i * 2.5] for i in range(30)]),
        "hindi and chinese values": _csv([["शहर", "city", "revenue"]] + [[["पुणे", "दिल्ली"][i % 2], ["北京", "上海"][i % 2], i * 7] for i in range(20)]),
        "negative, huge and infinite numbers": _csv([["a", "b"], [-5, 1e15], [3, -1e15], [float("inf"), 2], [4, float("nan")], [5, 6]]),
        "numbers as ids and codes": _csv([["zip", "phone", "revenue"]] + [[f"0{i:04d}", f"+91 98{i:08d}", i] for i in range(15)]),
    }
    return files


@dataclass
class FileOutcome:
    name: str
    stages: dict = field(default_factory=dict)        # stage -> (status, detail)

    @property
    def crashes(self) -> list:
        return [(s, d) for s, (st, d) in self.stages.items() if st == CRASH]

    @property
    def refused_at(self):
        return next((s for s, (st, _) in self.stages.items() if st == REFUSED), None)

    @property
    def ugly_at(self):
        return next((s for s, (st, _) in self.stages.items() if st == UGLY), None)


def _stage(out: FileOutcome, name: str, fn, *, refusal=()):
    """Runs one step. Returns its value, or None after recording REFUSED (an allowed exception) or CRASH."""
    try:
        value = fn()
    except refusal as e:
        friendly = type(e) is ValueError                     # our own message, not a library's subclass of ValueError
        out.stages[name] = (REFUSED if friendly else UGLY, f"{type(e).__name__}: {str(e)[:160]}")
        return None
    except Exception as e:
        last = traceback.extract_tb(e.__traceback__)[-1]
        out.stages[name] = (CRASH, f"{type(e).__name__}: {str(e)[:120]} ({last.filename.split('/')[-1]}:{last.lineno})")
        return None
    out.stages[name] = (OK, "")
    return value if value is not None else True


def run_file(name: str, data: bytes, runner=None) -> FileOutcome:
    from agents.chat_agent import default_runner, format_result, profile_text_for_chat, sanitize_profile_text, starter_questions
    from evals.chat_eval import build_questions
    from tools.charts import ChartError, build_chart, compute_kpis, suggest_charts
    from tools.cleaner import clean_dataframe
    from tools.filters import apply_filters, filter_options
    from tools.loader import load_csv_bytes
    from tools.profiler import profile_dataframe, profile_to_text

    out = FileOutcome(name)
    loaded = _stage(out, "load", lambda: load_csv_bytes(data), refusal=(ValueError,))
    if loaded is None:
        return out
    df = loaded[0] if isinstance(loaded, tuple) else loaded
    cleaned = _stage(out, "clean", lambda: clean_dataframe(df))
    if cleaned is None:
        return out
    df = cleaned[0] if isinstance(cleaned, tuple) else cleaned
    profile = _stage(out, "profile", lambda: profile_dataframe(df))
    if profile is None:
        return out

    def summary():
        text = profile_to_text(profile)
        chat = sanitize_profile_text(profile_text_for_chat(profile))
        assert isinstance(text, str) and isinstance(chat, str) and chat.strip(), "empty summary"
        assert max((len(ln) for ln in chat.split("\n")), default=0) <= 410, "a very long line reached the AI"
    _stage(out, "summary", summary)
    _stage(out, "starters", lambda: [str(q) for q in starter_questions(profile)])

    def filters():
        options = filter_options(df, profile)
        everything = {"values": {c: list(v) for c, v in options["values"].items()}}
        assert len(apply_filters(df, options, everything)) == len(df), "choosing everything changed the rows"
        for col, values in options["values"].items():
            part = apply_filters(df, options, {"values": {col: values[:1]}})
            assert len(part) <= len(df)
    _stage(out, "filters", filters)
    _stage(out, "questions", lambda: build_questions(df))

    def gallery():
        compute_kpis(df, profile)
        for spec in suggest_charts(profile):
            try:
                build_chart(df, spec)
            except ChartError:
                pass                                      # a clean refusal to draw one chart is fine
    _stage(out, "gallery", gallery)

    def sandbox():
        res = (runner or default_runner)("result = int(len(df))", df)
        assert getattr(res, "ok", False), f"the sandbox could not even count rows: {getattr(res, 'status', '?')}"
        assert str(len(df)) in format_result(res), "the row count did not come back"
    _stage(out, "sandbox", sandbox)
    return out


def run_sweep(files: dict | None = None, progress=None, runner=None) -> list:
    results = []
    for name, data in (files or make_files()).items():
        outcome = run_file(name, data, runner)
        results.append(outcome)
        if progress:
            progress(outcome)
    return results


def describe(outcome: FileOutcome) -> str:
    if outcome.crashes:
        return "CRASH   " + "; ".join(f"{s}: {d}" for s, d in outcome.crashes)
    if outcome.ugly_at:
        return f"UGLY    at {outcome.ugly_at} (raw library message): " + outcome.stages[outcome.ugly_at][1]
    if outcome.refused_at:
        return f"REFUSED at {outcome.refused_at}: " + outcome.stages[outcome.refused_at][1]
    return "OK"
