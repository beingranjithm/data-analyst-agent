"""Day 18: how accurate is the chat? 30 questions whose true answers are computed with pandas on the same table,
and a checker that reads the chat's answer and compares.

This file has no AI and no Streamlit in it: `build_questions(df)` makes the questions and their true answers,
`score(...)` judges one answer, `run_eval(...)` asks every question through any `ask(text)` function and keeps
going safely (it stops cleanly when the daily quota or your request budget runs out), and `render_markdown(...)`
writes the report. scripts/eval_chat.py wires it to the real chat.

How an answer is judged (it is automatic, so it has limits; see the report's "How to read this"):
- numbers: every expected number must appear in the answer as the AI would show it (1,394,618.70 or 1394619
  or 1.4M all match 1394618.7; 1.5M does not). Where two readings are fair (a standard deviation can divide by n or
  by n-1; a discount can be 0.05 or 5%) either one is accepted.
- names: every expected name (or date, or month) must appear; for a ranking they must appear in the right order.
- trap questions (no such column, no such region, a year with no data, a forecast): a good answer says it is not
  in the data and quotes no number the summary or calculation did not provide.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

PASS, MINOR, FAIL, NO_ANSWER, NOT_RUN = "PASS", "MINOR", "FAIL", "NO_ANSWER", "NOT_RUN"
NOT_FOUND_WORDS = ("no ", "not ", "n't", "none", "nothing", "cannot", "unable", "does not", "do not",
                   "only covers", "only contains")

_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
_NUM = re.compile(r"(?<![\w.])(-?(?:\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?))"
                  r"(?:\s?(%|k\b|m\b|bn\b|thousand\b|million\b|billion\b))?", re.IGNORECASE)


# ------------------------------------------------------------------ reading an answer
def parse_numbers(text: str) -> list:
    """[(value, decimals shown, scale, is_percent)] for every number in the text ("1.4M" -> 1.4, 1, 1e6, False)."""
    out = []
    for m in _NUM.finditer(text):
        tok, suffix = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        out.append((float(tok), decimals, _SCALES.get(suffix, 1.0), suffix == "%"))
    return out


def shows_number(text: str, truth: float) -> bool:
    """True when the text contains `truth` as written or rounded to the digits shown (also 1.4M, 97.19%)."""
    truth = float(truth)
    for value, decimals, scale, is_pct in parse_numbers(text):
        candidates = [value] if not is_pct else [value, value / 100]
        for v in candidates:
            if abs(v * scale - truth) <= 0.5 * 10 ** -decimals * scale + 1e-9 * max(1.0, abs(truth)):
                return True
    return False


def shows_any_number(text: str, truth) -> bool:
    options = truth if isinstance(truth, (tuple, list)) else (truth,)
    return any(shows_number(text, t) for t in options)


def _find_name(text: str, name: str) -> int:
    m = re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text, re.IGNORECASE)
    return m.start() if m else -1


def find_name(text: str, name) -> int:
    """Position of a name (or the first of several fair spellings) in the text; -1 when it is not there."""
    options = name if isinstance(name, (tuple, list)) else (name,)
    hits = [p for p in (_find_name(text, str(o)) for o in options) if p >= 0]
    return min(hits) if hits else -1


def says_not_found(text: str) -> bool:
    return any(w in text.lower() for w in NOT_FOUND_WORDS)


def claims_zero_total(text: str) -> bool:
    """True when the answer states a total of 0 ('the total revenue is 0.' counts; '0.5' does not)."""
    return bool(re.search(r"(revenue|total)\s+(is|was|of|equals)\s+\$?0(?:\.0+)?(?!\d|\.\d)", text.lower()))


# ------------------------------------------------------------------ the questions
@dataclass
class Expectation:
    numbers: list = field(default_factory=list)       # each item: a number, or a tuple of fair alternatives
    names: list = field(default_factory=list)         # each item: a text, or a tuple of fair spellings
    ordered: bool = False                             # the names must appear in this order
    trap: bool = False                                # the right answer is "that is not in the data"
    allow_zero: bool = False                          # a trap where quoting a zero total is only a minor slip
    describe: str = ""                                # the truth in words, for the report


@dataclass
class Question:
    id: int
    category: str
    text: str
    expect: Expectation


def _date_forms(t: pd.Timestamp) -> tuple:
    d, ordinal = t.day, "th" if 10 <= t.day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(t.day % 10, "th")
    return (t.strftime("%Y-%m-%d"), f"{t:%B} {d}, {t.year}", f"{t:%B} {d:02d}, {t.year}", f"{d} {t:%B} {t.year}",
            f"{t:%b} {d}, {t.year}", f"{d} {t:%b} {t.year}", f"{t:%B} {d}{ordinal}, {t.year}", f"{t:%d/%m/%Y}", f"{t:%m/%d/%Y}")


def _need(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("no column " + ", ".join(missing))


def _subset(df: pd.DataFrame, mask: pd.Series, what: str) -> pd.DataFrame:
    out = df[mask]
    if out.empty:
        raise ValueError(f"no rows for {what}")
    return out


def _pretty(x) -> str:
    return f"{x:,.2f}" if isinstance(x, float) and not float(x).is_integer() else f"{x:,.0f}"


def _month(period: pd.Period) -> tuple:
    return (period.strftime("%B"), period.strftime("%b"), str(period), period.strftime("%m/%Y"))


def _q_definitions() -> list:
    """(category, question text, function(df) -> Expectation). Every function raises when this data cannot answer it."""
    D = []

    def add(category, text):
        def deco(fn):
            D.append((category, text, fn))
            return fn
        return deco

    # ---- simple totals and averages
    @add("totals", "What is the total revenue?")
    def _(df):
        _need(df, "revenue"); v = float(df["revenue"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("totals", "How many units were sold in total?")
    def _(df):
        _need(df, "units"); v = float(df["units"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("totals", "How many orders (rows) are in the data?")
    def _(df):
        return Expectation([float(len(df))], describe=_pretty(float(len(df))))

    @add("totals", "What is the average revenue per order?")
    def _(df):
        _need(df, "revenue"); v = float(df["revenue"].mean())
        return Expectation([v], describe=_pretty(v))

    @add("totals", "What is the largest single-order revenue?")
    def _(df):
        _need(df, "revenue"); v = float(df["revenue"].max())
        return Expectation([v], describe=_pretty(v))

    @add("totals", "What is the average discount?")
    def _(df):
        _need(df, "discount"); v = float(df["discount"].mean())
        return Expectation([(v, v * 100)], describe=f"{v:.4f} (or {v * 100:.2f}%)")

    # ---- which group is the winner
    @add("groups", "Which region has the highest total revenue, and what is that total?")
    def _(df):
        _need(df, "region", "revenue"); s = df.groupby("region")["revenue"].sum()
        return Expectation([float(s.max())], [str(s.idxmax())], describe=f"{s.idxmax()}, {_pretty(float(s.max()))}")

    @add("groups", "Which product sold the fewest units, and how many units was that?")
    def _(df):
        _need(df, "product", "units"); s = df.groupby("product")["units"].sum()
        return Expectation([float(s.min())], [str(s.idxmin())], describe=f"{s.idxmin()}, {_pretty(float(s.min()))}")

    @add("groups", "Which category has the lowest total revenue?")
    def _(df):
        _need(df, "category", "revenue"); s = df.groupby("category")["revenue"].sum()
        return Expectation([], [str(s.idxmin())], describe=str(s.idxmin()))

    @add("groups", "What are the top 3 products by total revenue? List them from highest to lowest.")
    def _(df):
        _need(df, "product", "revenue"); top = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(3)
        if len(top) < 3:
            raise ValueError("fewer than 3 products")
        return Expectation([], [str(p) for p in top.index], ordered=True, describe=", ".join(map(str, top.index)))

    @add("groups", "Which region has the highest average revenue per order?")
    def _(df):
        _need(df, "region", "revenue"); s = df.groupby("region")["revenue"].mean()
        return Expectation([], [str(s.idxmax())], describe=str(s.idxmax()))

    @add("groups", "Which region has the most orders, and how many orders is that?")
    def _(df):
        _need(df, "region"); s = df["region"].value_counts()
        return Expectation([float(s.max())], [str(s.idxmax())], describe=f"{s.idxmax()}, {int(s.max())}")

    # ---- filters
    @add("filters", "What is the total revenue for the East region?")
    def _(df):
        _need(df, "region", "revenue"); v = float(_subset(df, df["region"] == "East", "East")["revenue"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("filters", "How many orders were placed in the West region?")
    def _(df):
        _need(df, "region"); n = float(len(_subset(df, df["region"] == "West", "West")))
        return Expectation([n], describe=_pretty(n))

    @add("filters", "What is the total revenue of the Electronics category in the North region?")
    def _(df):
        _need(df, "category", "region", "revenue")
        sub = _subset(df, (df["category"] == "Electronics") & (df["region"] == "North"), "Electronics in North")
        v = float(sub["revenue"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("filters", "How many units of Desk were sold?")
    def _(df):
        _need(df, "product", "units"); v = float(_subset(df, df["product"] == "Desk", "Desk")["units"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("filters", "How many orders had a discount greater than zero?")
    def _(df):
        _need(df, "discount"); n = float((df["discount"] > 0).sum())
        if n == 0:
            raise ValueError("no discounted orders")
        return Expectation([n], describe=_pretty(n))

    # ---- time
    @add("time", "What was the total revenue in March 2025?")
    def _(df):
        _need(df, "order_date", "revenue"); d = df["order_date"]
        v = float(_subset(df, (d.dt.year == 2025) & (d.dt.month == 3), "March 2025")["revenue"].sum())
        return Expectation([v], describe=_pretty(v))

    @add("time", "Which month had the highest total revenue, and what was that total?")
    def _(df):
        _need(df, "order_date", "revenue"); s = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
        return Expectation([float(s.max())], [_month(s.idxmax())], describe=f"{s.idxmax()}, {_pretty(float(s.max()))}")

    @add("time", "On how many different days were orders placed?")
    def _(df):
        _need(df, "order_date"); n = float(df["order_date"].dt.normalize().nunique())
        return Expectation([n], describe=_pretty(n))

    @add("time", "What is the date of the first order and the date of the last order?")
    def _(df):
        _need(df, "order_date"); lo, hi = df["order_date"].min(), df["order_date"].max()
        return Expectation([], [_date_forms(lo), _date_forms(hi)], describe=f"{lo:%Y-%m-%d} and {hi:%Y-%m-%d}")

    @add("time", "What was the total revenue in the first quarter of 2025 (January to March)?")
    def _(df):
        _need(df, "order_date", "revenue"); d = df["order_date"]
        v = float(_subset(df, (d.dt.year == 2025) & (d.dt.month <= 3), "Q1 2025")["revenue"].sum())
        return Expectation([v], describe=_pretty(v))

    # ---- statistics
    @add("statistics", "What is the median revenue per order?")
    def _(df):
        _need(df, "revenue"); v = float(df["revenue"].median())
        return Expectation([v], describe=_pretty(v))

    @add("statistics", "What is the standard deviation of units per order?")
    def _(df):
        _need(df, "units"); u = df["units"].dropna(); a, b = float(u.std(ddof=1)), float(u.std(ddof=0))
        return Expectation([(a, b)], describe=f"{a:.4f} (sample) or {b:.4f} (population)")

    @add("statistics", "What is the correlation between units and revenue?")
    def _(df):
        _need(df, "units", "revenue"); v = float(df["units"].corr(df["revenue"]))
        return Expectation([v], describe=f"{v:.4f}")

    # ---- traps: the right answer is "that is not in the data"
    @add("traps", "What is the average customer age?")
    def _(df):
        if any("age" in str(c).lower() for c in df.columns):
            raise ValueError("this data has an age column")
        return Expectation(trap=True, describe="no such column: say so, no number")

    @add("traps", "What was the profit margin?")
    def _(df):
        if any(w in str(c).lower() for c in df.columns for w in ("profit", "cost", "margin")):
            raise ValueError("this data has cost or profit columns")
        return Expectation(trap=True, describe="no cost or profit column: say so, no number")

    @add("traps", "What is the total revenue for the Atlantis region?")
    def _(df):
        _need(df, "region")
        if (df["region"] == "Atlantis").any():
            raise ValueError("this data has an Atlantis region")
        return Expectation(trap=True, allow_zero=True, describe="no such region: say so (quoting a zero total is a minor slip)")

    @add("traps", "What will the total revenue be next year?")
    def _(df):
        return Expectation(trap=True, describe="a forecast is not in the data: say so, no invented number")

    @add("traps", "What was the total revenue in 2030?")
    def _(df):
        _need(df, "order_date")
        if (df["order_date"].dt.year == 2030).any():
            raise ValueError("this data has 2030")
        return Expectation(trap=True, allow_zero=True, describe="no data for 2030: say so (quoting a zero total is a minor slip)")

    return D


def build_questions(df: pd.DataFrame) -> tuple:
    """(questions, skipped). A question this table cannot answer (no such column, no such group) is skipped, with
    the reason, instead of being asked."""
    questions, skipped = [], []
    for n, (category, text, fn) in enumerate(_q_definitions(), 1):
        try:
            exp = fn(df)
            for x in exp.numbers:
                if not all(math.isfinite(v) for v in (x if isinstance(x, tuple) else (x,))):
                    raise ValueError("the true answer is not a number (NaN)")
            questions.append(Question(n, category, text, exp))
        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            skipped.append((n, text, str(e).strip("'\"")))
    return questions, skipped


# ------------------------------------------------------------------ judging one answer
@dataclass
class Verdict:
    status: str
    reasons: list


def score(q: Question, answer: str, ungrounded: Sequence = ()) -> Verdict:
    e, reasons = q.expect, []
    if e.trap:
        if ungrounded:
            return Verdict(FAIL, ["it quoted numbers that were in neither the summary nor a calculation: " + ", ".join(map(str, ungrounded))])
        if not says_not_found(answer):
            return Verdict(FAIL, ["it did not say that this is not in the data"])
        if claims_zero_total(answer):
            return Verdict(MINOR if e.allow_zero else FAIL, ["it says nothing matched but still quotes a total of 0"])
        return Verdict(PASS, [])
    for x in e.numbers:
        if not shows_any_number(answer, x):
            shown = ", ".join(f"{v:g}" for v, *_ in parse_numbers(answer)[:6]) or "none"
            want = " or ".join(f"{v:,.4f}".rstrip("0").rstrip(".") for v in (x if isinstance(x, tuple) else (x,)))
            reasons.append(f"expected {want}; numbers in the answer: {shown}")
    positions = []
    for name in e.names:
        p = find_name(answer, name)
        if p < 0:
            label = name[0] if isinstance(name, (tuple, list)) else name
            reasons.append(f"expected {label!r} in the answer")
        positions.append(p)
    if e.ordered and not reasons and positions != sorted(positions):
        reasons.append("the names are in the wrong order")
    return Verdict(FAIL if reasons else PASS, reasons)


# ------------------------------------------------------------------ running the whole set
@dataclass
class Result:
    id: int
    category: str
    question: str
    status: str
    reasons: list = field(default_factory=list)
    expected: str = ""
    answer: str = ""
    tries: int = 0                        # how many times the AI's code was run for this question
    healed: bool = False                  # passed after a failed or empty first try
    corrected: bool = False               # Day 18b: the chat rewrote its answer once after a number check
    ungrounded: list = field(default_factory=list)
    error: str = ""


def _is_quota(error: str) -> bool:
    e = (error or "").lower()
    return "quota" in e and ("daily" in e or "used up" in e)


def run_eval(questions: Sequence[Question], ask: Callable, *, requests_used: Optional[Callable] = None,
             max_requests: Optional[int] = None, progress: Optional[Callable] = None) -> list:
    """Asks every question with `ask(text) -> ChatReply` (ok, text, ungrounded, error, calculations). Stops cleanly,
    marking the rest NOT_RUN, when the daily quota is used up or `requests_used()` reaches `max_requests`."""
    results, stop_reason = [], ""
    for q in questions:
        base = dict(id=q.id, category=q.category, question=q.text, expected=q.expect.describe)
        if not stop_reason and max_requests is not None and requests_used is not None and requests_used() >= max_requests:
            stop_reason = f"request budget of {max_requests} reached"
        if stop_reason:
            results.append(Result(status=NOT_RUN, error=stop_reason, **base))
            continue
        try:
            reply = ask(q.text)
        except Exception as e:                                       # one broken question must not end the run
            results.append(Result(status=NO_ANSWER, error=f"{type(e).__name__}: {e}"[:300], **base))
        else:
            calcs = list(getattr(reply, "calculations", None) or [])
            if not reply.ok:
                if _is_quota(reply.error):
                    stop_reason = "the free daily quota is used up"
                results.append(Result(status=NO_ANSWER, error=str(reply.error)[:300], tries=len(calcs), **base))
            else:
                v = score(q, reply.text, reply.ungrounded)
                healed = (v.status in (PASS, MINOR) and len(calcs) > 1
                          and (not calcs[0].ok or getattr(calcs[0], "empty", False) or getattr(calcs[0], "zero", False)))
                results.append(Result(status=v.status, reasons=v.reasons, answer=reply.text, tries=len(calcs),
                                      healed=healed, ungrounded=list(reply.ungrounded or []),
                                      corrected=bool(getattr(reply, "corrected", False)), **base))
        if progress:
            progress(results[-1])
    return results


# ------------------------------------------------------------------ the report
def summarize(results: Sequence[Result]) -> dict:
    count = lambda *s: sum(1 for r in results if r.status in s)          # noqa: E731
    answered = count(PASS, MINOR, FAIL)
    cats = {}
    for r in results:
        c = cats.setdefault(r.category, dict(asked=0, passed=0, minor=0, failed=0, other=0))
        if r.status == PASS: c["passed"] += 1
        elif r.status == MINOR: c["minor"] += 1
        elif r.status == FAIL: c["failed"] += 1
        else: c["other"] += 1
        c["asked"] += 1
    wrong = [r for r in results if r.status == FAIL]
    right = [r for r in results if r.status in (PASS, MINOR)]
    return dict(
        total=len(results), answered=answered, passed=count(PASS), minor=count(MINOR), failed=count(FAIL),
        no_answer=count(NO_ANSWER), not_run=count(NOT_RUN),
        strict=(count(PASS) / answered) if answered else None,
        lenient=((count(PASS) + count(MINOR)) / answered) if answered else None,
        healed=sum(1 for r in results if r.healed), categories=cats,
        corrected=sum(1 for r in results if r.corrected),
        corrected_ok=sum(1 for r in results if r.corrected and r.status in (PASS, MINOR)),
        false_alarms=sum(1 for r in right if r.ungrounded),               # a correct answer that was flagged
        wrong_flagged=sum(1 for r in wrong if r.ungrounded),             # a wrong answer the app flagged
        calc_runs=sum(r.tries for r in results if r.status in (PASS, MINOR, FAIL)),
    )


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_markdown(results: Sequence[Result], skipped: Sequence = (), meta: Optional[dict] = None) -> str:
    s, meta = summarize(results), meta or {}
    lines = ["# Chat accuracy evaluation", ""]
    if meta:
        lines += [" · ".join(f"{k}: {v}" for k, v in meta.items()), ""]
    lines += [f"**{s['passed']} of {s['answered']} answered questions fully correct ({_pct(s['strict'])})**"
              + (f", plus {s['minor']} with a minor slip ({_pct(s['lenient'])} counting those)" if s["minor"] else "") + ".", ""]
    if s["no_answer"] or s["not_run"]:
        lines += [f"> Partial run: {s['no_answer']} question(s) got no answer and {s['not_run']} were not run "
                  "(daily quota, request budget or a missing saved reply). Run the script again later: answered "
                  "questions replay for free.", ""]
    lines += ["| Category | Asked | Correct | Minor slip | Wrong | No answer / not run |", "|---|---|---|---|---|---|"]
    for name, c in s["categories"].items():
        lines.append(f"| {name} | {c['asked']} | {c['passed']} | {c['minor']} | {c['failed']} | {c['other']} |")
    lines += ["", f"- Self-healed (correct after a failed, empty or zero first try): {s['healed']}",
              f"- Number-copy guard: {s['corrected']} answer(s) rewritten once after a number check, "
              f"{s['corrected_ok']} of them correct afterwards",
              f"- Calculations run for the answered questions: {s['calc_runs']}",
              f"- The app's number check flagged {s['wrong_flagged']} of {s['failed']} wrong answers, and raised "
              f"{s['false_alarms']} false alarm(s) on correct answers", ""]
    bad = [r for r in results if r.status == FAIL]
    if bad:
        lines += ["## Wrong answers", ""]
        for r in bad:
            lines += [f"**{r.id}. {r.question}**", f"- true answer: {r.expected}", f"- the AI said: {r.answer[:400]!r}"]
            lines += [f"- {why}" for why in r.reasons]
            lines.append("")
    minor = [r for r in results if r.status == MINOR]
    if minor:
        lines += ["## Minor slips", ""] + [f"- {r.id}. {r.question} → {r.answer[:200]!r}" for r in minor] + [""]
    other = [r for r in results if r.status in (NO_ANSWER, NOT_RUN)]
    if other:
        lines += ["## No answer", ""] + [f"- {r.id}. {r.question} → {r.status}: {r.error[:160]}" for r in other] + [""]
    if skipped:
        lines += ["## Skipped (this table cannot answer them)", ""] + [f"- {n}. {t} ({why})" for n, t, why in skipped] + [""]
    lines += ["## All questions", "", "| # | Category | Question | Result | True answer |", "|---|---|---|---|---|"]
    lines += [f"| {r.id} | {r.category} | {r.question} | {r.status} | {r.expected} |" for r in results]
    lines += ["", "## How to read this", "",
              "- The true answers come from pandas on the same table the chat uses. The score is automatic, so it can be "
              "unfair in both directions: a correct answer worded in an unexpected way may be marked wrong, and an answer "
              "with the right number in the wrong context may be marked right. Read the wrong answers above.",
              "- Trap questions (no such column, region, year or forecast) use simple wording checks and the app's own "
              "number check. A WRONG there deserves a look at the answer before you worry.",
              "- One run is a snapshot: the model can answer the same question differently next time (use `--no-cache` "
              "to ask again).", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ saving the report
def save_report(results: Sequence[Result], skipped: Sequence, meta: dict, folder) -> tuple:
    """Writes chat_eval.md and chat_eval.json into `folder`. A run that answered FEWER questions than the report
    already there (a quota-limited or replay-only run) is written as chat_eval_partial.* instead, so a complete
    report is never overwritten by a smaller one. Returns (markdown path, replaced_existing_report)."""
    import json
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    answered = summarize(results)["answered"]
    existing = folder / "chat_eval.json"
    previous = -1
    if existing.exists():
        try:
            previous = int(json.loads(existing.read_text()).get("summary", {}).get("answered", -1))
        except (ValueError, OSError):
            previous = -1
    stem = "chat_eval" if answered >= previous else "chat_eval_partial"
    md = folder / f"{stem}.md"
    md.write_text(render_markdown(results, skipped, meta))
    data = dict(meta=meta, summary=summarize(results), results=[r.__dict__ for r in results],
                skipped=[dict(id=n, question=t, reason=why) for n, t, why in skipped])
    (folder / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str))
    return md, stem == "chat_eval"
