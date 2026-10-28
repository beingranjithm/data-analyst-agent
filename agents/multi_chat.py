"""Day 22: the chat can answer questions that need several tables (several uploaded CSV files).

Three small pieces, none of them calls a model:
  * multi_note()      - a text block added to the data summary: which tables exist, how they link, and the rules
                        for combining them. The chat agent's own instructions stay unchanged.
  * check_combination() - looks at code the AI wrote BEFORE it runs. If it merges two "event" tables (results,
                        lap_times ...) without summarising either first, the numbers would be inflated, so the
                        code is blocked with an explanation and the AI gets to fix it (the normal retry path).
  * make_multi_runner() - the sandbox runner for several tables: `df` is the analysed (flattened) table and
                        `tables` holds every table; the check above runs first.

Limit, stated honestly: the check is a rule of thumb on the code text. It stops the common mistake (merge two big
tables, then add things up). It cannot prove that every other piece of code is right; that is what the visible
calculation and the F1 test questions are for.
"""
from __future__ import annotations

import ast

from tools.relations import event_tables, lookup_tables

MAX_COLUMNS_LISTED = 40
MAX_LINKS_LISTED = 40
AGGREGATORS = {"groupby", "pivot_table", "drop_duplicates", "agg", "aggregate", "value_counts", "nunique",
               "idxmax", "idxmin", "crosstab", "resample"}
MERGERS = {"merge", "join", "merge_asof"}


def multi_note(tables: dict, links: list, active: str) -> str:
    """Text for the chat's data summary. `tables` maps name -> DataFrame, `active` is the one called `df`."""
    events = set(event_tables(tables, links))
    lookups = lookup_tables(tables, links)
    kids = {l.child for l in links}
    lines = ["SEVERAL TABLES",
             f"The person uploaded {len(tables)} tables. In your code `df` is the table '{active}' (its summary is "
             "above) and `tables` is a dict with every table: tables['name'] is a DataFrame."]
    for name, frame in tables.items():
        kind = ("event table: one row per thing that happened" if name in events
                else "lookup table: one row per item, other tables point to it" if name in lookups
                else "linked table: it points to another table and other tables point to it" if name in kids
                else "table")
        cols = list(map(str, frame.columns))
        shown = ", ".join(cols[:MAX_COLUMNS_LISTED]) + (" ..." if len(cols) > MAX_COLUMNS_LISTED else "")
        if name == active:
            lines.append(f"- {name} ({len(frame):,} rows): this is `df`; it already contains the columns of the lookup "
                         "tables it links to (their names start with the table name when they were generic).")
        else:
            lines.append(f"- {name} ({len(frame):,} rows, {kind}): columns {shown}")
    if links:
        lines.append("")
        lines.append("LINKS (every value of the first column exists in the second; the second has no repeats):")
        lines += [f"- {l.text()}" for l in links[:MAX_LINKS_LISTED]]
        if len(links) > MAX_LINKS_LISTED:
            lines.append(f"- ... and {len(links) - MAX_LINKS_LISTED} more")
    lines += ["",
              "RULES FOR USING SEVERAL TABLES",
              "- Prefer `df` when it has the columns you need.",
              "- To add columns from a lookup table, merge it on the link columns with how='left'. Rows stay the same.",
              "- NEVER merge two event tables directly: each row would be repeated for every match and every count or "
              "total becomes too big. Summarise one of them first with groupby (one row per key), then merge.",
              "- Count THINGS, not rows: when a table has several rows per thing (one row per result, per lap, per pit "
              "stop), count distinct ids with nunique(), for example the number of races is raceId.nunique(), not len(df).",
              "- Columns that were generic in a lookup table start with its table name in `df` (the year of a race is "
              "races_year, not year). Check the column list above before filtering, and filter numbers with numbers, not text.",
              "- After a merge, compare len() before and after and return both, so the person can see the rows stayed the same.",
              "- Columns of the other tables keep their original names (no table prefix). Copy names exactly.",
              "- Missing values written as \\N in the files are already treated as missing."]
    return "\n".join(lines)


def _event_names_used(code: str, active: str, origin: dict, events: set):
    """The event tables the code touches: tables['x'] / tables.get('x') for an event table x, plus `df` (the active
    table counts as its source event table)."""
    tree = ast.parse(code)
    used, merges, aggregates = set(), False, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "tables":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                used.add(origin.get(key.value, key.value))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "tables" and node.args
              and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            used.add(origin.get(node.args[0].value, node.args[0].value))
        elif isinstance(node, ast.Name) and node.id == "df":
            used.add(origin.get(active, active))
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            merges = merges or name in MERGERS
            aggregates = aggregates or name in AGGREGATORS
    return used & events, merges, aggregates


def check_combination(code: str, active: str, origin: dict, events: set):
    """None when fine, else the reason (a sentence the AI can act on). `origin` maps a flattened table's name to the
    event table it came from (results_flat -> results)."""
    try:
        used, merges, aggregates = _event_names_used(code, active, origin, set(events))
    except SyntaxError:
        return None                                    # the sandbox reports syntax errors itself
    if len(used) >= 2 and merges and not aggregates:
        return (f"this code merges two event tables ({', '.join(sorted(used))}) without summarising either one first. "
                "Every row of one would be repeated for each matching row of the other, so counts and totals come out "
                "too big. Summarise one table to one row per key with groupby first (for example one row per raceId "
                "and driverId), or use the columns already in `df`, then merge")
    return None


def check_number_as_text(code: str, tables: dict):
    """None when fine, else a sentence for the AI. Catches df['year'] == '2020' when year holds numbers: that matches
    no row, and the AI then reports 'nothing matched' for data that is there."""
    import pandas as pd
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    def column(n):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            return n.slice.value
        return None

    def numeric_text(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            try:
                float(n.value)
                return n.value
            except ValueError:
                return None
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left] + list(node.comparators)
        for op, a, b in zip(node.ops, sides, sides[1:]):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for col_side, text_side in ((a, b), (b, a)):
                col, text = column(col_side), numeric_text(text_side)
                if col is None or text is None:
                    continue
                kinds = [pd.api.types.is_numeric_dtype(t[col]) and not pd.api.types.is_bool_dtype(t[col])
                         for t in tables.values() if hasattr(t, "columns") and col in t.columns]
                if kinds and all(kinds):
                    return (f"the column '{col}' holds numbers, but the code compares it with the text '{text}'. That "
                            f"matches no rows. Compare with the number {text.strip()} (no quotes)")
    return None


class BlockedRun:
    """What the chat agent needs from a result that never ran (same shape as the sandbox's blocked result)."""
    status = "blocked"
    ok = False
    table = None
    value = None
    stdout = ""
    error = None
    figure = None

    def __init__(self, reason: str):
        self.problems = [reason]

    def message(self) -> str:
        return "Code was blocked before running: " + "; ".join(self.problems)


def make_multi_runner(tables: dict, links: list, active: str, *, inner=None, origin: dict | None = None):
    """A runner(code, tables) for ChatSession. `inner(code, tables)` runs the code (default: the real sandbox with
    `active` as `df`). `origin` says which event table the active table was built from."""
    events = set(event_tables(tables, links))
    origin = dict(origin or {})
    if inner is None:
        def inner(code, tbls):
            from agents.chat_agent import CALC_TIMEOUT
            from tools.sandbox import run_code
            return run_code(code, tbls, active=active, timeout=CALC_TIMEOUT)

    def runner(code, tbls):
        reason = check_combination(code, active, origin, events) or check_number_as_text(code, tbls)
        if reason:
            return BlockedRun(reason)
        return inner(code, tbls)
    return runner


def build_multi_chat(active: str, active_df, frames: dict, links: list, *, origin: dict | None = None) -> dict:
    """Everything the chat needs for several tables: {"tables", "note", "runner"}. `frames` = {name: DataFrame} of the
    uploaded tables; `active_df` is the analysed table (possibly filtered), named `active`, and comes first (= `df`)."""
    tables = {active: active_df, **{n: f for n, f in frames.items() if n != active}}
    return {"tables": tables, "note": multi_note(tables, links, active),
            "runner": make_multi_runner(tables, links, active, origin=origin)}
