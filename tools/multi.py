"""Day 21b: several CSV files as one data set. Pure functions (no Streamlit), so everything here is testable.

  load_tables   -> read + clean + profile every file; a bad file is reported and the others still load
  choose_focus  -> the table a dashboard is most likely about (an "event" table that reaches the most lookups)
  build_flat    -> the focus table plus the columns of the lookup tables it links to, profiled with link columns
                   treated as identifiers (so 'driverId' is never offered as a number to add up)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe
from tools.relations import Flat, Link, event_tables, flatten

MAX_FILES = 30
MAX_TOTAL_MB = 300


def build_table(name, data: bytes):
    """Load -> clean -> profile. Returns a dict describing one table."""
    df, load_notes = load_csv_bytes(data)
    df, clean_log = clean_dataframe(df)
    return {
        "name": name,
        "df": df,
        "profile": profile_dataframe(df),
        "cleaning_log": load_notes + clean_log,
    }


def table_name(filename: str) -> str:
    """'Race Results.csv' -> 'Race_Results': a name that is safe to use as a table name in code."""
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", Path(filename).stem).strip("_")
    return stem or "table"


def unique_names(names) -> list:
    out, seen = [], {}
    for n in names:
        if n in seen:
            seen[n] += 1
            n2 = f"{n}_{seen[n]}"
            while n2 in seen:
                seen[n] += 1
                n2 = f"{n}_{seen[n]}"
            seen[n2] = 1
            out.append(n2)
        else:
            seen[n] = 1
            out.append(n)
    return out


def files_signature(files) -> str:
    """Short text that changes when any file's name or content changes. files = [(filename, bytes)]."""
    h = hashlib.sha1()
    for name, data in files:
        h.update(name.encode("utf-8", "replace"))
        h.update(hashlib.sha1(data).digest())
    return h.hexdigest()[:12]


def check_limits(files) -> str:
    """'' when the upload is acceptable, otherwise a message for the person."""
    if len(files) > MAX_FILES:
        return f"That is {len(files)} files. The limit is {MAX_FILES} files at a time."
    total = sum(len(d) for _, d in files) / (1024 * 1024)
    if total > MAX_TOTAL_MB:
        return f"These files add up to {total:,.0f} MB. The limit is {MAX_TOTAL_MB} MB in total."
    return ""


def load_tables(files, build: Callable):
    """files = [(filename, bytes)]. build(name, bytes) -> table dict (raises ValueError with a friendly message).
    Returns (tables {name: table dict}, errors [(filename, message)]). One bad file never stops the others."""
    names = unique_names([table_name(f) for f, _ in files])
    tables, errors = {}, []
    for (filename, data), name in zip(files, names):
        try:
            tables[name] = build(name, data)
        except ValueError as e:
            errors.append((filename, str(e)))
        except Exception as e:                                   # noqa: BLE001  (an odd file must not crash the app)
            errors.append((filename, f"Something unexpected went wrong while reading this file: {type(e).__name__}: {str(e)[:150]}"))
    return tables, errors


def _reach(links: list, start: str) -> int:
    out, seen, queue = {}, {start}, [start]
    for l in links:
        if l.kind == "many-to-one":
            out.setdefault(l.child, []).append(l.parent)
    while queue:
        for p in out.get(queue.pop(0), []):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return len(seen) - 1


def table_order(frames: dict, links: list) -> list:
    """Table names for a picker: event tables first (most useful first), then the rest, largest first."""
    events = sorted(event_tables(frames, links), key=lambda t: (-_reach(links, t), -len(frames[t]), t))
    rest = sorted((t for t in frames if t not in events), key=lambda t: (-len(frames[t]), t))
    return events + rest


def choose_focus(frames: dict, links: list) -> str:
    """The table to analyse by default. frames = {name: DataFrame}."""
    if not frames:
        raise ValueError("No tables.")
    return table_order(frames, links)[0]


def links_signature(links) -> str:
    return hashlib.sha1("|".join(sorted(l.text() for l in links)).encode()).hexdigest()[:10]


_ID_TOKEN = re.compile(r"(?i:^id$)|(?i:[_\s.\-]id$)|[a-z0-9]I[dD]$")


def looks_like_id_name(name: str) -> bool:
    """'driverId', 'driver_id', 'ID' -> True; 'width', 'paid', 'Identity' -> False."""
    return bool(_ID_TOKEN.search(str(name).strip()))


TIME_HINTS = ("date", "time", "day", "month", "year", "created", "updated", "order")
_YEARISH = re.compile(r"year|yr|month|quarter|week")


# Numbers that label or locate something rather than measure an amount. They stay available, but come last, so the
# first measure (which the simple gallery uses for its headline) is a real quantity such as points or revenue.
LABEL_NUMBER_WORDS = {"number", "num", "no", "rank", "position", "round", "order", "grid", "index", "lat", "lng", "lon",
                      "latitude", "longitude", "alt", "altitude", "age", "year"}


def _is_label_number(name: str) -> bool:
    tokens = {t for t in re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name)).lower()) if t}
    return bool(tokens & LABEL_NUMBER_WORDS)


def calendar_time_column(columns: list):
    """The column to draw trends against. The profiler may call a lap time like '1:27.452' or a clock time like
    '06:00:00' a date; those all fall on one day, so they are not a calendar. Needs a span of at least two days."""
    real = [c["name"] for c in columns if c["role"] == "datetime" and (c.get("span_days") or 0) >= 2]
    hinted = [n for n in real if any(h in n.lower() for h in TIME_HINTS)]
    return (hinted or real or [None])[0]


def mark_ids(profile: dict, key_columns) -> dict:
    """A copy of `profile` for a table made from several: link columns (and whole-number columns whose name ends in
    id) become identifiers (role 'id') instead of numbers to add up; a year-like or text link column stays a category.
    Measures, dimensions, correlations and the time column are rebuilt to match."""
    keys = set(key_columns)
    prof = dict(profile)
    cols, changed = [], set()
    for c in profile["columns"]:
        c = dict(c)
        id_named = looks_like_id_name(c["name"]) and (c["role"] == "numeric" and c.get("integer_valued", True)
                                                      or c["role"] == "categorical")
        linked_number = c["name"] in keys and c["role"] == "numeric"
        if linked_number and not id_named and _YEARISH.search(c["name"].lower()):
            c["role"] = "categorical"                       # a year that links to a seasons table: a category, not a measure
            c["notes"] = list(c.get("notes", [])) + ["Year-like column treated as a category, not a measure"]
        elif id_named or linked_number:
            c["role"] = "id"
            c["notes"] = list(c.get("notes", [])) + ["Identifier that links tables; not a measurement"]
            c.pop("likely_discrete", None)
            changed.add(c["name"])
        cols.append(c)
    prof["columns"] = cols
    numeric = [c["name"] for c in cols if c["role"] == "numeric"]
    prof["measures"] = ([n for n in numeric if not _is_label_number(n)] + [n for n in numeric if _is_label_number(n)])
    prof["dimensions"] = [c["name"] for c in cols if c["role"] in ("categorical", "boolean")]
    prof["id_columns"] = [c["name"] for c in cols if c["role"] == "id"]
    prof["correlations"] = [p for p in profile.get("correlations", []) if p["a"] not in changed and p["b"] not in changed]
    prof["time_column"] = calendar_time_column(cols)
    return prof


def build_flat(frames: dict, focus: str, links: list, profile_fn: Callable, *, raw_tables: dict = None):
    """The focus table with its lookup columns. Returns (table dict like the app's single-table dict, Flat).
    frames = {name: DataFrame}; raw_tables = {name: table dict} (used for the cleaning log)."""
    flat = flatten(frames, focus, links)
    profile = mark_ids(profile_fn(flat.df), flat.key_columns)
    log = list((raw_tables or {}).get(focus, {}).get("cleaning_log", []))
    joined = [l.parent for l, _ in flat.joined]
    if joined:
        log.append(f"Added the columns of {', '.join(joined)} (rows unchanged: {flat.rows_after:,})")
    for t, why in flat.skipped:
        log.append(f"Did not add {t}: {why}")
    return {"name": focus, "df": flat.df, "profile": profile, "cleaning_log": log}, flat
