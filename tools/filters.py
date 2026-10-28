"""Filters (Day 17): choose a slice of the data (some categories, a date range) in one place.

Pure functions, no Streamlit. `filter_options` says what can be filtered, `apply_filters` cuts the table,
`describe_filters` writes the one-line label shown next to the chat and the dashboard, and `signature` is a
short text that changes whenever the slice changes (used to start a fresh chat for a new slice).
A choice that keeps everything counts as "no filter", so rows with missing values are never dropped silently.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

MAX_FILTER_VALUES = 20       # a column with more different values than this is not offered as a filter
MAX_FILTER_COLUMNS = 4
FILTER_ROLES = ("categorical", "boolean")


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype(str)


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def filter_options(df: pd.DataFrame, profile: dict) -> dict:
    """What can be filtered: {"values": {column: [text, ...]}, "date": (column, first_day, last_day) | None}."""
    values: dict = {}
    for c in profile.get("columns", []):
        if len(values) >= MAX_FILTER_COLUMNS:
            break
        name = c.get("name")
        if c.get("role") not in FILTER_ROLES or name not in df.columns:
            continue
        options = sorted(_as_text(df[name].dropna()).unique())
        if 2 <= len(options) <= MAX_FILTER_VALUES:
            values[name] = options
    date = None
    tc = profile.get("time_column")
    if tc and tc in df.columns:
        days = _dates(df[tc]).dropna()
        if len(days) and days.min() != days.max():
            date = (tc, days.min().date(), days.max().date())
    return {"values": values, "date": date}


def apply_filters(df: pd.DataFrame, options: dict, chosen: dict) -> pd.DataFrame:
    """Returns the rows that match every choice. `chosen` = {"values": {column: [text, ...]}, "date": (start, end)}."""
    keep = pd.Series(True, index=df.index)
    for col, wanted in (chosen.get("values") or {}).items():
        if col not in options["values"] or col not in df.columns:
            continue
        if set(wanted) >= set(options["values"][col]):          # everything is selected: no filter on this column
            continue
        keep &= _as_text(df[col]).isin(set(wanted))
    date, picked = options.get("date"), chosen.get("date")
    if date and picked:
        col, first, last = date
        start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        if not (start.date() <= first and end.date() >= last) and col in df.columns:
            days = _dates(df[col])
            keep &= (days >= start) & (days <= end)
    return df[keep].reset_index(drop=True)             # clean row numbers 0..n-1 for charts and the sandbox


def is_active(options: dict, chosen: dict) -> bool:
    """True when the choices cut anything away (ignoring choices that keep every value)."""
    for col, wanted in (chosen.get("values") or {}).items():
        if col in options["values"] and not set(wanted) >= set(options["values"][col]):
            return True
    date, picked = options.get("date"), chosen.get("date")
    if date and picked:
        return not (pd.Timestamp(picked[0]).date() <= date[1] and pd.Timestamp(picked[1]).date() >= date[2])
    return False


def describe_filters(options: dict, chosen: dict, rows: int, total: int) -> str:
    """e.g. "region: East, North | order_date: 2025-01-01 to 2025-03-31 (126 of 600 rows)". Empty when nothing is cut."""
    if not is_active(options, chosen):
        return ""
    parts = []
    for col, wanted in (chosen.get("values") or {}).items():
        if col in options["values"] and not set(wanted) >= set(options["values"][col]):
            parts.append(f"{col}: " + (", ".join(sorted(wanted)) if wanted else "(nothing selected)"))
    date, picked = options.get("date"), chosen.get("date")
    if date and picked and not (pd.Timestamp(picked[0]).date() <= date[1] and pd.Timestamp(picked[1]).date() >= date[2]):
        parts.append(f"{date[0]}: {pd.Timestamp(picked[0]).date()} to {pd.Timestamp(picked[1]).date()}")
    return " | ".join(parts) + f" ({rows:,} of {total:,} rows)"


def signature(options: dict, chosen: dict) -> str:
    """Changes whenever the slice changes; empty when nothing is cut."""
    if not is_active(options, chosen):
        return ""
    return describe_filters(options, chosen, 0, 0)
