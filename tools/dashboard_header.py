"""Dashboard header + data-health banner (dashboard SOP, Phase A).

Both are computed deterministically from the dataset's own profile, never guessed by a model:
- format_header_title(): makes the plan's title read as an "... overview" header.
- coverage_subtitle(): states what period/how many rows the dashboard actually covers.
- data_health_note(): one concrete, dataset-specific data-quality fact (not a generic disclaimer).
"""
from __future__ import annotations

from typing import Optional

_OVERVIEW_WORDS = ("overview", "dashboard", "report")


def format_header_title(title: str) -> str:
    """Add an "overview" framing to the plan's title, unless it already reads like one."""
    title = (title or "Dashboard").strip()
    low = title.lower()
    if any(w in low for w in _OVERVIEW_WORDS):
        return title
    return f"{title} overview"


def coverage_subtitle(profile: dict, time_col: Optional[str] = None) -> str:
    """What this dashboard actually covers: a date range when there is one, else row/column counts.
    `time_col` overrides the profile's time column (Day 26: the date the dashboard's trend chart is about,
    see tools.kpi_deltas.preferred_time_column)."""
    d = profile.get("dataset", {})
    rows = d.get("rows", 0)
    time_col = time_col or profile.get("time_column")
    if time_col:
        col = next((c for c in profile.get("columns", []) if c["name"] == time_col), None)
        if col and col.get("min") and col.get("max"):
            if col["min"] == col["max"]:
                return f"Data for {col['min']} · {rows:,} rows"
            return f"Data from {col['min']} to {col['max']} · {rows:,} rows"
    cols = d.get("columns", 0)
    return f"{rows:,} rows across {cols} columns"


def data_health_note(profile: dict) -> str:
    """One sentence stating a concrete, computed data-quality fact about THIS dataset."""
    d = profile.get("dataset", {})
    rows = d.get("rows", 0)
    duplicates = d.get("duplicate_rows", 0)
    candidates = [c for c in profile.get("columns", [])
                  if c.get("role") not in ("empty", "id") and c.get("missing_pct", 0) > 0]

    if candidates:
        worst = max(candidates, key=lambda c: c["missing_pct"])
        note = f"Data health: '{worst['name']}' is missing {worst['missing_pct']}% of rows"
        if duplicates:
            note += f"; {duplicates:,} duplicate row{'s' if duplicates != 1 else ''} found"
        return note + "."
    if duplicates and rows:
        return (f"Data health: {duplicates:,} duplicate row{'s' if duplicates != 1 else ''} found; "
                "all other columns are complete.")
    return "Data health: no missing values found in this dataset."
