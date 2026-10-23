"""Data profiler: turns a DataFrame into a compact, JSON-friendly report.

The AI agents read this report instead of the raw data. That keeps costs low
and avoids sending a whole file to a model.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pdt

ID_HINTS = {"id", "key", "uuid", "guid", "code", "no", "number", "num"}
TIME_HINTS = ("date", "time", "day", "month", "year", "created", "updated", "order")
BOOL_WORDS = {"yes", "no", "true", "false", "y", "n", "t", "f"}


# ---------- small helpers ----------

def _py(x):
    """Convert numpy/pandas values to plain Python (None for missing)."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if (np.isnan(x) or np.isinf(x)) else float(x)
    return str(x)


def _r(x, digits=4):
    """Round a number safely."""
    if x is None or pd.isna(x):
        return None
    return _py(round(float(x), digits))


def _name_tokens(name):
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _top_values(non_null, k=5):
    counts = non_null.astype(str).value_counts().head(k)
    total = len(non_null)
    return [
        {"value": str(v), "count": int(c), "pct": round(100 * c / total, 1)}
        for v, c in counts.items()
    ]


# ---------- type detection ----------

def _parse_dates(series):
    """Parse a text Series as dates. Return None if fewer than 90% parse."""
    needed = 0.9 * series.notna().sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() < needed:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed if parsed.notna().sum() >= needed else None


def _detect_dates(s):
    """If a text column looks like dates, return it parsed; otherwise None."""
    text = s.dropna().astype(str).str.strip()
    text = text[text != ""]
    if len(text) < 5:
        return None
    sample = text.sample(min(200, len(text)), random_state=0)
    if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return None  # plain numbers are not dates
    if sample.str.contains(r"\d").mean() < 0.9:
        return None  # dates contain digits
    if _parse_dates(sample) is None:
        return None
    return _parse_dates(text)


def _as_numeric_text(s):
    """If a text column holds numbers like '1,200' or '$75', convert it."""
    cleaned = s.astype(str).str.strip().str.replace(r"[,$€£%\s]", "", regex=True)
    nums = pd.to_numeric(cleaned, errors="coerce")
    nums[s.isna()] = np.nan
    rate = nums.notna().sum() / max(s.notna().sum(), 1)
    return nums if rate >= 0.95 else None


def _date_stats(parsed):
    lo, hi = parsed.min(), parsed.max()
    return {"min": _py(lo), "max": _py(hi), "span_days": int((hi - lo).days)}


def _numeric_profile(name, values, info):
    """Fill numeric stats into info. Return the role: numeric, id or empty."""
    v = values.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if v.empty:
        return "empty"
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    is_int = bool((v % 1 == 0).all())
    info.update({
        "min": _r(v.min()), "max": _r(v.max()), "mean": _r(v.mean()),
        "median": _r(med), "std": _r(v.std()), "q1": _r(q1), "q3": _r(q3),
        "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum()),
        "outliers": outliers, "integer_valued": is_int,
    })
    ratio = v.nunique() / len(v)
    looks_like_id = bool(_name_tokens(name) & ID_HINTS) or v.is_monotonic_increasing
    if is_int and len(v) >= 20 and ratio >= 0.98 and looks_like_id:
        return "id"
    if is_int and v.nunique() <= 10:
        info["likely_discrete"] = True
    return "numeric"


# ---------- per-column profile ----------

def _profile_column(name, s, n_rows):
    non_null = s.dropna()
    missing = int(s.isna().sum())
    info = {
        "name": name,
        "dtype": str(s.dtype),
        "missing": missing,
        "missing_pct": round(100 * missing / n_rows, 1) if n_rows else 0.0,
        "unique": int(non_null.nunique()),
    }
    notes = []
    numeric_values = None

    if non_null.empty:
        info["role"] = "empty"
        notes.append("Column has no data")
    elif pdt.is_bool_dtype(s):
        info["role"] = "boolean"
        info["top_values"] = _top_values(non_null)
    elif pdt.is_datetime64_any_dtype(s):
        info["role"] = "datetime"
        info.update(_date_stats(non_null))
    elif pdt.is_numeric_dtype(s):
        numeric_values = pd.to_numeric(s, errors="coerce").astype("float64")
        info["role"] = _numeric_profile(name, numeric_values, info)
    else:
        parsed = _detect_dates(s)
        as_num = None if parsed is not None else _as_numeric_text(s)
        if parsed is not None:
            info["role"] = "datetime"
            info["parsed_from_text"] = True
            info.update(_date_stats(parsed.dropna()))
            notes.append("Stored as text; parsed as dates")
        elif as_num is not None:
            numeric_values = as_num
            info["role"] = _numeric_profile(name, as_num, info)
            notes.append("Numbers stored as text")
        else:
            text = non_null.astype(str)
            lowered = set(text.str.strip().str.lower().unique())
            if lowered <= BOOL_WORDS:
                info["role"] = "boolean"
                info["top_values"] = _top_values(non_null)
            else:
                avg_len = float(text.str.len().mean())
                ratio = info["unique"] / len(non_null)
                info["avg_length"] = round(avg_len, 1)
                if ratio >= 0.98 and len(non_null) >= 20:
                    info["role"] = "id" if avg_len <= 40 else "text"
                elif avg_len > 50 and ratio > 0.5:
                    info["role"] = "text"
                elif info["unique"] <= max(30, 0.05 * len(non_null)):
                    info["role"] = "categorical"
                    info["top_values"] = _top_values(non_null)
                else:
                    info["role"] = "high_cardinality"
                    info["top_values"] = _top_values(non_null)
                    notes.append(f"Many distinct values ({info['unique']}); hard to chart")

    if info["unique"] == 1 and info["role"] != "empty":
        notes.append("Constant column (only one value)")
    if info["missing_pct"] >= 80:
        notes.append(f"Almost empty ({info['missing_pct']}% missing)")
    elif info["missing_pct"] >= 30:
        notes.append(f"High missing values ({info['missing_pct']}%)")
    info["notes"] = notes
    return info, numeric_values


# ---------- whole-dataset profile ----------

def profile_dataframe(df):
    """Profile a DataFrame. Returns a plain dict that is safe to turn into JSON."""
    n_rows, n_cols = df.shape
    columns, numeric_frames = [], {}

    for i in range(n_cols):  # by position, so duplicate column names don't break us
        name = str(df.columns[i])
        info, numeric_values = _profile_column(name, df.iloc[:, i], n_rows)
        columns.append(info)
        if numeric_values is not None and info["role"] == "numeric":
            numeric_frames[name] = numeric_values

    # time column: prefer names that sound like dates
    dt_cols = [c["name"] for c in columns if c["role"] == "datetime"]
    time_column = None
    if dt_cols:
        hinted = [n for n in dt_cols if any(h in n.lower() for h in TIME_HINTS)]
        time_column = (hinted or dt_cols)[0]

    # strong correlations between numeric columns
    corr_pairs = []
    if len(numeric_frames) >= 2 and n_rows >= 10:
        corr = pd.DataFrame(numeric_frames).corr()
        names = list(corr.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= 0.5:
                    corr_pairs.append({"a": names[i], "b": names[j], "r": round(float(r), 3)})
        corr_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        corr_pairs = corr_pairs[:10]

    # warnings
    warns = []
    duplicates = int(df.duplicated().sum()) if n_cols else 0
    if n_rows and n_rows < 30:
        warns.append(f"Very small dataset ({n_rows} rows); charts may not be meaningful")
    if duplicates:
        warns.append(f"{duplicates} duplicate rows")
    if len(set(df.columns)) != n_cols:
        warns.append("Duplicate column names found")
    for c in columns:
        for note in c["notes"]:
            warns.append(f"{c['name']}: {note}")

    cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum()) if n_cols else 0
    return {
        "dataset": {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "duplicate_rows": duplicates,
            "missing_cells": missing_cells,
            "missing_pct": round(100 * missing_cells / cells, 2) if cells else 0.0,
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
        },
        "columns": columns,
        "time_column": time_column,
        "measures": [c["name"] for c in columns if c["role"] == "numeric"],
        "dimensions": [c["name"] for c in columns if c["role"] in ("categorical", "boolean")],
        "id_columns": [c["name"] for c in columns if c["role"] == "id"],
        "correlations": corr_pairs,
        "warnings": warns,
    }


# ---------- human- and AI-readable text ----------

def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.2f}"
    return str(x)


def describe_column(c):
    """One short line describing a profiled column."""
    role = c["role"]
    if role == "numeric":
        return f"min {_fmt(c.get('min'))}, median {_fmt(c.get('median'))}, max {_fmt(c.get('max'))}"
    if role == "datetime":
        return f"{c.get('min')} to {c.get('max')} ({c.get('span_days')} days)"
    if role in ("categorical", "boolean", "high_cardinality"):
        top = c.get("top_values", [])[:3]
        return ", ".join(f"{t['value']} ({t['pct']}%)" for t in top)
    if role == "text":
        return f"free text, average {c.get('avg_length')} characters"
    if role == "id":
        return "unique identifier"
    return "no data"


def profile_to_text(profile):
    """Compact summary that the AI agents will read instead of the raw data."""
    d = profile["dataset"]
    lines = [
        f"Dataset: {d['rows']} rows x {d['columns']} columns; "
        f"{d['duplicate_rows']} duplicate rows; {d['missing_pct']}% of cells missing."
    ]
    if profile["time_column"]:
        lines.append(f"Time column: {profile['time_column']}")
    lines.append("Columns:")
    for c in profile["columns"]:
        lines.append(f"- {c['name']} [{c['role']}, {c['missing_pct']}% missing]: {describe_column(c)}")
    if profile["correlations"]:
        lines.append("Strong correlations:")
        for p in profile["correlations"]:
            lines.append(f"- {p['a']} vs {p['b']}: r={p['r']}")
    if profile["warnings"]:
        lines.append("Data quality warnings:")
        for w in profile["warnings"]:
            lines.append(f"- {w}")
    return "\n".join(lines)
