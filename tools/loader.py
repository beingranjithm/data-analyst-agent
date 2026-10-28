"""Robust CSV loading: guesses text encoding and delimiter, tidies headers."""
from __future__ import annotations

import io
import re
import statistics

import pandas as pd

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = [",", ";", "\t", "|"]
NA_VALUES = ["", "N/A", "n/a", "NA", "na", "NaN", "nan", "null", "NULL", "Null",
             "None", "none", "-", "--", "?", "#N/A", "missing", "Missing",
             "\\N"]                                 # Day 21: databases (MySQL dumps, the F1 data set) write NULL as \N


def _decode(data: bytes):
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _guess_delimiter(text: str) -> str:
    """Pick the delimiter that appears a consistent number of times per line."""
    lines = [ln for ln in text.splitlines()[:30] if ln.strip()]
    if not lines:
        return ","
    best, best_score = ",", 0.0
    for d in DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        med = statistics.median(counts)
        if med == 0:
            continue
        consistency = sum(c == med for c in counts) / len(counts)
        score = consistency * 10 + min(med, 10) / 10
        if score > best_score:
            best, best_score = d, score
    return best


def clean_column_names(columns):
    """Strip spaces, fill blank names, make names unique. Returns (names, notes)."""
    names, notes, seen = [], [], {}
    for i, raw in enumerate(columns, start=1):
        original = "" if raw is None else str(raw)
        name = re.sub(r"\s+", " ", original).strip()
        if name == "" or re.fullmatch(r"Unnamed: \d+(_level_\d+)?", name):
            name = f"column_{i}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        if name != original:
            notes.append(f"Column name {original!r} renamed to {name!r}")
        names.append(name)
    return names, notes


def load_csv_bytes(data: bytes):
    """Read CSV bytes into a DataFrame. Returns (df, notes). Raises ValueError if unusable."""
    if not data or not data.strip():
        raise ValueError("This file is empty. Please upload a CSV that contains data.")
    text, encoding = _decode(data)
    delimiter = _guess_delimiter(text)
    notes = []
    if encoding != "utf-8-sig":
        notes.append(f"File was not UTF-8; read as {encoding}")
    if delimiter != ",":
        notes.append(f"Detected delimiter {delimiter!r} instead of a comma")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            decimal="," if delimiter == ";" else ".",
            na_values=NA_VALUES,
            skipinitialspace=True,
            on_bad_lines="skip",
            engine="python",
        )
    except pd.errors.EmptyDataError:
        raise ValueError("This file is empty. Please upload a CSV that contains data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"This does not look like a valid CSV file ({e}).")

    if df.shape[1] == 0:
        raise ValueError("No columns were found in this file.")

    df = df.dropna(how="all")
    if len(df) == 0:
        raise ValueError("This file has column headers but no data rows.")

    names, rename_notes = clean_column_names(df.columns)
    df.columns = names
    notes.extend(rename_notes)
    return df.reset_index(drop=True), notes
