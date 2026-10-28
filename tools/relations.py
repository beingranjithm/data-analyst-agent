"""Day 21: find how several tables belong together, and join them without silently changing the row count.

A LINK says: every value in child.child_col should be found in parent.parent_col, and parent_col has no repeats
(so each child row has at most one parent row). Example: results.driverId -> drivers.driverId.

Why only this kind of link is joined automatically: following a many-to-one link never adds or removes rows, so the
totals stay right. Joining two big "event" tables to each other (for example lap_times and results) multiplies rows
and quietly inflates every total, so that is never done automatically. Instead each event table is FLATTENED: it keeps
its own rows and gains the columns of the lookup tables it points to (directly or through other lookup tables).

No AI is involved here. Everything is plain, testable pandas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

MIN_CONTAINMENT = 0.9          # at least this share of the child's distinct values must exist in the parent
ID_WORDS = ("id", "key", "ref", "code")
# Column names that say nothing without their table ("name", "number" ...): always shown with the table name in front.
GENERIC_NAMES = {"name", "code", "number", "type", "status", "title", "label", "description", "date", "time",
                 "value", "category", "notes", "comment", "rank", "position", "points", "id", "year", "country",
                 "nationality", "location"}


@dataclass(frozen=True)
class Link:
    child: str
    child_col: str
    parent: str
    parent_col: str
    containment: float          # share of the child's distinct values found in the parent (1.0 = all)
    kind: str = "many-to-one"   # or "one-to-one"
    reason: str = ""

    def text(self) -> str:
        return f"{self.child}.{self.child_col} → {self.parent}.{self.parent_col}"


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _singular(word: str) -> str:
    w = _norm(word)
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _keyable(s: pd.Series) -> bool:
    """Columns that can serve as a join key: whole numbers or text, not decimals, dates or booleans."""
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        return len(v) > 0 and bool((v % 1 == 0).all())
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _keys(s: pd.Series) -> pd.Series:
    """Values in a comparable form: whole-number floats become integers, text is stripped."""
    v = s.dropna()
    if pd.api.types.is_object_dtype(v) or pd.api.types.is_string_dtype(v):
        v = v.astype(str).str.strip()
    return v


def _is_unique_key(s: pd.Series) -> bool:
    if len(s) < 2 or not s.notna().all() or not _keyable(s):
        return False
    return bool(_keys(s).is_unique)                 # judged on the cleaned form: 'b2' and 'b2 ' are the same key


def _name_evidence(child_table: str, child_col: str, parent_table: str, parent_col: str) -> Optional[str]:
    cn, pn = _norm(child_col), _norm(parent_col)
    if cn == pn:
        return "same column name"
    if pn == "id" and cn in (_singular(parent_table) + "id", _norm(parent_table) + "id"):
        return f"'{child_col}' names the table '{parent_table}'"
    return None


def _id_like(name) -> bool:
    n = _norm(name)
    return any(n.endswith(w) for w in ID_WORDS)


def find_links(tables: dict, min_containment: float = MIN_CONTAINMENT) -> list:
    """All plausible links between the tables, best evidence first. A link needs ALL of: matching column names,
    a parent column without repeats or gaps, and (almost) every child value found in the parent."""
    uniq, key_sets = {}, {}
    for t, df in tables.items():
        for c in df.columns:
            if _is_unique_key(df[c]):
                uniq[(t, c)] = True
                key_sets[(t, c)] = set(_keys(df[c]))
    links = []
    for (pt, pc) in uniq:
        for ct, cdf in tables.items():
            if ct == pt:
                continue
            for cc in cdf.columns:
                why = _name_evidence(ct, cc, pt, pc)
                s = cdf[cc]
                if why is None or not _keyable(s):
                    continue
                vals = _keys(s)
                if vals.empty:
                    continue
                distinct = set(vals)
                contained = len(distinct & key_sets[(pt, pc)]) / len(distinct)
                if contained < min_containment:
                    continue
                if not (_id_like(cc) or (contained >= 0.999 and len(distinct) >= 3)):
                    continue                                    # a shared generic name such as 'name' is not enough
                child_unique = (ct, cc) in uniq
                if child_unique:
                    # both sides have no repeats: a real one-to-one link has about the same set of values on both
                    # sides. A small id column inside a big one (sprint_results.resultId in results.resultId) is a coincidence.
                    same_size = min(len(distinct), len(key_sets[(pt, pc)])) / max(len(distinct), len(key_sets[(pt, pc)]))
                    if same_size < 0.9 or contained < 0.999:
                        continue
                    kind = "one-to-one"
                else:
                    kind = "many-to-one"
                links.append(Link(ct, cc, pt, pc, round(contained, 4), kind, why))
    links.sort(key=lambda l: (-l.containment, l.child, l.parent, l.child_col))
    return links


def lookup_tables(tables: dict, links: list) -> set:
    """Tables that only other tables point to and that point to nothing themselves (like drivers or circuits)."""
    kids = {l.child for l in links if l.kind == "many-to-one"}
    parents = {l.parent for l in links if l.kind == "many-to-one"}
    return {t for t in tables if t in parents and t not in kids}


def event_tables(tables: dict, links: list) -> list:
    """Tables that point at other tables and that nothing points to (results, lap_times ...), largest first.
    These record things that happened, so they are what a dashboard is usually about."""
    kids = {l.child for l in links if l.kind == "many-to-one"}
    parents = {l.parent for l in links if l.kind == "many-to-one"}
    return sorted((t for t in tables if t in kids and t not in parents), key=lambda t: -len(tables[t]))


@dataclass
class Flat:
    focus: str
    df: pd.DataFrame
    joined: list = field(default_factory=list)       # [(Link, columns added)]
    skipped: list = field(default_factory=list)      # [(table, why)]
    rows_before: int = 0
    rows_after: int = 0
    key_columns: list = field(default_factory=list)  # columns of `df` that only link tables (ids), not measurements

    @property
    def ok(self) -> bool:
        return self.rows_before == self.rows_after


def _is_noise(name: str) -> bool:
    """Columns that never help an analysis when pulled in from a lookup table: web addresses."""
    n = _norm(name)
    return n == "url" or n.endswith("url")


def flatten(tables: dict, focus: str, links: list, *, max_columns: int = 200) -> Flat:
    """The focus table plus the columns of every table it reaches by following many-to-one links.
    Rows are never added or removed (left joins on columns without repeats); this is checked, not assumed.
    A table is joined once, through the shortest path. Columns pulled in from a lookup table get the table name in
    front when the name is generic ('name' -> 'constructors_name') or already taken. Web-address columns of lookup
    tables are left out. Only tables joined by a link you pass in are used, so you can leave out a wrong link."""
    if focus not in tables:
        raise KeyError(f"No table called {focus!r}")
    out = tables[focus].copy()
    where = {(focus, c): c for c in out.columns}           # (table, original column) -> its name in `out`
    flat = Flat(focus, out, rows_before=len(out))
    keys = set()
    out_links = {}
    for l in links:
        if l.kind == "many-to-one":
            out_links.setdefault(l.child, []).append(l)
    seen, queue = {focus}, [focus]
    while queue:
        cur = queue.pop(0)
        for l in sorted(out_links.get(cur, []), key=lambda x: (x.parent, x.child_col)):
            if l.parent in seen:
                continue
            key_in_out = where.get((cur, l.child_col))
            if key_in_out is None:
                flat.skipped.append((l.parent, f"the key column {l.child_col} of {cur} is not in the flattened table"))
                continue
            parent_df = tables[l.parent]
            take = [c for c in parent_df.columns if c != l.parent_col and not _is_noise(c)]
            if len(out.columns) + len(take) > max_columns:
                flat.skipped.append((l.parent, f"it would make the table wider than {max_columns} columns"))
                continue
            right_key = _keys_for_merge(parent_df[l.parent_col])
            if right_key.isna().any() or not right_key.is_unique:       # found links never have this; a hand-made one might
                flat.skipped.append((l.parent, "joining it would change the number of rows"))
                continue
            taken = set(out.columns)
            names = {}
            for c in take:
                n = c if (_norm(c) not in GENERIC_NAMES and c not in taken) else f"{l.parent}_{c}"
                while n in taken or n in names.values():
                    n += "_"
                names[c] = n
            right = parent_df[take].rename(columns=names).copy()
            right.index = pd.Index(right_key)
            out = pd.concat([out.reset_index(drop=True),
                             right.reindex(pd.Index(_keys_for_merge(out[key_in_out]))).reset_index(drop=True)], axis=1)
            for c, n in names.items():
                where[(l.parent, c)] = n
            keys.add(key_in_out)
            flat.joined.append((l, list(names.values())))
            seen.add(l.parent)
            queue.append(l.parent)
    flat.df = out
    flat.rows_after = len(out)
    flat.key_columns = sorted(keys)
    return flat


def _keys_for_merge(s: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.astype("string").str.strip()
    return s
