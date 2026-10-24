"""Turns text columns into real dates and numbers so later code can use them."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from tools.profiler import _as_numeric_text, _detect_dates, _guess_dayfirst


def clean_dataframe(df: pd.DataFrame):
    """Return (cleaned copy, log of what was changed)."""
    df = df.copy()
    log = []
    for i in range(df.shape[1]):
        s = df.iloc[:, i]
        name = str(df.columns[i])
        if pdt.is_numeric_dtype(s) or pdt.is_bool_dtype(s) or pdt.is_datetime64_any_dtype(s):
            continue
        s = s.astype("object").where(s.notna(), None)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: None if v == "" else v)
        df.isetitem(i, s)

        parsed = _detect_dates(s)
        if parsed is not None:
            text = s.dropna().astype(str)
            order = "day-first" if _guess_dayfirst(text) else "month-first"
            df.isetitem(i, parsed.reindex(s.index))
            log.append(f"{name}: text converted to dates ({order} where ambiguous)")
            continue

        numeric = _as_numeric_text(s)
        if numeric is not None:
            had_percent = s.dropna().astype(str).str.contains("%").any()
            df.isetitem(i, numeric)
            extra = "; percent signs removed (values are in percent)" if had_percent else ""
            log.append(f"{name}: text converted to numbers{extra}")
    return df, log
