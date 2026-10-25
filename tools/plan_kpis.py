"""Computes the KPI numbers a DashboardPlan asks for. Pure pandas: the model never sees these values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

_AGGS = ("sum", "mean", "median", "min", "max")


@dataclass
class KpiValue:
    label: str
    agg: str
    measure: str
    value: Optional[float]       # None when it could not be computed
    note: str = ""               # e.g. "15 rows with a missing value left out"


def compute_plan_kpis(df: pd.DataFrame, kpis) -> list[KpiValue]:
    out = []
    for k in kpis:
        if k.measure not in df.columns:
            out.append(KpiValue(k.label, k.agg, k.measure, None, f"column '{k.measure}' not found"))
            continue
        s = df[k.measure]
        missing = int(s.isna().sum())
        note = f"{missing} rows with a missing value left out" if missing else ""
        try:
            if k.agg == "count":
                v = float(s.count())
            elif k.agg in _AGGS:
                v = getattr(pd.to_numeric(s, errors="coerce"), k.agg)()
                v = None if pd.isna(v) else float(v)
            else:
                v, note = None, f"unknown aggregation '{k.agg}'"
        except Exception as e:  # never let one KPI break the dashboard
            v, note = None, f"could not compute: {type(e).__name__}"
        out.append(KpiValue(k.label, k.agg, k.measure, v, note))
    return out
