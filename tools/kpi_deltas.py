"""Period-over-period KPI deltas (dashboard SOP, Phase B).

Only attempted when the dataset has a usable, parsed datetime column with enough span to form
two comparable periods. When it doesn't (no date column, or too little span — e.g. the F1 demo,
which is season/race-based, not daily), compute_kpi_deltas() returns {} and every KPI falls back
to today's plain tile. A fabricated comparison period is never shown.

Direction-of-goodness (is a rise good or bad?) uses a conservative keyword heuristic: metrics that
sound like costs, churn or errors are flagged "lower is better"; everything else defaults to
"higher is better". A delta under NEUTRAL_PCT is shown as flat/neutral rather than colored, since a
tiny wobble is not really a "win" or a "risk".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from tools.charts import AGG_FUNCS

MIN_SPAN_DAYS = 4          # need at least this much span to form two periods worth comparing
MAX_PERIOD_DAYS = 30       # "vs previous 30 days" once there's enough data; shorter spans use half
NEUTRAL_PCT = 1.0          # a move smaller than this is shown as flat, not a win or a risk
MIN_PERIOD_SHARE = 0.10    # each period must hold at least this share of the dated rows (Day 26)

LOWER_IS_BETTER_HINTS = {
    "cost", "costs", "expense", "expenses", "churn", "bounce", "complaint", "complaints",
    "error", "errors", "defect", "defects", "refund", "refunds", "cancellation", "cancellations",
    "delay", "delays", "return", "returns", "downtime", "outage", "outages", "wastage", "loss",
    "losses", "attrition",
}


@dataclass
class ComparisonWindow:
    time_col: str
    period_days: int
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    prior_start: pd.Timestamp
    prior_end: pd.Timestamp
    label: str                 # e.g. "vs previous 30 days"


@dataclass
class KpiDelta:
    pct: Optional[float]       # None when a period had no computable value
    direction: str              # "up", "down" or "flat"
    good: Optional[bool]        # True/False to color it, None when flat (shown neutral/gray)


def _name_tokens(name: str) -> set:
    return set(name.lower().replace("-", " ").replace("_", " ").split())


def lower_is_better(*names: str) -> bool:
    """True when the KPI's name suggests a fall is the good outcome (cost, churn, errors, ...)."""
    tokens = set()
    for n in names:
        tokens |= _name_tokens(n or "")
    return bool(tokens & LOWER_IS_BETTER_HINTS)


def preferred_time_column(plan, profile: dict) -> Optional[str]:
    """The date column to compare periods on (Day 26).

    The profile's time_column is simply the first date-like column, which can be a deadline or a delivery date
    rather than when things happened (in the Olist demo it was shipping_limit_date, which runs two months past
    the orders and made revenue look like it fell 97%). The planner's own trend chart says which date the
    dashboard is about, so its time column wins when it is a real date column in the profile."""
    dates = {c["name"] for c in profile.get("columns", []) if c.get("role") == "datetime"}
    for chart in getattr(plan, "charts", None) or []:
        if getattr(chart, "type", None) == "line" and getattr(chart, "time", None) in dates:
            return chart.time
    return profile.get("time_column")


def pick_comparison_window(profile: dict, time_col: Optional[str] = None) -> Optional[ComparisonWindow]:
    """A usable "current period vs prior period" window, or None when the data can't support one.
    `time_col` overrides the profile's time column (see preferred_time_column)."""
    time_col = time_col or profile.get("time_column")
    if not time_col:
        return None
    col = next((c for c in profile.get("columns", []) if c["name"] == time_col), None)
    if not col or not col.get("min") or not col.get("max"):
        return None
    span_days = col.get("span_days", 0) or 0
    if span_days < MIN_SPAN_DAYS:
        return None
    period_days = min(MAX_PERIOD_DAYS, span_days // 2)
    if period_days < 1:
        return None
    end = pd.Timestamp(col["max"])
    current_start = end - pd.Timedelta(days=period_days)
    prior_end = current_start
    prior_start = prior_end - pd.Timedelta(days=period_days)
    label = f"vs previous {period_days} day{'s' if period_days != 1 else ''}"
    return ComparisonWindow(time_col, period_days, current_start, end, prior_start, prior_end, label)


def _agg_in_window(df: pd.DataFrame, time_col: str, measure: str, agg: str,
                   start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    mask = (df[time_col] > start) & (df[time_col] <= end)
    s = pd.to_numeric(df.loc[mask, measure], errors="coerce").dropna()
    if s.empty:
        return None
    if agg == "count":
        return float(len(s))
    v = getattr(s, agg)()
    return None if pd.isna(v) else float(v)


def compute_kpi_delta(df: pd.DataFrame, kpi, window: ComparisonWindow) -> Optional[KpiDelta]:
    """The delta for one KpiSpec/KpiValue-like object (needs .measure and .agg), or None if either
    period has no computable value (never guess a direction from an incomplete comparison)."""
    if kpi.measure not in df.columns or window.time_col not in df.columns:
        return None
    agg = kpi.agg if kpi.agg in AGG_FUNCS else "sum"
    current = _agg_in_window(df, window.time_col, kpi.measure, agg, window.current_start, window.current_end)
    prior = _agg_in_window(df, window.time_col, kpi.measure, agg, window.prior_start, window.prior_end)
    if current is None or prior is None:
        return None
    if prior == 0:
        return None            # a percentage change from zero is not a meaningful number to show
    pct = round(100 * (current - prior) / abs(prior), 1)
    if abs(pct) < NEUTRAL_PCT:
        return KpiDelta(pct, "flat", None)
    direction = "up" if pct > 0 else "down"
    inverted = lower_is_better(kpi.label, kpi.measure)
    good = (direction == "down") if inverted else (direction == "up")
    return KpiDelta(pct, direction, good)


def window_is_balanced(df: pd.DataFrame, window: ComparisonWindow, min_share: float = MIN_PERIOD_SHARE) -> bool:
    """True when both periods hold a fair share of the dated rows. A period with only a sliver of the data
    (e.g. the tail of a date that runs on after most rows end) would make any change look huge."""
    if window.time_col not in df.columns:
        return False
    t = pd.to_datetime(df[window.time_col], errors="coerce").dropna()
    if t.empty:
        return False
    current = ((t > window.current_start) & (t <= window.current_end)).sum()
    prior = ((t > window.prior_start) & (t <= window.prior_end)).sum()
    return min(current, prior) >= min_share * len(t)


def compute_kpi_deltas(df: pd.DataFrame, kpis, profile: dict, time_col: Optional[str] = None) -> dict:
    """{kpi.label: KpiDelta}, computed only for the KPIs where a clean comparison is possible.
    Returns {} outright when the dataset has no usable date/span, or when one of the two periods holds too
    little of the data for a fair comparison (graceful fallback: plain KPI tiles, no fabricated change)."""
    window = pick_comparison_window(profile, time_col)
    if window is None or not window_is_balanced(df, window):
        return {}
    out = {}
    for kpi in kpis:
        try:
            delta = compute_kpi_delta(df, kpi, window)
        except Exception:          # a delta must never break the dashboard
            delta = None
        if delta is not None:
            out[kpi.label] = delta
    return out
