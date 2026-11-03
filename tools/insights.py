"""Insight-led chart titles (dashboard SOP, Phase A).

A chart title should state what the chart actually shows, not just its axes. This module
computes that sentence directly from the real data with plain pandas, using only the columns
and aggregation already validated against the profile (tools.plan_schema.validate_plan_against_profile).

It never sees or trusts the planner LLM's guess: the same discipline as tools/plan_kpis.py.
If no confident, grounded sentence can be computed, it returns None and the caller keeps the
original title instead of showing something invented or shaky.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from tools.charts import AGG_FUNCS, default_agg, pick_frequency

MAX_LEN = 90


def _compact(x: float) -> str:
    ax = abs(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if ax >= limit:
            return f"{x / limit:,.1f}{suffix}"
    return f"{x:,.2f}" if x % 1 else f"{x:,.0f}"


def _bar_or_donut_insight(df: pd.DataFrame, dimension: str, measure: str, agg: str) -> Optional[str]:
    data = df[[dimension, measure]].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna(subset=[dimension, measure])
    if data.empty:
        return None
    grouped = data.groupby(dimension)[measure].agg(AGG_FUNCS[agg])
    grouped = grouped.dropna()
    if grouped.empty:
        return None
    top_label, top_value = grouped.idxmax(), float(grouped.max())
    if agg in ("sum", "count") and len(grouped) >= 2:
        total = float(grouped.sum())
        if total > 0:
            share = round(100 * top_value / total)
            return f"{top_label} leads with {share}% of {measure}"
    if agg == "mean":
        verb = "has the highest average"
    elif agg == "median":
        verb = "has the highest median"
    else:
        verb = "is the largest"
    return f"{top_label} {verb} {measure} ({_compact(top_value)})"


def _line_insight(df: pd.DataFrame, time: str, measure: str, agg: str) -> Optional[str]:
    data = df[[time, measure]].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna(subset=[time, measure]).sort_values(time)
    if len(data) < 2:
        return None
    span_days = int((data[time].max() - data[time].min()).days)
    freq, _ = pick_frequency(span_days)
    series = data.set_index(time)[measure].resample(freq).agg(AGG_FUNCS[agg]).dropna()
    if len(series) < 2:
        return None
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first == 0:
        return None
    pct = round(100 * (last - first) / abs(first))
    if pct == 0:
        return f"{measure} stayed roughly flat over the period"
    direction = "rose" if pct > 0 else "fell"
    return f"{measure} {direction} {abs(pct)}% over the period"


def _scatter_insight(df: pd.DataFrame, x: str, y: str) -> Optional[str]:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 5:
        return None
    r = data[x].corr(data[y])
    if pd.isna(r):
        return None
    strength = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.4 else "Weak"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction} relationship between {x} and {y} (r={r:.2f})"


def _histogram_insight(df: pd.DataFrame, measure: str) -> Optional[str]:
    s = pd.to_numeric(df[measure], errors="coerce").dropna()
    if len(s) < 5:
        return None
    return f"Most {measure} values cluster around {_compact(float(s.median()))}"


def chart_insight_title(df: pd.DataFrame, chart) -> Optional[str]:
    """A grounded, insight-led title for this chart, or None if one cannot be computed safely.

    `chart` is anything with the same attributes as tools.plan_schema.ChartSpec. Never raises:
    any problem computing an insight is treated the same as "no confident insight available".
    """
    try:
        t = chart.type
        if t in ("bar", "donut") and chart.dimension and chart.measure:
            insight = _bar_or_donut_insight(df, chart.dimension, chart.measure,
                                            chart.agg or default_agg(chart.measure))
        elif t == "line" and chart.time and chart.measure:
            insight = _line_insight(df, chart.time, chart.measure, chart.agg or default_agg(chart.measure))
        elif t == "scatter" and chart.x and chart.y:
            insight = _scatter_insight(df, chart.x, chart.y)
        elif t == "histogram" and chart.measure:
            insight = _histogram_insight(df, chart.measure)
        else:
            insight = None
    except Exception:
        insight = None
    if insight and len(insight) > MAX_LEN:
        insight = insight[:MAX_LEN - 1].rstrip() + "…"
    return insight
