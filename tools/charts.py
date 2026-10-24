"""Hand-built chart library with simple, explainable rules.

These functions are the reference for what a "good" chart looks like.
Later, the AI planner will follow the same rules (see CHART_RULES).
A chart is described by a small dict called a "spec", for example:
    {"type": "bar", "dimension": "region", "measure": "revenue", "agg": "sum"}
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]
MEAN_HINTS = {"price", "rate", "ratio", "pct", "percent", "percentage", "score", "avg",
              "average", "mean", "discount", "age", "temp", "temperature", "growth",
              "margin", "rating", "share"}
PRIMARY_HINTS = ("revenue", "sales", "amount", "total", "profit", "cost", "income",
                 "units", "quantity", "qty", "count", "price", "value")
AGG_FUNCS = {"sum": "sum", "mean": "mean", "median": "median", "count": "count",
             "min": "min", "max": "max"}
MAX_BARS = 10

CHART_RULES = """\
Chart selection rules (the AI planner will follow these):
- Trend over time: line chart of one measure by a time column, aggregated by day, week, month, quarter or year depending on the time span.
- Compare categories: horizontal bar chart of a measure by one category column; show the top 10 and group the rest as 'Other'.
- Share of a whole: donut chart only when a category has 6 or fewer values.
- Distribution of one number: histogram.
- Spread of a number across categories: box plot.
- Relationship between two numbers: scatter plot; add colour only for a category with 8 or fewer values.
- Overview of many numbers: correlation heatmap when there are 3 or more measures.
- Sum amounts (revenue, units); average rates, prices, scores and percentages.
- Never chart identifier or free-text columns. Always say how many rows were left out because of missing values.
"""


class ChartError(ValueError):
    """Raised with a friendly message when a chart cannot be built."""


# ---------- helpers ----------

def default_agg(measure: str) -> str:
    """Amounts are summed; rates, prices and scores are averaged."""
    tokens = set(measure.lower().replace("-", "_").replace(" ", "_").split("_"))
    return "mean" if tokens & MEAN_HINTS else "sum"


def pick_primary_measure(profile: dict):
    measures = profile.get("measures", [])
    if not measures:
        return None
    for hint in PRIMARY_HINTS:
        for m in measures:
            if hint in m.lower():
                return m
    return measures[0]


def _need(df, *cols):
    for c in cols:
        if c is not None and c not in df.columns:
            raise ChartError(f"Column '{c}' does not exist in this table.")


def _need_numeric(df, col):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ChartError(f"Column '{col}' is not numeric, so it cannot be used as a measure.")


def _need_datetime(df, col):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise ChartError(f"Column '{col}' is not a date column.")


def _style(fig, title, note=None):
    fig.update_layout(
        template="plotly_white", title=dict(text=title, x=0.01, xanchor="left"),
        colorway=PALETTE, margin=dict(l=40, r=20, t=60, b=60 if note else 40),
        height=380, legend_title_text="",
    )
    if note:
        fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=-0.18,
                           showarrow=False, font=dict(size=11, color="#888"), xanchor="left")
    return fig


def _excluded_note(n_excluded, what):
    return f"Excludes {n_excluded:,} rows with missing {what}" if n_excluded else None


def pick_frequency(span_days: int):
    """(pandas frequency code, label) based on how long the data spans."""
    if span_days <= 60:
        return "D", "daily"
    if span_days <= 400:
        return "MS", "monthly"
    if span_days <= 365 * 6:
        return "QS", "quarterly"
    return "YS", "yearly"


# ---------- chart builders ----------

def trend_chart(df, time, measure, agg=None):
    _need(df, time, measure); _need_datetime(df, time); _need_numeric(df, measure)
    agg = agg or default_agg(measure)
    data = df[[time, measure]].dropna(subset=[time])
    if data.empty:
        raise ChartError(f"No valid dates in '{time}'.")
    span = int((data[time].max() - data[time].min()).days)
    freq, label = pick_frequency(span)
    series = data.set_index(time)[measure].resample(freq).agg(AGG_FUNCS[agg]).reset_index()
    fig = px.line(series, x=time, y=measure, markers=True)
    fig.update_traces(line=dict(width=2.5))
    note = _excluded_note(int(df[time].isna().sum()), f"'{time}'")
    return _style(fig, f"{agg.title()} of {measure} ({label})", note)


def bar_chart(df, dimension, measure, agg=None, top_n=MAX_BARS):
    _need(df, dimension, measure); _need_numeric(df, measure)
    agg = agg or default_agg(measure)
    data = df[[dimension, measure]].dropna(subset=[dimension])
    grouped = data.groupby(dimension)[measure].agg(AGG_FUNCS[agg]).sort_values(ascending=False)
    if grouped.empty:
        raise ChartError(f"No data to chart for '{dimension}'.")
    if len(grouped) > top_n and agg in ("sum", "count"):
        other = grouped.iloc[top_n:].sum()
        grouped = pd.concat([grouped.iloc[:top_n], pd.Series({"Other": other})])
    elif len(grouped) > top_n:
        grouped = grouped.iloc[:top_n]
    plot = grouped.sort_values().reset_index()
    plot.columns = [dimension, measure]
    fig = px.bar(plot, x=measure, y=dimension, orientation="h")
    note = _excluded_note(int(df[dimension].isna().sum()), f"'{dimension}'")
    return _style(fig, f"{agg.title()} of {measure} by {dimension}", note)


def donut_chart(df, dimension, measure, agg=None):
    _need(df, dimension, measure); _need_numeric(df, measure)
    agg = agg or default_agg(measure)
    if agg not in ("sum", "count"):
        raise ChartError("A share-of-total chart needs a summed or counted measure.")
    data = df[[dimension, measure]].dropna(subset=[dimension])
    grouped = data.groupby(dimension)[measure].agg(AGG_FUNCS[agg])
    grouped = grouped[grouped > 0]
    if len(grouped) < 2:
        raise ChartError(f"'{dimension}' needs at least 2 values for a share chart.")
    fig = go.Figure(go.Pie(labels=grouped.index.astype(str), values=grouped.values, hole=0.5))
    note = _excluded_note(int(df[dimension].isna().sum()), f"'{dimension}'")
    return _style(fig, f"Share of {measure} by {dimension}", note)


def histogram_chart(df, measure, bins=30):
    _need(df, measure); _need_numeric(df, measure)
    data = df[[measure]].dropna()
    if data.empty:
        raise ChartError(f"No values in '{measure}'.")
    fig = px.histogram(data, x=measure, nbins=bins)
    note = _excluded_note(int(df[measure].isna().sum()), f"'{measure}'")
    return _style(fig, f"Distribution of {measure}", note)


def box_chart(df, dimension, measure, top_n=MAX_BARS):
    _need(df, dimension, measure); _need_numeric(df, measure)
    data = df[[dimension, measure]].dropna()
    keep = data[dimension].value_counts().head(top_n).index
    data = data[data[dimension].isin(keep)]
    if data.empty:
        raise ChartError("No data to chart.")
    fig = px.box(data, x=dimension, y=measure)
    return _style(fig, f"{measure} by {dimension}")


def scatter_chart(df, x, y, color=None):
    _need(df, x, y, color); _need_numeric(df, x); _need_numeric(df, y)
    cols = [c for c in (x, y, color) if c]
    data = df[cols].dropna()
    if len(data) > 5000:
        data = data.sample(5000, random_state=0)
    if data.empty:
        raise ChartError("No complete rows to plot.")
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7)
    return _style(fig, f"{y} vs {x}")


def heatmap_chart(df, measures):
    _need(df, *measures)
    for m in measures:
        _need_numeric(df, m)
    if len(measures) < 2:
        raise ChartError("A correlation heatmap needs at least 2 numeric columns.")
    corr = df[measures].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    return _style(fig, "Correlation between numeric columns")


BUILDERS = {
    "line": lambda df, s: trend_chart(df, s["time"], s["measure"], s.get("agg")),
    "bar": lambda df, s: bar_chart(df, s["dimension"], s["measure"], s.get("agg")),
    "donut": lambda df, s: donut_chart(df, s["dimension"], s["measure"], s.get("agg")),
    "histogram": lambda df, s: histogram_chart(df, s["measure"]),
    "box": lambda df, s: box_chart(df, s["dimension"], s["measure"]),
    "scatter": lambda df, s: scatter_chart(df, s["x"], s["y"], s.get("color")),
    "heatmap": lambda df, s: heatmap_chart(df, s["measures"]),
}


def build_chart(df: pd.DataFrame, spec: dict) -> go.Figure:
    """Build a Plotly figure from a spec dict. Raises ChartError with a friendly message."""
    kind = spec.get("type")
    if kind not in BUILDERS:
        raise ChartError(f"Unknown chart type '{kind}'. Choose from: {', '.join(BUILDERS)}.")
    try:
        return BUILDERS[kind](df, spec)
    except KeyError as e:
        raise ChartError(f"The chart spec is missing {e}.")


# ---------- rule-based chart suggestions ----------

def _usable_dimensions(profile, max_unique=15):
    cols = {c["name"]: c for c in profile["columns"]}
    dims = [cols[n] for n in profile["dimensions"] if 2 <= cols[n]["unique"] <= max_unique]
    return [d["name"] for d in sorted(dims, key=lambda d: (d["missing_pct"], d["unique"]))]


def suggest_charts(profile: dict, max_charts: int = 8):
    """Apply CHART_RULES to a profile. Returns a list of specs with a 'why' explanation."""
    specs = []
    cols = {c["name"]: c for c in profile["columns"]}
    primary = pick_primary_measure(profile)
    dims = _usable_dimensions(profile)
    measures = profile["measures"]
    time = profile["time_column"]

    if time and primary:
        specs.append({"type": "line", "time": time, "measure": primary,
                      "why": f"Shows how {primary} changes over time."})
    for dim in dims[:2]:
        if primary:
            specs.append({"type": "bar", "dimension": dim, "measure": primary,
                          "why": f"Compares {primary} across {dim}."})
    for dim in dims:
        if primary and cols[dim]["unique"] <= 6 and default_agg(primary) == "sum":
            specs.append({"type": "donut", "dimension": dim, "measure": primary,
                          "why": f"Shows each {dim}'s share of total {primary}."})
            break
    if primary:
        specs.append({"type": "histogram", "measure": primary,
                      "why": f"Shows the typical size and spread of {primary}."})
    if len(measures) >= 2 and profile["correlations"]:
        top = profile["correlations"][0]
        color = next((d for d in dims if cols[d]["unique"] <= 8), None)
        specs.append({"type": "scatter", "x": top["a"], "y": top["b"], "color": color,
                      "why": f"{top['a']} and {top['b']} move together (r={top['r']})."})
    if len(measures) >= 3:
        specs.append({"type": "heatmap", "measures": measures[:10],
                      "why": "Gives an overview of how all numeric columns relate."})
    if dims and primary:
        specs.append({"type": "box", "dimension": dims[0], "measure": primary,
                      "why": f"Shows the spread of {primary} within each {dims[0]}."})
    return specs[:max_charts]


def spec_title(spec: dict) -> str:
    t = spec["type"]
    if t == "line":
        return f"{spec['measure']} over time"
    if t in ("bar", "donut", "box"):
        return f"{spec['measure']} by {spec['dimension']}"
    if t == "histogram":
        return f"Distribution of {spec['measure']}"
    if t == "scatter":
        return f"{spec['y']} vs {spec['x']}"
    return "Correlation overview"


# ---------- KPI cards ----------

def _compact(x):
    ax = abs(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if ax >= limit:
            return f"{x / limit:,.1f}{suffix}"
    return f"{x:,.2f}" if x % 1 else f"{x:,.0f}"


def compute_kpis(df: pd.DataFrame, profile: dict):
    """A few headline numbers: list of {'label', 'value'}."""
    kpis = [{"label": "Rows analysed", "value": f"{len(df):,}"}]
    primary = pick_primary_measure(profile)
    if primary:
        agg = default_agg(primary)
        value = df[primary].sum() if agg == "sum" else df[primary].mean()
        kpis.append({"label": f"{'Total' if agg == 'sum' else 'Average'} {primary}", "value": _compact(float(value))})
    if profile["time_column"]:
        col = df[profile["time_column"]].dropna()
        if not col.empty:
            kpis.append({"label": "Period covered", "value": f"{col.min():%Y-%m-%d} to {col.max():%Y-%m-%d}"})
    dims = _usable_dimensions(profile)
    if dims and primary:
        g = df.groupby(dims[0])[primary].agg(AGG_FUNCS[default_agg(primary)])
        if not g.empty:
            kpis.append({"label": f"Top {dims[0]}", "value": str(g.idxmax())})
    return kpis[:4]
