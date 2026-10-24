from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from tools.charts import (ChartError, build_chart, compute_kpis, default_agg,
                          pick_primary_measure, suggest_charts)
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "sample_data" / "sales_sample.csv"
MESSY = sorted((ROOT / "sample_data" / "messy").glob("*.csv"))


def prepare(path):
    df, _ = load_csv_bytes(Path(path).read_bytes())
    df, _ = clean_dataframe(df)
    return df, profile_dataframe(df)


def test_default_aggregation_rules():
    assert default_agg("revenue") == "sum"
    assert default_agg("unit_price") == "mean"
    assert default_agg("discount") == "mean"
    assert default_agg("Profit Margin") == "mean"


def test_primary_measure_is_revenue():
    _, p = prepare(SAMPLE)
    assert pick_primary_measure(p) == "revenue"


def test_trend_totals_match_the_data():
    df, _ = prepare(SAMPLE)
    fig = build_chart(df, {"type": "line", "time": "order_date", "measure": "revenue"})
    assert len(fig.data[0].y) == 12  # monthly
    assert sum(fig.data[0].y) == pytest.approx(df["revenue"].sum())


def test_bar_totals_match_the_data_and_note_excluded_rows():
    df, _ = prepare(SAMPLE)
    fig = build_chart(df, {"type": "bar", "dimension": "region", "measure": "revenue"})
    expected = df.dropna(subset=["region"])["revenue"].sum()
    assert sum(fig.data[0].x) == pytest.approx(expected)
    assert "10 rows" in fig.layout.annotations[0].text


def test_suggestions_cover_the_main_chart_types():
    df, p = prepare(SAMPLE)
    types = {s["type"] for s in suggest_charts(p)}
    assert {"line", "bar", "histogram", "scatter", "heatmap"} <= types


def test_kpis():
    df, p = prepare(SAMPLE)
    labels = [k["label"] for k in compute_kpis(df, p)]
    assert "Rows analysed" in labels and "Total revenue" in labels


@pytest.mark.parametrize("spec,message", [
    ({"type": "bar", "dimension": "nope", "measure": "revenue"}, "does not exist"),
    ({"type": "bar", "dimension": "region", "measure": "product"}, "not numeric"),
    ({"type": "line", "time": "region", "measure": "revenue"}, "not a date"),
    ({"type": "pie3d"}, "Unknown chart type"),
    ({"type": "bar"}, "missing"),
])
def test_bad_specs_give_friendly_errors(spec, message):
    df, _ = prepare(SAMPLE)
    with pytest.raises(ChartError, match=message):
        build_chart(df, spec)


@pytest.mark.parametrize("path", MESSY, ids=[p.name for p in MESSY])
def test_every_suggested_chart_builds_on_messy_files(path):
    try:
        df, p = prepare(path)
    except ValueError:
        pytest.skip("file is rejected by the loader (expected)")
    for spec in suggest_charts(p):
        fig = build_chart(df, spec)
        assert isinstance(fig, go.Figure)
    compute_kpis(df, p)
