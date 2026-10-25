"""Checks that the plan schema agrees with your REAL charts/profiler code (uses sample_data)."""
from pathlib import Path

import pytest

from tools.charts import AGG_FUNCS, build_chart, suggest_charts
from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.plan_schema import AGGS, ChartSpec, DashboardPlan, validate_plan_against_profile
from tools.profiler import profile_dataframe

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"


@pytest.fixture(scope="module")
def data():
    df, _ = load_csv_bytes(SAMPLE.read_bytes())
    df, _ = clean_dataframe(df)
    return df, profile_dataframe(df)


def test_every_schema_agg_is_known_to_charts():
    assert set(AGGS) <= set(AGG_FUNCS), f"unknown to charts.py: {set(AGGS) - set(AGG_FUNCS)}"


def test_rule_based_gallery_fits_the_schema(data):
    df, profile = data
    charts = [ChartSpec(**{**s, "title": s.get("title", "t")}) for s in suggest_charts(profile, max_charts=8)]
    plan = DashboardPlan(title="gallery", charts=charts)
    assert validate_plan_against_profile(plan, profile) == []


def test_plan_specs_build_real_charts(data):
    df, profile = data
    charts = [ChartSpec(**{**s, "title": s.get("title", "t")}) for s in suggest_charts(profile, max_charts=8)]
    for c in charts:
        build_chart(df, c.to_spec())  # must not raise
