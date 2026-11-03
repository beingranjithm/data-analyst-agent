"""Period-over-period KPI deltas (dashboard SOP, Phase B): pure pandas, no model, graceful
fallback when there's no usable date column or not enough span to compare two periods."""
import pandas as pd

from tools.kpi_deltas import (compute_kpi_delta, compute_kpi_deltas, lower_is_better,
                              pick_comparison_window)
from tools.plan_schema import KpiSpec

PROFILE_NO_TIME = {"time_column": None, "columns": []}

PROFILE_SHORT_SPAN = {"time_column": "d", "columns": [
    {"name": "d", "role": "datetime", "min": "2026-01-01", "max": "2026-01-02", "span_days": 1}]}

PROFILE_60_DAYS = {"time_column": "d", "columns": [
    {"name": "d", "role": "datetime", "min": "2026-01-01", "max": "2026-03-01", "span_days": 60}]}

PROFILE_10_DAYS = {"time_column": "d", "columns": [
    {"name": "d", "role": "datetime", "min": "2026-01-01", "max": "2026-01-11", "span_days": 10}]}


def test_no_time_column_means_no_window():
    assert pick_comparison_window(PROFILE_NO_TIME) is None


def test_too_little_span_means_no_window():
    assert pick_comparison_window(PROFILE_SHORT_SPAN) is None


def test_a_span_just_under_the_minimum_also_means_no_window():
    profile = {"time_column": "d", "columns": [
        {"name": "d", "role": "datetime", "min": "2026-01-01", "max": "2026-01-03", "span_days": 2}]}
    assert pick_comparison_window(profile) is None


def test_long_span_uses_a_30_day_window():
    w = pick_comparison_window(PROFILE_60_DAYS)
    assert w is not None and w.period_days == 30 and w.label == "vs previous 30 days"


def test_short_span_uses_half_the_data_as_the_window():
    w = pick_comparison_window(PROFILE_10_DAYS)
    assert w is not None and w.period_days == 5


def test_lower_is_better_detects_cost_like_names():
    assert lower_is_better("Total Cost") is True
    assert lower_is_better("Churn rate") is True
    assert lower_is_better("Total Revenue") is False


def _daily_df(start, n, values):
    return pd.DataFrame({"d": pd.date_range(start, periods=n, freq="D"), "revenue": values})


def test_rising_revenue_is_a_good_up_delta():
    df = _daily_df("2026-01-01", 60, [10.0] * 30 + [20.0] * 30)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Revenue", measure="revenue", agg="sum")
    delta = compute_kpi_delta(df, kpi, w)
    assert delta is not None and delta.direction == "up" and delta.good is True
    assert delta.pct == 100.0           # 300 -> 600 (period sums)


def test_rising_cost_is_a_bad_up_delta():
    df = _daily_df("2026-01-01", 60, [10.0] * 30 + [20.0] * 30)
    df = df.rename(columns={"revenue": "cost"})
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Total Cost", measure="cost", agg="sum")
    delta = compute_kpi_delta(df, kpi, w)
    assert delta is not None and delta.direction == "up" and delta.good is False


def test_tiny_change_is_flat_and_uncolored():
    df = _daily_df("2026-01-01", 60, [10.0] * 30 + [10.05] * 30)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Revenue", measure="revenue", agg="sum")
    delta = compute_kpi_delta(df, kpi, w)
    assert delta is not None and delta.direction == "flat" and delta.good is None


def test_zero_prior_period_returns_none_rather_than_a_fake_pct():
    df = _daily_df("2026-01-01", 60, [0.0] * 30 + [10.0] * 30)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Revenue", measure="revenue", agg="sum")
    assert compute_kpi_delta(df, kpi, w) is None


def test_prior_period_with_no_rows_returns_none_rather_than_crashing():
    # all the data falls in the current window; the prior window has nothing at all
    df = _daily_df("2026-01-31", 30, [20.0] * 30)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Revenue", measure="revenue", agg="sum")
    assert compute_kpi_delta(df, kpi, w) is None


def test_negative_prior_period_uses_the_absolute_value_as_the_denominator():
    # prior period nets to -100 (a loss), current period nets to +60: a real improvement (pct > 0),
    # which only holds if the denominator is abs(-100) rather than the signed -100
    df = _daily_df("2026-01-01", 60, [-10.0] * 10 + [0.0] * 20 + [2.0] * 30)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="Profit", measure="revenue", agg="sum")
    delta = compute_kpi_delta(df, kpi, w)
    assert delta is not None and delta.pct == 160.0 and delta.direction == "up"


def test_missing_column_returns_none_rather_than_raising():
    df = _daily_df("2026-01-01", 60, [10.0] * 60)
    w = pick_comparison_window(PROFILE_60_DAYS)
    kpi = KpiSpec(label="X", measure="does_not_exist", agg="sum")
    assert compute_kpi_delta(df, kpi, w) is None


def test_compute_kpi_deltas_returns_empty_dict_with_no_usable_date_column():
    df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})
    kpis = [KpiSpec(label="Revenue", measure="revenue", agg="sum")]
    assert compute_kpi_deltas(df, kpis, PROFILE_NO_TIME) == {}


def test_compute_kpi_deltas_keyed_by_label_and_skips_unsupported_kpis():
    df = _daily_df("2026-01-01", 60, [10.0] * 30 + [20.0] * 30)
    kpis = [KpiSpec(label="Revenue", measure="revenue", agg="sum"),
            KpiSpec(label="Ghost", measure="nope", agg="sum")]
    out = compute_kpi_deltas(df, kpis, PROFILE_60_DAYS)
    assert list(out.keys()) == ["Revenue"] and out["Revenue"].direction == "up"
