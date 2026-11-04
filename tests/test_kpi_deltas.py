"""Period-over-period KPI deltas (dashboard SOP, Phase B): pure pandas, no model, graceful
fallback when there's no usable date column or not enough span to compare two periods."""
import pandas as pd

from tools.kpi_deltas import (compute_kpi_delta, compute_kpi_deltas, lower_is_better,
                              pick_comparison_window, preferred_time_column, window_is_balanced)
from tools.plan_schema import ChartSpec, DashboardPlan
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


# ---- Day 26: which date to compare on, and refusing lopsided comparisons ----
PROFILE_TWO_DATES = {"time_column": "deadline", "columns": [
    {"name": "deadline", "role": "datetime", "min": "2026-01-07", "max": "2026-03-10", "span_days": 62},
    {"name": "ordered", "role": "datetime", "min": "2026-01-01", "max": "2026-01-30", "span_days": 29},
    {"name": "revenue", "role": "numeric"}]}


def _plan(*charts):
    return DashboardPlan(title="t", summary="s", charts=list(charts) or [
        ChartSpec(type="histogram", title="h", measure="revenue")])


def test_the_planners_trend_chart_picks_the_date_to_compare_on():
    plan = _plan(ChartSpec(type="line", title="t", time="ordered", measure="revenue", agg="sum"))
    assert preferred_time_column(plan, PROFILE_TWO_DATES) == "ordered"


def test_without_a_trend_chart_the_profiles_time_column_is_used():
    assert preferred_time_column(_plan(), PROFILE_TWO_DATES) == "deadline"


def test_a_trend_chart_on_a_column_that_is_not_a_date_is_ignored():
    plan = _plan(ChartSpec(type="line", title="t", time="revenue", measure="revenue", agg="sum"))
    assert preferred_time_column(plan, PROFILE_TWO_DATES) == "deadline"


def test_the_override_changes_the_window_and_its_label():
    w = pick_comparison_window(PROFILE_TWO_DATES, "ordered")
    assert w.time_col == "ordered" and w.period_days == 14 and w.label == "vs previous 14 days"


def test_a_window_where_the_current_period_holds_a_sliver_of_rows_is_refused():
    # the Olist demo bug: a deadline date running two months past the orders put almost no rows in the
    # "current 30 days", so revenue looked like it fell 97%
    d = list(pd.date_range("2026-01-07", periods=60, freq="D").repeat(10)) + [pd.Timestamp("2026-03-10")]
    df = pd.DataFrame({"deadline": d, "revenue": [1.0] * len(d)})
    w = pick_comparison_window(PROFILE_TWO_DATES)
    df.loc[df["deadline"] > w.current_start, "deadline"] = pd.Timestamp("2026-03-10")
    df.loc[df.index[:-3], "deadline"] = df.loc[df.index[:-3], "deadline"].where(
        df.loc[df.index[:-3], "deadline"] <= w.current_start, w.current_start)
    assert not window_is_balanced(df, w)
    assert compute_kpi_deltas(df, [KpiSpec(label="Revenue", measure="revenue", agg="sum")], PROFILE_TWO_DATES) == {}


def test_an_even_spread_of_rows_is_balanced():
    df = _daily_df("2026-01-01", 60, [1.0] * 60)
    assert window_is_balanced(df, pick_comparison_window(PROFILE_60_DAYS))


def test_compute_kpi_deltas_uses_the_given_time_column():
    df = pd.DataFrame({"ordered": pd.date_range("2026-01-01", periods=30, freq="D"),
                       "deadline": pd.date_range("2026-01-07", periods=30, freq="2D"),
                       "revenue": [10.0] * 15 + [20.0] * 15})
    kpis = [KpiSpec(label="Revenue", measure="revenue", agg="sum")]
    out = compute_kpi_deltas(df, kpis, PROFILE_TWO_DATES, "ordered")
    assert out["Revenue"].direction == "up" and out["Revenue"].good is True
