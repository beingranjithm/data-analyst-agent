"""Insight-led chart titles: grounded sentences computed straight from the data (no model)."""
import pandas as pd

from tools.insights import _line_insight, chart_insight_title
from tools.plan_schema import ChartSpec


def test_bar_sum_states_leader_and_share():
    df = pd.DataFrame({"region": ["North", "South", "North", "East"], "revenue": [10.0, 20.0, 45.0, 25.0]})
    chart = ChartSpec(type="bar", title="x", dimension="region", measure="revenue", agg="sum")
    assert chart_insight_title(df, chart) == "North leads with 55% of revenue"


def test_donut_sum_states_leader_and_share():
    df = pd.DataFrame({"channel": ["Organic", "Paid", "Organic"], "revenue": [38.0, 32.0, 30.0]})
    chart = ChartSpec(type="donut", title="x", dimension="channel", measure="revenue", agg="sum")
    assert chart_insight_title(df, chart) == "Organic leads with 68% of revenue"


def test_bar_count_states_leader_and_share():
    df = pd.DataFrame({"status": ["Open", "Open", "Open", "Closed"], "order_id": [1, 2, 3, 4]})
    chart = ChartSpec(type="bar", title="x", dimension="status", measure="order_id", agg="count")
    assert chart_insight_title(df, chart) == "Open leads with 75% of order_id"


def test_bar_mean_states_highest_average_not_a_share():
    df = pd.DataFrame({"region": ["A", "A", "B"], "price": [10.0, 10.0, 5.0]})
    chart = ChartSpec(type="bar", title="x", dimension="region", measure="price", agg="mean")
    assert chart_insight_title(df, chart) == "A has the highest average price (10)"


def test_line_rising_trend_states_pct_change():
    df = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=10, freq="D"),
                        "revenue": list(range(1, 11))})
    chart = ChartSpec(type="line", title="x", time="d", measure="revenue", agg="sum")
    insight = chart_insight_title(df, chart)
    assert insight is not None and "rose" in insight and "revenue" in insight


def test_line_zero_starting_value_returns_none_rather_than_a_pct_change():
    df = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=10, freq="D"),
                        "revenue": [0.0] + list(range(1, 10))})
    chart = ChartSpec(type="line", title="x", time="d", measure="revenue", agg="sum")
    assert chart_insight_title(df, chart) is None
    # called directly (bypassing chart_insight_title's catch-all) so a reintroduced
    # division-by-zero actually fails this test instead of being silently swallowed
    assert _line_insight(df, "d", "revenue", "sum") is None


def test_line_with_a_single_valid_point_returns_none():
    df = pd.DataFrame({"d": [pd.Timestamp("2026-01-01"), pd.NaT], "revenue": [10.0, 20.0]})
    chart = ChartSpec(type="line", title="x", time="d", measure="revenue", agg="sum")
    assert chart_insight_title(df, chart) is None


def test_scatter_states_correlation_strength_and_direction():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    chart = ChartSpec(type="scatter", title="x", x="x", y="y")
    insight = chart_insight_title(df, chart)
    assert insight == "Strong positive relationship between x and y (r=1.00)"


def test_histogram_states_median_cluster():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    chart = ChartSpec(type="histogram", title="x", measure="v")
    assert chart_insight_title(df, chart) == "Most v values cluster around 3"


def test_too_few_rows_returns_none_rather_than_guessing():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    chart = ChartSpec(type="histogram", title="x", measure="v")
    assert chart_insight_title(df, chart) is None


def test_box_and_heatmap_are_not_rewritten_yet():
    df = pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]})
    chart = ChartSpec(type="box", title="x", dimension="region", measure="revenue")
    assert chart_insight_title(df, chart) is None


def test_missing_column_never_raises():
    df = pd.DataFrame({"region": ["A", "B"], "revenue": [1.0, 2.0]})
    chart = ChartSpec(type="bar", title="x", dimension="region", measure="does_not_exist", agg="sum")
    assert chart_insight_title(df, chart) is None


def test_long_insight_is_truncated():
    df = pd.DataFrame({"a_very_long_dimension_name_indeed": ["a_very_long_category_name_here_too"] * 3
                        + ["other"], "a_very_long_measure_name_for_testing_purposes": [10.0, 10.0, 10.0, 1.0]})
    chart = ChartSpec(type="bar", title="x", dimension="a_very_long_dimension_name_indeed",
                       measure="a_very_long_measure_name_for_testing_purposes", agg="sum")
    insight = chart_insight_title(df, chart)
    assert insight is not None and len(insight) <= 90
