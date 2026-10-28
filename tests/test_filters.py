"""Day 17: the filter functions (no Streamlit)."""
import datetime as dt

import pandas as pd

from tools.filters import (MAX_FILTER_COLUMNS, MAX_FILTER_VALUES, apply_filters, describe_filters, filter_options,
                           is_active, signature)
from tools.profiler import profile_dataframe


def make_df():
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "order_date": days,
        "region": (["East", "West", "North"] * 20),
        "product": [f"p{i % 25}" for i in range(60)],            # 25 different values: too many to filter on
        "channel": (["web", "shop"] * 30),
        "revenue": [float(i) for i in range(60)],
    })
    df.loc[0, "region"] = None                                    # a missing region
    return df


def make_profile(df, time_column="order_date"):
    p = profile_dataframe(df)
    for c in p["columns"]:                                        # the stand-in profiler has no date role: say it by hand
        if c["name"] == "order_date":
            c["role"] = "datetime"
    p["time_column"] = time_column
    return p


def test_it_offers_small_category_columns_and_the_date_range_only():
    df = make_df()
    o = filter_options(df, make_profile(df))
    assert list(o["values"]) == ["region", "channel"]                                 # not product (25), not revenue
    assert o["values"]["region"] == ["East", "North", "West"]                         # the missing value is not an option
    assert o["date"] == ("order_date", dt.date(2025, 1, 1), dt.date(2025, 3, 1))


def test_it_offers_nothing_for_columns_that_cannot_be_filtered_usefully():
    df = pd.DataFrame({"a": ["x"] * 5, "d": pd.to_datetime(["2025-01-01"] * 5)})
    p = make_profile(df, time_column="d")
    o = filter_options(df, p)
    assert "a" not in o["values"]                                                     # one value: nothing to choose
    assert o["date"] is None                                                          # one single day: nothing to choose
    big = pd.DataFrame({"b": [f"v{i}" for i in range(MAX_FILTER_VALUES + 1)] * 2})
    assert filter_options(big, profile_dataframe(big))["values"] == {}                # too many different values


def test_it_offers_at_most_a_few_columns():
    df = pd.DataFrame({f"c{i}": ["a", "b"] * 5 for i in range(MAX_FILTER_COLUMNS + 3)})
    assert len(filter_options(df, profile_dataframe(df))["values"]) == MAX_FILTER_COLUMNS


def test_choosing_some_values_keeps_exactly_those_rows():
    df = make_df()
    o = filter_options(df, make_profile(df))
    out = apply_filters(df, o, {"values": {"region": ["East"], "channel": ["web", "shop"]}})
    assert len(out) == int((df["region"] == "East").sum()) and set(out["region"]) == {"East"}


def test_choosing_everything_is_no_filter_and_keeps_rows_with_missing_values():
    df = make_df()
    o = filter_options(df, make_profile(df))
    chosen = {"values": {"region": ["East", "North", "West"]}, "date": (dt.date(2025, 1, 1), dt.date(2025, 3, 1))}
    out = apply_filters(df, o, chosen)
    assert len(out) == len(df) and out["region"].isna().sum() == 1
    assert not is_active(o, chosen) and describe_filters(o, chosen, 60, 60) == "" and signature(o, chosen) == ""


def test_the_date_range_includes_both_end_days():
    df = make_df()
    o = filter_options(df, make_profile(df))
    out = apply_filters(df, o, {"date": (dt.date(2025, 1, 10), dt.date(2025, 1, 12))})
    assert list(out["order_date"].dt.day) == [10, 11, 12]


def test_the_date_range_also_cuts_rows_that_have_a_time_of_day():
    df = make_df()
    df["order_date"] = df["order_date"] + pd.Timedelta(hours=23)                       # 2025-01-12 23:00 is still the 12th
    o = filter_options(df, make_profile(df))
    out = apply_filters(df, o, {"date": (dt.date(2025, 1, 10), dt.date(2025, 1, 12))})
    assert len(out) == 3


def test_dates_stored_as_text_work_too():
    df = make_df()
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    o = filter_options(df, make_profile(df))
    assert o["date"] == ("order_date", dt.date(2025, 1, 1), dt.date(2025, 3, 1))
    assert len(apply_filters(df, o, {"date": (dt.date(2025, 2, 1), dt.date(2025, 2, 28))})) == 28


def test_filters_combine_with_and_and_an_empty_choice_leaves_no_rows():
    df = make_df()
    o = filter_options(df, make_profile(df))
    out = apply_filters(df, o, {"values": {"region": ["East"], "channel": ["web"]}, "date": (dt.date(2025, 1, 1), dt.date(2025, 1, 31))})
    assert len(out) > 0 and set(out["region"]) == {"East"} and set(out["channel"]) == {"web"} and out["order_date"].max().month == 1
    assert len(out) < int((df["region"] == "East").sum())
    assert len(apply_filters(df, o, {"values": {"region": []}})) == 0


def test_choices_for_unknown_columns_are_ignored_and_the_input_is_not_changed():
    df = make_df()
    before = df.copy()
    o = filter_options(df, make_profile(df))
    assert len(apply_filters(df, o, {"values": {"nope": ["x"]}})) == len(df)
    apply_filters(df, o, {"values": {"region": ["East"]}})
    pd.testing.assert_frame_equal(df, before)


def test_the_label_names_every_cut_and_the_row_counts():
    df = make_df()
    o = filter_options(df, make_profile(df))
    chosen = {"values": {"region": ["West", "East"], "channel": ["web", "shop"]}, "date": (dt.date(2025, 2, 1), dt.date(2025, 2, 28))}
    label = describe_filters(o, chosen, 12, 60)
    assert label == "region: East, West | order_date: 2025-02-01 to 2025-02-28 (12 of 60 rows)"


def test_the_signature_changes_with_the_slice_and_not_with_the_order_of_choices():
    df = make_df()
    o = filter_options(df, make_profile(df))
    a = signature(o, {"values": {"region": ["East", "West"]}})
    b = signature(o, {"values": {"region": ["West", "East"]}})
    c = signature(o, {"values": {"region": ["East"]}})
    d = signature(o, {"date": (dt.date(2025, 1, 2), dt.date(2025, 3, 1))})
    assert a == b and a and len({a, c, d}) == 3 and "East" in c and "2025-01-02" in d


def test_the_result_has_clean_row_numbers():
    df = make_df()
    o = filter_options(df, make_profile(df))
    out = apply_filters(df, o, {"values": {"region": ["West"]}})
    assert list(out.index) == list(range(len(out))) and 0 < len(out) < len(df)
