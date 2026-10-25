import pandas as pd

from tools.plan_kpis import compute_plan_kpis
from tools.plan_schema import KpiSpec


def df():
    return pd.DataFrame({"revenue": [10.0, 20.0, None, 30.0], "region": ["a", "b", "c", None]})


def test_aggregations_and_missing_note():
    k = compute_plan_kpis(df(), [KpiSpec(label="Revenue", measure="revenue", agg="sum"),
                                 KpiSpec(label="Avg", measure="revenue", agg="mean"),
                                 KpiSpec(label="Orders", measure="revenue", agg="count")])
    assert [x.value for x in k] == [60.0, 20.0, 3.0]
    assert "1 rows with a missing value" in k[0].note


def test_missing_column_and_all_missing_do_not_crash():
    d = pd.DataFrame({"revenue": [None, None]})
    k = compute_plan_kpis(d, [KpiSpec(label="X", measure="nope", agg="sum"),
                              KpiSpec(label="Y", measure="revenue", agg="sum")])
    assert k[0].value is None and "not found" in k[0].note
    assert k[1].value == 0.0 or k[1].value is None


def test_count_works_on_a_text_column():
    k = compute_plan_kpis(df(), [KpiSpec(label="Regions", measure="region", agg="count")])
    assert k[0].value == 3.0
