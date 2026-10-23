from pathlib import Path

import numpy as np
import pandas as pd

from tools.profiler import profile_dataframe

SAMPLE = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"


def roles(profile):
    return {c["name"]: c["role"] for c in profile["columns"]}


def test_sample_sales_roles():
    p = profile_dataframe(pd.read_csv(SAMPLE))
    r = roles(p)
    assert p["dataset"]["rows"] == 600
    assert p["dataset"]["columns"] == 8
    assert r["order_date"] == "datetime"
    assert r["region"] == "categorical"
    assert r["category"] == "categorical"
    assert r["revenue"] == "numeric"
    assert p["time_column"] == "order_date"
    assert "revenue" in p["measures"]
    assert "region" in p["dimensions"]


def test_sample_sales_missing_values():
    p = profile_dataframe(pd.read_csv(SAMPLE))
    missing = {c["name"]: c["missing"] for c in p["columns"]}
    assert missing["region"] == 10
    assert missing["discount"] == 15


def test_id_column_detected():
    df = pd.DataFrame({
        "customer_id": range(1, 101),
        "amount": np.random.default_rng(0).normal(100, 10, 100),
    })
    p = profile_dataframe(df)
    assert "customer_id" in p["id_columns"]
    assert "amount" in p["measures"]


def test_duplicates_and_constant_column():
    df = pd.DataFrame({"a": [1, 2, 3] * 10, "b": ["x"] * 30})
    p = profile_dataframe(df)
    assert p["dataset"]["duplicate_rows"] == 27
    assert any("Constant" in w for w in p["warnings"])


def test_dates_stored_as_text():
    df = pd.DataFrame({"when": ["2025-01-01", "2025-01-15", "2025-02-03",
                                "2025-03-09", "2025-04-20", "2025-05-11"]})
    assert roles(profile_dataframe(df))["when"] == "datetime"


def test_numbers_stored_as_text():
    df = pd.DataFrame({"price": ["1,200", "950", "3,400", "75", "10", "20"]})
    p = profile_dataframe(df)
    assert roles(p)["price"] == "numeric"
    assert any("stored as text" in w for w in p["warnings"])


def test_empty_dataframe_does_not_crash():
    p = profile_dataframe(pd.DataFrame())
    assert p["dataset"]["rows"] == 0
