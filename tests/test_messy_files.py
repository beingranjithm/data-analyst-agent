from pathlib import Path

import pandas as pd
import pytest

from tools.cleaner import clean_dataframe
from tools.loader import load_csv_bytes
from tools.profiler import profile_dataframe

MESSY = Path(__file__).parent.parent / "sample_data" / "messy"


def run(filename):
    df, notes = load_csv_bytes((MESSY / filename).read_bytes())
    df, log = clean_dataframe(df)
    return df, notes + log, profile_dataframe(df)


def roles(p):
    return {c["name"]: c["role"] for c in p["columns"]}


def test_mixed_date_formats_are_read_correctly():
    df, _, p = run("1_mixed_dates.csv")
    expected = pd.date_range("2025-01-01", periods=80, freq="3D")
    got = df["order_date"]
    wrong = [i for i in range(80) if pd.notna(got[i]) and got[i] != expected[i]]
    assert wrong == []
    assert p["time_column"] == "order_date"


def test_ugly_headers_are_cleaned():
    df, notes, _ = run("2_ugly_headers.csv")
    assert "Sales Amount" in df.columns
    assert "column_2" in df.columns
    assert len(set(df.columns)) == len(df.columns)
    assert any("renamed" in n for n in notes)


def test_currency_and_percent_text_become_numbers():
    df, _, p = run("3_currency_text.csv")
    assert pd.api.types.is_float_dtype(df["price"])
    assert df["price"].isna().sum() == 3  # the "-" placeholders
    assert p["columns"][0]["missing"] == 3
    assert "price" in p["measures"] and "growth" in p["measures"]


def test_empty_and_constant_columns_flagged():
    _, _, p = run("4_empty_constant.csv")
    r = roles(p)
    assert r["empty_col"] == "empty"
    assert r["notes"] == "text"
    assert any("Constant" in w for w in p["warnings"])


def test_tiny_file_warns():
    _, _, p = run("5_tiny.csv")
    assert any("Very small" in w for w in p["warnings"])


def test_semicolon_latin1_file_is_read():
    df, notes, p = run("6_semicolon_latin1.csv")
    assert df.shape == (80, 4)
    assert df["produkt"].iloc[0].startswith("Käse")
    assert pd.api.types.is_float_dtype(df["preis"])
    assert p["time_column"] == "datum"
    assert any("delimiter" in n for n in notes)


def test_header_only_file_gives_friendly_error():
    with pytest.raises(ValueError, match="no data rows"):
        load_csv_bytes((MESSY / "7_header_only.csv").read_bytes())


def test_empty_bytes_give_friendly_error():
    with pytest.raises(ValueError, match="empty"):
        load_csv_bytes(b"")


def test_wide_file_warns():
    _, _, p = run("8_wide.csv")
    assert p["dataset"]["columns"] == 200
    assert any("wide" in w for w in p["warnings"])


def test_year_column_is_not_a_measure_and_dayfirst_dates_are_right():
    df, _, p = run("9_year_zip_dayfirst.csv")
    assert "year" not in p["measures"]
    assert "zip_code" in p["id_columns"]
    for i in range(80):
        assert df["joined"][i] == pd.Timestamp(2024, (i % 12) + 1, (i % 28) + 1)
