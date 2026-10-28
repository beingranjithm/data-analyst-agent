"""Day 17: the sidebar filter panel, clicked through with Streamlit's AppTest."""
import datetime as dt
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))

HARNESS = str(Path(__file__).resolve().parent / "filter_panel_harness.py")


def new_app(key="k1"):
    at = AppTest.from_file(HARNESS, default_timeout=60)
    at.session_state["data_key"] = key
    return at.run()


def reported(at):
    vals = {}
    for m in at.markdown:
        for name in ("ROWS", "LABEL", "SIG"):
            if m.value.startswith(name + "="):
                vals[name] = m.value[len(name) + 1:]
    return vals


def region_box(at):
    return [m for m in at.sidebar.multiselect if m.label == "region"][0]


def test_with_nothing_chosen_there_is_no_filter_and_no_notice():
    at = new_app()
    assert not at.exception
    assert reported(at) == {"ROWS": "40", "LABEL": "", "SIG": ""}
    assert len(at.sidebar.info) == 0
    assert region_box(at).value == ["East", "North", "South", "West"]                  # everything starts selected


def test_taking_a_region_out_cuts_the_rows_and_shows_the_filter_in_the_sidebar():
    at = new_app()
    region_box(at).unselect("South").run()
    r = reported(at)
    assert not at.exception and r["ROWS"] == "30"
    assert r["LABEL"] == "region: East, North, West (30 of 40 rows)" and r["SIG"].startswith("region: East, North, West")
    assert "Filter on: region: East, North, West (30 of 40 rows)" in at.sidebar.info[0].value


def test_the_date_range_cuts_rows():
    at = new_app()
    at.sidebar.date_input[0].set_value((dt.date(2025, 1, 1), dt.date(2025, 1, 10))).run()
    assert not at.exception and reported(at)["ROWS"] == "10"
    assert "order_date: 2025-01-01 to 2025-01-10" in reported(at)["LABEL"]


def test_choosing_only_a_first_date_does_not_break_anything():
    at = new_app()
    at.sidebar.date_input[0].set_value((dt.date(2025, 1, 5),)).run()                    # still choosing the second date
    assert not at.exception and reported(at)["ROWS"] == "40"


def test_clear_filters_puts_everything_back():
    at = new_app()
    region_box(at).unselect("South").unselect("West").run()
    assert reported(at)["ROWS"] == "20"
    at.sidebar.button(key="flt_clear").click().run()
    assert not at.exception and reported(at)["ROWS"] == "40" and region_box(at).value == ["East", "North", "South", "West"]


def test_a_different_dataset_starts_with_no_filters():
    at = new_app("data-A")
    region_box(at).unselect("South").run()
    assert reported(at)["ROWS"] == "30"
    at.session_state["data_key"] = "data-B"
    at.run()
    assert reported(at)["ROWS"] == "40" and region_box(at).value == ["East", "North", "South", "West"]
