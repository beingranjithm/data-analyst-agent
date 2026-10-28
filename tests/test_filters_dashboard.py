"""Day 17: the AI dashboard says which rows it was built for, and warns when the filter changed afterwards."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))

HARNESS = str(Path(__file__).resolve().parent / "dash_scope_harness.py")
SCOPE = "region: East (10 of 40 rows)"


def new_app(scope=""):
    at = AppTest.from_file(HARNESS, default_timeout=90)
    at.session_state["scope"] = scope
    return at.run()


def notes(at):
    return " ".join(e.value for e in list(at.info) + list(at.warning) + list(at.caption))


def build(at):
    at.button(key="ai_plan").click().run()
    at.button(key="ai_build").click().run()
    return at


def test_before_planning_the_tab_says_which_rows_the_dashboard_will_use():
    at = new_app(SCOPE)
    assert not at.exception and any(SCOPE in i.value and "will be built only" in i.value for i in at.info)
    assert not any("filtered rows" in i.value for i in new_app("").info)


def test_a_built_dashboard_says_which_rows_it_covers_and_does_not_warn():
    at = build(new_app(SCOPE))
    assert not at.exception and at.session_state["ai_stage"] == "done"
    assert any(f"Built only for the filtered rows: {SCOPE}" in c.value for c in at.caption)
    assert not any("filter changed" in w.value for w in at.warning)


def test_changing_the_filter_after_the_build_warns_instead_of_silently_showing_old_charts():
    at = build(new_app(SCOPE))
    at.session_state["scope"] = "region: West (10 of 40 rows)"
    at.run()
    warning = " ".join(w.value for w in at.warning)
    assert "filter changed" in warning and SCOPE in warning and "region: West" in warning
    assert at.session_state["ai_stage"] == "done"                                       # the dashboard itself is kept


def test_removing_the_filter_after_the_build_also_warns():
    at = build(new_app(SCOPE))
    at.session_state["scope"] = ""
    at.run()
    assert any("all rows" in w.value and "filter changed" in w.value for w in at.warning)


def test_starting_over_builds_for_the_new_rows_and_the_warning_goes_away():
    at = build(new_app(SCOPE))
    at.session_state["scope"] = "region: West (10 of 40 rows)"
    at.run()
    at.button(key="ai_restart").click().run()
    assert at.session_state["ai_stage"] == "idle" and at.session_state["ai_scope"] == "region: West (10 of 40 rows)"
    assert not any("filter changed" in w.value for w in at.warning)


def test_without_any_filter_nothing_extra_is_shown():
    at = build(new_app(""))
    assert not at.exception and "filtered rows" not in notes(at) and "filter changed" not in notes(at)
