"""The AI dashboard tab, clicked through with Streamlit's AppTest. Fake models: no Gemini, no real waiting."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.llm_cache import CachingLlm  # noqa: E402
from agents.llm_fallback import FallbackLlm  # noqa: E402
from ui import dashboard_tab as tab  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "ui_harness.py")


def new_app(scenario="ok", key="k1"):
    at = AppTest.from_file(HARNESS, default_timeout=90)
    at.session_state["scenario"] = scenario
    at.session_state["data_key"] = key
    return at.run()


def to_review(scenario="ok"):
    at = new_app(scenario)
    at.button(key="ai_plan").click().run()
    return at


def texts(at):
    return " ".join(str(getattr(e, "value", "")) for e in
                    list(at.markdown) + list(at.caption) + list(at.info) + list(at.error) + list(at.warning)
                    + list(at.subheader))


def charts(at):
    return len(at.get("plotly_chart"))


def test_the_tab_starts_idle_with_sensible_settings():
    at = new_app()
    assert not at.exception and at.session_state["ai_stage"] == "idle"
    assert at.button(key="ai_plan").label.endswith("Plan a dashboard")
    assert at.checkbox(key="ai_use_ai").value is True and at.number_input(key="ai_max_charts").value == 6
    assert "never your rows" in texts(at)                       # the privacy note is shown


def test_planning_shows_the_proposal_for_review():
    at = to_review()
    assert not at.exception and at.session_state["ai_stage"] == "review"
    assert [at.checkbox(key=f"ai_keep_1_{i}").value for i in (1, 2, 3)] == [True, True, True]
    assert at.text_input(key="ai_title_1").value == "Sales"
    assert at.button(key="ai_build").label.endswith("Build the dashboard") and charts(at) == 0
    assert "Compare regions." in texts(at)


def test_building_everything_shows_kpis_charts_and_a_download():
    at = to_review()
    at.button(key="ai_build").click().run()
    assert not at.exception and at.session_state["ai_stage"] == "done"
    assert charts(at) == 3 and at.subheader[0].value == "Sales"
    assert at.metric[0].label == "Revenue" and at.metric[0].value not in ("n/a", "")
    assert len(at.get("download_button")) == 1
    assert "AI requests in this run" in texts(at)


def test_an_unticked_chart_is_not_built():
    at = to_review()
    at.checkbox(key="ai_keep_1_2").uncheck()
    at.button(key="ai_build").click().run()
    assert not at.exception and charts(at) == 2
    shown = texts(at)
    assert "Compare regions." in shown and "See the spread." not in shown        # chart 2's caption is absent


def test_the_title_can_be_changed_and_an_empty_title_is_refused():
    at = to_review()
    at.text_input(key="ai_title_1").set_value("Board pack")
    at.button(key="ai_build").click().run()
    assert at.subheader[0].value == "Board pack"

    at2 = to_review()
    at2.text_input(key="ai_title_1").set_value("   ")
    at2.button(key="ai_build").click().run()
    assert at2.session_state["ai_stage"] == "review" and "title cannot be empty" in texts(at2)


def test_nothing_selected_is_refused_and_the_plan_stays_open():
    at = to_review()
    for i in (1, 2, 3):
        at.checkbox(key=f"ai_keep_1_{i}").uncheck()
    at.button(key="ai_build").click().run()
    assert at.session_state["ai_stage"] == "review" and "Select at least one chart" in texts(at)
    at.checkbox(key="ai_keep_1_1").check()                       # and it still works afterwards
    at.button(key="ai_build").click().run()
    assert at.session_state["ai_stage"] == "done" and charts(at) == 1


def test_cancel_builds_nothing_and_start_over_returns_to_the_start():
    at = to_review()
    at.button(key="ai_cancel").click().run()
    assert at.session_state["ai_stage"] == "cancelled" and "Nothing was built" in texts(at) and charts(at) == 0
    at.button(key="ai_restart").click().run()
    assert at.session_state["ai_stage"] == "idle" and at.button(key="ai_plan")


def test_a_failed_planner_still_gives_a_rule_based_proposal():
    at = to_review("planner_fails")
    assert not at.exception and at.session_state["ai_stage"] == "review"
    assert "rule-based" in texts(at) and len(at.checkbox) >= 1


def test_a_crashing_session_shows_an_error_and_can_be_retried():
    at = to_review("factory_error")
    assert not at.exception and at.session_state["ai_stage"] == "error"
    assert "boom" in texts(at)
    at.button(key="ai_restart").click().run()
    assert at.session_state["ai_stage"] == "idle"


def test_a_new_dataset_starts_over_but_keeps_the_settings():
    at = new_app()
    at.number_input(key="ai_max_charts").set_value(3)
    at.button(key="ai_plan").click().run()
    assert at.session_state["ai_stage"] == "review"
    at.session_state["data_key"] = "k2"
    at.run()
    assert at.session_state["ai_stage"] == "idle" and at.number_input(key="ai_max_charts").value == 3


def test_settings_survive_cancel_and_start_over():
    at = new_app()
    at.checkbox(key="ai_use_ai").uncheck()
    at.number_input(key="ai_max_charts").set_value(4)
    at.button(key="ai_plan").click().run()                        # the settings widgets are not drawn in this stage
    at.button(key="ai_cancel").click().run()
    at.button(key="ai_restart").click().run()
    assert at.checkbox(key="ai_use_ai").value is False and at.number_input(key="ai_max_charts").value == 4


def test_a_second_plan_after_starting_over_uses_fresh_widgets():
    at = to_review()
    at.checkbox(key="ai_keep_1_2").uncheck()
    at.button(key="ai_cancel").click().run()
    at.button(key="ai_restart").click().run()
    at.button(key="ai_plan").click().run()
    assert at.checkbox(key="ai_keep_2_2").value is True            # a new run id: no leftover unticked boxes


def test_without_ai_the_tab_uses_the_rule_based_gallery_and_no_model():
    at = new_app()
    at.checkbox(key="ai_use_ai").uncheck()
    at.button(key="ai_plan").click().run()
    assert not at.exception and "rule-based" in texts(at)


# ---------------------------------------------------------------- small helpers
def test_kpi_formatting():
    assert tab.format_kpi(None) == "n/a" and tab.format_kpi(1708450.5) == "1,708,450"
    assert tab.format_kpi(6094.0) == "6,094" and tab.format_kpi(0.05) == "0.05" and tab.format_kpi(294.27) == "294.27"


def test_usage_counts_requests_and_replays(tmp_path):
    a = CachingLlm(model="a", inner=None, path=str(tmp_path / "c.json"), real_calls=2, hits=1)
    b = CachingLlm(model="b", inner=None, path=str(tmp_path / "c.json"), real_calls=1, hits=4)
    assert tab.usage_of(None) == (0, 0) and tab.usage_of(a) == (2, 1)
    assert tab.usage_of(FallbackLlm(chain=[a, b])) == (3, 5)


def test_the_key_check_reads_the_environment(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert tab.api_key_available() is False
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert tab.api_key_available() is True


def test_the_cache_mode_labels_map_to_the_factory_modes():
    assert sorted(tab.CACHE_MODES.values()) == ["off", "replay", "use"]
