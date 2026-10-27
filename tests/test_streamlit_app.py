"""The WHOLE app (app/streamlit_app.py) with the AI dashboard tab, run with Streamlit's AppTest.
The AI models are replaced by fakes, so no Gemini request is ever made."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_fakes  # noqa: E402
from ui import dashboard_tab  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def open_sample():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    return at


def test_the_app_still_runs_and_has_five_tabs_with_the_ai_tab_second():
    at = open_sample()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "✨ AI dashboard", "Chart gallery", "Build a chart", "Code sandbox"]


def test_plan_review_and_build_inside_the_real_app(monkeypatch):
    monkeypatch.setattr(dashboard_tab, "default_session_factory", ui_fakes.make_factory("ok"))
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: True)
    at = open_sample()
    at.button(key="ai_plan").click().run()
    assert not at.exception and at.session_state["ai_stage"] == "review"
    before = len(at.get("plotly_chart"))                       # charts of the other tabs
    at.checkbox(key="ai_keep_1_2").uncheck()
    at.button(key="ai_build").click().run()
    assert not at.exception and at.session_state["ai_stage"] == "done"
    assert len(at.get("plotly_chart")) - before == 2             # exactly the two approved charts were added
    assert at.session_state["tables"] and at.session_state["active_table"] == "sales_sample"   # existing state intact


def test_switching_away_and_back_keeps_the_pending_plan(monkeypatch):
    monkeypatch.setattr(dashboard_tab, "default_session_factory", ui_fakes.make_factory("ok"))
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: True)
    at = open_sample()
    at.button(key="ai_plan").click().run()
    assert at.session_state["ai_stage"] == "review"
    at.sidebar.radio[0].set_value("Upload my own CSV").run()          # no data now: the app stops early, no crash
    assert not at.exception
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert at.session_state["ai_stage"] == "review"                    # the pending plan was not lost


def test_without_a_key_the_plan_button_is_disabled_and_explains_why(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: False)
    at = open_sample()
    assert at.button(key="ai_plan").disabled is True
    assert any("No Gemini API key" in e.value for e in at.error)
