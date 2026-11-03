"""The WHOLE app (app/streamlit_app.py) with the AI dashboard tab, run with Streamlit's AppTest.
The AI models are replaced by fakes, so no Gemini request is ever made."""
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_fakes  # noqa: E402
from ui import dashboard_tab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")
DEMO_LABEL = "Olist e-commerce demo set (Nov 2017)"


class U:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


def open_sample(monkeypatch):
    sample = (ROOT / "sample_data" / "sales_sample.csv").read_bytes()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U("sales_sample.csv", sample)])
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()                        # "Upload my own CSV" is the default source; the mocked upload loads immediately
    return at


def test_the_app_still_runs_and_has_six_tabs_with_ai_dashboard_second_and_chat_third(monkeypatch):
    at = open_sample(monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "✨ AI dashboard", "💬 Chat", "Chart gallery",
                                          "Build a chart", "Code sandbox"]


def test_plan_review_and_build_inside_the_real_app(monkeypatch):
    monkeypatch.setattr(dashboard_tab, "default_session_factory", ui_fakes.make_factory("ok"))
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: True)
    at = open_sample(monkeypatch)
    at.button(key="ai_plan").click().run()
    assert not at.exception and at.session_state["ai_stage"] == "review"
    before = len(at.get("plotly_chart"))                       # charts of the other tabs
    at.checkbox(key="ai_keep_1_2").uncheck()
    at.button(key="ai_build").click().run()
    assert not at.exception and at.session_state["ai_stage"] == "done"
    assert len(at.get("plotly_chart")) - before == 2             # exactly the two approved charts were added
    assert at.session_state["tables"] and at.session_state["active_table"] == "sales_sample"   # existing state intact


def test_switching_the_data_source_resets_the_ai_tab(monkeypatch):
    monkeypatch.setattr(dashboard_tab, "default_session_factory", ui_fakes.make_factory("ok"))
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: True)
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value(DEMO_LABEL).run()                    # has data: the Olist demo set loads on its own
    at.button(key="ai_plan").click().run()
    assert at.session_state["ai_stage"] == "review"
    at.sidebar.radio[0].set_value("Upload my own CSV").run()          # no data now (nothing uploaded): the app stops
    assert not at.exception                                            # early, no crash
    at.sidebar.radio[0].set_value(DEMO_LABEL).run()
    assert at.session_state["ai_stage"] == "review"                    # the pending plan was not lost


def test_without_a_key_the_plan_button_is_disabled_and_explains_why(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dashboard_tab, "api_key_available", lambda: False)
    at = open_sample(monkeypatch)
    assert at.button(key="ai_plan").disabled is True
    assert any("No Gemini API key" in e.value for e in at.error)
