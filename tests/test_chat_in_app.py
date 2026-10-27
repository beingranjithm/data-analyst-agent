"""The Chat tab inside the WHOLE app (app/streamlit_app.py). A scripted fake model: no Gemini request is made."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_fakes  # noqa: E402
from agents.chat_agent import ChatSession  # noqa: E402
from ui import chat_tab  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_the_chat_tab_works_inside_the_real_app(monkeypatch):
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: True)
    monkeypatch.setattr(chat_tab, "default_session_factory",
                        lambda profile, settings: ChatSession(
                            profile, ui_fakes.ScriptedLlm(replies=["Hello from the fake."]), sleep=ui_fakes._no_sleep))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert at.session_state["chat_session"].transcript == [("user", "hi"), ("model", "Hello from the fake.")]
    assert at.session_state["tables"] and at.session_state["active_table"] == "sales_sample"   # existing state intact
