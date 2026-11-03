"""The Chat tab inside the WHOLE app (app/streamlit_app.py). A scripted fake model: no Gemini request is made."""
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_fakes  # noqa: E402
from agents.chat_agent import ChatSession  # noqa: E402
from ui import chat_tab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")


class U:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


def test_the_chat_tab_works_inside_the_real_app(monkeypatch):
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: True)
    monkeypatch.setattr(chat_tab, "default_session_factory",
                        lambda profile, settings: ChatSession(
                            profile, ui_fakes.ScriptedLlm(replies=["Hello from the fake."]), sleep=ui_fakes._no_sleep))
    sample = (ROOT / "sample_data" / "sales_sample.csv").read_bytes()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U("sales_sample.csv", sample)])
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()                        # "Upload my own CSV" is the default source; the mocked upload loads immediately
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert at.session_state["chat_session"].transcript == [("user", "hi"), ("model", "Hello from the fake.")]
    assert at.session_state["tables"] and at.session_state["active_table"] == "sales_sample"   # existing state intact
