"""The calculation chat inside the WHOLE app: the tab must hand the app's real table to the chat."""
import sys
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_fakes  # noqa: E402
from agents.chat_agent import ChatSession  # noqa: E402
from calc_fakes import local_runner  # noqa: E402
from ui import chat_tab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app" / "streamlit_app.py")


class U:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


def test_the_app_gives_the_chat_its_table_and_a_calculation_runs_on_it(monkeypatch):
    seen = {}

    def factory(profile, settings):
        seen["df"] = settings["df"]
        llm = ui_fakes.ScriptedLlm(replies=["```python\nresult = int(len(df))\n```", "The table has rows."])
        return ChatSession(profile, llm, tables=settings["df"], runner=local_runner, sleep=ui_fakes._no_sleep)
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: True)
    monkeypatch.setattr(chat_tab, "default_session_factory", factory)
    sample = (ROOT / "sample_data" / "sales_sample.csv").read_bytes()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U("sales_sample.csv", sample)])
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()                        # "Upload my own CSV" is the default source; the mocked upload loads immediately
    at.chat_input[0].set_value("how many rows?").run()
    assert not at.exception
    n = len(seen["df"])
    assert n > 0 and "revenue" in seen["df"].columns                        # the app's real table, not a stand-in
    reply = at.session_state["chat_session"]
    assert [c.ok for c in reply.transcript and at.session_state["chat_calcs"][1]] == [True]
    assert f"{n}" in at.session_state["chat_calcs"][1][0].result_text
