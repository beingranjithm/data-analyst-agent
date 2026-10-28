"""Day 17: the chat knows which rows it is talking about, and starts fresh when the slice changes."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import ChatSession, scope_note  # noqa: E402
from test_chat_agent import PROFILE_TEXT  # noqa: E402
from test_chat_calc import make  # noqa: E402
import ui_fakes  # noqa: E402
from calc_fakes import local_runner  # noqa: E402
from ui import chat_tab  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "chat_scope_harness.py")
APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
SCOPE = "region: East, West (20 of 40 rows)"
CODE = "```python\nresult = int(len(df))\n```"


@pytest.fixture
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


# ------------------------------------------------------------------ what the model is told
def test_without_a_filter_the_model_is_told_nothing_extra(simple_profile_text):
    llm, s = make(["Answer."])
    s.ask("q")
    assert "FILTER ACTIVE" not in llm.systems[0] and scope_note("") == "" and s.scope == ""


def test_with_a_filter_the_model_is_told_which_rows_it_is_talking_about(simple_profile_text):
    llm, s = make(["Answer."], scope=SCOPE)
    s.ask("q")
    system = llm.systems[0]
    assert "FILTER ACTIVE" in system and SCOPE in system and "ONLY" in system
    assert system.index("FILTER ACTIVE") < system.index(PROFILE_TEXT.splitlines()[0])      # before the profile itself


def test_numbers_in_the_filter_label_are_not_flagged_as_invented(simple_profile_text):
    _, s = make(["Your filter keeps 20 of 40 rows."], scope=SCOPE)
    assert s.ask("how many rows are left?").ungrounded == []


def test_the_filter_label_with_curly_braces_cannot_break_the_prompt(simple_profile_text):
    llm, s = make(["Answer."], scope="region: {East} (1 of 2 rows)")
    assert s.ask("q").ok and "{East}" in llm.systems[0]


# ------------------------------------------------------------------ the chat tab
def test_the_tab_shows_the_filter_and_hands_it_to_the_session():
    at = AppTest.from_file(HARNESS, default_timeout=60)
    at.session_state["scope"] = SCOPE
    at.run()
    assert not at.exception
    assert any(SCOPE in i.value and "new conversation" in i.value for i in at.info)
    at.session_state["replies"] = ["Fine."]
    at.chat_input[0].set_value("hello").run()
    assert at.session_state["settings_seen"] == {"scope": SCOPE, "rows": 40}


def test_no_filter_means_no_notice():
    at = AppTest.from_file(HARNESS, default_timeout=60).run()
    assert not at.exception and not any("filtered rows" in i.value for i in at.info)


# ------------------------------------------------------------------ the whole app
def test_in_the_whole_app_the_chat_calculates_on_the_filtered_rows_and_starts_fresh_when_the_filter_changes(monkeypatch):
    seen = []

    def factory(profile, settings):
        seen.append(dict(rows=len(settings["df"]), scope=settings.get("scope"), df=settings["df"]))
        llm = ui_fakes.ScriptedLlm(replies=[CODE, "Counted.", CODE, "Counted again."])
        return ChatSession(profile, llm, tables=settings["df"], runner=local_runner, sleep=ui_fakes._no_sleep,
                           scope=settings.get("scope", ""))
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: True)
    monkeypatch.setattr(chat_tab, "default_session_factory", factory)
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert not at.exception
    box = at.sidebar.multiselect[0]
    column, options = box.label, list(box.options)
    at.chat_input[0].set_value("how many rows?").run()
    total = seen[-1]["rows"]
    assert seen[-1]["scope"] == "" and total > 0
    first_answer = at.session_state["chat_calcs"][1][0].result_text
    assert str(total) in first_answer

    dropped = options[0]                                                                  # take one value out of the filter
    at.sidebar.multiselect[0].unselect(dropped).run()
    assert not at.exception
    assert not at.session_state["chat_session"].transcript                                # a new slice = a new conversation
    full = seen[0]["df"]
    expected = int(full[column].astype(str).isin(set(options) - {dropped}).sum())
    at.chat_input[0].set_value("how many rows?").run()
    assert seen[-1]["rows"] == expected and 0 < expected < total
    assert column in seen[-1]["scope"] and dropped not in seen[-1]["scope"] and f"{expected:,} of {total:,} rows" in seen[-1]["scope"]
    assert str(expected) in at.session_state["chat_calcs"][1][0].result_text
    assert at.metric[0].value == f"{total:,}"                                              # the Overview still counts all rows

