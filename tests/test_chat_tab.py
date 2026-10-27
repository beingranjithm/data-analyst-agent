"""The Chat tab, clicked through with Streamlit's AppTest. A scripted fake model: no Gemini, no real waiting."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import chat_tab  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "chat_harness.py")
OVERVIEW = "Give me a short overview of this dataset."


def new_app(replies=(), scenario="ok", key="k1"):
    at = AppTest.from_file(HARNESS, default_timeout=60)
    at.session_state["replies"] = list(replies)
    at.session_state["scenario"] = scenario
    at.session_state["data_key"] = key
    return at.run()


def say(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def shown(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_the_tab_starts_with_starter_questions_and_the_limits_note():
    at = new_app()
    assert not at.exception and at.session_state["chat_session"].transcript == []
    assert any(b.label == OVERVIEW for b in at.button)
    assert any("cannot calculate new numbers" in c.value for c in at.caption)
    assert not at.chat_message                                   # nothing said yet
    assert any("never your rows" in c.value for c in at.caption)


def test_typing_a_question_shows_both_sides_and_remembers_them():
    at = say(new_app(["It has two columns."]), "What is in here?")
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "What is in here?" in shown(at) and "It has two columns." in shown(at)
    assert not any(b.label == OVERVIEW for b in at.button)       # starters disappear once the chat has started
    assert at.session_state["chat_session"].transcript == [("user", "What is in here?"), ("model", "It has two columns.")]


def test_a_starter_button_asks_that_question():
    at = new_app(["An overview."])
    [b for b in at.button if b.label == OVERVIEW][0].click().run()
    assert not at.exception and "An overview." in shown(at)
    assert at.session_state["llm"].calls == 1


def test_the_second_question_carries_the_first_to_the_model():
    at = say(new_app(["a1", "a2"]), "q1")
    say(at, "q2")
    assert at.session_state["chat_session"].transcript[-2:] == [("user", "q2"), ("model", "a2")]
    assert at.session_state["llm"].calls == 2


def test_numbers_missing_from_the_summary_get_a_warning():
    at = say(new_app(["Total revenue is 9,876,543 overall."]), "total?")
    assert any("9876543" in w.value and "double-check" in w.value for w in at.warning)
    at2 = say(new_app(["It has 40 rows."]), "rows?")                  # 40 rows: a small number, no warning
    assert not at2.warning


def test_a_model_failure_shows_a_message_and_leaves_no_trace():
    at = say(new_app([RuntimeError("401 API key not valid"), "fine now"]), "first")
    assert not at.exception and at.session_state["chat_session"].transcript == []
    assert any("401" in e.value for e in at.error)
    say(at, "second")
    assert not at.error and at.session_state["chat_session"].transcript[0] == ("user", "second")


def test_start_a_new_conversation_clears_it():
    at = say(new_app(["a1"]), "q1")
    at.button(key="chat_reset").click().run()
    assert not at.exception and at.session_state["chat_session"].transcript == [] and not at.chat_message
    assert any(b.label == OVERVIEW for b in at.button)


def test_a_new_dataset_starts_a_new_conversation_but_keeps_the_reply_setting():
    at = say(new_app(["a1"]), "q1")
    labels = list(chat_tab.CACHE_MODES)
    at.selectbox(key="chat_cache_mode").select(labels[2]).run()
    assert at.session_state["chat_saved_mode"] == labels[2]
    assert at.session_state["chat_session"].transcript == [("user", "q1"), ("model", "a1")]   # kept across a setting change
    at.session_state["data_key"] = "k2"
    at.run()
    assert at.session_state["chat_session"].transcript == [] and at.session_state["chat_saved_mode"] == labels[2]


def test_a_factory_that_breaks_gives_a_message_not_a_crash():
    at = new_app(scenario="factory_error")
    assert not at.exception and any("Could not start the chat" in e.value and "boom" in e.value for e in at.error)


def test_without_a_key_the_real_factory_path_explains_and_shows_no_input(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: False)
    script = ("import sys\nsys.path.insert(0, %r)\nimport pandas as pd\nimport streamlit as st\n"
              "from tools.profiler import profile_dataframe\nfrom ui.chat_tab import render_chat\n"
              "df = pd.DataFrame({'a': [1, 2]})\nrender_chat(profile_dataframe(df), 'k')\n"
              % str(Path(__file__).resolve().parent.parent))
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception and any("No Gemini API key" in e.value for e in at.error)
    assert len(at.chat_input) == 0
