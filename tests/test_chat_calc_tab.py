"""The Chat tab WITH calculations, clicked through with Streamlit's AppTest (fake model, fake runner)."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import chat_tab  # noqa: E402

HARNESS = str(Path(__file__).resolve().parent / "chat_calc_harness.py")
EAST_SUM = "```python\nresult = df[df['region'] == 'East']['revenue'].sum()\n```"
BAD = "```python\nresult = df['nope'].sum()\n```"


def new_app(replies=(), key="k1", **state):
    at = AppTest.from_file(HARNESS, default_timeout=60)
    at.session_state["replies"] = list(replies)
    at.session_state["data_key"] = key
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def say(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def expander_labels(at):
    return [e.label for e in at.expander]


def test_the_tab_explains_what_the_ai_sees_when_it_can_calculate():
    at = new_app()
    assert not at.exception
    notes = " ".join(c.value for c in at.caption)
    assert "locked-down sandbox" in notes and "never the whole table" in notes and "Show the calculation" in notes
    assert "cannot calculate new numbers" not in notes


def test_a_calculation_answer_comes_with_a_look_at_the_code_and_the_result():
    at = say(new_app([EAST_SUM, "East revenue is 480 in total."]), "East total?")
    assert not at.exception
    assert any("East revenue" in m.value for m in at.markdown)
    assert expander_labels(at) == ["🧮 Show the calculation"]
    assert any("df[df['region'] == 'East']" in c.value for c in at.code)
    assert any("Result shown to the AI" in t.value for t in at.text)
    assert at.session_state["llm"].calls == 2


def test_a_failed_calculation_is_labelled_as_failed_and_the_error_is_shown():
    at = say(new_app([BAD, "Sorry, that could not be worked out."]), "q")
    assert expander_labels(at) == ["⚠ The calculation failed"]
    assert any("KeyError" in t.value and "Error:" in t.value for t in at.text)
    assert any("could not be worked out" in m.value for m in at.markdown)


def test_two_tries_show_both_calculations_in_one_expander():
    at = say(new_app([BAD, EAST_SUM, "East is 480."], max_code_runs=2), "q")
    assert expander_labels(at) == ["🧮 Show the calculation"]
    assert len(at.code) == 2 and any("(failed)" in c.value for c in at.caption)


def test_an_answer_without_a_calculation_has_no_expander():
    at = say(new_app(["There are 40 rows."]), "rows?")
    assert not at.exception and expander_labels(at) == []


def test_an_invented_number_is_still_flagged_next_to_a_calculated_answer():
    at = say(new_app([EAST_SUM, "East is 480 and the whole company made 424,242."]), "q")
    assert any("424242" in w.value for w in at.warning)
    assert not any("480" in w.value for w in at.warning)


def test_a_calculation_stays_visible_after_the_next_question_and_goes_with_a_reset():
    at = say(new_app([EAST_SUM, "East is 480.", "Fine."]), "q1")
    say(at, "q2")
    assert expander_labels(at) == ["🧮 Show the calculation"]            # still under the first answer
    at.button(key="chat_reset").click().run()
    assert not at.exception and expander_labels(at) == [] and at.session_state["chat_calcs"] == {}


def test_a_calculation_starter_question_is_offered():
    at = new_app()
    assert any(b.label == "Which region has the highest total revenue?" for b in at.button)
    assert chat_tab._starters({"measures": ["revenue"], "dimensions": ["region"]}, False) != \
        chat_tab._starters({"measures": ["revenue"], "dimensions": ["region"]}, True)


def test_the_real_session_factory_gives_the_chat_its_data():
    import pandas as pd
    from agents.chat_agent import CALC_INSTRUCTION, INSTRUCTION
    frame = pd.DataFrame({"a": [1, 2]})
    from tools.profiler import profile_dataframe
    profile = profile_dataframe(frame)                                      # a real profile, as the app builds it
    with_data = chat_tab.default_session_factory(profile, {"cache_mode": "replay", "df": frame})   # replay: no Gemini
    without = chat_tab.default_session_factory(profile, {"cache_mode": "replay", "df": None})
    assert with_data.tables is frame and with_data.instruction == CALC_INSTRUCTION
    assert without.tables is None and without.instruction == INSTRUCTION
