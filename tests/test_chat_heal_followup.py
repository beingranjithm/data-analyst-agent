"""Day 16d: after a failed / empty / zero try the model may take one more step (look, then redo) while it has runs left."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from test_chat_agent import PROFILE_TEXT  # noqa: E402
from test_chat_calc import CODE, make  # noqa: E402

TYPO = "```python\nresult = df[df['region'] == 'east']\n```"                      # empty table
LOOK = "```python\nresult = df['region'].value_counts().reset_index()\n```"       # a diagnostic step that works
BAD = "```python\nresult = df['nope'].sum()\n```"
ZERO = "```python\nresult = 0\n```"
OFFER = "If you still need another calculation"


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


def test_after_an_empty_try_a_successful_look_around_may_be_followed_by_another_calculation():
    llm, s = make([TYPO, LOOK, CODE, "East revenue is 626.25."], max_code_runs=3)
    r = s.ask("East revenue?")
    assert r.ok and r.text == "East revenue is 626.25." and llm.calls == 4
    assert [(c.ok, c.empty) for c in r.calculations] == [(True, True), (True, False), (True, False)]
    told = llm.seen[2][4][1]                                                   # what came back after the look-around
    assert OFFER in told and "RESULT of your calculation" in told and "prose only" in told


def test_after_a_failed_try_a_successful_result_may_also_be_followed_by_another_calculation():
    llm, s = make([BAD, LOOK, CODE, "East revenue is 626.25."], max_code_runs=3)
    assert s.ask("q").ok and OFFER in llm.seen[2][4][1]


def test_after_a_zero_the_model_may_ask_again_too():
    llm, s = make([ZERO, CODE, "East revenue is 626.25."], max_code_runs=3)
    assert s.ask("q").ok and llm.calls == 3
    assert "exactly 0" in llm.seen[1][2][1] and OFFER in llm.seen[2][4][1]


def test_a_normal_first_result_still_gets_the_prose_only_instruction():
    llm, s = make([CODE, "East revenue is 626.25."], max_code_runs=3)
    s.ask("q")
    told = llm.seen[1][2][1]
    assert "Reply with prose only, no code." in told and OFFER not in told


def test_on_the_last_run_the_model_must_answer_in_prose_whatever_happened_before():
    llm, s = make([TYPO, LOOK, "Region names are Europe, East, North ... no 'east'."], max_code_runs=2)
    r = s.ask("q")
    assert r.ok and llm.calls == 3
    assert OFFER not in llm.seen[2][4][1] and "Reply with prose only, no code." in llm.seen[2][4][1]


def test_the_number_of_requests_stays_bounded_even_if_the_model_never_stops_asking():
    llm, s = make([TYPO, LOOK, LOOK, LOOK, LOOK, LOOK], max_code_runs=3)
    r = s.ask("q")
    assert not r.ok and llm.calls <= 3 + 2
