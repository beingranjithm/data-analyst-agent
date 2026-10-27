"""Day 16: the chat fixes its own failed or empty calculations. Fake models, fake runner, no waiting."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import (CALC_INSTRUCTION, MAX_CODE_RUNS, ChatSession, is_empty_result,  # noqa: E402
                               ungrounded_numbers)
from agents.llm_cache import CachingLlm  # noqa: E402
from calc_fakes import FakeRun, local_runner  # noqa: E402
from test_chat_agent import PROFILE, PROFILE_TEXT, ScriptedLlm, _no_sleep  # noqa: E402
from test_chat_calc import CODE, DF, make  # noqa: E402

TYPO = "```python\nresult = df[df['region'] == 'east']['revenue']\n```"                 # 'east' matches nothing
NAN = "```python\nresult = float('nan')\n```"
NO_RESULT = "```python\nx = 1\n```"
BAD = "```python\nresult = df['nope'].sum()\n```"


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


# ------------------------------------------------------------------ the app setting
def test_the_app_lets_the_model_try_three_times():
    assert MAX_CODE_RUNS == 3
    from ui import chat_tab
    profile = {"dataset": {"rows": 2, "columns": 1, "duplicate_rows": 0, "missing_pct": 0},
               "columns": [{"name": "a", "role": "numeric"}]}
    s = chat_tab.default_session_factory(profile, {"cache_mode": "replay", "df": pd.DataFrame({"a": [1]})})
    assert s.max_code_runs == MAX_CODE_RUNS


def test_the_default_for_a_bare_session_is_still_one_try():
    assert ChatSession(PROFILE, ScriptedLlm(replies=[], seen=[], systems=[]), tables=DF).max_code_runs == 1


# ------------------------------------------------------------------ what counts as "found nothing"
def test_what_counts_as_an_empty_result():
    assert is_empty_result(FakeRun(table=pd.DataFrame({"a": []}), total=0))
    assert is_empty_result(FakeRun(value=float("nan")))
    assert is_empty_result(FakeRun())                                       # nothing assigned to result
    assert not is_empty_result(FakeRun(value=0.0))                          # a real zero is an answer
    assert not is_empty_result(FakeRun(value=0))
    assert not is_empty_result(FakeRun(value=""))
    assert not is_empty_result(FakeRun(stdout="printed something"))
    assert not is_empty_result(FakeRun(table=pd.DataFrame({"a": [1]}), total=1))
    assert not is_empty_result(FakeRun(table=pd.DataFrame({"a": []}), total=40))   # the sandbox cut a big table


def test_a_filter_typo_is_noticed_and_fixed():
    llm, s = make([TYPO, CODE, "East revenue is 626.25."], max_code_runs=3)
    r = s.ask("East revenue?")
    assert r.ok and llm.calls == 3 and r.text == "East revenue is 626.25."
    assert [(c.ok, c.empty) for c in r.calculations] == [(True, True), (True, False)]
    told = llm.seen[1][2][1]
    assert "found nothing" in told and "case-sensitive" in told and "Fix the code" in told


def test_nan_and_a_missing_result_are_retried_too():
    for first in (NAN, NO_RESULT):
        llm, s = make([first, CODE, "East revenue is 626.25."], max_code_runs=2)
        r = s.ask("q")
        assert r.ok and [c.empty for c in r.calculations] == [True, False], first


def test_a_real_zero_is_accepted_but_the_model_is_told_to_double_check_its_filter():
    zero = "```python\nresult = 0\n```"
    llm, s = make([zero, "The total is 0."], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and llm.calls == 2 and r.text == "The total is 0."              # prose after the warning: accepted
    assert not r.calculations[0].empty and r.calculations[0].zero
    told = llm.seen[1][2][1]
    assert "exactly 0" in told and "typo in a filter value" in told and "rows_matched" in told


def test_a_zero_from_a_filter_typo_is_corrected_when_the_model_takes_the_hint():
    typo_sum = "```python\nresult = df[df['region'] == 'east']['revenue'].sum()\n```"          # 0.0 for 'east'
    llm, s = make([typo_sum, CODE, "East revenue is 626.25."], max_code_runs=3)
    r = s.ask("East revenue?")
    assert r.ok and r.text == "East revenue is 626.25." and [c.zero for c in r.calculations] == [True, False]


def test_with_no_tries_left_a_zero_is_just_the_answer():
    zero = "```python\nresult = 0\n```"
    llm, s = make([zero, "The total is 0."], max_code_runs=1)
    r = s.ask("q")
    assert r.ok and "exactly 0" not in llm.seen[1][2][1] and "RESULT of your calculation" in llm.seen[1][2][1]


def test_only_a_plain_zero_counts():
    from agents.chat_agent import is_zero_scalar
    assert is_zero_scalar(FakeRun(value=0)) and is_zero_scalar(FakeRun(value=0.0))
    assert not is_zero_scalar(FakeRun(value=False)) and not is_zero_scalar(FakeRun(value=0.5))
    assert not is_zero_scalar(FakeRun(value="0")) and not is_zero_scalar(FakeRun())
    assert not is_zero_scalar(FakeRun(table=pd.DataFrame({"a": [0]}), total=1))


def test_when_nothing_matches_and_no_tries_are_left_the_model_must_say_so():
    llm, s = make([TYPO, "No rows matched that region."], max_code_runs=1)
    r = s.ask("q")
    assert r.ok and r.calculations[0].empty and r.text == "No rows matched that region."
    told = llm.seen[1][2][1]
    assert "no matching data was found" in told and "You cannot run another calculation" in told
    assert "Do not guess a number" in told


# ------------------------------------------------------------------ errors
def test_an_error_is_fixed_on_the_second_or_third_try():
    llm, s = make([BAD, BAD, CODE, "East is 626.25."], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and llm.calls == 4 and [c.ok for c in r.calculations] == [False, False, True]
    assert "Fix the code" in llm.seen[1][2][1] and "Fix the code" in llm.seen[2][4][1]
    assert "case-sensitive" in llm.seen[1][2][1]


def test_after_the_last_failed_try_the_model_must_admit_it_and_not_guess():
    llm, s = make([BAD, BAD, BAD, "Sorry, I could not work that out."], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and llm.calls == 4 and [c.ok for c in r.calculations] == [False, False, False]
    told = llm.seen[3][6][1]
    assert "You cannot run another calculation" in told and "Do not guess a number" in told
    assert s.transcript[-1] == ("model", "Sorry, I could not work that out.")


def test_the_number_of_model_requests_for_one_question_is_bounded():
    llm, s = make([CODE] * 20, max_code_runs=3)
    r = s.ask("q")
    assert not r.ok and "could not finish" in r.error
    assert llm.calls == 3 + 2 and s.transcript == []                          # three runs, one nudge, one last look


def test_a_mixed_failure_then_empty_then_success_path():
    llm, s = make([BAD, TYPO, CODE, "East is 626.25."], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and llm.calls == 4
    assert [(c.ok, c.empty) for c in r.calculations] == [(False, False), (True, True), (True, False)]


def test_numbers_from_an_empty_or_failed_try_are_never_trusted():
    llm, s = make([BAD, CODE, "East is 626.25 and 987,654 elsewhere."], max_code_runs=2)
    assert s.ask("q").ungrounded == ["987654"]


# ------------------------------------------------------------------ the percentage rule
def test_a_fraction_written_as_a_percentage_is_not_flagged():
    assert ungrounded_numbers("about 4.98% on average", PROFILE_TEXT, results="0.04982905982905983") == []
    assert ungrounded_numbers("about 4.99% on average", PROFILE_TEXT, results="0.04982905982905983") == ["4.99"]
    assert ungrounded_numbers("share is 77.9%", PROFILE_TEXT, results="0.779202558107478") == []
    assert ungrounded_numbers("about 498%", PROFILE_TEXT, results="4.98") == ["498"]       # only fractions are scaled
    assert ungrounded_numbers("1.5 million", PROFILE_TEXT, results="1500000") == ["1.5"]  # unit words still flagged


# ------------------------------------------------------------------ the rules the model is given
def test_the_instruction_asks_for_match_counts_and_for_the_figure_next_to_a_winner():
    assert CALC_INSTRUCTION.count("{") == 1                                    # still only {profile_text}
    assert "pd.Series(dict(total=total, rows_matched=n))" in CALC_INSTRUCTION
    assert "typo in a filter cannot pass for a real zero" in CALC_INSTRUCTION and "If rows_matched is 0" in CALC_INSTRUCTION
    assert "names a top or bottom item" in CALC_INSTRUCTION and "quote the number" in CALC_INSTRUCTION


# ------------------------------------------------------------------ replay
def test_a_healed_conversation_is_free_the_second_time(tmp_path):
    path = str(tmp_path / "c.json")

    def session(replies):
        inner = ScriptedLlm(replies=list(replies), seen=[], systems=[])
        return inner, ChatSession(PROFILE, CachingLlm(model="chat-model", inner=inner, path=path), tables=DF,
                                  runner=local_runner, sleep=_no_sleep, max_code_runs=3)
    i1, s1 = session([TYPO, CODE, "East is 626.25."])
    assert s1.ask("q").text == "East is 626.25." and i1.calls == 3
    i2, s2 = session([])
    r = s2.ask("q")
    assert r.text == "East is 626.25." and i2.calls == 0 and [c.empty for c in r.calculations] == [True, False]


def test_numpy_zeros_count_and_odd_objects_do_not():
    import numpy as np
    from agents.chat_agent import is_zero_scalar

    class AlwaysEqual:
        def __eq__(self, other):
            return True
    assert is_zero_scalar(FakeRun(value=np.int64(0))) and is_zero_scalar(FakeRun(value=np.float64(0.0)))
    assert not is_zero_scalar(FakeRun(value=np.int64(3))) and not is_zero_scalar(FakeRun(value=AlwaysEqual()))
