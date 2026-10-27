"""Day 15: the chat can ask for a calculation, which runs in the sandbox. Fake models and a fake runner:
no Gemini, no real sandbox, no real waiting."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import (CALC_INSTRUCTION, ChatSession, format_result, ungrounded_numbers)  # noqa: E402
from agents.llm_cache import CachingLlm  # noqa: E402
from calc_fakes import FakeRun, local_runner  # noqa: E402
from test_chat_agent import BUSY, DAILY, PROFILE, PROFILE_TEXT, ScriptedLlm, _no_sleep  # noqa: E402

DF = pd.DataFrame({"region": ["East", "West", "East", "North"] * 5, "revenue": [100.0, 50.5, 25.25, 10.0] * 5,
                   "secret_note": ["Marketing-Secret-123"] * 20})
CODE = "```python\nresult = df[df['region'] == 'East']['revenue'].sum()\n```"        # East: 5 x 125.25 = 626.25


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


def make(replies, runner=local_runner, **kw):
    llm = ScriptedLlm(replies=list(replies), seen=[], systems=[])
    kw.setdefault("sleep", _no_sleep)
    return llm, ChatSession(PROFILE, llm, tables=DF, runner=runner, **kw)


def text_of(content):
    return "".join(p.text or "" for p in content.parts)


# ------------------------------------------------------------------ the normal flow
def test_a_question_needing_numbers_runs_the_code_and_explains_the_result():
    llm, s = make([CODE, "East revenue is 626.25 in total."])
    r = s.ask("What is the total revenue of East?")
    assert r.ok and r.text == "East revenue is 626.25 in total." and llm.calls == 2
    assert len(r.calculations) == 1 and r.calculations[0].ok and "626.25" in r.calculations[0].result_text
    assert "df[df['region'] == 'East']" in r.calculations[0].code
    assert r.ungrounded == []                                     # 626.25 is in the result
    second = llm.seen[1]                                          # what the model saw on its second request
    assert [role for role, _ in second] == ["user", "model", "user"]
    assert second[0][1] == "What is the total revenue of East?" and "```python" in second[1][1]
    assert "RESULT of your calculation" in second[2][1] and "626.25" in second[2][1]


def test_only_the_question_and_the_answer_are_remembered_not_the_code():
    llm, s = make([CODE, "East is 626.25.", "Same for West?"])
    s.ask("East total?")
    assert s.transcript == [("user", "East total?"), ("model", "East is 626.25.")]
    s.ask("and the average?")
    assert llm.seen[2] == [("user", "East total?"), ("model", "East is 626.25."), ("user", "and the average?")]


def test_a_question_the_profile_can_answer_makes_one_request_and_no_calculation():
    calls = []
    llm, s = make(["There are 600 rows."], runner=lambda c, t: calls.append(c))
    r = s.ask("how many rows?")
    assert r.ok and llm.calls == 1 and calls == [] and r.calculations == []


def test_the_runner_gets_the_data_the_session_was_given():
    seen = []

    def runner(code, tables):
        seen.append(tables)
        return FakeRun(value=3)
    llm, s = make([CODE, "three"], runner=runner)
    s.ask("q")
    assert seen[0] is DF


def test_without_data_a_code_reply_is_just_an_answer_and_the_old_instruction_is_used():
    llm = ScriptedLlm(replies=["```python\nprint(1)\n```"], seen=[], systems=[])
    s = ChatSession(PROFILE, llm, sleep=_no_sleep)
    r = s.ask("q")
    assert r.ok and llm.calls == 1 and r.calculations == []
    assert "CODE RULES" not in llm.systems[0] and "Answer only from the profile" in llm.systems[0]


def test_the_calculation_instruction_has_the_rules_and_one_placeholder():
    llm, s = make(["fine"])
    s.ask("q")
    system = llm.systems[0]
    assert CALC_INSTRUCTION.count("{") == 1 and "{profile_text}" in CALC_INSTRUCTION and "{profile_text}" not in system
    assert PROFILE_TEXT in system and "CODE RULES" in system and "boolean masks" in system
    assert "Never return the raw table" in system and "result" in system


# ------------------------------------------------------------------ what the model is allowed to see
def test_a_big_table_is_cut_before_the_model_sees_it():
    big = pd.DataFrame({"k": range(50), "v": [1.5] * 50})
    text = format_result(FakeRun(table=big, total=50))
    assert text.count("\n") < 25 and "only the first 20 of 50 rows are shown" in text and "\n49," not in text
    assert len(format_result(FakeRun(value="x" * 10000))) == chat_agent.MAX_RESULT_CHARS
    assert "printed output" in format_result(FakeRun(stdout="hello", value=None))
    assert "result was empty" in format_result(FakeRun())
    assert "3.5" in format_result(FakeRun(value=pd.Series([3.5])))


def test_through_the_whole_flow_the_second_request_never_carries_more_than_twenty_rows():
    code = "```python\nresult = df.copy()\n```"                      # a bad request: the whole table
    llm, s = make([code, "Here is a summary."])
    s.ask("show everything")
    shown = llm.seen[1][2][1]
    assert shown.count("\n") < 30 and "only the first 20 of 20 rows" not in shown
    big = pd.concat([DF] * 3, ignore_index=True)
    llm2 = ScriptedLlm(replies=[code, "ok"], seen=[], systems=[])
    ChatSession(PROFILE, llm2, tables=big, runner=local_runner, sleep=_no_sleep).ask("show everything")
    assert "only the first 20 of 60 rows are shown" in llm2.seen[1][2][1]


# ------------------------------------------------------------------ failures
def test_a_failed_calculation_is_reported_and_values_from_the_data_are_hidden():
    bad = "```python\nresult = df['Marketing-Secret-123']\n```"       # KeyError: 'Marketing-Secret-123' (a data value)
    llm, s = make([bad, "Sorry, I could not work that out."])
    r = s.ask("q")
    assert r.ok and r.text == "Sorry, I could not work that out." and llm.calls == 2
    assert not r.calculations[0].ok and r.ungrounded == []
    told = llm.seen[1][2][1]
    assert "The calculation failed" in told and "KeyError" in told
    assert "You cannot run another calculation" in told and "Do not guess a number" in told


def test_error_text_from_the_data_never_reaches_the_model():
    def runner(code, tables):
        return FakeRun("error", error={"type": "ValueError", "message": "bad value 'Marketing-Secret-123' in column"})
    llm, s = make([CODE, "could not"], runner=runner)
    s.ask("q")
    assert "Marketing-Secret-123" not in llm.seen[1][2][1] and "<value>" in llm.seen[1][2][1]


def test_blocked_code_is_explained_to_the_model():
    runner = lambda c, t: FakeRun("blocked", problems=["import os is not allowed"])      # noqa: E731
    llm, s = make([CODE, "I cannot do that."], runner=runner)
    s.ask("q")
    assert "import os is not allowed" in llm.seen[1][2][1]


def test_a_crashing_tool_is_a_failed_calculation_not_a_crash():
    def runner(code, tables):
        raise RuntimeError("secret internal path /home/x")
    llm, s = make([CODE, "could not"], runner=runner)
    r = s.ask("q")
    assert r.ok and not r.calculations[0].ok and "RuntimeError" in llm.seen[1][2][1]
    assert "/home/x" not in llm.seen[1][2][1]


def test_with_two_tries_allowed_a_failure_is_fixed():
    bad = "```python\nresult = df['nope'].sum()\n```"
    llm, s = make([bad, CODE, "East is 626.25."], max_code_runs=2)
    r = s.ask("q")
    assert r.ok and llm.calls == 3 and [c.ok for c in r.calculations] == [False, True]
    assert "Fix the code" in llm.seen[1][2][1]


def test_a_model_that_keeps_asking_for_calculations_is_stopped():
    llm, s = make([CODE, CODE, CODE])
    r = s.ask("q")
    assert not r.ok and "could not finish" in r.error and llm.calls == 3 and s.transcript == []
    assert "No more calculations are possible" in llm.seen[2][4][1]


def test_a_nudge_can_rescue_the_answer():
    llm, s = make([CODE, CODE, "East is 626.25."])
    r = s.ask("q")
    assert r.ok and r.text == "East is 626.25." and len(r.calculations) == 1


# ------------------------------------------------------------------ model problems in the middle of a calculation
def test_a_busy_server_during_the_second_step_is_waited_out_and_the_code_runs_only_once():
    runs = []

    def runner(code, tables):
        runs.append(code)
        return FakeRun(value=626.25)
    llm, s = make([CODE, RuntimeError(BUSY), "East is 626.25."], runner=runner)
    r = s.ask("q")
    assert r.ok and len(runs) == 1 and llm.calls == 3


def test_a_daily_quota_stop_in_the_second_step_leaves_no_trace():
    llm, s = make([CODE, RuntimeError(DAILY)])
    r = s.ask("q")
    assert not r.ok and "daily" in r.error and s.transcript == []


def test_an_empty_final_answer_is_an_error():
    llm, s = make([CODE, "  "])
    assert "empty answer" in s.ask("q").error and s.transcript == []


# ------------------------------------------------------------------ the honesty check now knows about results
def test_numbers_must_come_from_the_result_or_the_summary():
    llm, s = make([CODE, "East revenue is 626.25 and the company total is 99,999."])
    r = s.ask("q")
    assert r.ungrounded == ["99999"]


def test_rounding_of_a_known_number_is_not_flagged_but_other_numbers_are():
    assert ungrounded_numbers("about 1,708,451 in total", PROFILE_TEXT, results="1708450.5") == []
    assert ungrounded_numbers("about 1,708,452 in total", PROFILE_TEXT, results="1708450.5") == ["1708452"]
    assert ungrounded_numbers("share is 12.35", PROFILE_TEXT, results="12.3456") == []
    assert ungrounded_numbers("share is 12.4", PROFILE_TEXT, results="12.3456") == ["12.4"]       # wrongly rounded


# ------------------------------------------------------------------ replay
def test_an_identical_calculation_conversation_costs_nothing_the_second_time(tmp_path):
    path = str(tmp_path / "c.json")

    def session(replies):
        inner = ScriptedLlm(replies=list(replies), seen=[], systems=[])
        return inner, ChatSession(PROFILE, CachingLlm(model="chat-model", inner=inner, path=path),
                                  tables=DF, runner=local_runner, sleep=_no_sleep)
    i1, s1 = session([CODE, "East is 626.25."])
    assert s1.ask("q").text == "East is 626.25." and i1.calls == 2
    i2, s2 = session([])
    r = s2.ask("q")
    assert r.text == "East is 626.25." and i2.calls == 0 and r.calculations[0].ok


def test_numbers_that_only_appear_in_an_error_message_are_not_trusted():
    def runner(code, tables):
        return FakeRun("error", error={"type": "ValueError", "message": "cannot convert 987654 rows"})
    llm, s = make([CODE, "It failed after 987,654 rows."], runner=runner)
    r = s.ask("q")
    assert r.ok and r.ungrounded == ["987654"]
