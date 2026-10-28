"""Day 18b: the number-copy guard. The model computed the right result but copied a number wrongly (the evaluation's
question 15: 280050 became 280505). The chat now asks the model ONCE to check its numbers against the result."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.llm_cache import CachingLlm  # noqa: E402
from test_chat_agent import PROFILE_TEXT  # noqa: E402
from test_chat_calc import CODE, make  # noqa: E402

WRONG = "East revenue is 626.52 in total."         # the true result is 626.25: two digits swapped
RIGHT = "East revenue is 626.25 in total."
BAD_CODE = "```python\nresult = df['nope'].sum()\n```"


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


def test_a_wrongly_copied_number_is_corrected_with_one_extra_request():
    llm, s = make([CODE, WRONG, RIGHT])
    r = s.ask("East revenue?")
    assert r.ok and r.text == RIGHT and r.corrected is True and r.ungrounded == [] and llm.calls == 3
    asked = llm.seen[2]                                                       # what the model saw when asked to check
    assert asked[-2] == ("model", WRONG)                                      # it sees its own wrong answer ...
    assert "626.52" in asked[-1][1] and "digit for digit" in asked[-1][1] and "prose only" in asked[-1][1]
    assert any("RESULT of your calculation" in text and "626.25" in text for _, text in asked)    # ... and the result


def test_only_the_corrected_answer_is_remembered():
    llm, s = make([CODE, WRONG, RIGHT, "Fine."])
    s.ask("East revenue?")
    assert s.transcript == [("user", "East revenue?"), ("model", RIGHT)]
    s.ask("thanks")
    assert ("model", RIGHT) in llm.seen[3] and not any(WRONG in t for _, t in llm.seen[3])


def test_a_correct_answer_costs_no_extra_request():
    llm, s = make([CODE, RIGHT])
    r = s.ask("East revenue?")
    assert r.ok and r.corrected is False and llm.calls == 2 and r.text == RIGHT


def test_if_the_second_try_is_still_wrong_the_first_answer_and_its_warning_stay():
    llm, s = make([CODE, WRONG, "East revenue is 626.53 in total."])
    r = s.ask("East revenue?")
    assert r.ok and r.text == WRONG and r.ungrounded == ["626.52"] and r.corrected is False and llm.calls == 3


def test_it_asks_only_once_even_when_the_model_keeps_getting_it_wrong():
    llm, s = make([CODE, WRONG, "626.99", "626.98", "626.97"], max_code_runs=3)
    s.ask("East revenue?")
    assert llm.calls == 3


def test_a_rewrite_with_fewer_wrong_numbers_is_taken_even_if_one_remains():
    llm, s = make([CODE, "East is 626.52 and West is 777.77.", "East is 626.25 and West is 777.78."])
    r = s.ask("q")
    assert r.corrected and r.text == "East is 626.25 and West is 777.78." and r.ungrounded == ["777.78"]


def test_a_rewrite_that_asks_for_more_code_is_ignored():
    llm, s = make([CODE, WRONG, BAD_CODE], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and r.text == WRONG and r.corrected is False


def test_an_empty_rewrite_is_ignored():
    llm, s = make([CODE, WRONG, "   "])
    r = s.ask("q")
    assert r.ok and r.text == WRONG and not r.corrected


def test_a_model_problem_while_checking_never_costs_the_person_their_answer():
    llm, s = make([CODE, WRONG, RuntimeError("503 UNAVAILABLE")])
    r = s.ask("q")
    assert r.ok and r.text == WRONG and r.ungrounded == ["626.52"] and not r.corrected


def test_the_daily_quota_running_out_during_the_check_also_keeps_the_answer():
    llm, s = make([CODE, WRONG, RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for requests per day")])
    r = s.ask("q")
    assert r.ok and r.text == WRONG


def test_no_calculation_means_nothing_to_check_against_so_no_extra_request():
    llm, s = make(["The summary shows 777.77 in total."])                   # a made-up number, but no result to compare
    r = s.ask("q")
    assert r.ok and llm.calls == 1 and not r.corrected and r.ungrounded == ["777.77"]


def test_it_can_be_switched_off():
    llm, s = make([CODE, WRONG], fix_numbers=False)
    r = s.ask("q")
    assert r.ok and r.text == WRONG and llm.calls == 2 and not r.corrected


def test_the_check_after_a_self_healed_calculation_uses_the_good_result():
    llm, s = make([BAD_CODE, CODE, WRONG, RIGHT], max_code_runs=3)
    r = s.ask("q")
    assert r.ok and r.text == RIGHT and r.corrected and llm.calls == 4 and [c.ok for c in r.calculations] == [False, True]


def test_the_total_number_of_requests_stays_bounded():
    llm, s = make([BAD_CODE, BAD_CODE, BAD_CODE, BAD_CODE, BAD_CODE, BAD_CODE], max_code_runs=3)
    s.ask("q")
    assert llm.calls <= 3 + 2 + 1


def test_the_check_is_one_cached_request_so_a_replay_costs_nothing(tmp_path):
    from test_chat_calc import DF
    from agents.chat_agent import ChatSession
    from calc_fakes import local_runner
    from test_chat_agent import PROFILE, ScriptedLlm, _no_sleep
    path = tmp_path / "c.db"
    inner = ScriptedLlm(replies=[CODE, WRONG, RIGHT], seen=[], systems=[])
    first = ChatSession(PROFILE, CachingLlm(model="chat-model", inner=inner, path=str(path)), tables=DF, runner=local_runner, sleep=_no_sleep)
    assert first.ask("East revenue?").text == RIGHT
    again = ChatSession(PROFILE, CachingLlm(model="chat-model", inner=None, mode="replay", path=str(path)),
                        tables=DF, runner=local_runner, sleep=_no_sleep)
    r = again.ask("East revenue?")
    assert r.ok and r.text == RIGHT and r.corrected


# ------------------------------------------------------------------ the chat tab
HARNESS = str(Path(__file__).resolve().parent / "chat_calc_harness.py")
EAST_SUM = "```python\nresult = df[df['region'] == 'East']['revenue'].sum()\n```"


def tab(replies):
    at = AppTest.from_file(HARNESS, default_timeout=60)
    at.session_state["replies"] = list(replies)             # must be set before the first run: the session is built then
    at.session_state["data_key"] = "k1"
    return at.run()


def captions(at):
    return " ".join(c.value for c in at.caption)


def test_the_tab_tells_the_person_when_an_answer_was_rewritten_and_shows_no_warning():
    at = tab([EAST_SUM, "East revenue is 408 in total.", "East revenue is 480 in total."])
    at.chat_input[0].set_value("East total?").run()
    assert not at.exception
    drafts = [e for e in at.expander if "first draft" in e.label]       # Day 20: the replaced answer is kept, but only in here
    assert len(drafts) == 1 and any("408" in m.value for m in drafts[0].markdown)
    outside = [m.value for m in at.markdown if not any(m is x for x in drafts[0].markdown)]
    assert any("East revenue is 480" in v for v in outside) and not any("408" in v for v in outside)
    assert "rewritten once" in captions(at) and len(at.warning) == 0


def test_the_tab_says_nothing_extra_for_a_normal_answer():
    at = tab([EAST_SUM, "East revenue is 480 in total."])
    at.chat_input[0].set_value("East total?").run()
    assert "rewritten once" not in captions(at) and len(at.warning) == 0


def test_the_warning_stays_when_the_check_did_not_help_and_a_new_conversation_clears_the_note():
    at = tab([EAST_SUM, "East revenue is 408 in total.", "East revenue is 409 in total."])
    at.chat_input[0].set_value("East total?").run()
    assert "rewritten once" not in captions(at) and any("408" in w.value for w in at.warning)
    at2 = tab([EAST_SUM, "East revenue is 408 in total.", "East revenue is 480 in total."])
    at2.chat_input[0].set_value("East total?").run()
    assert "rewritten once" in captions(at2)
    at2.button(key="chat_reset").click().run()
    assert "rewritten once" not in captions(at2) and at2.session_state["chat_fixed"] == {}
