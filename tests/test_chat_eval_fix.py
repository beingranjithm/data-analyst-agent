"""Day 18b: the evaluation counts answers that the number-copy guard rewrote."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.chat_agent import ChatReply  # noqa: E402
from evals.chat_eval import PASS, FAIL, build_questions, render_markdown, run_eval, summarize  # noqa: E402
from test_chat_eval import make_df, oracle_reply  # noqa: E402


def test_rewritten_answers_are_counted_and_reported():
    qs, _ = build_questions(make_df())
    a, b, c = qs[0], qs[1], qs[2]
    replies = {a.text: oracle_reply(a), b.text: oracle_reply(b), c.text: oracle_reply(c, text="It is 1.")}
    replies[a.text].corrected = True
    replies[c.text].corrected = True
    results = run_eval([a, b, c], lambda t: replies[t])
    s = summarize(results)
    assert [r.status for r in results] == [PASS, PASS, FAIL] and [r.corrected for r in results] == [True, False, True]
    assert (s["corrected"], s["corrected_ok"]) == (2, 1)
    assert "Number-copy guard: 2 answer(s) rewritten once after a number check, 1 of them correct afterwards" in render_markdown(results)


def test_a_reply_without_the_new_field_still_works():
    qs, _ = build_questions(make_df())

    class Old:                                   # a reply object from before Day 18b
        ok, text, ungrounded, error, calculations = True, "Answer: 1", [], None, []
    r = run_eval([qs[2]], lambda t: Old())[0]
    assert r.corrected is False
