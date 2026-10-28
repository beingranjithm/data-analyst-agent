"""Day 18: the accuracy evaluation (question builder, scorer, runner, report). No AI involved."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.chat_agent import Calculation, ChatReply  # noqa: E402
from evals import chat_eval as ev  # noqa: E402
from evals.chat_eval import (FAIL, MINOR, NO_ANSWER, NOT_RUN, PASS, Expectation, Question, build_questions,  # noqa: E402
                             find_name, parse_numbers, render_markdown, run_eval, save_report, score, shows_any_number,
                             shows_number, summarize)


def make_df(n=300, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 330, n), unit="D"),
        "region": rng.choice(["East", "West", "North", "South", "Europe"], n),
        "category": rng.choice(["Electronics", "Furniture", "Stationery"], n),
        "product": rng.choice(["Desk", "Chair", "Phone", "Laptop", "Pen Set", "Notebook"], n),
        "units": rng.integers(1, 20, n), "unit_price": rng.uniform(5, 500, n).round(2),
        "discount": rng.choice([0.0, 0.05, 0.1], n)})
    df["revenue"] = (df.units * df.unit_price * (1 - df.discount)).round(2)
    return df


# ------------------------------------------------------------------ reading numbers
@pytest.mark.parametrize("text,truth,expected", [
    ("The total is 1,394,618.70.", 1394618.7, True),
    ("About 1394619 in total.", 1394618.7, True),
    ("Roughly 1.4M.", 1394618.7, True),
    ("Roughly 1.4 million.", 1394618.7, True),
    ("Roughly 1.5M.", 1394618.7, False),
    ("It is 1,394,619.70.", 1394618.7, False),                       # one unit off at the shown precision
    ("It is 11,394,618.7.", 1394618.7, False),                       # a longer number is not a smaller one
    ("Growth was 97.19%.", 97.1877, True),
    ("Growth was 97.19%.", 0.971877, True),                          # a percent may stand for a fraction
    ("The correlation is 0.65.", 0.6537, True),
    ("The correlation is 0.66.", 0.6537, False),
    ("It is -3.5.", -3.5, True),
    ("Orders ran from 2025-01-01 to 2025-03-05.", -1, False),        # hyphens in dates are not minus signs
    ("There were 600 orders.", 600, True),
    ("There were 601 orders.", 600, False),
    ("no numbers here", 5, False),
])
def test_reading_numbers(text, truth, expected):
    assert shows_number(text, truth) is expected


def test_parse_numbers_reports_what_was_shown():
    assert parse_numbers("1,250.50 and 3 and 2.5k and 4%") == [(1250.5, 2, 1.0, False), (3.0, 0, 1.0, False),
                                                              (2.5, 1, 1e3, False), (4.0, 0, 1.0, True)]


def test_alternative_readings_are_both_accepted():
    assert shows_any_number("population std is 5.54", (5.5430, 5.5383)) and shows_any_number("sample std is 5.54", (5.5430, 5.5383))
    assert shows_any_number("the average discount is 5%", (0.05, 5.0)) and shows_any_number("it is 0.05", (0.05, 5.0))
    assert not shows_any_number("it is 7", (0.05, 5.0))


def test_names_match_whole_words_only_and_any_fair_spelling():
    assert find_name("Sales were highest in North.", "North") > 0
    assert find_name("Sales were highest in Northeast.", "North") == -1
    assert find_name("november was best", ("November", "Nov")) == 0
    assert find_name("the pen set sold well", "Pen Set") > 0
    assert find_name("nothing", ("a", "b")) == -1


# ------------------------------------------------------------------ the questions
def test_there_are_thirty_questions_in_six_groups_with_truths_computed_from_the_table():
    df = make_df()
    qs, skipped = build_questions(df)
    assert skipped == [] and [q.id for q in qs] == list(range(1, 31))
    cats = [q.category for q in qs]
    assert {c: cats.count(c) for c in dict.fromkeys(cats)} == {"totals": 6, "groups": 6, "filters": 5, "time": 5,
                                                                 "statistics": 3, "traps": 5}
    by = {q.text: q.expect for q in qs}
    east = df[df.region == "East"].revenue.sum()
    assert by["What is the total revenue for the East region?"].numbers == [pytest.approx(east)]
    assert by["What is the total revenue?"].numbers == [pytest.approx(df.revenue.sum())]
    best = df.groupby("region").revenue.sum()
    e = by["Which region has the highest total revenue, and what is that total?"]
    assert e.names == [best.idxmax()] and e.numbers == [pytest.approx(best.max())]
    top3 = df.groupby("product").revenue.sum().sort_values(ascending=False).head(3).index.tolist()
    e = by["What are the top 3 products by total revenue? List them from highest to lowest."]
    assert e.names == top3 and e.ordered
    mar = df[(df.order_date.dt.month == 3)].revenue.sum()
    assert by["What was the total revenue in March 2025?"].numbers == [pytest.approx(mar)]
    assert by["What is the average customer age?"].trap and by["What was the total revenue in 2030?"].allow_zero
    assert sum(q.expect.trap for q in qs) == 5


def test_a_question_the_table_cannot_answer_is_skipped_with_a_reason():
    df = make_df().drop(columns=["product", "discount"])
    qs, skipped = build_questions(df)
    reasons = {t: why for _, t, why in skipped}
    assert "no column product" in reasons["Which product sold the fewest units, and how many units was that?"]
    assert "no column discount" in reasons["What is the average discount?"]
    assert len(qs) + len(skipped) == 30 and len(skipped) >= 5 and qs                          # the rest still run
    assert {q.id for q in qs}.isdisjoint({n for n, _, _ in skipped})


def test_questions_about_groups_this_data_lacks_are_skipped_not_asked_with_a_made_up_truth():
    df = make_df()
    df["product"] = df["product"].replace("Desk", "Table")
    _, skipped = build_questions(df)
    assert any("Desk" in t and "no rows" in why for _, t, why in skipped)
    df2 = make_df()
    df2["region"] = df2["region"].replace("East", "Atlantis")
    _, skipped2 = build_questions(df2)
    assert any("Atlantis" in why for _, _, why in skipped2)                                   # the Atlantis trap is not a trap here


def test_a_question_whose_true_answer_is_not_a_number_is_skipped():
    df = make_df()
    df["revenue"] = np.nan
    qs, skipped = build_questions(df)
    assert any("What is the total revenue?" == t for _, t, _ in skipped) is False             # a sum of nothing is 0, not NaN
    assert any("average revenue per order" in t and "NaN" in why for _, t, why in skipped)


# ------------------------------------------------------------------ judging an answer
def q_of(**kw):
    return Question(1, "x", "q?", Expectation(**kw))


def test_a_correct_answer_passes_and_every_missing_piece_fails_with_a_reason():
    q = q_of(numbers=[100.0], names=["East"])
    assert score(q, "East had 100 in total.").status == PASS
    v = score(q, "East had 99 in total.")
    assert v.status == FAIL and "expected 100" in v.reasons[0] and "99" in v.reasons[0]
    v = score(q, "West had 100 in total.")
    assert v.status == FAIL and "'East'" in v.reasons[0]
    assert score(q, "I do not know.").status == FAIL


def test_a_ranking_must_be_in_order():
    q = q_of(names=["Desk", "Chair", "Phone"], ordered=True)
    assert score(q, "1. Desk 2. Chair 3. Phone").status == PASS
    v = score(q, "1. Chair 2. Desk 3. Phone")
    assert v.status == FAIL and "wrong order" in v.reasons[0]
    assert score(q, "Desk and Chair only").status == FAIL


def test_trap_questions():
    q = q_of(trap=True)
    assert score(q, "The data has no customer age column.").status == PASS
    assert score(q, "The average age is probably 35.").status == FAIL                          # invents instead of declining
    v = score(q, "The data has no age column, but about 35 would be typical.", ungrounded=["35"])
    assert v.status == FAIL and "35" in v.reasons[0]
    z = q_of(trap=True, allow_zero=True)
    assert score(z, "No rows matched Atlantis, so the total revenue is 0.").status == MINOR
    assert score(z, "No rows matched Atlantis.").status == PASS
    assert score(q_of(trap=True), "No rows matched, so the total revenue is 0.").status == FAIL   # a minor slip only where allowed


# ------------------------------------------------------------------ running them
def oracle_reply(q, text=None, ungrounded=(), calcs=()):
    e = q.expect
    if e.trap:
        text = text or "That is not in the data, so I cannot answer it."
    else:
        parts = [str(n[0] if isinstance(n, tuple) else n) for n in e.numbers]
        names = [str(n[0] if isinstance(n, (tuple, list)) else n) for n in e.names]
        text = text or ("Answer: " + ", ".join(names + parts))
    return ChatReply(True, text, list(ungrounded), None, list(calcs))


def oracle_ask(questions):
    table = {q.text: q for q in questions}
    return lambda text: oracle_reply(table[text])


def test_a_perfect_chat_scores_everything_correct():
    qs, _ = build_questions(make_df())
    results = run_eval(qs, oracle_ask(qs))
    s = summarize(results)
    assert [r.status for r in results] == [PASS] * 30 and s["strict"] == 1.0 and s["answered"] == 30


def test_a_chat_that_is_slightly_wrong_everywhere_is_caught():
    qs, _ = build_questions(make_df())
    table = {q.text: q for q in qs}

    def sloppy(text):
        q = table[text]
        if q.expect.trap:
            return oracle_reply(q, "The value is 42.")
        return oracle_reply(q, "Answer: " + ", ".join(["East"] * len(q.expect.names)) + " 12345.678")
    results = run_eval(qs, sloppy)
    assert summarize(results)["passed"] == 0 and all(r.status == FAIL for r in results)


def test_one_question_that_raises_does_not_stop_the_run():
    qs, _ = build_questions(make_df())
    table = {q.text: q for q in qs}

    def flaky(text):
        if text == qs[3].text:
            raise RuntimeError("boom")
        return oracle_reply(table[text])
    results = run_eval(qs, flaky)
    assert results[3].status == NO_ANSWER and "RuntimeError: boom" in results[3].error
    assert [r.status for i, r in enumerate(results) if i != 3] == [PASS] * 29


def test_the_run_stops_cleanly_when_the_daily_quota_is_used_up():
    qs, _ = build_questions(make_df())
    table = {q.text: q for q in qs}
    calls = []

    def ask(text):
        calls.append(text)
        if len(calls) == 4:
            return ChatReply(False, "", [], "The free daily Gemini quota is used up. Try again tomorrow.", [])
        return oracle_reply(table[text])
    results = run_eval(qs, ask)
    assert len(calls) == 4                                                                    # nothing is asked after the quota error
    assert [r.status for r in results[:4]] == [PASS, PASS, PASS, NO_ANSWER]
    assert all(r.status == NOT_RUN and "daily quota" in r.error for r in results[4:]) and len(results) == 30


def test_other_errors_do_not_stop_the_run():
    qs, _ = build_questions(make_df())
    table = {q.text: q for q in qs}

    def ask(text):
        if text == qs[1].text:
            return ChatReply(False, "", [], "CacheMiss: No saved reply for this request (replay-only mode).", [])
        return oracle_reply(table[text])
    results = run_eval(qs, ask)
    assert results[1].status == NO_ANSWER and sum(r.status == PASS for r in results) == 29


def test_the_request_budget_stops_the_run_between_questions():
    qs, _ = build_questions(make_df())
    table = {q.text: q for q in qs}
    used = {"n": 0}

    def ask(text):
        used["n"] += 2
        return oracle_reply(table[text])
    results = run_eval(qs, ask, requests_used=lambda: used["n"], max_requests=10)
    assert used["n"] == 10 and sum(r.status == PASS for r in results) == 5
    assert all(r.status == NOT_RUN and "budget of 10" in r.error for r in results[5:])


def test_progress_is_reported_after_every_question():
    qs, _ = build_questions(make_df())
    seen = []
    run_eval(qs[:4], oracle_ask(qs), progress=seen.append)
    assert [r.id for r in seen] == [1, 2, 3, 4]


def test_healing_and_the_number_check_are_counted():
    qs, _ = build_questions(make_df())
    q1, q2, q3 = qs[0], qs[1], qs[2]
    bad, good = Calculation("x", False, "ValueError", False, False), Calculation("y", True, "1", False, False)
    replies = {
        q1.text: oracle_reply(q1, calcs=[bad, good]),                                          # healed after an error
        q2.text: oracle_reply(q2, ungrounded=["999"], calcs=[good]),                           # correct but flagged: a false alarm
        q3.text: oracle_reply(q3, text="It is 1.", ungrounded=["1"], calcs=[good]),            # wrong and flagged: caught
    }
    results = run_eval([q1, q2, q3], lambda t: replies[t])
    s = summarize(results)
    assert [r.status for r in results] == [PASS, PASS, FAIL] and [r.healed for r in results] == [True, False, False]
    assert (s["healed"], s["false_alarms"], s["wrong_flagged"], s["failed"]) == (1, 1, 1, 1) and s["calc_runs"] == 4


# ------------------------------------------------------------------ the report
def test_the_report_names_wrong_answers_with_the_truth_and_explains_partial_runs():
    qs, skipped = build_questions(make_df().drop(columns=["discount"]))
    table = {q.text: q for q in qs}
    ask_n = {"n": 0}

    def ask(text):
        ask_n["n"] += 1
        if ask_n["n"] == 2:
            return oracle_reply(table[text], text="Answer: 7")
        if ask_n["n"] == 5:
            return ChatReply(False, "", [], "The free daily Gemini quota is used up.", [])
        return oracle_reply(table[text])
    results = run_eval(qs, ask)
    md = render_markdown(results, skipped, {"date": "2026-10-28", "mode": "use"})
    assert md.startswith("# Chat accuracy evaluation") and "date: 2026-10-28 · mode: use" in md
    assert "3 of 4 answered questions fully correct (75%)" in md
    assert "## Wrong answers" in md and f"**2. {qs[1].text}**" in md and f"true answer: {qs[1].expect.describe}" in md
    assert "Partial run: 1 question(s) got no answer and" in md and "## Skipped" in md and "no column discount" in md
    assert md.count("\n| ") >= len(results) + 2 and "## How to read this" in md


def test_a_perfect_report_has_no_wrong_answer_section():
    qs, _ = build_questions(make_df())
    md = render_markdown(run_eval(qs, oracle_ask(qs)))
    assert "30 of 30 answered questions fully correct (100%)" in md and "## Wrong answers" not in md and "Partial run" not in md


def test_saving_never_replaces_a_bigger_report_with_a_smaller_one(tmp_path):
    qs, _ = build_questions(make_df())
    full = run_eval(qs, oracle_ask(qs))
    path, replaced = save_report(full, [], {"mode": "use"}, tmp_path / "eval_results")
    assert replaced and path.name == "chat_eval.md" and json.loads((tmp_path / "eval_results/chat_eval.json").read_text())["summary"]["answered"] == 30
    small = run_eval(qs[:5], oracle_ask(qs))
    path2, replaced2 = save_report(small, [], {"mode": "replay"}, tmp_path / "eval_results")
    assert not replaced2 and path2.name == "chat_eval_partial.md"
    assert json.loads((tmp_path / "eval_results/chat_eval.json").read_text())["summary"]["answered"] == 30     # kept
    again, replaced3 = save_report(full, [], {"mode": "use"}, tmp_path / "eval_results")
    assert replaced3 and again.name == "chat_eval.md"                                                         # an equal run may replace it


def test_a_corrupt_saved_report_is_replaced(tmp_path):
    folder = tmp_path / "eval_results"
    folder.mkdir()
    (folder / "chat_eval.json").write_text("not json at all")
    qs, _ = build_questions(make_df())
    _, replaced = save_report(run_eval(qs[:3], oracle_ask(qs)), [], {}, folder)
    assert replaced


def test_healed_means_a_good_answer_after_a_failed_empty_or_zero_first_try_and_nothing_else():
    qs, _ = build_questions(make_df())
    q = qs[0]
    ok = Calculation("y", True, "1", False, False)
    cases = {
        "two fine calculations": ([ok, ok], False),                                            # more than one try is not healing
        "a failed first try": ([Calculation("x", False, "boom", False, False), ok], True),
        "an empty first try": ([Calculation("x", True, "", True, False), ok], True),
        "a zero first try": ([Calculation("x", True, "0", False, True), ok], True),
        "one fine calculation": ([ok], False),
    }
    for label, (calcs, expected) in cases.items():
        r = run_eval([q], lambda t, calcs=calcs: oracle_reply(q, calcs=calcs))[0]
        assert r.status == PASS and r.healed is expected, label
    wrong = run_eval([q], lambda t: oracle_reply(q, text="It is 1.", calcs=cases["a failed first try"][0]))[0]
    assert wrong.status == FAIL and wrong.healed is False                                       # a wrong answer is never "healed"


def test_the_true_answers_are_independent_of_how_the_questions_are_built():
    df = make_df()
    by = {q.text: q.expect for q in build_questions(df)[0]}
    assert by["What is the total revenue for the East region?"].numbers == [pytest.approx(df[df.region == "East"].revenue.sum())]
    assert by["How many units were sold in total?"].numbers == [pytest.approx(df.units.sum())]
    assert by["What is the median revenue per order?"].numbers == [pytest.approx(df.revenue.median())]
    assert by["Which category has the lowest total revenue?"].names == [df.groupby("category").revenue.sum().idxmin()]
    assert by["Which region has the highest average revenue per order?"].names == [df.groupby("region").revenue.mean().idxmax()]
    assert by["How many orders had a discount greater than zero?"].numbers == [float((df.discount > 0).sum())]
    assert by["On how many different days were orders placed?"].numbers == [float(df.order_date.nunique())]
    assert by["What is the correlation between units and revenue?"].numbers == [pytest.approx(df.units.corr(df.revenue))]
    first, last = df.order_date.min(), df.order_date.max()
    names = by["What is the date of the first order and the date of the last order?"].names
    assert first.strftime("%Y-%m-%d") in names[0] and last.strftime("%Y-%m-%d") in names[1]


def test_the_standard_deviation_accepts_both_the_sample_and_the_population_reading():
    df = make_df()
    (alts,) = {q.text: q.expect for q in build_questions(df)[0]}["What is the standard deviation of units per order?"].numbers
    assert alts == (pytest.approx(df.units.std(ddof=1)), pytest.approx(df.units.std(ddof=0))) and alts[0] != alts[1]
