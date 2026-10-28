"""Day 22: the true answers of the Formula 1 questions are computed correctly (tiny made-up tables, hand-checked)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evals.chat_eval import PASS, FAIL, run_eval, score  # noqa: E402
from evals.f1_eval import DEFINITIONS, build_f1_questions, read_raw  # noqa: E402

RAW = {
    "drivers": pd.DataFrame({"driverId": [1, 2, 3, 4], "forename": ["Lewis", "Max", "Jos", "Michael"],
                             "surname": ["Hamilton", "Verstappen", "Verstappen", "Schumacher"],
                             "nationality": ["British", "Dutch", "Dutch", "Brazilian"]}),
    "constructors": pd.DataFrame({"constructorId": [1, 2], "name": ["Mercedes", "Ferrari"]}),
    "circuits": pd.DataFrame({"circuitId": [1, 2], "name": ["Monaco", "Monza"], "country": ["Monaco", "Italy"]}),
    "races": pd.DataFrame({"raceId": [10, 11, 12], "year": [2019, 2019, 2021], "circuitId": [1, 2, 2],
                           "name": ["Monaco Grand Prix", "Italian Grand Prix", "Italian Grand Prix"]}),
    "results": pd.DataFrame({"raceId": [10, 10, 11, 11, 12, 12], "driverId": [1, 2, 1, 4, 2, 4],
                             "constructorId": [1, 1, 1, 2, 1, 2], "position": [1, 2, 1, 2, 1, 2],
                             "points": [25, 18, 25, 18, 25, 18]}),
    "pit_stops": pd.DataFrame({"raceId": [10, 10, 10, 12], "driverId": [1, 1, 2, 2], "stop": [1, 2, 1, 1]}),
    "lap_times": pd.DataFrame({"raceId": [10, 10, 11, 12, 12], "driverId": [1, 2, 1, 2, 1], "lap": [1, 1, 1, 1, 1],
                               "milliseconds": [90000, 80000, 100000, 95000, 120000]}),
}


def expect(n):
    qs, _ = build_f1_questions(RAW)
    return next(q for q in qs if q.id == n).expect


def number_of(text):
    return next(i for i, (_, t, _) in enumerate(DEFINITIONS, 1) if t == text)


def test_each_question_has_the_hand_computed_answer():
    assert expect(number_of("How many races were held in 2019?")).numbers == [2]
    assert expect(number_of("How many drivers are Brazilian?")).numbers == [1]
    e = expect(number_of("Which driver has won the most races, and how many?"))
    assert e.numbers == [2] and e.names == ["Hamilton"]
    assert expect(number_of("How many different drivers have ever won a race?")).numbers == [2]
    assert expect(number_of("How many races did Michael Schumacher win?")).numbers == [0]     # he only finished 2nd here
    assert expect(number_of("What is the total number of points Lewis Hamilton has scored?")).numbers == [50.0]
    e = expect(number_of("Which circuit has hosted the most races, and how many?"))
    assert e.names == ["Monza"] and e.numbers == [2]
    assert expect(number_of("How many pit stops did Max Verstappen make in 2021?")).numbers == [1]
    assert expect(number_of("How many laps did Lewis Hamilton drive in 2019 according to the lap times?")).numbers == [2]
    assert expect(number_of("What was Lewis Hamilton's average lap time in milliseconds in 2019?")).numbers == [(95000, 95000.0, 95000.0)]
    assert expect(number_of("Which driver set the fastest lap in the lap times data?")).names == ["Verstappen"]


def test_the_two_event_table_question_uses_the_winner_of_that_race():
    e = expect(number_of("How many pit stops did the winner of the 2019 Monaco Grand Prix make, and who was it?"))
    assert e.numbers == [2] and e.names == ["Hamilton"]          # Hamilton won race 10 and has 2 stops there (not Verstappen's)


def test_a_driver_who_is_not_one_person_is_skipped_not_guessed():
    raw = dict(RAW, drivers=pd.concat([RAW["drivers"], RAW["drivers"].iloc[[0]]]))
    qs, skipped = build_f1_questions(raw)
    assert any("not exactly one driver" in why for _, _, why in skipped) and len(qs) < len(DEFINITIONS)


def test_a_missing_file_skips_only_the_questions_that_need_it():
    raw = {k: v for k, v in RAW.items() if k != "pit_stops"}
    qs, skipped = build_f1_questions(raw)
    assert {t for _, t, _ in skipped} >= {"How many pit stops did Max Verstappen make in 2021?"}
    assert len(qs) >= 10


def test_a_tie_for_first_is_skipped():
    raw = dict(RAW, results=RAW["results"].assign(position=[1, 2, 2, 1, 1, 2]))
    qs, skipped = build_f1_questions(raw)
    assert any("tied" in why for _, _, why in skipped)


def test_the_traps_are_traps_and_scoring_works_with_the_shared_harness():
    qs, _ = build_f1_questions(RAW)
    traps = [q for q in qs if q.category == "traps"]
    assert len(traps) == 2 and all(q.expect.trap for q in traps)
    assert [q.expect.allow_zero for q in traps] == [False, True]
    q = next(q for q in qs if q.text.startswith("How many races were held in 2019"))
    assert score(q, "There were 2 races in 2019.").status == PASS and score(q, "There were 7.").status == FAIL
    res = run_eval(qs[:1], lambda t: type("R", (), {"ok": True, "text": "2 races", "ungrounded": [], "calculations": []})())
    assert res[0].status == PASS


def test_read_raw_reads_backslash_n_as_missing(tmp_path):
    (tmp_path / "results.csv").write_text("raceId,position\n1,1\n1,\\N\n")
    r = read_raw(tmp_path)["results"]
    assert r.position.isna().sum() == 1 and list(r) == ["raceId", "position"]
    assert set(read_raw(tmp_path)) == {"results"}


def test_the_report_goes_to_its_own_file(tmp_path):
    from evals.chat_eval import save_report
    qs, _ = build_f1_questions(RAW)
    res = run_eval(qs[:2], lambda t: type("R", (), {"ok": True, "text": "x", "ungrounded": [], "calculations": []})())
    path, replaced = save_report(res, [], {"k": 1}, tmp_path, name="f1_eval", title="Across tables")
    assert path.name == "f1_eval.md" and replaced and (tmp_path / "f1_eval.json").exists()
    assert path.read_text().startswith("# Across tables") and not (tmp_path / "chat_eval.md").exists()
    smaller = save_report(res[:1], [], {}, tmp_path, name="f1_eval")[0]
    assert smaller.name == "f1_eval_partial.md"
