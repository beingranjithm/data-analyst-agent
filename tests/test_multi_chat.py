"""Day 22: the chat across several tables. Fake models, no sandbox, no waiting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import ChatSession  # noqa: E402
from agents.multi_chat import (BlockedRun, check_combination, make_multi_runner, multi_note)  # noqa: E402
from calc_fakes import FakeRun  # noqa: E402
from test_chat_agent import PROFILE, PROFILE_TEXT, ScriptedLlm, _no_sleep  # noqa: E402
from tools.relations import find_links  # noqa: E402

drivers = pd.DataFrame({"driverId": [1, 2, 3], "surname": ["Aaa", "Bbb", "Ccc"]})
races = pd.DataFrame({"raceId": [10, 11], "year": [2020, 2021]})
results = pd.DataFrame({"resultId": range(1, 7), "raceId": [10, 10, 10, 11, 11, 11],
                        "driverId": [1, 2, 3, 1, 2, 3], "points": [10, 6, 2, 12, 8, 1]})
laps = pd.DataFrame({"raceId": [10] * 6 + [11] * 6, "driverId": [1, 2, 3] * 4, "lap": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
                     "milliseconds": [90, 95, 99, 91, 94, 98, 92, 96, 97, 93, 93, 99]})
TABLES = {"results": results, "drivers": drivers, "races": races, "lap_times": laps}
LINKS = find_links(TABLES)
EVENTS = {"results", "lap_times"}
BAD = ('m = tables["results"].merge(tables["lap_times"], on=["raceId", "driverId"])\n'
       'result = m["points"].sum()')
GOOD = ('lap = tables["lap_times"].groupby(["raceId", "driverId"], as_index=False)["milliseconds"].mean()\n'
        'm = tables["results"].merge(lap, on=["raceId", "driverId"])\nresult = m["points"].sum()')


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


def test_the_test_data_has_the_links_and_event_tables_we_think():
    assert {l.text() for l in LINKS} >= {"results.driverId → drivers.driverId", "results.raceId → races.raceId",
                                         "lap_times.driverId → drivers.driverId"}
    from tools.relations import event_tables
    assert set(event_tables(TABLES, LINKS)) == EVENTS


# ------------------------------------------------------------------ the note
def test_the_note_names_tables_links_and_rules_but_no_values():
    note = multi_note(TABLES, LINKS, "results")
    for word in ("results", "drivers", "races", "lap_times", "results.driverId → drivers.driverId", "NEVER merge two event"):
        assert word in note
    assert "Aaa" not in note and "event table" in note and "lookup table" in note
    assert "this is `df`" in note
    assert "nunique()" in note and "races_year" in note           # Day 22b: count things, not rows; prefixed names


def test_the_note_is_short_even_with_many_columns_and_links():
    wide = {f"t{i}": pd.DataFrame({f"c{j}": [1, 2] for j in range(100)}) for i in range(3)}
    assert len(multi_note(wide, [], "t0")) < 3000


def test_without_a_note_the_prompt_is_exactly_what_it_was():
    a = ChatSession(PROFILE, ScriptedLlm(replies=[], seen=[], systems=[]), tables=results)
    b = ChatSession(PROFILE, ScriptedLlm(replies=[], seen=[], systems=[]), tables=results, extra_note="")
    assert a.profile_text == b.profile_text
    c = ChatSession(PROFILE, ScriptedLlm(replies=[], seen=[], systems=[]), tables=TABLES, extra_note="SEVERAL TABLES x")
    assert c.profile_text == a.profile_text + "\n\nSEVERAL TABLES x"
    assert c.instruction == a.instruction                       # the instructions themselves never change


# ------------------------------------------------------------------ the check
def test_merging_two_event_tables_without_summarising_is_blocked():
    reason = check_combination(BAD, "results", {}, EVENTS)
    assert reason and "lap_times" in reason and "results" in reason and "groupby" in reason


def test_summarising_first_is_allowed():
    assert check_combination(GOOD, "results", {}, EVENTS) is None


def test_merging_a_lookup_table_is_allowed():
    code = 'result = tables["results"].merge(tables["drivers"], on="driverId").groupby("surname")["points"].sum()'
    assert check_combination(code, "results", {}, EVENTS) is None
    assert check_combination('result = tables["results"].merge(tables["drivers"], on="driverId")', "results", {}, EVENTS) is None


def test_two_event_tables_without_any_merge_are_fine():
    code = 'result = (tables["results"]["points"].sum(), tables["lap_times"]["lap"].max())'
    assert check_combination(code, "results", {}, EVENTS) is None


def test_get_form_and_double_quotes_are_seen():
    code = "m = tables.get('results').merge(tables.get('lap_times'), on='raceId')\nresult = len(m)"
    assert check_combination(code, "results", {}, EVENTS)


def test_df_counts_as_its_source_event_table():
    code = 'm = df.merge(tables["lap_times"], on=["raceId", "driverId"])\nresult = m["points"].sum()'
    assert check_combination(code, "results_flat", {"results_flat": "results"}, EVENTS)
    assert check_combination(code, "results_flat", {}, EVENTS) is None        # unknown origin: nothing to compare


def test_the_flat_table_named_in_tables_counts_as_its_source_event_table():
    code = 'm = tables["results_flat"].merge(tables["lap_times"], on=["raceId", "driverId"])\nresult = m["points"].sum()'
    assert check_combination(code, "results_flat", {"results_flat": "results"}, EVENTS)
    assert check_combination(code, "results_flat", {}, EVENTS) is None


def test_the_flat_table_and_its_own_source_do_not_count_twice():
    code = 'm = df.merge(tables["results"][["resultId"]], on="resultId")\nresult = len(m)'
    assert check_combination(code, "results_flat", {"results_flat": "results"}, EVENTS) is None


def test_broken_python_is_left_to_the_sandbox():
    assert check_combination("result = (", "results", {}, EVENTS) is None


# ------------------------------------------------------------------ the runner
def _exec_inner(active):
    def run(code, tbls):
        ns = {"df": tbls[active].copy(), "tables": {k: v.copy() for k, v in tbls.items()}, "pd": pd, "np": np}
        try:
            exec(code, ns)
        except Exception as e:
            return FakeRun("error", error={"type": type(e).__name__, "message": str(e)})
        r = ns.get("result")
        return FakeRun(value=r.item() if hasattr(r, "item") else r)
    return run


def test_the_runner_blocks_before_running_and_passes_good_code_on():
    calls = []

    def inner(code, tbls):
        calls.append(code)
        return FakeRun(value=1)
    run = make_multi_runner(TABLES, LINKS, "results", inner=inner)
    blocked = run(BAD, TABLES)
    assert isinstance(blocked, BlockedRun) and not blocked.ok and blocked.status == "blocked"
    assert "blocked" in blocked.message() and calls == []
    assert run(GOOD, TABLES).ok and calls == [GOOD]


def test_the_blocked_message_names_the_fix():
    msg = make_multi_runner(TABLES, LINKS, "results", inner=lambda c, t: None)(BAD, TABLES).message()
    assert "groupby" in msg and "too big" in msg


def test_the_default_runner_uses_the_sandbox_with_the_active_table(monkeypatch):
    seen = {}
    import tools.sandbox as sb
    monkeypatch.setattr(sb, "run_code", lambda code, tables, **kw: seen.update(kw) or FakeRun(value=1))
    assert make_multi_runner(TABLES, LINKS, "drivers")("result = 1", TABLES).ok
    assert seen["active"] == "drivers" and seen["timeout"] == chat_agent.CALC_TIMEOUT


# ------------------------------------------------------------------ a whole conversation
def _session(replies):
    seen, systems = [], []
    llm = ScriptedLlm(replies=list(replies), seen=seen, systems=systems)
    runner = make_multi_runner(TABLES, LINKS, "results", inner=_exec_inner("results"))
    s = ChatSession(PROFILE, llm, tables=TABLES, runner=runner, sleep=_no_sleep, max_code_runs=3,
                    extra_note=multi_note(TABLES, LINKS, "results"))
    return s, llm


def test_a_bad_merge_is_blocked_then_fixed_by_the_model_and_the_answer_comes_from_the_good_result():
    s, llm = _session([f"```python\n{BAD}\n```", f"```python\n{GOOD}\n```", "The points total is 39."])
    reply = s.ask("What is the total of points?")
    assert reply.ok and "39" in reply.text
    assert [c.ok for c in reply.calculations] == [False, True]
    assert "blocked" in reply.calculations[0].result_text.lower()
    assert "groupby" in llm.seen[1][-1][1] and "blocked" in llm.seen[1][-1][1]   # the model was told why, to fix it
    assert "SEVERAL TABLES" in llm.systems[0] and "NEVER merge two event" in llm.systems[0]


def test_the_good_code_really_gives_the_right_total_and_the_bad_code_the_wrong_one():
    inner = _exec_inner("results")
    assert inner(GOOD, TABLES).value == 39
    assert inner(BAD, TABLES).value != 39                              # why the guard exists: 2 laps per driver doubles it


# ------------------------------------------------------------------ the tab and the app
def test_the_tab_hands_the_tables_note_and_runner_to_the_session():
    from ui import chat_tab
    profile = {"dataset": {"rows": 2, "columns": 1, "duplicate_rows": 0, "missing_pct": 0},
               "columns": [{"name": "a", "role": "numeric"}]}
    runner = make_multi_runner(TABLES, LINKS, "results", inner=_exec_inner("results"))
    multi = {"tables": TABLES, "note": "SEVERAL TABLES here", "runner": runner}
    s = chat_tab.default_session_factory(profile, {"cache_mode": "replay", "df": results, "multi": multi})
    assert s.tables is TABLES and s._runner is runner and s.profile_text.endswith("SEVERAL TABLES here")
    one = chat_tab.default_session_factory(profile, {"cache_mode": "replay", "df": results})
    assert one.tables is results and "SEVERAL TABLES" not in one.profile_text


def test_the_app_builds_the_multi_chat_only_for_several_files(monkeypatch):
    import glob
    import os
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parent.parent

    class U:
        def __init__(self, n, d):
            self.name, self._d, self.size = n, d, len(d)

        def getvalue(self):
            return self._d
    sets = {"several": [("races.csv", races.to_csv(index=False).encode()), ("results.csv", results.to_csv(index=False).encode()),
                        ("drivers.csv", drivers.to_csv(index=False).encode())],
            "one": [("results.csv", results.to_csv(index=False).encode())]}
    seen = {}
    for name, files in sets.items():
        monkeypatch.setattr(st, "file_uploader", lambda *a, _f=files, **k: [U(n, d) for n, d in _f])
        at = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=120).run()
        assert not at.exception
        seen[name] = [c.value for c in at.caption if "other uploaded tables" in c.value]
    assert len(seen["several"]) == 1 and "'results_flat'" in seen["several"][0]
    assert seen["one"] == []


def test_the_real_chat_session_in_the_app_knows_the_links_and_blocks_bad_merges(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import ui.chat_tab as chat_tab
    root = Path(__file__).resolve().parent.parent

    class U:
        def __init__(self, n, d):
            self.name, self._d, self.size = n, d, len(d)

        def getvalue(self):
            return self._d
    files = [(f"{n}.csv", t.to_csv(index=False).encode()) for n, t in TABLES.items()]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U(n, d) for n, d in files])
    monkeypatch.setattr(chat_tab, "api_key_available", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    at = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    session = at.session_state["chat_session"]
    assert "results.driverId → drivers.driverId" in session.profile_text and "SEVERAL TABLES" in session.profile_text
    first = next(iter(session.tables))
    assert first.endswith("_flat") and "lap_times" in session.tables
    blocked = session._runner(BAD, session.tables)
    assert blocked.status == "blocked"
    flat_bad = f'm = df.merge(tables["lap_times"], on=["raceId", "driverId"])\nresult = len(m)'
    if first == "lap_times_flat":
        flat_bad = f'm = df.merge(tables["results"], on=["raceId", "driverId"])\nresult = len(m)'
    assert session._runner(flat_bad, session.tables).status == "blocked"      # `df` is known to come from an event table


# ------------------------------------------------------------------ Day 22c: a number column compared with quoted text
from agents.multi_chat import check_number_as_text  # noqa: E402


def test_a_number_column_compared_with_quoted_text_is_blocked():
    reason = check_number_as_text("x = df[df['year'] == '2020']\nresult = len(x)", {"races": races})
    assert reason and "'year'" in reason and "2020" in reason and "no quotes" in reason
    assert check_number_as_text('result = races["year"] != "2019"', {"races": races})
    assert check_number_as_text("result = '2020' == races['year']", {"races": races})
    assert check_number_as_text("m = (races['raceId'] > 1) & (races['year'] == ' 2020 ')", {"races": races})


def test_the_check_leaves_good_code_alone():
    t = {"races": races, "drivers": drivers}
    assert check_number_as_text("result = races[races['year'] == 2020]", t) is None             # a number
    assert check_number_as_text("result = drivers[drivers['surname'] == '2020']", t) is None    # a text column
    assert check_number_as_text("result = drivers[drivers['surname'] == 'Aaa']", t) is None     # text that is not a number
    assert check_number_as_text("result = races['nope'] == '5'", t) is None                     # unknown column
    assert check_number_as_text("result = (", t) is None
    assert check_number_as_text("result = races['year'] > '2000'", t) is None                   # only equality is checked
    mixed = {"a": pd.DataFrame({"k": [1, 2]}), "b": pd.DataFrame({"k": ["1", "2"]})}
    assert check_number_as_text("result = df['k'] == '1'", mixed) is None                       # one table has text: not sure
    flags = {"a": pd.DataFrame({"f": [True, False]})}
    assert check_number_as_text("result = df['f'] == '1'", flags) is None                       # booleans are not numbers here


def test_the_runner_blocks_it_and_lets_the_fixed_code_through():
    code = "x = df[df['year'] == '2019']\nresult = x['points'].sum()"
    tables = {"results_flat": results.merge(races, on="raceId"), **TABLES}
    run = make_multi_runner(tables, LINKS, "results_flat", inner=lambda c, t: FakeRun(value=1), origin={"results_flat": "results"})
    assert run(code, tables).status == "blocked" and run(code.replace("'2019'", "2019"), tables).ok


def test_chained_comparisons_and_odd_subscripts_are_handled():
    t = {"races": races}
    assert check_number_as_text("result = 1 < races['year'] == '2020'", t)                     # the second pair of a chain
    assert check_number_as_text("result = races[races['year'] > 1] == '5'", t) is None          # a subscript that is not a column name
    assert check_number_as_text("result = races[0] == '5'", t) is None
