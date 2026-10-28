"""Day 23b: the small F1 demo set, its questions, and recorded answers that replay in the app (fake models only)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_demo_f1 as bd  # noqa: E402
import tools.hosting as hosting  # noqa: E402
from agents.llm_cache import CachingLlm  # noqa: E402
from agents.llm_factory import DEMO_CACHE_PATH, ROLE_MODELS, default_cache_path  # noqa: E402
from agents.multi_chat import chat_session_for, prepare_chat  # noqa: E402
from calc_fakes import FakeRun  # noqa: E402
from evals.f1_eval import DEMO_DEFINITIONS, DEMO_QUESTIONS, build_demo_questions, read_raw  # noqa: E402
from test_chat_agent import ScriptedLlm, _no_sleep  # noqa: E402
from tools.relations import find_links  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "sample_data" / "f1_demo"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("DAA_HOSTED", raising=False)
    monkeypatch.delenv("DAA_CACHE_PATH", raising=False)
    monkeypatch.setattr(hosting, "_secret_flag", lambda: False)
    hosting._PROBE.clear()


# ------------------------------------------------------------------ the builder (tiny made-up files)
def write_f1(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "seasons": "year,url\n2019,u\n2021,u\n2022,u\n",
        "circuits": "circuitId,name,country\n1,Monaco,Monaco\n2,Monza,Italy\n3,Old,France\n",
        "races": "raceId,year,round,circuitId,name\n10,2019,1,3,Old GP\n11,2021,1,1,Monaco Grand Prix\n12,2022,1,2,Italian Grand Prix\n13,2022,2,1,Monaco Grand Prix\n",
        "drivers": "driverId,forename,surname\n1,Lewis,Hamilton\n2,Max,Verstappen\n3,Old,Timer\n",
        "constructors": "constructorId,name\n1,Mercedes\n2,Red Bull\n3,Gone\n",
        "status": "statusId,status\n1,Finished\n2,Retired\n3,Unused\n",
        "results": "resultId,raceId,driverId,constructorId,position,points,statusId\n1,10,3,3,1,25,1\n2,11,1,1,1,25,1\n3,11,2,2,\\N,0,2\n4,12,2,2,1,25,1\n5,13,2,2,1,25,1\n",
        "qualifying": "qualifyId,raceId,driverId,constructorId,position\n1,10,3,3,1\n2,12,1,1,1\n",
        "pit_stops": "raceId,driverId,stop,milliseconds\n10,3,1,20000\n13,2,1,21000\n13,2,2,22000\n",
        "lap_times": "raceId,driverId,lap,milliseconds\n10,3,1,90000\n11,1,1,91000\n12,2,1,92000\n13,2,1,93000\n",
        "driver_standings": "driverStandingsId,raceId,driverId,points\n1,10,3,25\n2,12,2,25\n",
        "constructor_standings": "constructorStandingsId,raceId,constructorId,points\n1,10,3,25\n2,12,2,25\n",
        "constructor_results": "constructorResultsId,raceId,constructorId,points\n1,10,3,25\n2,12,2,25\n",
        "sprint_results": "resultId,raceId,driverId,constructorId,position,points,statusId\n1,10,3,3,1,3,1\n2,12,2,2,1,3,1\n",
    }
    for n, text in files.items():
        (folder / f"{n}.csv").write_text(text)


def test_the_builder_keeps_only_the_chosen_seasons_and_what_they_point_to(tmp_path):
    write_f1(tmp_path / "src")
    counts = bd.build_demo(tmp_path / "src", tmp_path / "out", [2021, 2022])
    assert counts["races"] == 3 and counts["seasons"] == 2 and counts["results"] == 4
    assert counts["drivers"] == 2 and counts["constructors"] == 2 and counts["circuits"] == 2 and counts["status"] == 2
    assert counts["qualifying"] == 1 and counts["pit_stops"] == 2 and counts["sprint_results"] == 1
    out = {p.stem: pd.read_csv(p, dtype=str, keep_default_na=False) for p in (tmp_path / "out").glob("*.csv")}
    assert set(out) == set(bd.ALL) and set(out["races"].year) == {"2021", "2022"}
    assert set(out["drivers"].surname) == {"Hamilton", "Verstappen"}                  # the 2019-only driver is gone
    assert set(out["circuits"].name) == {"Monaco", "Monza"}


def test_the_builder_keeps_values_exactly_including_backslash_n(tmp_path):
    write_f1(tmp_path / "src")
    bd.build_demo(tmp_path / "src", tmp_path / "out", [2021])
    assert "\\N" in (tmp_path / "out" / "results.csv").read_text()


def test_every_link_between_the_tables_survives_the_cut(tmp_path):
    write_f1(tmp_path / "src")
    bd.build_demo(tmp_path / "src", tmp_path / "out", [2021, 2022])
    frames = {p.stem: pd.read_csv(p, na_values=["\\N"]) for p in (tmp_path / "out").glob("*.csv")}
    texts = {l.text() for l in find_links(frames)}
    assert {"results.driverId → drivers.driverId", "results.raceId → races.raceId",
            "races.circuitId → circuits.circuitId"} <= texts


def test_the_builder_says_what_went_wrong(tmp_path):
    write_f1(tmp_path / "src")
    with pytest.raises(ValueError, match="No races found"):
        bd.build_demo(tmp_path / "src", tmp_path / "out", [1950])
    (tmp_path / "src" / "pit_stops.csv").unlink()
    with pytest.raises(FileNotFoundError, match="pit_stops.csv"):
        bd.build_demo(tmp_path / "src", tmp_path / "out", [2021])


def test_a_rebuild_removes_files_from_an_earlier_build(tmp_path):
    write_f1(tmp_path / "src")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.csv").write_text("a\n1\n")
    bd.build_demo(tmp_path / "src", tmp_path / "out", [2021])
    assert not (tmp_path / "out" / "old.csv").exists()


# ------------------------------------------------------------------ the shipped demo set and its questions
def test_the_shipped_demo_set_is_small_complete_and_linked():
    files = sorted(DEMO_DIR.glob("*.csv"))
    assert {p.stem for p in files} == set(bd.ALL)
    assert sum(p.stat().st_size for p in files) < 3_000_000
    frames = {p.stem: pd.read_csv(p, na_values=["\\N"]) for p in files}
    assert set(frames["races"].year) == {2021, 2022}
    assert len(find_links(frames)) >= 20


def test_the_demo_questions_all_have_true_answers_on_the_shipped_set():
    qs, skipped = build_demo_questions(read_raw(DEMO_DIR))
    assert skipped == [] and len(qs) == len(DEMO_DEFINITIONS) == len(DEMO_QUESTIONS) == len(set(DEMO_QUESTIONS))
    by = {q.text: q.expect for q in qs}
    assert by["How many races were held in 2022?"].numbers == [22]
    assert by["Which driver won the most races in 2021, and how many?"].names == ["Verstappen"]
    assert by["Which constructor won the most races in 2022, and how many?"].names == ["Red Bull"]
    monaco = by["How many pit stops did the winner of the 2022 Monaco Grand Prix make, and who was it?"]
    assert monaco.names == [("Pérez", "Perez")]                        # the accent is an allowed spelling difference
    assert sum(q.expect.trap for q in qs) == 2


# ------------------------------------------------------------------ where saved replies live
def test_saved_reply_locations(monkeypatch):
    from agents.llm_cache import DEFAULT_CACHE_PATH
    assert default_cache_path("use") == Path(DEFAULT_CACHE_PATH) == default_cache_path("replay")
    monkeypatch.setenv("DAA_CACHE_PATH", "~/demo.json")
    assert default_cache_path("use") == Path("~/demo.json").expanduser()
    monkeypatch.delenv("DAA_CACHE_PATH")
    monkeypatch.setenv("DAA_HOSTED", "1")
    assert default_cache_path("replay") == DEMO_CACHE_PATH and DEMO_CACHE_PATH.name == "demo_cache.json"
    assert default_cache_path("off") == Path(DEFAULT_CACHE_PATH)         # never the demo file, never written


# ------------------------------------------------------------------ the app
class U:
    def __init__(self, n, d):
        self.name, self._d, self.size = n, d, len(d)

    def getvalue(self):
        return self._d


def app(monkeypatch, probe=(True, ""), hosted=False):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    if hosted:
        monkeypatch.setenv("DAA_HOSTED", "1")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    monkeypatch.setattr(hosting, "isolation_ok", lambda *a, **k: probe)
    return AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)


def starter_buttons(at):
    return [b.label for b in at.button if b.key and b.key.startswith("chat_starter_")]


def test_locally_the_demo_is_one_choice_among_several(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-key-so-the-chat-opens")          # locally the chat needs a key unless 'Replay only'
    at = app(monkeypatch).run()
    radio = at.sidebar.radio[0]
    assert radio.value == "Upload my own CSV" and "Formula 1 demo set (2021-2022)" in radio.options and "Messy test file" in radio.options
    radio.set_value("Formula 1 demo set (2021-2022)").run()
    assert not at.exception
    assert starter_buttons(at) == DEMO_QUESTIONS
    assert any("saved answers" in c.value for c in at.caption)
    assert at.session_state["active_table"] == "results_flat" and len(at.session_state["tables"]) == 15


def test_the_hosted_app_opens_on_the_demo_without_the_messy_file(monkeypatch):
    at = app(monkeypatch, hosted=True).run()
    assert not at.exception
    radio = at.sidebar.radio[0]
    assert radio.value == "Formula 1 demo set (2021-2022)" and "Messy test file" not in radio.options
    assert starter_buttons(at) == DEMO_QUESTIONS
    chat_modes = next(s for s in at.selectbox if s.key == "chat_cache_mode")
    assert chat_modes.value.startswith("Replay only")                        # no key needed to start


def test_without_the_sandbox_lock_the_demo_questions_are_not_offered(monkeypatch):
    at = app(monkeypatch, probe=(False, "no lock"), hosted=True).run()
    assert not at.exception and not set(starter_buttons(at)) & set(DEMO_QUESTIONS)      # replayed code would be shown as text
    assert starter_buttons(at)                                                          # the ordinary starters remain


def test_other_data_gets_the_usual_starters_not_the_demo_ones(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-key-so-the-chat-opens")
    at = app(monkeypatch).run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert not at.exception and starter_buttons(at) and not set(starter_buttons(at)) & set(DEMO_QUESTIONS)


# ------------------------------------------------------------------ recorded by a script, replayed in the app
def run_in_test_sandbox(code, tables, *, active=None, **kw):
    ns = {"df": tables[active or next(iter(tables))].copy(), "tables": {k: v.copy() for k, v in tables.items()}, "pd": pd, "np": np}
    try:
        exec(code, ns)
    except Exception as e:
        return FakeRun("error", error={"type": type(e).__name__, "message": str(e)})
    r = ns.get("result")
    if isinstance(r, pd.Series):
        r = r.reset_index()
    if isinstance(r, pd.DataFrame):
        return FakeRun(table=r.head(500), total=len(r))
    return FakeRun(value=r.item() if hasattr(r, "item") else r)


CODE = '```python\nresult = df["raceId"].nunique()\n```'
ANSWER = "There were 22 races in 2022."


def demo_files():
    return [(p.name, p.read_bytes()) for p in sorted(DEMO_DIR.glob("*.csv"))]


def test_answers_recorded_by_the_script_path_replay_in_the_app(monkeypatch, tmp_path):
    import tools.sandbox as sb
    monkeypatch.setattr(sb, "run_code", run_in_test_sandbox)
    cache = tmp_path / "demo_cache.json"
    # 1) RECORD, the way scripts/eval_f1.py --demo does: its set-up code, a (fake) model behind the saved-replies cache
    prep = prepare_chat(demo_files())
    inner = ScriptedLlm(replies=[CODE, ANSWER], seen=[], systems=[])
    recorder = CachingLlm(model=ROLE_MODELS["chat"][0], inner=inner, mode="use", path=str(cache), min_interval=0)
    session = chat_session_for(prep, recorder, sleep=_no_sleep)
    reply = session.ask(DEMO_QUESTIONS[0])
    assert reply.ok and ANSWER in reply.text and len(json.loads(cache.read_text())) == 2
    # 2) REPLAY in the real app (public host, 'Replay only'): the saved file is the only source of answers
    monkeypatch.setenv("DAA_CACHE_PATH", str(cache))
    at = app(monkeypatch, hosted=True).run()
    assert not at.exception
    next(b for b in at.button if b.label == DEMO_QUESTIONS[0]).click().run()
    assert not at.exception
    assert not [e.value for e in at.error], [e.value for e in at.error]
    assert any(ANSWER in m.value for m in at.markdown)
    assert any("0 real, 2 replayed" in c.value for c in at.caption)           # both requests were found in the saved file


def test_a_question_nobody_recorded_gets_a_friendly_message_not_a_crash(monkeypatch, tmp_path):
    import tools.sandbox as sb
    monkeypatch.setattr(sb, "run_code", run_in_test_sandbox)
    monkeypatch.setenv("DAA_CACHE_PATH", str(tmp_path / "empty.json"))
    at = app(monkeypatch, hosted=True).run()
    next(b for b in at.button if b.label == DEMO_QUESTIONS[0]).click().run()
    assert not at.exception
    assert any("saved" in e.value.lower() or "replay" in e.value.lower() for e in at.error)


def test_text_that_looks_like_missing_is_copied_untouched(tmp_path):
    write_f1(tmp_path / "src")
    d = tmp_path / "src" / "drivers.csv"
    d.write_text(d.read_text().replace("Hamilton", "NA").replace("Verstappen", "null"))
    bd.build_demo(tmp_path / "src", tmp_path / "out", [2021, 2022])
    out = pd.read_csv(tmp_path / "out" / "drivers.csv", dtype=str, keep_default_na=False)
    assert set(out.surname) == {"NA", "null"}


def test_the_chat_set_up_uses_only_joinable_links():
    twin_a = pd.DataFrame({"code": ["a", "b", "c", "d"], "x": [1, 2, 3, 4]})
    twin_b = pd.DataFrame({"code": ["a", "b", "c", "d"], "y": [5, 6, 7, 8]})
    assert any(l.kind == "one-to-one" for l in find_links({"twin_a": twin_a, "twin_b": twin_b}))
    files = [("twin_a.csv", twin_a.to_csv(index=False).encode()), ("twin_b.csv", twin_b.to_csv(index=False).encode())]
    prep = prepare_chat(files)
    assert prep["links"] == [] and set(prep["frames"]) == {"twin_a", "twin_b"}
