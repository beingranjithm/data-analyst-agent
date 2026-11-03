"""Day 25: the Olist e-commerce demo set in the app -- picking it, its starter questions, and recorded answers
that replay in the app (fake models only). The demo set's own builder/shipped-file/true-answer tests live in
tests/test_olist_eval.py; this file covers what's unique to the live app: AppTest-level selection, starters,
and the record/replay plumbing around deploy/demo_cache.json."""
import json

import numpy as np
import pandas as pd

import tools.hosting as hosting  # noqa: E402
from agents.llm_cache import CachingLlm  # noqa: E402
from agents.llm_factory import DEMO_CACHE_PATH, ROLE_MODELS, default_cache_path  # noqa: E402
from agents.multi_chat import chat_session_for, prepare_chat  # noqa: E402
from calc_fakes import FakeRun  # noqa: E402
from evals.olist_eval import DEMO_QUESTIONS  # noqa: E402
from test_chat_agent import ScriptedLlm, _no_sleep  # noqa: E402
from tools.relations import find_links  # noqa: E402

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "sample_data" / "olist_demo"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("DAA_HOSTED", raising=False)
    monkeypatch.delenv("DAA_CACHE_PATH", raising=False)
    monkeypatch.setattr(hosting, "_secret_flag", lambda: False)
    hosting._PROBE.clear()


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


def test_locally_the_demo_is_the_other_choice_besides_uploading(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-key-so-the-chat-opens")          # locally the chat needs a key unless 'Replay only'
    at = app(monkeypatch).run()
    radio = at.sidebar.radio[0]
    assert radio.value == "Upload my own CSV" and radio.options == ["Upload my own CSV", "Olist e-commerce demo set (Nov 2017)"]
    radio.set_value("Olist e-commerce demo set (Nov 2017)").run()
    assert not at.exception
    assert starter_buttons(at) == DEMO_QUESTIONS
    assert any("saved answers" in c.value for c in at.caption)
    assert at.session_state["active_table"] == "order_items_flat" and len(at.session_state["tables"]) == 9


def test_the_hosted_app_opens_on_the_demo_with_only_the_two_sources(monkeypatch):
    at = app(monkeypatch, hosted=True).run()
    assert not at.exception
    radio = at.sidebar.radio[0]
    assert radio.value == "Olist e-commerce demo set (Nov 2017)" and radio.options == ["Upload my own CSV", "Olist e-commerce demo set (Nov 2017)"]
    assert starter_buttons(at) == DEMO_QUESTIONS
    chat_modes = next(s for s in at.selectbox if s.key == "chat_cache_mode")
    assert chat_modes.value.startswith("Replay only")                        # no key needed to start


def test_without_the_sandbox_lock_the_demo_questions_still_work_in_replay_mode(monkeypatch):
    at = app(monkeypatch, probe=(False, "no lock"), hosted=True).run()           # Day 23c: recorded code may run
    assert not at.exception and starter_buttons(at) == DEMO_QUESTIONS
    assert any("Live calculations are switched off" in w.value for w in at.warning)


def test_without_the_sandbox_lock_a_live_ai_gets_no_demo_questions_and_no_data(monkeypatch):
    at = app(monkeypatch, probe=(False, "no lock"), hosted=True)
    at.session_state["user_api_key"] = "a-visitor-key-123"
    at.run()
    next(s for s in at.selectbox if s.key == "chat_cache_mode").set_value("Ask the AI again: ignore saved replies").run()
    assert not at.exception
    assert not set(starter_buttons(at)) & set(DEMO_QUESTIONS) and starter_buttons(at)
    assert any("I answer from a summary of the data" in c.value for c in at.caption)


def test_other_data_gets_the_usual_starters_not_the_demo_ones(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "a-key-so-the-chat-opens")
    import streamlit as st
    at = app(monkeypatch)
    sample = (ROOT / "sample_data" / "sales_sample.csv").read_bytes()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U("sales_sample.csv", sample)])
    at.run()
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


CODE = '```python\nresult = (df["order_status"] == "canceled").sum()\n```'
ANSWER = "There were 37 canceled orders."


def demo_files():
    return [(p.name, p.read_bytes()) for p in sorted(DEMO_DIR.glob("*.csv"))]


def test_answers_recorded_by_the_script_path_replay_in_the_app(monkeypatch, tmp_path):
    import tools.sandbox as sb
    monkeypatch.setattr(sb, "run_code", run_in_test_sandbox)
    cache = tmp_path / "demo_cache.json"
    # 1) RECORD, the way scripts/eval_olist.py --demo does: its set-up code, a (fake) model behind the saved-replies cache
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


def test_the_chat_set_up_uses_only_joinable_links():
    twin_a = pd.DataFrame({"code": ["a", "b", "c", "d"], "x": [1, 2, 3, 4]})
    twin_b = pd.DataFrame({"code": ["a", "b", "c", "d"], "y": [5, 6, 7, 8]})
    assert any(l.kind == "one-to-one" for l in find_links({"twin_a": twin_a, "twin_b": twin_b}))
    files = [("twin_a.csv", twin_a.to_csv(index=False).encode()), ("twin_b.csv", twin_b.to_csv(index=False).encode())]
    prep = prepare_chat(files)
    assert prep["links"] == [] and set(prep["frames"]) == {"twin_a", "twin_b"}
