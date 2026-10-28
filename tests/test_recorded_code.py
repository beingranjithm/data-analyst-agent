"""Day 23c: code that is word for word in the shipped demo replies may run on a host without the file-system lock;
everything else still needs it. Fake models and fake sandbox; nothing real is called."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools.hosting as hosting  # noqa: E402
from tools.hosting import HOSTED_SANDBOX_MB, is_recorded, recorded_code, sandbox_options  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FENCE = "```"
RECORDED = "result = df['points'].sum()"


def reply(code):
    return f"Here is the code.\n{FENCE}python\n{code}\n{FENCE}\n"


@pytest.fixture
def demo_file(tmp_path, monkeypatch):
    p = tmp_path / "demo_cache.json"
    p.write_text(json.dumps({
        "k1": {"text": reply(RECORDED), "model": "m"},
        "k2": {"text": '{"title": "A dashboard", "charts": []}', "model": "m"},          # planner JSON: no code
        "k3": {"text": "Plain words, no code at all.", "model": "m"},
        "k4": {"text": reply("   "), "model": "m"},                                       # an empty block counts as nothing
        "k5": "not a dict",
    }))
    monkeypatch.setattr(hosting, "DEMO_CACHE_FILE", p)
    hosting._RECORDED.clear()
    monkeypatch.delenv("DAA_HOSTED", raising=False)
    monkeypatch.setattr(hosting, "_secret_flag", lambda: False)
    yield p
    hosting._RECORDED.clear()


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("DAA_HOSTED", "1")


def test_only_real_code_blocks_are_collected(demo_file):
    assert recorded_code() == frozenset({RECORDED})


def test_a_missing_or_broken_file_records_nothing(tmp_path, monkeypatch):
    hosting._RECORDED.clear()
    assert recorded_code(tmp_path / "nope.json") == frozenset()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert recorded_code(bad) == frozenset()
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]")
    assert recorded_code(lst) == frozenset()
    hosting._RECORDED.clear()


def test_the_file_is_read_once(demo_file):
    assert recorded_code() == frozenset({RECORDED})
    demo_file.write_text("{}")
    assert recorded_code() == frozenset({RECORDED})                     # cached for the process


def test_the_shipped_path_is_the_demo_file_in_the_deploy_folder():
    assert hosting.DEMO_CACHE_FILE == ROOT / "deploy" / "demo_cache.json"


def test_recorded_means_the_same_program_ignoring_blank_edges(demo_file):
    assert is_recorded(RECORDED) and is_recorded("\n  " + RECORDED + "  \n")
    assert not is_recorded(RECORDED + " + 1") and not is_recorded("result = 2")
    assert not is_recorded(None) and not is_recorded(5) and not is_recorded("")


def test_locally_nothing_changes(demo_file):
    assert sandbox_options(RECORDED) == {} and sandbox_options("anything") == {} and sandbox_options() == {}


def test_hosted_recorded_code_does_not_demand_the_lock_but_keeps_the_memory_cap(demo_file, hosted):
    assert sandbox_options(RECORDED) == {"memory_mb": HOSTED_SANDBOX_MB}


@pytest.mark.parametrize("code", ["result = 2", RECORDED + "\nimport os", None, ""])
def test_hosted_other_code_still_demands_the_lock(demo_file, hosted, code):
    assert sandbox_options(code) == {"require_isolation": True, "memory_mb": HOSTED_SANDBOX_MB}


def test_hosted_without_a_demo_file_everything_demands_the_lock(tmp_path, monkeypatch, hosted):
    monkeypatch.setattr(hosting, "DEMO_CACHE_FILE", tmp_path / "missing.json")
    hosting._RECORDED.clear()
    assert sandbox_options(RECORDED)["require_isolation"] is True
    hosting._RECORDED.clear()


class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, code, tables, **kw):
        self.calls.append(kw)
        return type("R", (), {"ok": False, "status": "error", "value": None, "table": None, "figure": None,
                              "figure_json": None, "problems": [], "error": {"type": "X", "message": "m"}, "stdout": "",
                              "message": lambda s: "spy"})()


@pytest.mark.parametrize("code,locked", [(RECORDED, False), ("result = 2", True)])
def test_the_chat_runner_passes_its_code(demo_file, hosted, monkeypatch, code, locked):
    import tools.sandbox as sb
    from agents.chat_agent import default_runner
    spy = Spy()
    monkeypatch.setattr(sb, "run_code", spy)
    default_runner(code, pd.DataFrame({"a": [1]}))
    assert ("require_isolation" in spy.calls[0]) is locked


@pytest.mark.parametrize("code,locked", [(RECORDED, False), ("result = 2", True)])
def test_the_multi_table_runner_passes_its_code(demo_file, hosted, monkeypatch, code, locked):
    import tools.sandbox as sb
    from agents.multi_chat import make_multi_runner
    spy = Spy()
    monkeypatch.setattr(sb, "run_code", spy)
    t = {"a": pd.DataFrame({"x": [1]})}
    make_multi_runner(t, [], "a")(code, t)
    assert ("require_isolation" in spy.calls[0]) is locked


@pytest.mark.parametrize("code,locked", [(RECORDED, False), ("result = 2", True)])
def test_the_chart_agent_passes_its_code(demo_file, hosted, monkeypatch, code, locked):
    import asyncio

    import agents.chart_agent as ca
    from test_chat_agent import ScriptedLlm
    spy = Spy()
    monkeypatch.setattr(ca, "run_code", spy)
    llm = ScriptedLlm(replies=[reply(code)], seen=[], systems=[])
    chart = type("C", (), {"title": "t", "why": "w", "to_spec": lambda self: {"kind": "bar"}})()
    from tools.profiler import profile_dataframe
    df = pd.DataFrame({"a": [1, 2]})
    res = asyncio.run(ca.make_chart_async(df, profile_dataframe(df), chart, llm))
    assert not res.ok and spy.calls and ("require_isolation" in spy.calls[0]) is locked


# ------------------------------------------------------------------ the dashboard tab on a host without the lock
from test_hosting import SMALL, open_app, texts  # noqa: E402

LIVE = "Ask the AI again: ignore saved replies"


def _mode(at, label):
    next(s for s in at.selectbox if s.key == "ai_cache_mode").set_value(label).run()


def test_dashboard_live_ai_is_switched_off_without_the_lock(hosted, monkeypatch):
    at = open_app(monkeypatch, SMALL, probe=(False, "x")).run()
    assert not any("chart code cannot run on this host" in t for t in texts(at))       # Replay only is the default
    _mode(at, LIVE)
    assert not at.exception and any("chart code cannot run on this host" in t for t in texts(at))
    assert not any("Paste your own Gemini key" in e.value for e in at.error)            # no key is needed for fixed rules


def test_dashboard_live_ai_works_when_the_lock_works(hosted, monkeypatch):
    at = open_app(monkeypatch, SMALL, probe=(True, "")).run()
    _mode(at, LIVE)
    assert not at.exception and not any("chart code cannot run on this host" in t for t in texts(at))
    assert any("Paste your own Gemini key" in e.value for e in at.error)


def test_dashboard_without_ai_says_nothing_about_the_lock(hosted, monkeypatch):
    at = open_app(monkeypatch, SMALL, probe=(False, "x")).run()
    _mode(at, LIVE)
    next(c for c in at.checkbox if c.key == "ai_use_ai").set_value(False).run()
    assert not at.exception and not any("chart code cannot run on this host" in t for t in texts(at))


def test_dashboard_locally_never_asks_about_the_lock(monkeypatch):
    called = []
    at = open_app(monkeypatch, SMALL, probe=(False, "x"))
    monkeypatch.setattr(hosting, "isolation_ok", lambda *a, **k: called.append(1) or (False, "x"))
    at.run()
    _mode(at, LIVE)
    assert not at.exception and not called


# ------------------------------------------------------------------ small gaps found by mutation testing
def test_spaces_around_a_recorded_program_do_not_matter(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"k": {"text": f"{FENCE}python\n  result = 3  \n{FENCE}"}}))
    hosting._RECORDED.clear()
    assert recorded_code(p) == frozenset({"result = 3"})
    hosting._RECORDED.clear()


def test_an_error_half_way_through_the_file_records_nothing(tmp_path, monkeypatch):
    import agents.chart_agent as ca
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": {"text": reply("result = 1")}, "b": {"text": reply("result = 2")}}))
    real, n = ca.extract_code, []

    def flaky(text):
        n.append(1)
        if len(n) == 2:
            raise RuntimeError("boom")
        return real(text)
    monkeypatch.setattr(ca, "extract_code", flaky)
    hosting._RECORDED.clear()
    assert recorded_code(p) == frozenset()              # never a half-read list
    hosting._RECORDED.clear()


def test_render_chat_by_default_hands_the_data_to_the_session():
    from streamlit.testing.v1 import AppTest

    def script():
        import pandas as pd
        import streamlit as st
        from ui.chat_tab import render_chat
        seen = st.session_state.setdefault("seen", [])

        def factory(profile, settings):
            seen.append(settings["df"] is not None)
            raise RuntimeError("stop here")
        render_chat({"dataset": {}}, "k", df=pd.DataFrame({"a": [1]}), session_factory=factory)
    at = AppTest.from_function(script, default_timeout=60)
    at.session_state["chat_saved_mode"] = LIVE
    at.run()
    assert at.session_state["seen"] == [True]
