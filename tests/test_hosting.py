"""Day 23a: the public hosted version. Fake models and fake sandbox; nothing real is called."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools.hosting as hosting  # noqa: E402
from tools.hosting import (HOSTED_FILE_MB, HOSTED_FILES, HOSTED_SANDBOX_MB, HOSTED_TOTAL_MB, file_limit_mb,  # noqa: E402
                           is_hosted, isolation_ok, redact, sandbox_options, upload_limits)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("DAA_HOSTED", raising=False)
    monkeypatch.setattr(hosting, "_secret_flag", lambda: False)
    hosting._PROBE.clear()
    yield
    hosting._PROBE.clear()


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("DAA_HOSTED", "1")


# ------------------------------------------------------------------ the switch and the limits
def test_hosted_mode_is_off_by_default_and_on_with_the_setting(monkeypatch):
    assert not is_hosted()
    for v in ("1", "true", "YES", " on "):
        monkeypatch.setenv("DAA_HOSTED", v)
        assert is_hosted()
    for v in ("", "0", "false", "no"):
        monkeypatch.setenv("DAA_HOSTED", v)
        assert not is_hosted()


def test_the_secret_flag_also_switches_it_on(monkeypatch):
    monkeypatch.setattr(hosting, "_secret_flag", lambda: True)
    assert is_hosted()


def test_a_missing_secrets_file_does_not_break_the_check(monkeypatch):
    monkeypatch.undo()
    monkeypatch.delenv("DAA_HOSTED", raising=False)
    assert is_hosted() in (True, False)                     # no exception, whatever the machine has


def test_local_runs_keep_the_normal_limits_and_sandbox_settings():
    assert sandbox_options() == {} and file_limit_mb(50) == 50 and upload_limits(30, 300) == (30, 300)


def test_hosted_runs_use_smaller_limits_and_demand_the_file_lock(hosted):
    assert sandbox_options() == {"require_isolation": True, "memory_mb": HOSTED_SANDBOX_MB}
    assert file_limit_mb(50) == HOSTED_FILE_MB and upload_limits(30, 300) == (HOSTED_FILES, HOSTED_TOTAL_MB)
    assert file_limit_mb(5) == 5 and upload_limits(3, 10) == (3, 10)           # never raises a smaller limit


def test_the_upload_check_uses_the_hosted_limits(hosted):
    from tools.multi import check_limits
    assert check_limits([("a.csv", b"x" * 1000)]) == ""
    big = b"x" * (HOSTED_TOTAL_MB * 1024 * 1024 + 1)
    assert "MB in total" in check_limits([("a.csv", big)])
    assert "files at a time" in check_limits([(f"{i}.csv", b"x") for i in range(HOSTED_FILES + 1)])


def test_the_upload_check_is_unchanged_locally():
    from tools.multi import check_limits
    assert check_limits([(f"{i}.csv", b"x") for i in range(HOSTED_FILES + 1)]) == ""


# ------------------------------------------------------------------ every place that runs AI code passes the options
class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, code, tables, **kw):
        self.calls.append(kw)
        return type("R", (), {"ok": True, "status": "ok", "value": 1, "table": None, "figure_json": None,
                              "problems": [], "error": None, "stdout": "", "message": lambda s: ""})()


@pytest.mark.parametrize("on", [False, True])
def test_chat_calculations_pass_the_hosting_options(monkeypatch, on):
    import tools.sandbox as sb
    from agents.chat_agent import CALC_TIMEOUT, default_runner
    if on:
        monkeypatch.setenv("DAA_HOSTED", "1")
    spy = Spy()
    monkeypatch.setattr(sb, "run_code", spy)
    default_runner("result = 1", pd.DataFrame({"a": [1]}))
    expected = {"timeout": CALC_TIMEOUT, **({"require_isolation": True, "memory_mb": HOSTED_SANDBOX_MB} if on else {})}
    assert spy.calls == [expected]


@pytest.mark.parametrize("on", [False, True])
def test_multi_table_calculations_pass_the_hosting_options(monkeypatch, on):
    import tools.sandbox as sb
    from agents.multi_chat import make_multi_runner
    if on:
        monkeypatch.setenv("DAA_HOSTED", "1")
    spy = Spy()
    monkeypatch.setattr(sb, "run_code", spy)
    t = {"a": pd.DataFrame({"x": [1]})}
    make_multi_runner(t, [], "a")("result = 1", t)
    assert ("require_isolation" in spy.calls[0]) is on and spy.calls[0]["active"] == "a"


@pytest.mark.parametrize("on", [False, True])
def test_ai_chart_code_passes_the_hosting_options(monkeypatch, on):
    import agents.chart_agent as ca
    if on:
        monkeypatch.setenv("DAA_HOSTED", "1")
    spy = Spy()
    monkeypatch.setattr(ca, "run_code", spy)
    src = open(ROOT / "agents" / "chart_agent.py").read()
    assert "sandbox_options()" in src and "run_code(code, df, **sandbox_options())" in src
    from tools.hosting import sandbox_options
    ca.run_code("x", pd.DataFrame(), **sandbox_options())
    assert ("require_isolation" in spy.calls[0]) is on


# ------------------------------------------------------------------ the sandbox probe fails closed
def test_the_probe_passes_when_the_locked_run_works():
    seen = []

    def run(code, df, **kw):
        seen.append(kw)
        return Spy()(code, df)
    assert isolation_ok(run) == (True, "") and seen[0]["require_isolation"] is True


def test_the_probe_fails_closed_when_the_lock_is_missing_or_the_run_breaks():
    refused = lambda code, df, **kw: type("R", (), {"ok": False})()          # noqa: E731
    ok, why = isolation_ok(refused)
    assert not ok and "file-system lock" in why

    def boom(code, df, **kw):
        raise OSError("no")
    ok, why = isolation_ok(boom, refresh=True)
    assert not ok and "OSError" in why


def test_the_probe_runs_once_unless_asked_again():
    n = []
    run = lambda code, df, **kw: n.append(1) or Spy()(code, df)              # noqa: E731
    isolation_ok(run)
    isolation_ok(run)
    assert len(n) == 1
    isolation_ok(run, refresh=True)
    assert len(n) == 2


# ------------------------------------------------------------------ keys
def test_redact_removes_the_key_from_a_message():
    assert redact("bad request for AIzaSyABCDEFGH123 sorry", "AIzaSyABCDEFGH123") == "bad request for [your key] sorry"
    assert redact("nothing here", "") == "nothing here" and redact("abc", "abc") == "abc"      # too short to be a key


def test_each_visitors_key_stays_in_their_own_model_and_never_in_the_environment(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    before = dict(os.environ)
    from agents.llm_factory import make_llm
    a = make_llm("chat", mode="off", api_key="KEY-AAAAAAAA")
    b = make_llm("chat", mode="off", api_key="KEY-BBBBBBBB")
    ka = [m.inner.api_client._api_client.api_key for m in a.chain]
    kb = [m.inner.api_client._api_client.api_key for m in b.chain]
    assert set(ka) == {"KEY-AAAAAAAA"} and set(kb) == {"KEY-BBBBBBBB"}
    assert dict(os.environ) == before and "KEY-AAAAAAAA" not in "".join(os.environ.values())


def test_without_a_key_the_model_is_built_exactly_as_before(monkeypatch):
    import agents.llm_factory as lf
    made = []
    monkeypatch.setattr("google.adk.models.Gemini", lambda **kw: made.append(kw) or object())
    lf.make_llm("planner", mode="off")
    assert made and all(set(k) == {"model"} for k in made)


def test_replay_mode_builds_no_client_even_with_a_key(monkeypatch):
    import agents.llm_factory as lf
    monkeypatch.setattr("google.adk.models.Gemini", lambda **kw: pytest.fail("must not be built"))
    lf.make_llm("chat", mode="replay", api_key="KEY-AAAAAAAA")


# ------------------------------------------------------------------ the tabs
def test_the_disk_cache_setting_is_hidden_on_the_host_only(hosted, monkeypatch):
    from ui.dashboard_tab import CACHE_MODES, cache_modes
    assert set(cache_modes().values()) == {"replay", "off"}
    monkeypatch.delenv("DAA_HOSTED")
    assert cache_modes() == CACHE_MODES


def test_the_server_key_is_never_used_on_the_host(hosted, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "the-servers-own-key-123")
    import ui.hosted_ui as hu
    from ui.dashboard_tab import api_key_available
    monkeypatch.setattr(hu, "user_key", lambda: "")
    assert api_key_available() is False
    monkeypatch.setattr(hu, "user_key", lambda: "visitor-key-12345")
    assert api_key_available() is True


def test_the_sessions_use_the_visitors_key(hosted, monkeypatch):
    import ui.hosted_ui as hu
    import agents.llm_factory as lf
    seen = []
    monkeypatch.setattr(lf, "make_llm", lambda role, **kw: seen.append((role, kw.get("api_key"))) or object())
    monkeypatch.setattr(hu, "user_key", lambda: "visitor-key-12345")
    from ui import chat_tab, dashboard_tab
    from tools.profiler import profile_dataframe
    profile = profile_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
    chat_tab.default_session_factory(profile, {"cache_mode": "off", "df": pd.DataFrame({"a": [1]})})
    try:
        dashboard_tab.default_session_factory(pd.DataFrame({"a": [1]}), profile,
                                              {"use_ai": True, "cache_mode": "off", "max_charts": 3}, lambda *a: None)
    except Exception:                                      # the object() stand-in cannot plan; only the key matters
        pass
    assert ("chat", "visitor-key-12345") in seen and ("planner", "visitor-key-12345") in seen and ("chart", "visitor-key-12345") in seen


# ------------------------------------------------------------------ the whole app
class U:
    def __init__(self, n, d, size=None):
        self.name, self._d, self.size = n, d, size if size is not None else len(d)

    def getvalue(self):
        return self._d


def open_app(monkeypatch, files=None, probe=(True, "")):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U(n, d) for n, d in (files or [])])
    monkeypatch.setattr(hosting, "isolation_ok", lambda *a, **k: probe)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.session_state["data_source"] = "Upload my own CSV"          # the hosted app opens on the demo set; these tests upload
    return at


def texts(at):
    return [e.value for e in list(at.info) + list(at.warning) + list(at.error) + list(at.caption)]


def test_the_local_app_has_no_banner_no_key_box_and_the_sandbox_box(monkeypatch):
    at = open_app(monkeypatch).run()
    assert not at.exception and not any("Public demo" in t for t in texts(at))
    assert not [t for t in at.sidebar.text_input if t.label == "Paste your key"]
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert any(t.label.startswith("Python code") for t in at.text_area)


def test_the_hosted_app_shows_the_banner_and_key_box_and_no_python_box(hosted, monkeypatch):
    at = open_app(monkeypatch).run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert not at.exception
    assert any("Public demo" in t for t in texts(at))
    assert [t for t in at.sidebar.text_input if t.label == "Paste your key"]
    assert not any(t.label.startswith("Python code") for t in at.text_area)
    assert any("code sandbox tab is switched off" in t for t in texts(at))
    assert not [b for b in at.button if b.label == "▶ Run in sandbox"]


def test_the_visitors_key_is_kept_in_the_session_only(hosted, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = open_app(monkeypatch).run()
    box = next(t for t in at.sidebar.text_input if t.label == "Paste your key")
    box.input("AIza-visitor-secret-123").run()
    assert at.session_state["user_api_key"] == "AIza-visitor-secret-123"
    assert "AIza-visitor-secret-123" not in "".join(os.environ.values())
    assert any("A key is set" in t for t in texts(at))
    next(b for b in at.sidebar.button if b.label == "Forget my key").click().run()
    assert "user_api_key" not in at.session_state


def test_a_file_over_the_hosted_limit_is_refused(hosted, monkeypatch):
    big = U("big.csv", b"a,b\n1,2\n", size=(HOSTED_FILE_MB + 1) * 1024 * 1024)
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [big])
    monkeypatch.setattr(hosting, "isolation_ok", lambda *a, **k: (True, ""))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.session_state["data_source"] = "Upload my own CSV"
    at.run()
    assert any("larger than 15 MB" in e.value for e in at.error)


SMALL = [("t.csv", b"region,revenue\nN,10\nS,20\nE,30\nW,40\nN,50\nS,60\n")]


def test_when_the_sandbox_lock_is_missing_the_chat_does_not_calculate(hosted, monkeypatch):
    at = open_app(monkeypatch, SMALL, probe=(False, "the sandbox's file-system lock is not available on this host")).run()
    assert not at.exception
    t = texts(at)
    assert any("Calculations are switched off on this host" in x for x in t)
    assert any("I answer from a summary of the data" in x for x in t)
    assert not any("locked-down sandbox on your computer" in x for x in t)


def test_when_the_sandbox_lock_works_the_chat_calculates(hosted, monkeypatch):
    at = open_app(monkeypatch, SMALL).run()
    assert not at.exception
    t = texts(at)
    assert not any("Calculations are switched off" in x for x in t)
    assert any("locked-down sandbox" in x for x in t)


def test_locally_the_probe_is_never_run(monkeypatch):
    called = []
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [U(n, d) for n, d in SMALL])
    monkeypatch.setattr(hosting, "isolation_ok", lambda *a, **k: called.append(1) or (False, "x"))
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and not called


def test_a_key_in_the_session_is_ignored_on_a_local_run(monkeypatch):
    import types
    import ui.hosted_ui as hu
    monkeypatch.setattr(hu, "st", types.SimpleNamespace(session_state={"user_api_key": "stray-key-1234"}))
    assert hu.user_key() == ""
    monkeypatch.setenv("DAA_HOSTED", "1")
    assert hu.user_key() == "stray-key-1234"


TWO = [("races.csv", b"raceId,year\n1,2020\n2,2021\n3,2022\n"),
       ("results.csv", b"resultId,raceId,points\n1,1,10\n2,1,8\n3,2,10\n4,2,6\n5,3,10\n6,3,4\n")]


def test_several_tables_and_no_sandbox_lock_means_no_multi_table_chat(hosted, monkeypatch):
    at = open_app(monkeypatch, TWO, probe=(False, "x")).run()
    assert not at.exception
    assert not any("The chat works on" in t for t in texts(at))
    at = open_app(monkeypatch, TWO, probe=(True, "")).run()
    assert any("The chat works on" in t for t in texts(at))


# ------------------------------------------------------------------ "off" must leave nothing on the server's disk
def _ask_once(mode, path):
    import asyncio
    from agents.llm_cache import CachingLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from test_chat_agent import ScriptedLlm
    inner = ScriptedLlm(replies=["an answer about someone's data"], seen=[], systems=[])
    llm = CachingLlm(model="fake", inner=inner, mode=mode, path=str(path), min_interval=0)
    req = LlmRequest(model="fake", contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    async def go():
        return [r async for r in llm.generate_content_async(req)]
    asyncio.run(go())


def test_off_mode_writes_no_cache_file(tmp_path):
    _ask_once("off", tmp_path / "c.json")
    assert not (tmp_path / "c.json").exists()


def test_use_mode_still_saves_the_reply(tmp_path):
    _ask_once("use", tmp_path / "c.json")
    import json
    saved = json.loads((tmp_path / "c.json").read_text())
    assert len(saved) == 1 and "null" not in saved and "an answer about someone's data" in str(saved)


# ------------------------------------------------------------------ Day 23b: the key really goes away
def test_safe_text_takes_the_visitors_key_out(hosted, monkeypatch):
    import ui.hosted_ui as hu
    monkeypatch.setattr(hu, "user_key", lambda: "AIza-visitor-secret-123")
    assert hu.safe_text("error for AIza-visitor-secret-123 happened") == "error for [your key] happened"
    monkeypatch.setattr(hu, "user_key", lambda: "")
    assert hu.safe_text("plain") == "plain"


def test_the_key_box_asks_the_browser_not_to_remember_it(hosted, monkeypatch):
    at = open_app(monkeypatch).run()
    box = next(t for t in at.sidebar.text_input if t.label == "Paste your key")
    assert box.proto.autocomplete == "off" and box.proto.type == 1          # 1 = password


def test_a_start_up_error_never_shows_the_key(hosted, monkeypatch):
    import ui.chat_tab as ct
    key = "AIza-visitor-secret-123"

    def boom(profile, settings):
        raise RuntimeError(f"rejected the key {key} for this request")
    monkeypatch.setattr(ct, "default_session_factory", boom)
    at = open_app(monkeypatch, SMALL).run()
    next(t for t in at.sidebar.text_input if t.label == "Paste your key").input(key).run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("Could not start the chat" in e for e in errors) and not any(key in e for e in errors)
    assert any("[your key]" in e for e in errors)


def test_forgetting_the_key_also_closes_the_chat_and_dashboard_that_held_it(hosted, monkeypatch):
    key = "AIza-visitor-secret-123"
    at = open_app(monkeypatch, SMALL).run()
    next(t for t in at.sidebar.text_input if t.label == "Paste your key").input(key).run()
    chat_mode = next(s for s in at.selectbox if s.key == "chat_cache_mode")
    chat_mode.select("Ask the AI again: ignore saved replies").run()
    session = at.session_state["chat_session"]
    held = [m.inner.api_client._api_client.api_key for m in session.model.chain]
    assert set(held) == {key}                                         # the chat really holds a connection made with the key
    at.session_state["ai_session"] = object()                          # and so may a dashboard run
    next(b for b in at.sidebar.button if b.label == "Forget my key").click().run()
    assert not at.exception
    for gone in ("user_api_key", "chat_session", "chat_kept", "ai_session"):
        assert gone not in at.session_state, gone
    assert at.session_state["key_box"] == ""                           # the empty box comes back, holding nothing
    assert all(key not in repr(at.session_state[k]) for k in at.session_state.keys() if k != "key_box")


def test_a_chat_error_message_never_shows_the_key(hosted, monkeypatch):
    key = "AIza-visitor-secret-123"
    at = open_app(monkeypatch, SMALL).run()
    next(t for t in at.sidebar.text_input if t.label == "Paste your key").input(key).run()
    at.session_state["chat_error"] = f"The model refused the key {key}."
    at.run()
    errors = [e.value for e in at.error]
    assert errors and not any(key in e for e in errors) and any("[your key]" in e for e in errors)
