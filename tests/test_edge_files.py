"""Day 19: odd CSV files through the whole no-AI pipeline. On YOUR computer this runs against your real loader,
cleaner, profiler, charts and sandbox; a failure here is a real bug with a file name and a step name."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools.profiler as profiler_mod  # noqa: E402
import tools.loader as loader_mod  # noqa: E402
from calc_fakes import local_runner  # noqa: E402
from evals import edge_files  # noqa: E402
from evals.edge_files import CRASH, OK, REFUSED, UGLY, describe, make_files, run_file, run_sweep  # noqa: E402
from evals.edge_live import HOSTILE  # noqa: E402

FILES = make_files()


def _real_sandbox_returns_values() -> bool:
    """The real sandbox hands back `result`. (The simplified stand-in used while this was written does not, so then the
    fake runner is used for that one step.)"""
    try:
        from agents.chat_agent import default_runner
        return getattr(default_runner("result = 41 + 1", pd.DataFrame({"a": [1]})), "value", None) == 42
    except Exception:
        return False


RUNNER = None if _real_sandbox_returns_values() else local_runner


# ------------------------------------------------------------------ the files themselves
def test_there_are_twenty_odd_files_with_the_cases_that_matter():
    assert len(FILES) >= 20 and len(set(FILES)) == len(FILES)
    assert FILES["empty file (0 bytes)"] == b"" and b"\xef\xbb\xbf" == FILES["starts with a BOM"][:3]
    assert HOSTILE.encode() in FILES["hostile text in a category"] and "HACKED" in HOSTILE
    assert b";" in FILES["semicolon separated"] and b"\xe9" in FILES["accents in latin-1 encoding"]


@pytest.mark.parametrize("name", list(FILES))
def test_no_odd_file_crashes_any_step(name):
    out = run_file(name, FILES[name], RUNNER)
    assert not out.crashes, describe(out)
    assert out.stages["load"][0] in (OK, REFUSED, UGLY)


def test_a_file_that_cannot_be_read_stops_after_the_load_step_and_the_rest_are_not_attempted():
    out = run_file("empty file (0 bytes)", b"", RUNNER)
    if out.stages["load"][0] != OK:
        assert list(out.stages) == ["load"]


# ------------------------------------------------------------------ the sweep itself reports honestly
def good_csv():
    return FILES["semicolon separated"].replace(b";", b",")


def test_a_clean_file_passes_every_step():
    out = run_file("clean", good_csv(), RUNNER)
    assert not out.crashes and out.refused_at is None and list(out.stages) == list(edge_files.STAGES)
    assert all(st == OK for st, _ in out.stages.values()) and describe(out) == "OK"


def test_a_crash_in_any_step_is_reported_with_its_step_and_later_steps_that_need_it_are_skipped(monkeypatch):
    def boom(df):
        raise KeyError("columns")
    monkeypatch.setattr(profiler_mod, "profile_dataframe", boom)
    out = run_file("x", good_csv(), RUNNER)
    assert [s for s, _ in out.crashes] == ["profile"] and "KeyError" in out.crashes[0][1] and list(out.stages) == ["load", "clean", "profile"]
    assert describe(out).startswith("CRASH") and "profile" in describe(out)


def test_our_own_clear_refusal_is_fine_but_a_raw_library_message_is_flagged_as_ugly(monkeypatch):
    monkeypatch.setattr(loader_mod, "load_csv_bytes", lambda b: (_ for _ in ()).throw(ValueError("This file has no rows.")))
    friendly = run_file("x", b"a\n", RUNNER)
    assert friendly.stages["load"][0] == REFUSED and not friendly.crashes and describe(friendly).startswith("REFUSED")
    monkeypatch.setattr(loader_mod, "load_csv_bytes", lambda b: pd.read_csv(__import__("io").BytesIO(b)))
    raw = run_file("x", b"", RUNNER)
    assert raw.stages["load"][0] == UGLY and not raw.crashes and describe(raw).startswith("UGLY")


def test_an_unexpected_exception_type_while_loading_is_a_crash_not_a_refusal(monkeypatch):
    monkeypatch.setattr(loader_mod, "load_csv_bytes", lambda b: (_ for _ in ()).throw(RuntimeError("boom")))
    out = run_file("x", b"a\n", RUNNER)
    assert out.crashes and out.crashes[0][0] == "load"


def test_the_sandbox_step_notices_when_the_row_count_does_not_come_back():
    out = run_file("clean", good_csv(), lambda code, tables: type("R", (), {"ok": True, "value": None, "table": None,
                                                                           "stdout": "", "table_total_rows": None})())
    assert [s for s, _ in out.crashes] == ["sandbox"]


def test_a_summary_with_a_very_long_line_is_caught(monkeypatch):
    import agents.chat_agent as chat_agent
    monkeypatch.setattr(chat_agent, "sanitize_profile_text", lambda t: t + "\n" + "x" * 600)
    out = run_file("clean", good_csv(), RUNNER)
    assert [s for s, _ in out.crashes] == ["summary"] and "long line" in out.crashes[0][1]


def test_run_sweep_covers_every_file_and_reports_progress():
    seen = []
    results = run_sweep({"a": good_csv(), "b": b"", "c": FILES["one data row"]}, progress=seen.append, runner=RUNNER)
    assert [r.name for r in results] == ["a", "b", "c"] == [r.name for r in seen]


def test_the_sandbox_step_notices_a_failed_run_even_if_something_that_looks_like_the_count_was_printed():
    n = len(pd.read_csv(__import__("io").BytesIO(good_csv())))
    failed = type("R", (), {"ok": False, "status": "error", "value": None, "table": None, "stdout": str(n), "table_total_rows": None})()
    out = run_file("clean", good_csv(), lambda code, tables: failed)
    assert [s for s, _ in out.crashes] == ["sandbox"] and "could not even count rows" in out.crashes[0][1]
