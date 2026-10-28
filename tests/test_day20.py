"""Day 20: UI polish. Plain message for 'no saved reply', whole numbers without '.0', first draft kept and shown,
chart-title check, KPI cards that are not cut off, captions that are no longer stale."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import ChatReply, format_result  # noqa: E402
from agents.dashboard import _finish  # noqa: E402
from agents.llm_cache import CacheMiss  # noqa: E402
from agents.resilience import describe_model_error  # noqa: E402
from calc_fakes import FakeRun  # noqa: E402
from evals.chat_eval import build_questions, render_markdown, run_eval  # noqa: E402
from test_chat_agent import PROFILE_TEXT  # noqa: E402
from test_chat_calc import CODE, make  # noqa: E402
from test_chat_eval import make_df, oracle_reply  # noqa: E402
from test_chat_number_fix import RIGHT, WRONG  # noqa: E402
from tools.plan_schema import ChartSpec  # noqa: E402
from tools.title_check import title_warnings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def simple_profile_text(monkeypatch):
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: PROFILE_TEXT)


# ---- A: friendly message
def test_a_missing_saved_reply_gets_plain_words():
    msg = describe_model_error(CacheMiss("no saved reply for key abc123"))
    assert "no saved answer" in msg.lower() and "Replay only" in msg and "abc123" not in msg and "CacheMiss" not in msg


def test_other_errors_are_unchanged():
    assert describe_model_error(ValueError("boom")) == "ValueError: boom"
    assert describe_model_error(RuntimeError("x" * 500)) == ("RuntimeError: " + "x" * 500)[:300]


# ---- C: whole numbers
def test_whole_valued_decimals_show_without_point_zero():
    t = pd.DataFrame({"total": [349969.65, 12.0], "rows_matched": [126.0, 99.0]})
    out = format_result(FakeRun(table=t, total=2))
    assert "rows_matched" in out and "126" in out and "126.0" not in out and "99.0" not in out
    assert "349969.65" in out and "12.0" not in out and out.splitlines()[2].startswith("12,")


def test_scalars_and_series_show_whole_numbers_plainly_but_keep_real_decimals():
    assert format_result(FakeRun(value=126.0)).strip().endswith("126") and "126.0" not in format_result(FakeRun(value=126.0))
    assert "3.5" in format_result(FakeRun(value=3.5))
    assert "0.1" in format_result(FakeRun(value=0.1))


def test_missing_values_stay_blank_and_text_is_untouched():
    t = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x.0", "y"]})
    out = format_result(FakeRun(table=t, total=2))
    assert "x.0" in out and "nan" not in out.lower() and out.splitlines()[-2:] == ["1,x.0", ",y"]


def test_huge_and_inexact_floats_are_not_mangled():
    out = format_result(FakeRun(value=1e20))
    assert "1e+20" in out or "100000000000000000000" in out
    assert "0.30000000000000004" in format_result(FakeRun(value=0.1 + 0.2))


# ---- E: first draft
def test_the_first_draft_is_kept_when_the_guard_rewrites(simple_profile_text):
    _, s = make([CODE, WRONG, RIGHT])
    r = s.ask("East revenue?")
    assert r.corrected and r.first_draft == WRONG and r.text == RIGHT


def test_no_first_draft_when_nothing_was_rewritten(simple_profile_text):
    _, s = make([CODE, RIGHT])
    r = s.ask("East revenue?")
    assert not r.corrected and r.first_draft == ""


def test_the_eval_report_lists_rewritten_answers():
    qs, _ = build_questions(make_df())
    a, b = qs[0], qs[1]
    ra, rb = oracle_reply(a), oracle_reply(b)
    ra.corrected, ra.first_draft = True, "FIRST DRAFT WITH 111"
    results = run_eval([a, b], lambda t: {a.text: ra, b.text: rb}[t])
    assert results[0].first_draft == "FIRST DRAFT WITH 111" and results[1].first_draft == ""
    md = render_markdown(results)
    assert "Rewritten answers" in md and "FIRST DRAFT WITH 111" in md
    assert "Rewritten answers" not in render_markdown(run_eval([b], lambda t: rb))


def test_a_reply_object_without_first_draft_still_works_in_the_eval():
    qs, _ = build_questions(make_df())

    class Old:
        ok, text, ungrounded, error, calculations, corrected = True, "Answer: 1", [], None, [], True
    assert run_eval([qs[2]], lambda t: Old())[0].first_draft == ""


# ---- D: title check
def test_money_measure_under_a_volume_title_warns():
    w = title_warnings("Top Products by Sales Volume", "revenue")
    assert len(w) == 1 and "volume" in w[0] and "revenue" in w[0]


def test_units_measure_under_a_money_title_warns():
    w = title_warnings("Revenue by Region", "units")
    assert len(w) == 1 and "revenue" in w[0].lower()


@pytest.mark.parametrize("title,measure", [
    ("Revenue by Region", "revenue"), ("Units sold by Product", "units"), ("Sales Volume (units)", "units"),
    ("Orders by Region", "orders"), ("Revenue and units", "units"), ("Anything", None), (None, "revenue"),
    ("Distribution of Discount", "discount")])
def test_no_warning_when_title_and_measure_agree_or_are_unclear(title, measure):
    assert title_warnings(title, measure) == []


def test_finish_adds_the_title_warning_without_blocking():
    chart = ChartSpec(type="bar", title="Top Products by Sales Volume", dimension="product", measure="revenue", agg="sum")
    fig = px.bar(pd.DataFrame({"product": ["a", "b"], "revenue": [1, 2]}), x="product", y="revenue", title=chart.title)
    _, problems, warnings = _finish(fig, chart)
    assert problems == [] and any("volume" in w.lower() for w in warnings)


def test_finish_stays_quiet_for_a_matching_title():
    chart = ChartSpec(type="bar", title="Revenue by Product", dimension="product", measure="revenue", agg="sum")
    fig = px.bar(pd.DataFrame({"product": ["a", "b"], "revenue": [1, 2]}), x="product", y="revenue", title=chart.title)
    assert not any("title says" in w for w in _finish(fig, chart)[2])


# ---- B + F: app captions and KPI cards
def test_stale_captions_are_gone():
    src = (ROOT / "app" / "streamlit_app.py").read_text()
    for stale in ("Chat comes next", "On Day 8", "AI agents will produce later", "AI-written code will run."):
        assert stale not in src, stale


KPI_HARNESS = """
import streamlit as st
from ui.kpi_card import show_kpi
cols = st.columns(3)
show_kpi(cols[0], {"label": "Rows", "value": "600"})
show_kpi(cols[1], {"label": "Date range", "value": "2025-01-01 to 2025-12-31"})
show_kpi(cols[2], {"label": "Evil", "value": "<script>alert(1)</script> & more text"})
"""


def test_kpi_cards_short_values_stay_metrics_and_long_ones_are_written_out_in_full():
    at = AppTest.from_string(KPI_HARNESS, default_timeout=30).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["600"]
    shown = " ".join(m.value for m in at.markdown)
    assert "2025-01-01 to 2025-12-31" in shown and "Date range" in shown


def test_kpi_text_from_the_data_is_escaped():
    from ui.kpi_card import kpi_html
    out = kpi_html("<b>L</b>", "<script>alert(1)</script>")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<b>L" not in out


def test_the_gallery_in_the_real_app_writes_long_kpi_values_out_in_full(monkeypatch):
    import tools.charts as charts
    monkeypatch.setattr(charts, "compute_kpis", lambda df, profile: [
        {"label": "Rows", "value": "600"}, {"label": "Date range", "value": "2025-01-01 to 2025-12-31"}])
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=90).run()
    at.sidebar.radio[0].set_value("Built-in sample sales data").run()
    assert not at.exception
    assert any("2025-01-01 to 2025-12-31" in m.value for m in at.markdown)
    assert "600" in [m.value for m in at.metric] and "2025-01-01 to 2025-12-31" not in [m.value for m in at.metric]


def test_true_and_false_are_not_turned_into_one_and_zero():
    assert "True" in format_result(FakeRun(value=True)) and "False" in format_result(FakeRun(value=False))


def test_a_per_unit_measure_gets_no_title_warning():
    assert title_warnings("Sales Volume by Region", "revenue_per_unit") == []
    assert title_warnings("Revenue by Region", "revenue_per_unit") == []
