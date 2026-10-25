"""Dashboard assembler tests: fake model, no Gemini, no real waiting."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import agents.dashboard as dash
from agents.dashboard import build_dashboard
from tools.charts import ChartError
from tools.figure_checks import check_figure

PROFILE = {"dataset": {"rows": 4, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}

BAR = {"type": "bar", "title": "Revenue by region", "dimension": "region", "measure": "revenue", "agg": "sum"}
HIST = {"type": "histogram", "title": "Revenue distribution", "measure": "revenue"}
BOX = {"type": "box", "title": "Revenue spread by region", "dimension": "region", "measure": "revenue"}
BUSY = "503 UNAVAILABLE. This model is currently experiencing high demand."

GOOD_BAR = ("```python\nd = df.groupby('region', as_index=False)['revenue'].sum()\n"
            "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')\n```")
GOOD_HIST = "```python\nfig = px.histogram(df, x='revenue')\n```"          # no title on purpose
EMPTY_FIG = "```python\nfig = px.bar(df[df['revenue'] > 1e9], x='region', y='revenue')\n```"


def bad(i):
    return f"```python\nfig = px.bar(df, x='nope{i}', y='revenue')\n```"


def plan_json(*charts, kpis=None):
    return json.dumps({"title": "Sales", "summary": "A summary.", "charts": list(charts),
                       "kpis": [{"label": "Revenue", "measure": "revenue", "agg": "sum"}] if kpis is None else kpis})


class ScriptedLlm(BaseLlm):
    """Each reply is a string to return or an exception to raise."""
    model: str = "fake"
    replies: list = []
    calls: int = 0

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=item)]))


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    import agents.chart_agent as chart_agent
    import agents.planner as planner
    text = lambda p: "\n".join(f"- {c['name']} ({c['role']})" for c in p["columns"])
    monkeypatch.setattr(chart_agent, "profile_to_text", text)
    monkeypatch.setattr(planner, "profile_to_text", text)


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace the rule-based chart builder and gallery with simple doubles."""
    monkeypatch.setattr(dash, "build_chart", lambda df, spec: px.bar(df, x="region", y="revenue", title="rule-based"))
    monkeypatch.setattr(dash, "suggest_charts", lambda profile, max_charts=8: [
        {"type": "bar", "dimension": "region", "measure": "revenue", "agg": "sum"},
        {"type": "histogram", "measure": "revenue"}])


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South", "North", "East"], "revenue": [10.0, 20.0, 30.0, 40.0]})


@pytest.fixture
def sleeps():
    return []


def run(df, sleeps, replies, **kw):
    async def fake_sleep(seconds):
        sleeps.append(seconds)
    llm = ScriptedLlm(replies=list(replies))
    return llm, build_dashboard(df, PROFILE, model=llm, sleep=fake_sleep, **kw)


def test_happy_path_ai(df, sleeps):
    llm, d = run(df, sleeps, [plan_json(BAR, HIST), GOOD_BAR, GOOD_HIST])
    assert d.mode == "ai" and d.title == "Sales" and d.rows == 4 and not d.notes
    assert [c.source for c in d.charts] == ["ai", "ai"] and len(d.ok_charts) == 2 and not d.failed_charts
    assert d.kpis[0].label == "Revenue" and d.kpis[0].value == 100.0
    assert d.charts[1].figure.layout.title.text == "Revenue distribution"      # missing title was filled in
    assert d.charts[0].code and llm.calls == 3 and sleeps == []


def test_progress_reports_each_stage_and_survives_a_broken_callback(df, sleeps):
    events = []
    run(df, sleeps, [plan_json(BAR, HIST), GOOD_BAR, GOOD_HIST], progress=lambda *a: events.append(a))
    assert [e[0] for e in events] == ["planning", "chart", "chart", "done"]
    assert events[1][1:3] == (0, 2) and events[2][1:3] == (1, 2) and events[-1][1:3] == (2, 2)

    def boom(*a):
        raise RuntimeError("UI crashed")
    _, d = run(df, sleeps, [plan_json(BAR), GOOD_BAR], progress=boom)
    assert len(d.ok_charts) == 1


def test_rule_based_builder_rescues_a_chart_the_ai_cannot_make(df, sleeps, fake_tools):
    _, d = run(df, sleeps, [plan_json(BAR), bad(1), bad(2), bad(3)])
    c = d.charts[0]
    assert c.ok and c.source == "rule-based" and c.attempts == 3 and "Gave up" in c.ai_message
    assert d.mode == "ai" and not d.failed_charts


def test_one_failed_chart_does_not_stop_the_others(df, sleeps, monkeypatch):
    def always_fails(df, spec):
        raise ChartError("cannot draw this")
    monkeypatch.setattr(dash, "build_chart", always_fails)
    _, d = run(df, sleeps, [plan_json(BAR, HIST), bad(1), bad(2), bad(3), GOOD_HIST])
    assert len(d.failed_charts) == 1 and len(d.ok_charts) == 1
    f = d.failed_charts[0]
    assert not f.ok and f.figure is None and "AI:" in f.message and "Rule-based:" in f.message


def test_validator_rejects_an_empty_ai_chart(df, sleeps, fake_tools):
    _, d = run(df, sleeps, [plan_json(BAR), EMPTY_FIG])
    c = d.charts[0]
    assert c.ok and c.source == "rule-based" and "rejected" in c.ai_message and "no data points" in c.ai_message


def test_garbage_plan_falls_back_to_the_gallery(df, sleeps, fake_tools):
    llm, d = run(df, sleeps, ["garbage", "garbage"])
    assert d.mode == "rule-based (planner failed)" and llm.calls == 2
    assert len(d.ok_charts) == 2 and all(c.source == "rule-based" for c in d.charts)
    assert any("rule-based" in n for n in d.notes)


def test_planner_waits_out_a_busy_server(df, sleeps):
    llm, d = run(df, sleeps, [RuntimeError(BUSY), RuntimeError(BUSY), plan_json(BAR), GOOD_BAR])
    assert d.mode == "ai" and sleeps == [5.0, 15.0] and len(d.ok_charts) == 1


def test_planner_down_for_good_falls_back(df, sleeps, fake_tools):
    llm, d = run(df, sleeps, [RuntimeError(BUSY)] * 4)
    assert d.mode == "rule-based (planner failed)" and sleeps == [5.0, 15.0, 30.0]
    assert any("could not run" in n for n in d.notes) and len(d.ok_charts) == 2


def test_model_outage_stops_further_ai_calls(df, sleeps, fake_tools):
    llm, d = run(df, sleeps, [plan_json(BAR, HIST, BOX)] + [RuntimeError(BUSY)] * 4)
    assert llm.calls == 5 and sleeps == [5.0, 15.0, 30.0]          # only the first chart waited
    assert [c.source for c in d.charts] == ["rule-based"] * 3
    assert any("unavailable" in n for n in d.notes)


def test_use_ai_false_never_calls_the_model(df, sleeps, fake_tools):
    llm, d = run(df, sleeps, [], use_ai=False)
    assert llm.calls == 0 and d.mode == "rule-based" and len(d.ok_charts) == 2


def test_max_charts(df, sleeps):
    llm, d = run(df, sleeps, [plan_json(BAR, HIST), GOOD_BAR], max_charts=1)
    assert len(d.charts) == 1 and llm.calls == 2


def test_real_chart_builder_makes_a_full_dashboard_from_the_sample_file():
    """No model, no doubles: the REAL gallery and chart builder on sample_data."""
    pytest.importorskip("tools.loader")
    from tools.cleaner import clean_dataframe
    from tools.loader import load_csv_bytes
    from tools.profiler import profile_dataframe
    sample = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"
    data, _ = load_csv_bytes(sample.read_bytes())
    data, _ = clean_dataframe(data)
    d = build_dashboard(data, profile_dataframe(data), use_ai=False)
    assert d.mode == "rule-based" and d.ok_charts and not d.failed_charts
    assert all(check_figure(c.figure).ok for c in d.ok_charts)
