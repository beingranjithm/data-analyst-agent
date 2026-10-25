"""Chart agent tests with a fake model (no Gemini calls). Uses your real sandbox."""
import json

import pandas as pd
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.chart_agent import extract_code, make_chart
from tools.plan_schema import ChartSpec

PROFILE = {"dataset": {"rows": 4}, "columns": [
    {"name": "region", "role": "categorical"},
    {"name": "revenue", "role": "numeric"}]}

CHART = ChartSpec(type="bar", title="Revenue by region", why="Compare regions",
                  dimension="region", measure="revenue", agg="sum")

GOOD = ("d = df.groupby('region', as_index=False)['revenue'].sum()\n"
        "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')")


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    """Keep these tests independent of profile_to_text (the real one is used in the last test)."""
    import agents.chart_agent as chart_agent
    monkeypatch.setattr(chart_agent, "profile_to_text", lambda p: "\n".join(
        f"- {c['name']} ({c['role']})" for c in p["columns"]))


def fence(code, lang="python"):
    return f"Here you go:\n```{lang}\n{code}\n```"


class FakeLlm(BaseLlm):
    model: str = "fake"
    replies: list = []
    seen: list = []
    explode: bool = False

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        if self.explode:
            raise RuntimeError("429 quota exceeded")
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South", "North", "SECRET-ROW-VALUE-123"],
                         "revenue": [10.0, 20.0, 30.0, 40.0]})


def go(df, reply, **kw):
    llm = FakeLlm(replies=[reply], seen=[], **kw)
    return llm, make_chart(df, PROFILE, CHART, model=llm)


def test_good_code_builds_a_figure(df):
    _, res = go(df, fence(GOOD))
    assert res.ok, res.message
    assert res.figure is not None and res.code == GOOD


def test_blocked_code_is_reported_not_run(df):
    _, res = go(df, fence("import os\nos.system('ls')\nfig = px.bar(df, x='region', y='revenue')"))
    assert not res.ok and res.run is not None and res.run.status == "blocked"
    assert "blocked" in res.message.lower()


def test_runtime_error_is_reported(df):
    _, res = go(df, fence("fig = px.bar(df, x='regoin', y='revenue')"))
    assert not res.ok and res.run.status == "error" and res.message


def test_reply_without_code(df):
    _, res = go(df, "Sorry, I cannot do that.")
    assert not res.ok and res.code is None and res.run is None and "no code" in res.message.lower()


def test_code_without_fig(df):
    _, res = go(df, fence("result = df['revenue'].sum()"))
    assert not res.ok and "fig" in res.message


def test_model_failure_is_caught(df):
    _, res = go(df, "unused", explode=True)
    assert not res.ok and "RuntimeError" in res.message and "429" in res.message


def test_model_sees_profile_and_spec_but_never_rows(df):
    llm, _ = go(df, fence(GOOD))
    sysmsg = str(llm.seen[0].config.system_instruction)
    assert "region (categorical)" in sysmsg and "Revenue by region" in sysmsg and '"dimension": "region"' in sysmsg
    assert "{profile_text}" not in sysmsg and "{chart_json}" not in sysmsg and "{feedback}" not in sysmsg
    assert "SECRET-ROW-VALUE-123" not in str(llm.seen[0].contents) + sysmsg


def test_extract_code_variants():
    assert extract_code("```python\na=1\n```") == "a=1"
    assert extract_code("```py\na=1\n```") == "a=1"
    assert extract_code("```\na=1\n```") == "a=1"
    assert extract_code("text\n```json\n{}\n```\n```python\nb=2\n```") == "b=2"   # prefers the python block
    assert extract_code("```python\na=1\nb=2") == "a=1\nb=2"                      # reply cut off
    assert extract_code("no code here") is None
    assert extract_code("```python\n```") is None


def test_works_with_the_real_profile(df, monkeypatch):
    """End to end with the REAL profiler and profile_to_text (no hand-built profile)."""
    from tools.profiler import profile_dataframe
    monkeypatch.undo()   # restore the real profile_to_text for this one test
    profile = profile_dataframe(df)
    llm = FakeLlm(replies=[fence(GOOD)], seen=[])
    res = make_chart(df, profile, CHART, model=llm)
    assert res.ok, res.message
    assert "revenue" in str(llm.seen[0].config.system_instruction)
