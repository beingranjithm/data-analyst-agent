"""Tests for the ForcedFirstFailure helper (fake inner model, no Gemini calls)."""
import pandas as pd
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.chart_loop import make_chart_healing
from agents.testing import ForcedFirstFailure
from tools.plan_schema import ChartSpec

PROFILE = {"dataset": {"rows": 3, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}
CHART = ChartSpec(type="bar", title="Revenue by region", dimension="region", measure="revenue", agg="sum")
GOOD = ("d = df.groupby('region', as_index=False)['revenue'].sum()\n"
        "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')")


class Inner(BaseLlm):
    model: str = "inner"
    got: list = []

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.got.append(str(llm_request.config.system_instruction))
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"```python\n{GOOD}\n```")]))


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    import agents.chart_agent as chart_agent
    monkeypatch.setattr(chart_agent, "profile_to_text", lambda p: "\n".join(
        f"- {c['name']} ({c['role']})" for c in p["columns"]))


def test_first_call_is_broken_then_inner_takes_over():
    df = pd.DataFrame({"region": ["North", "South", "SECRET-ROW-VALUE-123"], "revenue": [1.0, 2.0, 3.0]})
    inner = Inner(got=[])
    llm = ForcedFirstFailure(inner=inner, first_reply="```python\nfig = px.bar(df, x='regoin', y='revenue')\n```", prompts=[])
    res = make_chart_healing(df, PROFILE, CHART, model=llm)
    assert res.ok and res.attempts == 2 and llm.calls == 2 and len(inner.got) == 1
    assert "PREVIOUS ATTEMPT" not in llm.prompts[0] and "YOUR PREVIOUS ATTEMPT FAILED" in inner.got[0]


def test_data_value_in_the_forced_error_is_hidden_from_the_inner_model():
    df = pd.DataFrame({"region": ["SECRET-ROW-VALUE-123", "South", "North"], "revenue": [1.0, 2.0, 3.0]})
    boom = "raise ValueError(\"bad value '\" + str(df['region'].dropna().iloc[0]) + \"'\")"
    inner = Inner(got=[])
    llm = ForcedFirstFailure(inner=inner, first_reply=f"```python\n{boom}\n```", prompts=[])
    res = make_chart_healing(df, PROFILE, CHART, model=llm)
    assert res.ok and "SECRET-ROW-VALUE-123" in res.history[0].message
    assert "SECRET-ROW-VALUE-123" not in inner.got[0] and "<value>" in inner.got[0]
