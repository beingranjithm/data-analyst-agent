"""Waiting out temporary Google errors. No real waiting: sleep is replaced by a recorder."""
import asyncio

import pandas as pd
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

from agents.chart_loop import make_chart_healing
from agents.resilience import is_transient_error, with_backoff
from tools.plan_schema import ChartSpec

PROFILE = {"dataset": {"rows": 3, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}
CHART = ChartSpec(type="bar", title="Revenue by region", dimension="region", measure="revenue", agg="sum")
GOOD = ("```python\nd = df.groupby('region', as_index=False)['revenue'].sum()\n"
        "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')\n```")
BUSY = "503 UNAVAILABLE. This model is currently experiencing high demand."


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
    monkeypatch.setattr(chart_agent, "profile_to_text", lambda p: "\n".join(
        f"- {c['name']} ({c['role']})" for c in p["columns"]))


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South"], "revenue": [1.0, 2.0]})


def recorder():
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
    return sleeps, fake_sleep


# ---------- is_transient_error ----------
def test_real_google_errors_are_recognised():
    busy = errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}, None)
    quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}, None)
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}, None)
    assert is_transient_error(busy) and is_transient_error(quota) and not is_transient_error(bad)


def test_plain_errors():
    assert is_transient_error(RuntimeError("503 UNAVAILABLE"))
    assert not is_transient_error(KeyError("columns"))
    assert not is_transient_error(RuntimeError("401 API key not valid"))


# ---------- with_backoff ----------
def test_backoff_waits_then_succeeds():
    sleeps, fake_sleep = recorder()
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError(BUSY)
        return "done"
    waits = []
    out = asyncio.run(with_backoff(flaky, delays=(5.0, 15.0, 30.0), sleep=fake_sleep,
                                   on_wait=lambda n, s, why: waits.append((n, s))))
    assert out == "done" and sleeps == [5.0, 15.0] and waits == [(1, 5.0), (2, 15.0)]


def test_backoff_gives_up_and_raises_the_last_error():
    sleeps, fake_sleep = recorder()

    async def always_busy():
        raise RuntimeError(BUSY)
    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(with_backoff(always_busy, delays=(1.0, 2.0), sleep=fake_sleep))
    assert sleeps == [1.0, 2.0]


def test_backoff_does_not_wait_for_other_errors():
    sleeps, fake_sleep = recorder()

    async def broken():
        raise ValueError("a real bug")
    with pytest.raises(ValueError):
        asyncio.run(with_backoff(broken, sleep=fake_sleep))
    assert sleeps == []


# ---------- the chart loop ----------
def test_loop_waits_out_a_busy_server_without_using_up_attempts(df):
    sleeps, fake_sleep = recorder()
    llm = ScriptedLlm(replies=[RuntimeError(BUSY), RuntimeError(BUSY), GOOD])
    res = make_chart_healing(df, PROFILE, CHART, model=llm, sleep=fake_sleep)
    assert res.ok and res.attempts == 1 and llm.calls == 3 and sleeps == [5.0, 15.0]


def test_loop_gives_up_after_the_last_wait(df):
    sleeps, fake_sleep = recorder()
    llm = ScriptedLlm(replies=[RuntimeError(BUSY)] * 4)
    res = make_chart_healing(df, PROFILE, CHART, model=llm, sleep=fake_sleep)
    assert not res.ok and res.final.model_error and "503" in res.message
    assert llm.calls == 4 and sleeps == [5.0, 15.0, 30.0]


def test_loop_does_not_wait_for_a_bad_api_key(df):
    sleeps, fake_sleep = recorder()
    llm = ScriptedLlm(replies=[RuntimeError("401 API key not valid")])
    res = make_chart_healing(df, PROFILE, CHART, model=llm, sleep=fake_sleep)
    assert not res.ok and llm.calls == 1 and sleeps == []
