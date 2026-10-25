"""The free tier's DAILY quota must fail fast; busy servers and per-minute limits are waited out."""
import json
import logging

import pandas as pd
import plotly.express as px
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

import agents.dashboard as dash
from agents.chart_loop import make_chart_healing
from agents.dashboard import build_dashboard
from agents.resilience import describe_model_error, is_daily_quota_error, is_transient_error, quiet_adk_logs
from tools.plan_schema import ChartSpec

# the shape of the real error from the Day 11 live run (trimmed), as ADK wraps it
DAILY_TEXT = (
    "\nOn how to mitigate this issue, please refer to:\n\nhttps://google.github.io/adk-docs/agents/models/google-gemini/#error-code-429-resource_exhausted\n\n\n"
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. "
    "For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \\n"
    "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash, "
    "description: Generate content requests per day limit exceeded on Free Tier.\\nPlease retry in 16h12m27.783908591s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'retryDelay': '58347s'}]}}")


class _ResourceExhaustedError(Exception):     # ADK wraps 429s in a class like this (no .code attribute)
    pass


MINUTE = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                            "message": "Quota exceeded for metric: generate_content_free_tier_requests. Please retry in 16.2s.",
                                            "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                                        {"retryDelay": "16s"}]}}, None)
DAILY_REAL_CLASS = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                      "message": "Generate content requests per day limit exceeded on Free Tier.",
                                                      "details": [{"retryDelay": "58347s"}]}}, None)


def test_daily_quota_is_recognised_in_every_shape():
    assert is_daily_quota_error(DAILY_TEXT)
    assert not is_transient_error(_ResourceExhaustedError(DAILY_TEXT))
    assert not is_transient_error(DAILY_REAL_CLASS)           # code 429, but a 16-hour retry delay


def test_per_minute_limit_and_busy_server_are_still_waited_out():
    assert is_transient_error(MINUTE)
    assert is_transient_error(errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}}, None))


def test_describe_model_error():
    msg = describe_model_error(_ResourceExhaustedError(DAILY_TEXT))
    assert "daily" in msg and "16h 12m" in msg and "mitigate" not in msg and len(msg) < 120
    assert describe_model_error(RuntimeError("503 UNAVAILABLE")) == "RuntimeError: 503 UNAVAILABLE"


def test_quiet_adk_logs_sets_the_level():
    logger = logging.getLogger("google_adk")
    old = logger.level
    try:
        quiet_adk_logs()
        assert logger.level == logging.CRITICAL
    finally:
        logger.setLevel(old)


# ---------- the loop and the dashboard ----------
PROFILE = {"dataset": {"rows": 4, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}
BAR = {"type": "bar", "title": "Revenue by region", "dimension": "region", "measure": "revenue", "agg": "sum"}
HIST = {"type": "histogram", "title": "Revenue distribution", "measure": "revenue"}
CHART = ChartSpec(**BAR)


class ScriptedLlm(BaseLlm):
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
def doubles(monkeypatch):
    import agents.chart_agent as chart_agent
    import agents.planner as planner
    text = lambda p: "\n".join(f"- {c['name']} ({c['role']})" for c in p["columns"])
    monkeypatch.setattr(chart_agent, "profile_to_text", text)
    monkeypatch.setattr(planner, "profile_to_text", text)
    monkeypatch.setattr(dash, "build_chart", lambda df, spec: px.bar(df, x="region", y="revenue", title="rule-based"))
    monkeypatch.setattr(dash, "suggest_charts", lambda profile, max_charts=8: [
        {"type": "bar", "dimension": "region", "measure": "revenue", "agg": "sum"}])


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South"], "revenue": [1.0, 2.0]})


def sleeper():
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
    return sleeps, fake_sleep


def test_loop_does_not_wait_for_the_daily_quota(df):
    sleeps, fake_sleep = sleeper()
    llm = ScriptedLlm(replies=[_ResourceExhaustedError(DAILY_TEXT)])
    res = make_chart_healing(df, PROFILE, CHART, model=llm, sleep=fake_sleep)
    assert not res.ok and llm.calls == 1 and sleeps == []
    assert res.final.model_error and not res.final.transient and "daily" in res.message


def test_planner_hitting_the_daily_quota_falls_back_at_once(df):
    sleeps, fake_sleep = sleeper()
    llm = ScriptedLlm(replies=[_ResourceExhaustedError(DAILY_TEXT)])
    d = build_dashboard(df, PROFILE, model=llm, sleep=fake_sleep)
    assert sleeps == [] and llm.calls == 1 and d.mode == "rule-based (planner failed)"
    assert any("daily" in n for n in d.notes) and len(d.ok_charts) == 1


def test_quota_running_out_mid_dashboard_stops_the_ai_without_waiting(df):
    sleeps, fake_sleep = sleeper()
    plan = json.dumps({"title": "T", "summary": "s", "kpis": [], "charts": [BAR, HIST]})
    llm = ScriptedLlm(replies=[plan, _ResourceExhaustedError(DAILY_TEXT)])
    d = build_dashboard(df, PROFILE, model=llm, sleep=fake_sleep)
    assert sleeps == [] and llm.calls == 2
    assert [c.source for c in d.charts] == ["rule-based", "rule-based"]
    assert "daily" in d.charts[0].ai_message and any("unavailable" in n for n in d.notes)
