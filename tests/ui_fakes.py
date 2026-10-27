"""Fake models and a session factory for the UI tests (no Gemini)."""
import json

from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

BAR = {"type": "bar", "title": "Revenue by region", "dimension": "region", "measure": "revenue", "agg": "sum",
       "why": "Compare regions."}
HIST = {"type": "histogram", "title": "Revenue distribution", "measure": "revenue", "why": "See the spread."}
BOX = {"type": "box", "title": "Revenue spread by region", "dimension": "region", "measure": "revenue"}
GOOD_BAR = ("```python\nd = df.groupby('region', as_index=False)['revenue'].sum()\n"
            "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')\n```")
GOOD_HIST = "```python\nfig = px.histogram(df, x='revenue', title='Revenue distribution')\n```"
GOOD_BOX = "```python\nfig = px.box(df, x='region', y='revenue', title='Revenue spread by region')\n```"


def plan_json(*charts):
    return json.dumps({"title": "Sales", "summary": "A summary.", "charts": list(charts),
                       "kpis": [{"label": "Revenue", "measure": "revenue", "agg": "sum"}]})


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


async def _no_sleep(seconds):
    pass


def make_factory(scenario="ok"):
    """scenario: "ok" | "planner_fails" | "factory_error"."""
    def factory(df, profile, settings, progress):
        from agents.approval_workflow import ApprovalSession
        if scenario == "factory_error":
            raise RuntimeError("boom")
        if scenario == "planner_fails":
            plan_replies, chart_replies = ["garbage", "garbage"], []
        else:
            plan_replies, chart_replies = [plan_json(BAR, HIST, BOX)], [GOOD_BAR, GOOD_HIST, GOOD_BOX]
        planner_llm, chart_llm = ScriptedLlm(replies=plan_replies), ScriptedLlm(replies=chart_replies)
        session = ApprovalSession(df, profile, planner_model=planner_llm, chart_model=chart_llm, progress=progress,
                                  sleep=_no_sleep, use_ai=settings["use_ai"], max_charts=settings["max_charts"])
        session.models = (planner_llm, chart_llm)
        return session
    return factory
