"""Planner tests with a fake model (no Gemini calls, no network)."""
import json
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ValidationError

from agents.planner import plan_dashboard, plan_dashboard_async
from tools.plan_schema import DashboardPlan, ChartSpec, validate_plan_against_profile

PROFILE = {
    "dataset": {"rows": 600},
    "columns": [
        {"name": "order_date", "role": "datetime"},
        {"name": "region", "role": "categorical"},
        {"name": "category", "role": "categorical"},
        {"name": "units", "role": "numeric"},
        {"name": "revenue", "role": "numeric"},
        {"name": "discount", "role": "numeric"},
    ],
}


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    """Keep these tests independent of profile_to_text (it is tested with the real profile elsewhere)."""
    import agents.planner as planner
    monkeypatch.setattr(planner, "profile_to_text", lambda p: "\n".join(
        f"- {c['name']} ({c['role']})" for c in p["columns"]))


class FakeLlm(BaseLlm):
    model: str = "fake"
    replies: list = []
    seen: list = []

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


def good_plan(**over):
    d = {"title": "Sales overview", "summary": "s",
         "kpis": [{"label": "Revenue", "measure": "revenue", "agg": "sum"}],
         "charts": [{"type": "line", "title": "Revenue over time", "time": "order_date", "measure": "revenue", "agg": "sum"},
                    {"type": "bar", "title": "By region", "dimension": "region", "measure": "revenue"}]}
    d.update(over)
    return d


def run(replies, **kw):
    llm = FakeLlm(replies=[json.dumps(r) if isinstance(r, dict) else r for r in replies], seen=[])
    return llm, plan_dashboard(PROFILE, model=llm, **kw)


def test_valid_plan_first_try():
    llm, res = run([good_plan()])
    assert res.ok and res.attempts == 1 and len(res.plan.charts) == 2
    assert res.plan.charts[0].to_spec() == {"type": "line", "time": "order_date", "measure": "revenue", "agg": "sum"}


def test_model_sees_profile_not_data():
    llm, _ = run([good_plan()])
    sysmsg = str(llm.seen[0].config.system_instruction)
    assert "order_date (datetime)" in sysmsg and "{profile_text}" not in sysmsg and "{feedback}" not in sysmsg


def test_unknown_column_retries_with_feedback_then_succeeds():
    bad = good_plan(charts=[{"type": "bar", "title": "x", "dimension": "regoin", "measure": "revenue"}])
    llm, res = run([bad, good_plan()])
    assert res.ok and res.attempts == 2
    assert "regoin" in str(llm.seen[1].config.system_instruction)


def test_wrong_role_rejected():
    bad = good_plan(charts=[{"type": "bar", "title": "x", "dimension": "revenue", "measure": "units"}])
    _, res = run([bad, bad])
    assert not res.ok and res.plan is None and "role" in res.problems[0]


def test_malformed_json_does_not_crash():
    _, res = run(["not json at all", "still not json"])
    assert not res.ok and res.attempts == 2 and "plan format" in res.problems[0]


def test_missing_required_field_rejected():
    bad = good_plan(charts=[{"type": "line", "title": "x", "measure": "revenue"}])
    _, res = run([bad, good_plan()])
    assert res.ok and res.attempts == 2


def test_max_attempts_one():
    _, res = run(["garbage"], max_attempts=1)
    assert not res.ok and res.attempts == 1


def test_schema_rejects_bad_type_and_empty_charts():
    with pytest.raises(ValidationError):
        ChartSpec(type="pie", title="x")
    with pytest.raises(ValidationError):
        DashboardPlan(title="x", charts=[])


def test_heatmap_needs_two_measures():
    with pytest.raises(ValidationError):
        DashboardPlan(title="x", charts=[{"type": "heatmap", "title": "h", "measures": ["revenue"]}])


def test_count_kpi_allows_any_column():
    plan = DashboardPlan.model_validate(good_plan(kpis=[{"label": "Regions", "measure": "region", "agg": "count"}]))
    assert validate_plan_against_profile(plan, PROFILE) == []
