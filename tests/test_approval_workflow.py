"""Plan -> approval -> build workflow. Fake models, no Gemini calls, no real waiting."""
import asyncio
import json

import pandas as pd
import plotly.express as px
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import agents.approval_workflow as aw
import agents.dashboard as dash
from agents.approval_workflow import (ApprovalDecision, ApprovalSession, Proposal, apply_decision,
                                      validate_decision)
from agents.llm_cache import CachingLlm

PROFILE = {"dataset": {"rows": 4, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}

BAR = {"type": "bar", "title": "Revenue by region", "dimension": "region", "measure": "revenue", "agg": "sum"}
HIST = {"type": "histogram", "title": "Revenue distribution", "measure": "revenue"}
BOX = {"type": "box", "title": "Revenue spread by region", "dimension": "region", "measure": "revenue"}
GOOD_BAR = ("```python\nd = df.groupby('region', as_index=False)['revenue'].sum()\n"
            "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')\n```")
GOOD_HIST = "```python\nfig = px.histogram(df, x='revenue', title='Revenue distribution')\n```"
GOOD_BOX = "```python\nfig = px.box(df, x='region', y='revenue', title='Revenue spread by region')\n```"
SECRET = "SECRET-ROW-VALUE-123"


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


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    import agents.chart_agent as chart_agent
    import agents.planner as planner
    text = lambda p: "\n".join(f"- {c['name']} ({c['role']})" for c in p["columns"])
    monkeypatch.setattr(chart_agent, "profile_to_text", text)
    monkeypatch.setattr(planner, "profile_to_text", text)


@pytest.fixture
def fake_tools(monkeypatch):
    monkeypatch.setattr(dash, "build_chart", lambda df, spec: px.bar(df, x="region", y="revenue", title="rule-based"))
    monkeypatch.setattr(dash, "suggest_charts", lambda profile, max_charts=8: [
        {"type": "bar", "dimension": "region", "measure": "revenue", "agg": "sum"},
        {"type": "histogram", "measure": "revenue"}])


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South", "North", SECRET], "revenue": [10.0, 20.0, 30.0, 40.0]})


async def _no_sleep(seconds):
    pass


def make(df, plan_replies, chart_replies, **kw):
    planner_llm, chart_llm = ScriptedLlm(replies=list(plan_replies)), ScriptedLlm(replies=list(chart_replies))
    s = ApprovalSession(df, PROFILE, planner_model=planner_llm, chart_model=chart_llm, sleep=_no_sleep, **kw)
    return s, planner_llm, chart_llm


THREE = [plan_json(BAR, HIST, BOX)]
THREE_CODES = [GOOD_BAR, GOOD_HIST, GOOD_BOX]


def test_start_pauses_with_a_proposal_and_builds_nothing(df):
    s, planner_llm, chart_llm = make(df, THREE, THREE_CODES)
    proposal = s.start()
    assert s.state == "awaiting" and s.dashboard is None
    assert proposal.mode == "ai" and [c.title for c in proposal.plan.charts] == [
        "Revenue by region", "Revenue distribution", "Revenue spread by region"]
    assert planner_llm.calls == 1 and chart_llm.calls == 0


def test_approving_everything_builds_all_charts_without_planning_again(df):
    s, planner_llm, chart_llm = make(df, THREE, THREE_CODES)
    s.start()
    d = s.decide(ApprovalDecision())
    assert s.state == "done" and d is s.dashboard and d.mode == "ai" and d.title == "Sales"
    assert len(d.ok_charts) == 3 and [c.source for c in d.charts] == ["ai"] * 3 and d.kpis[0].value == 100.0
    assert planner_llm.calls == 1 and chart_llm.calls == 3


def test_a_dropped_chart_is_never_built(df):
    s, planner_llm, chart_llm = make(df, THREE, [GOOD_BAR, GOOD_BOX])
    s.start()
    d = s.decide(ApprovalDecision(drop=[2]))
    assert [c.spec.title for c in d.charts] == ["Revenue by region", "Revenue spread by region"]
    assert chart_llm.calls == 2


def test_the_title_can_be_changed(df):
    s, _, _ = make(df, [plan_json(BAR)], [GOOD_BAR])
    s.start()
    assert s.decide(ApprovalDecision(title="  Board pack  ")).title == "Board pack"


def test_cancelling_builds_nothing(df):
    s, planner_llm, chart_llm = make(df, THREE, [])
    s.start()
    assert s.decide(ApprovalDecision(approve=False)) is None
    assert s.state == "cancelled" and s.dashboard is None and chart_llm.calls == 0 and planner_llm.calls == 1


def test_bad_decisions_are_refused_and_the_session_keeps_waiting(df):
    s, _, chart_llm = make(df, THREE, THREE_CODES)
    s.start()
    with pytest.raises(ValueError, match="no chart number 9"):
        s.decide(ApprovalDecision(drop=[9]))
    with pytest.raises(ValueError, match="drop every chart"):
        s.decide(ApprovalDecision(drop=[1, 2, 3]))
    with pytest.raises(ValueError, match="title"):
        s.decide(ApprovalDecision(title="   "))
    assert s.state == "awaiting" and chart_llm.calls == 0
    assert len(s.decide(ApprovalDecision()).ok_charts) == 3          # still works afterwards


def test_the_session_cannot_be_driven_out_of_order(df):
    s, _, _ = make(df, [plan_json(BAR)], [GOOD_BAR])
    with pytest.raises(RuntimeError, match="Nothing is waiting"):
        s.decide(ApprovalDecision())
    s.start()
    with pytest.raises(RuntimeError, match="already started"):
        s.start()
    s.decide(ApprovalDecision())
    with pytest.raises(RuntimeError, match="Nothing is waiting"):
        s.decide(ApprovalDecision())


def test_planner_failure_gives_a_rule_based_proposal_that_needs_no_model(df, fake_tools):
    s, planner_llm, chart_llm = make(df, ["garbage", "garbage"], [])
    proposal = s.start()
    assert proposal.mode == "rule-based (planner failed)" and len(proposal.plan.charts) == 2
    assert any("rule-based" in n for n in proposal.notes)
    d = s.decide(ApprovalDecision())
    assert [c.source for c in d.charts] == ["rule-based"] * 2 and chart_llm.calls == 0 and planner_llm.calls == 2
    assert any("rule-based" in n for n in d.notes)                    # the planning notes travel with the plan


def test_auto_approve_runs_straight_through_without_pausing(df):
    s, _, chart_llm = make(df, [plan_json(BAR)], [GOOD_BAR], auto_approve=True)
    proposal = s.start()
    assert s.state == "done" and s.dashboard is not None and len(proposal.plan.charts) == 1
    assert chart_llm.calls == 1
    with pytest.raises(RuntimeError):
        s.decide(ApprovalDecision())


def test_max_charts_limits_what_is_proposed(df):
    s, _, chart_llm = make(df, THREE, [GOOD_BAR, GOOD_HIST], max_charts=2)
    assert len(s.start().plan.charts) == 2
    assert len(s.decide(ApprovalDecision()).charts) == 2 and chart_llm.calls == 2


def test_no_raw_data_ever_enters_the_workflow_events(df):
    s, _, _ = make(df, THREE, THREE_CODES)
    s.start()
    s.decide(ApprovalDecision(drop=[3]))
    log = "".join(s.event_log())
    assert "adk_request_input" in log and "Revenue by region" in log          # the log is real and not empty
    assert SECRET not in log and "North" not in log


def test_the_async_api_works_and_can_be_mixed_with_the_sync_one(df):
    s, _, _ = make(df, [plan_json(BAR)], [GOOD_BAR])

    async def main():
        proposal = await s.start_async()
        return (await s.decide_async(ApprovalDecision())), proposal
    d, proposal = asyncio.run(main())
    assert d.mode == "ai" and proposal.plan.title == "Sales"

    s2, _, _ = make(df, [plan_json(BAR)], [GOOD_BAR])
    s2.start()                                                    # one event loop ...
    assert asyncio.run(s2.decide_async(ApprovalDecision())).ok_charts      # ... and another one later


def test_progress_reports_planning_approval_and_each_chart(df):
    events = []
    s, _, _ = make(df, [plan_json(BAR, HIST)], [GOOD_BAR, GOOD_HIST], progress=lambda *a: events.append(a))
    s.start()
    assert [e[0] for e in events] == ["planning", "approval"]
    s.decide(ApprovalDecision())
    assert [e[0] for e in events] == ["planning", "approval", "chart", "chart", "done"]


def test_decision_helpers():
    from tools.plan_schema import DashboardPlan
    plan = DashboardPlan.model_validate(json.loads(plan_json(BAR, HIST, BOX)))
    proposal = Proposal(plan=plan, mode="ai", notes=["n"])
    approved = apply_decision(proposal, ApprovalDecision(drop=[1, 3], title="New"))
    assert [c.title for c in approved.plan.charts] == ["Revenue distribution"] and approved.plan.title == "New"
    assert len(proposal.plan.charts) == 3 and proposal.plan.title == "Sales"            # the original is untouched
    validate_decision(ApprovalDecision(approve=False, drop=[99]), proposal)             # cancelling is always fine
    with pytest.raises(ValueError):
        validate_decision(ApprovalDecision(drop=[0]), proposal)                         # numbering starts at 1


def test_the_hand_built_resume_message_matches_adks_own_helper(monkeypatch):
    from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
    expected = create_request_input_response("approval", {"approve": True, "drop": [2]}).model_dump(exclude_none=True)
    monkeypatch.setattr(aw, "create_request_input_response", None)
    assert aw._response_part("approval", {"approve": True, "drop": [2]}).model_dump(exclude_none=True) == expected


def test_a_second_session_is_free_when_the_replies_are_cached(df, tmp_path):
    path = str(tmp_path / "cache.json")

    def session(planner_inner, chart_inner):
        return ApprovalSession(df, PROFILE, sleep=_no_sleep,
                               planner_model=CachingLlm(model="p", inner=planner_inner, path=path),
                               chart_model=CachingLlm(model="c", inner=chart_inner, path=path))
    p1, c1 = ScriptedLlm(replies=[plan_json(BAR, HIST)]), ScriptedLlm(replies=[GOOD_BAR, GOOD_HIST])
    s1 = session(p1, c1)
    s1.start()
    s1.decide(ApprovalDecision())
    assert p1.calls == 1 and c1.calls == 2

    p2, c2 = ScriptedLlm(replies=[]), ScriptedLlm(replies=[])
    s2 = session(p2, c2)
    s2.start()
    d = s2.decide(ApprovalDecision(drop=[2]))                    # even a different decision reuses the cached chart
    assert p2.calls == 0 and c2.calls == 0 and [c.source for c in d.charts] == ["ai"]
