"""Plan -> human approval -> build, as an ADK 2.x Workflow (Day 12).

   START -> plan_step -> approval_step --approved--> build_step
                              \\--cancelled--> cancel_step

The approval step PAUSES the workflow (ADK RequestInput) until a person answers. The DataFrame, the
models and the finished Dashboard live in Python closures: they never enter the workflow's events or
session state, so no raw data is ever written to an event log. Use ApprovalSession:

    s = ApprovalSession(df, profile, planner_model=..., chart_model=...)
    proposal = s.start()                                  # runs the planner, then pauses
    dashboard = s.decide(ApprovalDecision(drop=[2]))      # resumes: builds the approved charts
"""
import asyncio
from typing import Any, Optional

import pandas as pd
from google.adk import Context, Event, Workflow
from google.adk.events import EventActions, RequestInput
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, node
from google.genai import types
from pydantic import BaseModel, Field

try:   # a private ADK helper; the fallback below builds the identical message by hand
    from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
except ImportError:   # pragma: no cover
    create_request_input_response = None

from agents.dashboard import Dashboard, Progress, build_dashboard_async, make_plan_async
from agents.planner import DEFAULT_MODEL
from agents.resilience import DEFAULT_DELAYS
from tools.plan_schema import DashboardPlan

INTERRUPT_ID = "approval"
APP_NAME = "dashboard_approval"
USER_ID = "local_user"


class Proposal(BaseModel):
    """What the person is asked to approve."""
    plan: DashboardPlan
    mode: str = "ai"                          # "ai", "rule-based" or "rule-based (planner failed)"
    notes: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    """The person's answer. Chart numbers start at 1, as shown to the person."""
    approve: bool = True                      # False = cancel everything
    drop: list[int] = Field(default_factory=list)
    title: Optional[str] = None               # optional new dashboard title


def validate_decision(decision: ApprovalDecision, proposal: Proposal) -> None:
    """Raises ValueError with a readable message when the decision cannot be applied."""
    if not decision.approve:
        return
    n = len(proposal.plan.charts)
    bad = sorted({d for d in decision.drop if not 1 <= d <= n})
    if bad:
        raise ValueError(f"There is no chart number {', '.join(map(str, bad))}: choose from 1 to {n}.")
    if len(set(decision.drop)) >= n:
        raise ValueError("That would drop every chart. Cancel instead (approve=False).")
    if decision.title is not None and not decision.title.strip():
        raise ValueError("The title cannot be empty.")


def apply_decision(proposal: Proposal, decision: ApprovalDecision) -> Proposal:
    """The proposal as approved: dropped charts removed, new title applied."""
    drop = set(decision.drop)
    charts = [c for i, c in enumerate(proposal.plan.charts, 1) if i not in drop]
    update: dict[str, Any] = {"charts": charts}
    if decision.title:
        update["title"] = decision.title.strip()
    return proposal.model_copy(update={"plan": proposal.plan.model_copy(update=update)})


def _response_part(interrupt_id: str, response: dict) -> types.Part:
    if create_request_input_response is not None:
        return create_request_input_response(interrupt_id, response)
    return types.Part(function_response=types.FunctionResponse(          # pragma: no cover
        id=interrupt_id, name="adk_request_input", response=response))


def build_workflow(df: pd.DataFrame, profile: dict, box: dict, *, model=DEFAULT_MODEL, planner_model=None,
                   chart_model=None, use_ai: bool = True, max_charts: Optional[int] = None, max_attempts: int = 3,
                   progress: Optional[Progress] = None, delays=DEFAULT_DELAYS, sleep=asyncio.sleep,
                   auto_approve: bool = False) -> Workflow:
    """The graph. `box` is a plain dict that carries the non-serialisable results out of the workflow."""

    def emit(stage: str, done: int, total: int, message: str):
        if progress:
            try:
                progress(stage, done, total, message)
            except Exception:
                pass

    def on_wait(n: int, seconds: float, reason: str):
        emit("waiting", 0, 0, f"Google is busy. Waiting {seconds:.0f}s before retry {n}. ({reason[:80]})")

    async def plan_step(node_input: str):
        notes: list[str] = []
        plan, mode = await make_plan_async(profile, model, use_ai=use_ai, max_charts=max_charts,
                                           planner_model=planner_model, delays=delays, sleep=sleep,
                                           emit=emit, on_wait=on_wait, notes=notes)
        if max_charts:
            plan = plan.model_copy(update={"charts": plan.charts[:max_charts]})
        proposal = Proposal(plan=plan, mode=mode, notes=notes)
        box["proposal"] = proposal
        return proposal.model_dump(mode="json")

    @node(rerun_on_resume=True)
    async def approval_step(ctx: Context, node_input: Proposal):
        if auto_approve:
            yield Event(output=node_input.model_dump(mode="json"), actions=EventActions(route="approved"))
            return
        answer = ctx.resume_inputs.get(INTERRUPT_ID)
        if answer is None:
            yield RequestInput(interrupt_id=INTERRUPT_ID, payload=node_input.model_dump(mode="json"),
                               message="Review the planned dashboard: drop charts you do not want, or cancel.",
                               response_schema=ApprovalDecision)
            return
        decision = ApprovalDecision.model_validate({k: v for k, v in answer.items() if v is not None})
        approved = apply_decision(node_input, decision) if decision.approve else node_input
        if not decision.approve or not approved.plan.charts:
            yield Event(output={"cancelled": True}, actions=EventActions(route="cancelled"))
            return
        yield Event(output=approved.model_dump(mode="json"), actions=EventActions(route="approved"))

    async def build_step(node_input: Proposal):
        box["approved"] = node_input
        dashboard = await build_dashboard_async(
            df, profile, model, use_ai=use_ai, max_attempts=max_attempts, progress=progress, delays=delays,
            sleep=sleep, planner_model=planner_model, chart_model=chart_model,
            plan_override=node_input.plan, plan_mode=node_input.mode, plan_notes=node_input.notes)
        box["dashboard"] = dashboard
        return {"charts_shown": len(dashboard.ok_charts), "charts_failed": len(dashboard.failed_charts)}

    async def cancel_step(node_input: dict):
        box["cancelled"] = True
        return {"cancelled": True}

    return Workflow(name="dashboard_workflow",
                    edges=[(START, plan_step, approval_step, {"approved": build_step, "cancelled": cancel_step})])


class ApprovalSession:
    """One dashboard, from planning to the approved build. Works from sync code (start / decide) and
    async code (start_async / decide_async); the two may even be mixed between calls.
    state: "new" -> "awaiting" -> "done" or "cancelled"."""

    def __init__(self, df: pd.DataFrame, profile: dict, model=DEFAULT_MODEL, *, auto_approve: bool = False,
                 progress: Optional[Progress] = None, **kw):
        self._box: dict[str, Any] = {}
        self._progress = progress
        self._auto = auto_approve
        self._workflow = build_workflow(df, profile, self._box, model=model, progress=progress,
                                        auto_approve=auto_approve, **kw)
        self._runner = InMemoryRunner(node=self._workflow, app_name=APP_NAME)
        self._session_id: Optional[str] = None
        self._interrupt_id: Optional[str] = None
        self.state = "new"

    # ---- results ----
    @property
    def proposal(self) -> Optional[Proposal]:
        return self._box.get("proposal")

    @property
    def dashboard(self) -> Optional[Dashboard]:
        return self._box.get("dashboard")

    # ---- driving the workflow ----
    async def _run(self, message: types.Content) -> None:
        async for ev in self._runner.run_async(user_id=USER_ID, session_id=self._session_id, new_message=message):
            if ev.long_running_tool_ids:
                self._interrupt_id = sorted(ev.long_running_tool_ids)[0]

    async def start_async(self) -> Proposal:
        """Runs the planner, then pauses for approval. Returns what the person must approve."""
        if self.state != "new":
            raise RuntimeError(f"This session was already started (state: {self.state}).")
        session = await self._runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        self._session_id = session.id
        await self._run(types.Content(role="user", parts=[types.Part(text="Plan the dashboard.")]))
        if self.proposal is None:
            raise RuntimeError("The workflow ended without a proposal.")
        if self._auto:
            self.state = "done" if self.dashboard is not None else "cancelled"
        else:
            self.state = "awaiting"
            if self._progress:
                try:
                    self._progress("approval", 0, 0, "Waiting for your approval ...")
                except Exception:
                    pass
        return self.proposal

    async def decide_async(self, decision: ApprovalDecision) -> Optional[Dashboard]:
        """Resumes the workflow with the person's answer. Returns the dashboard, or None if cancelled.
        A decision that does not fit the proposal raises ValueError and leaves the session waiting."""
        if self.state != "awaiting":
            raise RuntimeError(f"Nothing is waiting for a decision (state: {self.state}).")
        validate_decision(decision, self.proposal)
        part = _response_part(self._interrupt_id or INTERRUPT_ID, decision.model_dump())
        await self._run(types.Content(role="user", parts=[part]))
        if self.dashboard is not None:
            self.state = "done"
        elif self._box.get("cancelled"):
            self.state = "cancelled"
        else:
            raise RuntimeError("The workflow ended without a result.")
        return self.dashboard

    def start(self) -> Proposal:
        return asyncio.run(self.start_async())

    def decide(self, decision: ApprovalDecision) -> Optional[Dashboard]:
        return asyncio.run(self.decide_async(decision))

    # ---- inspection ----
    async def event_log_async(self) -> list[str]:
        """Every workflow event as JSON. It never contains your data rows (useful for audits and tests)."""
        s = await self._runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID,
                                                           session_id=self._session_id)
        return [e.model_dump_json() for e in s.events] + [str(s.state)]

    def event_log(self) -> list[str]:
        return asyncio.run(self.event_log_async())
