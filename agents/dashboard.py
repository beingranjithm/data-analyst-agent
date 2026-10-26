"""Dashboard assembler (Day 11): plan -> charts -> validation -> one Dashboard object.

For every chart: AI route first (healing loop), then the rule-based chart builder as a safety net.
If the planner itself fails, the whole dashboard is built by the rule-based gallery. Nothing here
ever sends raw data to a model.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import ValidationError

from agents.chart_loop import MAX_ATTEMPTS, make_chart_healing_async
from agents.planner import DEFAULT_MODEL, plan_dashboard_async
from agents.resilience import DEFAULT_DELAYS, describe_model_error, with_backoff
from tools.charts import build_chart, suggest_charts
from tools.figure_checks import check_figure
from tools.plan_kpis import KpiValue, compute_plan_kpis
from tools.plan_schema import ChartSpec, DashboardPlan

Progress = Callable[[str, int, int, str], None]   # (stage, done, total, message)


@dataclass
class ChartOutcome:
    spec: ChartSpec
    ok: bool
    figure: Any = None
    source: str = ""                  # "ai" or "rule-based" ("" when the chart failed)
    attempts: int = 0                 # AI attempts used (0 = the AI route was not used)
    code: Optional[str] = None        # the code the AI wrote (for the "show code" view)
    message: str = ""                 # what happened (unscrubbed: for the UI and logs)
    ai_message: str = ""              # why the AI route failed, when it did
    warnings: list = field(default_factory=list)


@dataclass
class Dashboard:
    title: str
    summary: str
    rows: int
    mode: str                         # "ai", "rule-based" or "rule-based (planner failed)"
    kpis: list = field(default_factory=list)
    charts: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok_charts(self) -> list:
        return [c for c in self.charts if c.ok]

    @property
    def failed_charts(self) -> list:
        return [c for c in self.charts if not c.ok]


def _title_for(spec: dict) -> str:
    parts = [str(spec[k]) for k in ("measure", "dimension", "time", "x", "y") if spec.get(k)]
    parts += [str(m) for m in (spec.get("measures") or [])]
    return f"{str(spec.get('type', 'chart')).title()}: " + ", ".join(parts) if parts else "Chart"


def _gallery_plan(profile: dict, max_charts: Optional[int], notes: list) -> DashboardPlan:
    charts = []
    for s in suggest_charts(profile, max_charts=max_charts or 8):
        try:
            charts.append(ChartSpec(**{**s, "title": s.get("title") or _title_for(s)}))
        except ValidationError as e:
            notes.append(f"Skipped a suggested chart that did not fit the schema: {e.errors()[0]['msg']}")
    if not charts:
        raise ValueError("The rule-based gallery found no usable charts for this dataset.")
    return DashboardPlan(title="Dashboard", summary="Built by the rule-based chart gallery.", charts=charts)


def _finish(fig, chart: ChartSpec):
    """Check a figure. Returns (figure, blocking problems, warnings); fills in a missing title."""
    check = check_figure(fig)
    if check.ok and check.missing_title:
        fig.update_layout(title_text=chart.title)
    return fig, check.problems, check.warnings


async def make_plan_async(profile: dict, model=DEFAULT_MODEL, *, use_ai: bool = True,
                          max_charts: Optional[int] = None, planner_model=None, delays=DEFAULT_DELAYS,
                          sleep=asyncio.sleep, emit=None, on_wait=None, notes: Optional[list] = None):
    """Step 1 of a dashboard: the plan. Returns (plan, mode).
    Model problems never raise: the rule-based gallery takes over and the reason is added to `notes`."""
    notes = notes if notes is not None else []
    plan: Optional[DashboardPlan] = None
    mode = "rule-based"
    if use_ai:
        if emit:
            emit("planning", 0, 0, "Planning the dashboard ...")
        try:
            res = await with_backoff(lambda: plan_dashboard_async(profile, planner_model or model), delays=delays,
                                     sleep=sleep, on_wait=on_wait)
            if res.ok:
                plan, mode = res.plan, "ai"
            else:
                notes.append("The AI plan was rejected: " + "; ".join(res.problems)[:300])
        except Exception as e:
            notes.append(f"The planner could not run: {describe_model_error(e)}")
        if plan is None:
            mode = "rule-based (planner failed)"
            notes.append("Used the rule-based chart gallery instead of an AI plan.")
    if plan is None:
        plan = _gallery_plan(profile, max_charts, notes)
    return plan, mode


async def build_dashboard_async(df: pd.DataFrame, profile: dict, model=DEFAULT_MODEL, *, use_ai: bool = True,
                                max_charts: Optional[int] = None, max_attempts: int = MAX_ATTEMPTS,
                                progress: Optional[Progress] = None, delays=DEFAULT_DELAYS,
                                sleep=asyncio.sleep, planner_model=None, chart_model=None,
                                plan_override: Optional[DashboardPlan] = None, plan_mode: str = "ai",
                                plan_notes: Optional[list] = None) -> Dashboard:
    notes: list[str] = list(plan_notes or [])

    def emit(stage: str, done: int, total: int, message: str):
        if progress:
            try:
                progress(stage, done, total, message)
            except Exception:       # a broken progress display must never break the build
                pass

    def on_wait(n: int, seconds: float, reason: str):
        emit("waiting", 0, 0, f"Google is busy. Waiting {seconds:.0f}s before retry {n}. ({reason[:80]})")

    # ---- 1. the plan ----
    if plan_override is not None:             # a plan that a person already approved (see approval_workflow.py)
        plan, mode = plan_override, plan_mode
    else:
        plan, mode = await make_plan_async(profile, model, use_ai=use_ai, max_charts=max_charts,
                                           planner_model=planner_model, delays=delays, sleep=sleep,
                                           emit=emit, on_wait=on_wait, notes=notes)
    charts = list(plan.charts[:max_charts] if max_charts else plan.charts)

    # ---- 2. KPIs ----
    try:
        kpis = compute_plan_kpis(df, plan.kpis)
    except Exception as e:
        kpis = []
        notes.append(f"KPIs could not be computed: {type(e).__name__}")

    # ---- 3. the charts, one by one (free-tier friendly) ----
    ai_available = use_ai and mode == "ai"
    outcomes: list[ChartOutcome] = []
    for i, chart in enumerate(charts, 1):
        emit("chart", i - 1, len(charts), f"Building chart {i} of {len(charts)}: {chart.title}")
        ai_message, attempts, code = "", 0, None
        if ai_available:
            heal = await make_chart_healing_async(df, profile, chart, chart_model or model, max_attempts,
                                                  delays=delays, sleep=sleep, on_wait=on_wait)
            attempts, code = heal.attempts, heal.code
            if heal.ok:
                fig, problems, warns = _finish(heal.figure, chart)
                if not problems:
                    outcomes.append(ChartOutcome(chart, True, fig, "ai", attempts, code,
                                                 f"Built by the AI on attempt {attempts}.", "", warns))
                    continue
                ai_message = "The AI chart was rejected: " + "; ".join(problems)
            else:
                ai_message = heal.message
                if heal.final is not None and heal.final.model_error:
                    ai_available = False      # the model service is down: stop waiting on it for later charts
                    notes.append("The AI model was unavailable, so the remaining charts use the rule-based builder.")
        try:
            fig, problems, warns = _finish(build_chart(df, chart.to_spec()), chart)
            error = "; ".join(problems)
        except Exception as e:
            fig, warns, error = None, [], f"{type(e).__name__}: {str(e)[:200]}"
        if fig is not None and not error:
            msg = "Built by the rule-based chart builder." + (f" (AI route: {ai_message})" if ai_message else "")
            outcomes.append(ChartOutcome(chart, True, fig, "rule-based", attempts, code, msg, ai_message, warns))
        else:
            msg = (f"AI: {ai_message} | " if ai_message else "") + f"Rule-based: {error}"
            outcomes.append(ChartOutcome(chart, False, None, "", attempts, code, msg, ai_message))
    emit("done", len(charts), len(charts), "Dashboard ready.")
    return Dashboard(plan.title, plan.summary, len(df), mode, kpis, outcomes, notes)


def build_dashboard(df: pd.DataFrame, profile: dict, model=DEFAULT_MODEL, **kw) -> Dashboard:
    return asyncio.run(build_dashboard_async(df, profile, model, **kw))
