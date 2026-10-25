"""Planner agent (ADK 2.x). Sees ONLY the text profile, never raw data."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import ValidationError

from tools.charts import CHART_RULES
from tools.plan_schema import DashboardPlan, validate_plan_against_profile
from tools.profiler import profile_to_text

DEFAULT_MODEL = "gemini-3.5-flash"

INSTRUCTION = """You are a data analyst planning a dashboard for one dataset.
You see ONLY the profile below, never the raw rows. Use column names EXACTLY as written.

DATASET PROFILE
{profile_text}

CHART RULES
""" + CHART_RULES.replace("{", "{{").replace("}", "}}") + """

Return a dashboard plan: a short title, a one or two sentence summary, up to 4 KPIs
and 4 to 8 charts. Each chart needs a clear title and a one-sentence reason (why).
Only use columns that exist. Respect each column's role: time = datetime column,
dimension = categorical, measure/x/y/measures = numeric. Do not repeat the same chart.
{feedback}"""


@dataclass
class PlanResult:
    ok: bool
    plan: Optional[DashboardPlan] = None
    problems: list[str] = field(default_factory=list)
    raw: str = ""
    attempts: int = 1


def build_planner(model=DEFAULT_MODEL) -> Agent:
    return Agent(name="planner", model=model, instruction=INSTRUCTION,
                 output_schema=DashboardPlan, output_key="plan")


async def _run_once(model, profile_text: str, feedback: str) -> tuple[Optional[dict], str]:
    agent = build_planner(model)
    runner = InMemoryRunner(agent=agent, app_name="planner")
    sess = await runner.session_service.create_session(
        app_name="planner", user_id="u",
        state={"profile_text": profile_text, "feedback": feedback})
    msg = types.Content(role="user", parts=[types.Part(text="Plan the dashboard.")])
    raw = ""
    try:
        async for ev in runner.run_async(user_id="u", session_id=sess.id, new_message=msg):
            if ev.content and ev.content.parts and ev.is_final_response():
                raw = "".join(p.text or "" for p in ev.content.parts)
    except ValidationError as e:  # ADK raises this when the model output breaks the schema
        return None, "Model output did not match the plan format: " + "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5]) + f" | raw: {raw[:200]}"
    s = await runner.session_service.get_session(app_name="planner", user_id="u", session_id=sess.id)
    return s.state.get("plan"), raw


async def plan_dashboard_async(profile: dict, model=DEFAULT_MODEL, max_attempts: int = 2) -> PlanResult:
    text = profile_to_text(profile)
    feedback, last = "", PlanResult(ok=False)
    for attempt in range(1, max_attempts + 1):
        data, raw = await _run_once(model, text, feedback)
        if data is None:
            last = PlanResult(False, None, [raw], raw, attempt)
        else:
            try:
                plan = DashboardPlan.model_validate(data)
            except ValidationError as e:
                last = PlanResult(False, None, [f"{err['loc']}: {err['msg']}" for err in e.errors()[:5]], raw, attempt)
            else:
                probs = validate_plan_against_profile(plan, profile)
                last = PlanResult(not probs, plan if not probs else None, probs, raw, attempt)
                if not probs:
                    return last
        feedback = ("\nYOUR PREVIOUS ANSWER WAS REJECTED. Fix these problems and answer again:\n- "
                    + "\n- ".join(last.problems))
    return last


def plan_dashboard(profile: dict, model=DEFAULT_MODEL, max_attempts: int = 2) -> PlanResult:
    return asyncio.run(plan_dashboard_async(profile, model, max_attempts))
