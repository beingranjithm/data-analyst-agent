"""Chart agent (ADK 2.x): writes Plotly code for ONE planned chart, then runs it in the sandbox.

The model sees ONLY the text profile and the chart spec, never raw rows.
Single shot on purpose: no retries yet (Day 10 adds the self-healing loop).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.resilience import describe_model_error, is_transient_error
from tools.plan_schema import ChartSpec
from tools.profiler import profile_to_text
from tools.sandbox import RunResult, run_code

DEFAULT_MODEL = "gemini-3.5-flash"

# NOTE: no literal curly braces in this text. ADK treats {name} as a state placeholder.
INSTRUCTION = """You write Plotly code for ONE chart in a dashboard.
You see only a profile of the dataset, never the rows.

DATASET PROFILE
{profile_text}

THE CHART TO BUILD
{chart_json}
{feedback}

ENVIRONMENT
- A pandas DataFrame called df holds the data. pd, np, px (plotly.express) and go
  (plotly.graph_objects) are already imported. Do not import anything.
- Assign the finished Plotly figure to a variable named fig.
- No files, no network, no classes. Do not use df.query or df.eval: use boolean masks.
- Do not modify df in place. Work on copies.

RULES
- Use column names exactly as written in the profile.
- Sum amounts (revenue, units, cost). Average rates, prices, discounts and scores.
- Do not assume a currency: use no currency symbol ($, EUR, GBP ...) unless the profile names one.
- Use the given chart title as the figure title and label the axes clearly.
- Show at most 15 categories (the largest). Sort bars from largest to smallest.
- If rows are left out because a needed column is missing, add a small annotation
  under the chart saying how many rows were excluded.
- Reply with ONE python code block and nothing else.
"""


@dataclass
class ChartResult:
    ok: bool
    message: str                      # short human-readable outcome
    code: Optional[str] = None        # the code the model wrote (None if none was found)
    run: Optional[RunResult] = None   # sandbox result (None if code never reached the sandbox)
    raw: str = ""                     # the model's raw reply
    model_error: bool = False         # True when the MODEL CALL failed (rate limit, network), not the code
    transient: bool = False           # True when waiting and asking again may work (busy server, per-minute limit)

    @property
    def figure(self):
        return self.run.figure if self.run is not None else None


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> Optional[str]:
    """Pull the python code out of a model reply. Returns None when there is no code."""
    if not text or "```" not in text:
        return None
    blocks = _FENCE.findall(text)
    if blocks:
        for lang, body in blocks:
            if lang.lower() in ("python", "py", "python3"):
                return body.strip("\n") or None
        return blocks[0][1].strip("\n") or None
    # opening fence but no closing fence (reply was cut off): take everything after it
    after = text.split("```", 1)[1]
    after = after.split("\n", 1)[1] if "\n" in after else ""
    return after.strip("\n") or None


def build_chart_agent(model=DEFAULT_MODEL) -> Agent:
    return Agent(name="chart_agent", model=model, instruction=INSTRUCTION)


async def _ask_model(model, profile_text: str, chart_json: str, feedback: str = "") -> str:
    runner = InMemoryRunner(agent=build_chart_agent(model), app_name="chart_agent")
    sess = await runner.session_service.create_session(
        app_name="chart_agent", user_id="u",
        state={"profile_text": profile_text, "chart_json": chart_json, "feedback": feedback})
    msg = types.Content(role="user", parts=[types.Part(text="Write the chart code.")])
    raw = ""
    async for ev in runner.run_async(user_id="u", session_id=sess.id, new_message=msg):
        if ev.content and ev.content.parts and ev.is_final_response():
            raw = "".join(p.text or "" for p in ev.content.parts)
    return raw


async def make_chart_async(df: pd.DataFrame, profile: dict, chart: ChartSpec,
                           model=DEFAULT_MODEL, feedback: str = "") -> ChartResult:
    chart_json = json.dumps({"title": chart.title, "why": chart.why, "spec": chart.to_spec()}, indent=2)
    profile_text = profile_to_text(profile)  # outside the try: a bug here must not look like a model failure
    try:
        raw = await _ask_model(model, profile_text, chart_json, feedback)
    except Exception as e:  # rate limit, network, bad key, ...
        return ChartResult(False, "Model call failed: " + describe_model_error(e), model_error=True,
                           transient=is_transient_error(e))

    code = extract_code(raw)
    if code is None:
        return ChartResult(False, "The model's reply contained no code block.", None, None, raw)

    from tools.hosting import sandbox_options
    run = run_code(code, df, **sandbox_options())          # the sandbox does the static safety check itself
    if not run.ok:
        return ChartResult(False, run.message(), code, run, raw)
    if run.figure is None:
        return ChartResult(False, "Code ran but did not create a Plotly figure named fig.", code, run, raw)
    return ChartResult(True, "Chart built.", code, run, raw)


def make_chart(df: pd.DataFrame, profile: dict, chart: ChartSpec, model=DEFAULT_MODEL) -> ChartResult:
    return asyncio.run(make_chart_async(df, profile, chart, model))
