"""Self-healing loop tests with a fake model (no Gemini calls). Uses your real sandbox."""
import pandas as pd
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.chart_loop import build_feedback, make_chart_healing, scrub_message, Attempt
from tools.plan_schema import ChartSpec

PROFILE = {"dataset": {"rows": 4, "columns": 2}, "columns": [
    {"name": "region", "role": "categorical"},
    {"name": "revenue", "role": "numeric"}]}

CHART = ChartSpec(type="bar", title="Revenue by region", why="Compare regions",
                  dimension="region", measure="revenue", agg="sum")

GOOD = ("d = df.groupby('region', as_index=False)['revenue'].sum()\n"
        "fig = px.bar(d, x='region', y='revenue', title='Revenue by region')")
BAD_COLUMN = "fig = px.bar(df, x='regoin', y='revenue')"


@pytest.fixture(autouse=True)
def simple_profile_text(monkeypatch):
    import agents.chart_agent as chart_agent
    monkeypatch.setattr(chart_agent, "profile_to_text", lambda p: "\n".join(
        f"- {c['name']} ({c['role']})" for c in p["columns"]))


def fence(code):
    return f"```python\n{code}\n```"


class FakeLlm(BaseLlm):
    model: str = "fake"
    replies: list = []
    seen: list = []
    explode: bool = False

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        if self.explode:
            raise RuntimeError("429 quota exceeded")
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.replies.pop(0))]))


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["North", "South", "North", "SECRET-ROW-VALUE-123"],
                         "revenue": [10.0, 20.0, 30.0, 40.0]})


def heal(df, replies, **kw):
    llm = FakeLlm(replies=list(replies), seen=[], explode=kw.pop("explode", False))
    return llm, make_chart_healing(df, PROFILE, CHART, model=llm, **kw)


def prompt(llm, i):
    return str(llm.seen[i].config.system_instruction)


def test_first_try_needs_no_retry(df):
    llm, res = heal(df, [fence(GOOD)])
    assert res.ok and res.attempts == 1 and len(llm.seen) == 1 and res.figure is not None
    assert "PREVIOUS ATTEMPT" not in prompt(llm, 0)


def test_recovers_after_a_runtime_error(df):
    llm, res = heal(df, [fence(BAD_COLUMN), fence(GOOD)])
    assert res.ok and res.attempts == 2 and res.code == GOOD
    p2 = prompt(llm, 1)
    assert "YOUR PREVIOUS ATTEMPT FAILED" in p2 and BAD_COLUMN in p2 and "regoin" in p2
    assert [a.ok for a in res.history] == [False, True]


def test_recovers_after_blocked_code(df):
    _, res = heal(df, [fence("import os\nos.system('ls')\nfig = px.bar(df, x='region', y='revenue')"), fence(GOOD)])
    assert res.ok and res.attempts == 2 and res.history[0].message.lower().startswith("code was blocked")


def test_recovers_after_reply_without_code(df):
    llm, res = heal(df, ["Sorry, I can't.", fence(GOOD)])
    assert res.ok and res.attempts == 2 and "contained no code block" in prompt(llm, 1)


def test_recovers_after_missing_fig(df):
    llm, res = heal(df, [fence("result = df['revenue'].sum()"), fence(GOOD)])
    assert res.ok and res.attempts == 2 and "fig" in prompt(llm, 1)


def test_gives_up_after_max_attempts(df):
    llm, res = heal(df, [fence(f"fig = px.bar(df, x='a{i}', y='revenue')") for i in range(3)])
    assert not res.ok and res.attempts == 3 and len(llm.seen) == 3
    assert res.message.startswith("Gave up after 3 attempts")
    assert "Earlier failed attempts" in prompt(llm, 2)


def test_max_attempts_is_respected(df):
    llm, res = heal(df, [fence(BAD_COLUMN), fence(GOOD)], max_attempts=1)
    assert not res.ok and res.attempts == 1 and len(llm.seen) == 1


def test_stops_when_the_model_repeats_itself(df):
    llm, res = heal(df, [fence(BAD_COLUMN), fence(BAD_COLUMN + "  "), fence(GOOD)])
    assert not res.ok and res.attempts == 2 and len(llm.seen) == 2 and "repeated" in res.message


def test_model_failure_is_not_retried(df):
    llm, res = heal(df, ["unused"], explode=True, delays=())
    assert not res.ok and res.attempts == 1 and len(llm.seen) == 1 and "429" in res.message


def test_data_values_never_reach_the_model(df):
    boom = "raise ValueError(\"bad value '\" + str(df['region'].iloc[3]) + \"' found\")"
    llm, res = heal(df, [fence(boom), fence(GOOD)])
    assert res.ok and res.attempts == 2
    assert "SECRET-ROW-VALUE-123" in res.history[0].message        # kept for YOUR logs
    p2 = prompt(llm, 1) + str(llm.seen[1].contents)
    assert "SECRET-ROW-VALUE-123" not in p2 and "<value>" in p2    # hidden from the model


def test_scrub_message_rules():
    df = pd.DataFrame({"region": ["O'Brien Ltd", "South"], "revenue": [1.0, 2.0]})
    code = "fig = px.bar(df, x='region', y='revenue')"
    # exact data values hidden (even with an apostrophe), unquoted too
    assert "O'Brien" not in scrub_message("bad value O'Brien Ltd in row 1", code, PROFILE, df)
    # quoted text that is neither a column nor in the code is hidden even if not in df
    assert scrub_message("can't convert '12,50 EUR' to float", code, PROFILE, df) == "can't convert <value> to float"
    # column names and the model's own literals are kept
    assert scrub_message("KeyError: 'region' and 'revenue'", code, PROFILE, df) == "KeyError: 'region' and 'revenue'"
    # a data value the model wrote itself is kept
    assert "South" in scrub_message("no rows for 'South'", "x = df[df.region == 'South']", PROFILE, df)
    # length cap
    assert len(scrub_message("x" * 2000, code, PROFILE, df)) == 500


def test_build_feedback_without_code():
    fb = build_feedback([Attempt(1, False, None, "The model's reply contained no code block.")], PROFILE, None)
    assert "no code block" in fb and "```python" not in fb.split("What went wrong")[0]
