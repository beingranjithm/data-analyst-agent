"""Day 18: the evaluation talks to the REAL ChatSession (fake model, real pandas code run by a stand-in runner).
A model that writes the right code scores correct; one that writes subtly wrong code is caught."""
import sys
from pathlib import Path

import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents.chat_agent as chat_agent  # noqa: E402
from agents.chat_agent import ChatSession  # noqa: E402
from calc_fakes import local_runner  # noqa: E402
from evals.chat_eval import FAIL, PASS, build_questions, run_eval, summarize  # noqa: E402
from test_chat_eval import make_df  # noqa: E402
from tools.profiler import profile_dataframe  # noqa: E402
from ui_fakes import _no_sleep  # noqa: E402

RIGHT = {
    "What is the total revenue?": "result = df['revenue'].sum()",
    "How many orders (rows) are in the data?": "result = len(df)",
    "What is the total revenue for the East region?": "result = df[df['region'] == 'East']['revenue'].sum()",
    "Which region has the highest total revenue, and what is that total?":
        "s = df.groupby('region')['revenue'].sum()\nresult = s.sort_values(ascending=False).head(1).reset_index()",
}
WRONG = {   # the kind of mistake a model really makes: mean for sum, a wrong filter, lowest instead of highest, a miscount
    "What is the total revenue?": "result = df['revenue'].mean()",
    "How many orders (rows) are in the data?": "result = len(df) - 1",
    "What is the total revenue for the East region?": "result = df[df['region'] == 'West']['revenue'].sum()",
    "Which region has the highest total revenue, and what is that total?":
        "s = df.groupby('region')['revenue'].sum()\nresult = s.sort_values().head(1).reset_index()",
}
TRAP = "What is the average customer age?"


class EchoModel(BaseLlm):
    """Writes the code in `codes` for a question, then repeats the RESULT back in a sentence."""
    model: str = "fake"
    codes: dict = {}
    calls: int = 0

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        last = "".join(p.text or "" for p in llm_request.contents[-1].parts)
        first = "".join(p.text or "" for p in llm_request.contents[0].parts)
        if last.startswith("RESULT of your calculation"):
            body = last.split("RESULT of your calculation:\n", 1)[1].split("\n\n", 1)[0]
            text = "The answer is " + " ".join(body.replace(",", " ").split())
        elif first in self.codes:
            text = f"```python\n{self.codes[first]}\n```"
        else:
            text = "The data has no customer age column, so I cannot say."
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


@pytest.fixture
def setup(monkeypatch):
    df = make_df()
    qs, _ = build_questions(df)
    wanted = [q for q in qs if q.text in RIGHT or q.text == TRAP]
    profile = profile_dataframe(df)
    monkeypatch.setattr(chat_agent, "profile_to_text", lambda p: "- columns: " + ", ".join(c["name"] for c in p["columns"]))
    return df, wanted, profile


def run(df, wanted, profile, codes):
    model = EchoModel(codes=dict(codes))
    chat = ChatSession(profile, model, tables=df, runner=local_runner, sleep=_no_sleep)

    def ask(text):
        chat.reset()
        return chat.ask(text)
    return run_eval(wanted, ask), model


def test_a_model_that_writes_the_right_code_is_scored_correct_through_the_real_chat(setup):
    df, wanted, profile = setup
    results, model = run(df, wanted, profile, RIGHT)
    assert len(wanted) == 5 and [r.status for r in results] == [PASS] * 5, [(r.question, r.reasons, r.answer) for r in results]
    assert [r.tries for r in results] == [1, 1, 1, 1, 0]                                                # the trap needs no calculation
    assert model.calls == 4 * 2 + 1


def test_a_model_that_writes_subtly_wrong_code_is_caught_through_the_real_chat(setup):
    df, wanted, profile = setup
    results, _ = run(df, wanted, profile, WRONG)
    status = {r.question: r.status for r in results}
    assert [status[q] for q in WRONG] == [FAIL] * 4
    assert status[TRAP] == PASS and summarize(results)["passed"] == 1                                   # only the honest "no such column"
