"""Model chains and the model-role factory. Fake models only."""
import asyncio

import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agents.llm_cache import CacheMiss, CachingLlm
from agents.llm_factory import MIN_INTERVAL, ROLE_MODELS, make_llm
from agents.llm_fallback import FallbackLlm


class Inner(BaseLlm):
    """Replies with each item of `replies` in turn (a string, or an exception to raise)."""
    model: str = "inner"
    replies: list = []
    calls: int = 0
    seen_models: list = []

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        self.seen_models.append(llm_request.model)
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=item)]))


def req(text="hi", system="sys", model="label"):
    return LlmRequest(model=model, contents=[types.Content(role="user", parts=[types.Part(text=text)])],
                      config=types.GenerateContentConfig(system_instruction=system))


def ask(llm, request):
    async def go():
        return [r async for r in llm.generate_content_async(request)]
    return asyncio.run(go())


def text_of(responses):
    return "".join(p.text for p in responses[0].content.parts)


DAILY = RuntimeError("429 RESOURCE_EXHAUSTED. Generate content requests per day limit exceeded on Free Tier. "
                     "Please retry in 16h12m27s. 'retryDelay': '58347s'")


def chain_of(*pairs):
    """pairs of (model name, replies) -> FallbackLlm over plain fake models + the fakes."""
    fakes = [Inner(model=name, replies=list(replies), seen_models=[]) for name, replies in pairs]
    return fakes, FallbackLlm(chain=fakes)


def test_first_model_answers_and_the_second_is_never_touched():
    (a, b), llm = chain_of(("A", ["from A"]), ("B", ["from B"]))
    assert text_of(ask(llm, req())) == "from A" and a.calls == 1 and b.calls == 0


def test_daily_quota_moves_on_to_the_next_model_and_remembers_it():
    (a, b), llm = chain_of(("A", [DAILY]), ("B", ["b1", "b2"]))
    assert text_of(ask(llm, req(text="q1"))) == "b1"
    assert text_of(ask(llm, req(text="q2"))) == "b2"
    assert a.calls == 1 and b.calls == 2                     # A was only tried once
    assert llm.exhausted == {"A"} and "daily quota used up: A" in llm.summary()


def test_each_model_receives_its_own_name():
    (a, b), llm = chain_of(("A", [DAILY]), ("B", ["ok"]))
    ask(llm, req(model="label"))
    assert a.seen_models == ["A"] and b.seen_models == ["B"]


def test_other_errors_are_raised_not_skipped():
    (a, b), llm = chain_of(("A", [RuntimeError("401 API key not valid")]), ("B", ["never"]))
    with pytest.raises(RuntimeError, match="401"):
        ask(llm, req())
    assert b.calls == 0


def test_a_busy_server_is_left_to_the_waiting_layer():
    (a, b), llm = chain_of(("A", [RuntimeError("503 UNAVAILABLE high demand")]), ("B", ["never"]))
    with pytest.raises(RuntimeError, match="503"):
        ask(llm, req())
    assert b.calls == 0 and llm.exhausted == set()


def test_when_every_model_is_used_up_the_daily_error_is_raised():
    (a, b), llm = chain_of(("A", [DAILY]), ("B", [DAILY]))
    with pytest.raises(RuntimeError, match="per day"):
        ask(llm, req())
    with pytest.raises(RuntimeError):                         # now nothing is even tried
        ask(llm, req())
    assert a.calls == 1 and b.calls == 1


def test_replay_looks_in_every_models_cache(tmp_path):
    path = str(tmp_path / "c.json")
    # the answer was recorded while model B was in use (A's daily quota was gone)
    rec_b = CachingLlm(model="B", inner=Inner(model="B", replies=["recorded by B"], seen_models=[]), path=path)
    ask(rec_b, req())
    a = CachingLlm(model="A", inner=None, mode="replay", path=path)
    b = CachingLlm(model="B", inner=None, mode="replay", path=path)
    llm = FallbackLlm(chain=[a, b])
    assert text_of(ask(llm, req())) == "recorded by B"
    with pytest.raises(CacheMiss):
        ask(llm, req(text="never recorded"))


# ---------- factory ----------
def test_factory_builds_the_configured_chains_without_a_key(tmp_path):
    planner = make_llm("planner", mode="use", path=tmp_path / "c.json")
    chart = make_llm("chart", mode="use", path=tmp_path / "c.json")
    assert [m.model for m in planner.chain] == ROLE_MODELS["planner"]
    assert [m.model for m in chart.chain] == ROLE_MODELS["chart"]
    assert chart.chain[0].min_interval == MIN_INTERVAL["gemini-3.5-flash-lite"]
    assert planner.chain[0].min_interval == MIN_INTERVAL["gemini-3.6-flash"]
    assert planner.chain[0].inner is not None


def test_factory_replay_mode_needs_no_model_and_single_model_is_not_wrapped(tmp_path):
    replay = make_llm("chart", mode="replay", path=tmp_path / "c.json")
    assert all(m.inner is None for m in replay.chain)
    single = make_llm("chart", mode="replay", path=tmp_path / "c.json", models=["only-one"])
    assert isinstance(single, CachingLlm) and single.model == "only-one"


def test_factory_rejects_an_unknown_role_and_reads_the_environment(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="unknown role"):
        make_llm("poet")
    monkeypatch.setenv("LLM_CACHE_MODE", "replay")
    assert make_llm("chat", path=tmp_path / "c.json").chain[0].mode == "replay"


def test_dashboard_can_use_a_different_model_for_the_planner_and_for_the_charts(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import agents.chart_agent as chart_agent
    import agents.planner as planner
    from agents.dashboard import build_dashboard
    text = lambda p: "\n".join(f"- {c['name']} ({c['role']})" for c in p["columns"])
    monkeypatch.setattr(chart_agent, "profile_to_text", text)
    monkeypatch.setattr(planner, "profile_to_text", text)
    profile = {"dataset": {"rows": 2, "columns": 2}, "columns": [
        {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}
    df = pd.DataFrame({"region": ["N", "S"], "revenue": [1.0, 2.0]})
    plan = json.dumps({"title": "T", "summary": "", "kpis": [], "charts": [
        {"type": "bar", "title": "By region", "dimension": "region", "measure": "revenue", "agg": "sum"}]})
    code = ("```python\nfig = px.bar(df.groupby('region', as_index=False)['revenue'].sum(), "
            "x='region', y='revenue', title='By region')\n```")
    planner_model = Inner(model="planner-model", replies=[plan], seen_models=[])
    chart_model = Inner(model="chart-model", replies=[code], seen_models=[])
    d = build_dashboard(df, profile, planner_model=planner_model, chart_model=chart_model)
    assert planner_model.calls == 1 and chart_model.calls == 1 and d.mode == "ai"
