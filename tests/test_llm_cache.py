"""Record/replay cache + pacing. Fake models and a fake clock: no Gemini calls, no real waiting."""
import asyncio
import json

import pandas as pd
import pytest
from google.adk.models.base_llm import BaseLlm, LlmCapabilities
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel

import agents.llm_cache as llm_cache
from agents.llm_cache import CacheMiss, CachingLlm, request_key


class Inner(BaseLlm):
    """Replies with each item of `replies` in turn (a string, an LlmResponse, or an exception to raise)."""
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
        if isinstance(item, LlmResponse):
            yield item
        else:
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=item)]))


def req(text="hi", system="sys", model="label", schema=None):
    cfg = types.GenerateContentConfig(system_instruction=system, response_schema=schema,
                                      response_mime_type="application/json" if schema else None)
    return LlmRequest(model=model, contents=[types.Content(role="user", parts=[types.Part(text=text)])], config=cfg)


def ask(llm, request):
    async def go():
        return [r async for r in llm.generate_content_async(request)]
    return asyncio.run(go())


def text_of(responses):
    return "".join(p.text for p in responses[0].content.parts)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "sub" / "cache.json")


def make(path, replies, **kw):
    inner = Inner(replies=list(replies), seen_models=[])
    return inner, CachingLlm(model="gemini-3.6-flash", inner=inner, path=path, **kw)


def test_second_identical_request_is_replayed_without_calling_the_model(path):
    inner, llm = make(path, ["answer"])
    assert text_of(ask(llm, req())) == "answer"
    assert text_of(ask(llm, req())) == "answer"
    assert inner.calls == 1 and llm.real_calls == 1 and llm.hits == 1
    assert "1 real request" in llm.summary() and "1 replayed" in llm.summary()


def test_cache_survives_a_restart_and_creates_its_folder(path):
    _, a = make(path, ["answer"])
    ask(a, req())
    inner, b = make(path, [])                      # a brand-new object, same file
    assert text_of(ask(b, req())) == "answer" and inner.calls == 0
    saved = json.loads(open(path).read())
    assert len(saved) == 1 and list(saved.values())[0]["text"] == "answer"
    assert "sys" not in open(path).read().replace("saved", "")      # the prompt itself is never stored


def test_any_change_in_the_request_is_a_miss(path):
    inner, llm = make(path, ["a1", "a2", "a3", "a4"])
    ask(llm, req())
    ask(llm, req(text="other question"))
    ask(llm, req(system="other system prompt"))
    other_model = CachingLlm(model="gemini-3.5-flash-lite", inner=inner, path=path)
    ask(other_model, req())
    assert inner.calls == 4


def test_schema_is_part_of_the_key():
    class A(BaseModel):
        x: int

    class B(BaseModel):
        y: str
    assert request_key("m", req(schema=A)) == request_key("m", req(schema=A))
    assert request_key("m", req(schema=A)) != request_key("m", req(schema=B))
    assert request_key("m", req(schema=A)) != request_key("m", req())


def test_the_real_model_name_is_sent_to_the_inner_model(path):
    inner, llm = make(path, ["x"])
    ask(llm, req(model="fallback-label"))
    assert inner.seen_models == ["gemini-3.6-flash"]


def test_replay_mode_never_calls_the_model(path):
    _, a = make(path, ["answer"])
    ask(a, req())
    b = CachingLlm(model="gemini-3.6-flash", inner=None, mode="replay", path=path)      # no inner model at all
    assert text_of(ask(b, req())) == "answer"
    with pytest.raises(CacheMiss, match="replay-only"):
        ask(b, req(text="never asked before"))


def test_off_mode_ignores_and_never_writes_the_cache(path):
    inner, llm = make(path, ["a", "b"], mode="off")
    ask(llm, req())
    ask(llm, req())
    assert inner.calls == 2 and llm.hits == 0
    import os
    assert not os.path.exists(path)


def test_errors_are_not_cached(path):
    inner, llm = make(path, [RuntimeError("503 UNAVAILABLE"), "fine"])
    with pytest.raises(RuntimeError):
        ask(llm, req())
    assert text_of(ask(llm, req())) == "fine" and inner.calls == 2


def test_thoughts_are_dropped_and_tool_calls_or_empty_replies_are_not_saved(path):
    thoughtful = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(text="let me think", thought=True), types.Part(text="the answer")]))
    tool = LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name="f", args={}))]))
    empty = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="")]))
    inner, llm = make(path, [thoughtful, tool, empty])
    ask(llm, req(text="q1"))
    ask(llm, req(text="q2"))
    ask(llm, req(text="q3"))
    saved = json.loads(open(path).read())
    assert [v["text"] for v in saved.values()] == ["the answer"]


def test_a_corrupt_cache_file_is_ignored_not_fatal(path):
    import os
    os.makedirs(os.path.dirname(path))
    open(path, "w").write("{this is not json")
    inner, llm = make(path, ["ok"])
    assert text_of(ask(llm, req())) == "ok"
    assert len(json.loads(open(path).read())) == 1


def test_clear_empties_the_cache(path):
    inner, llm = make(path, ["a", "b"])
    ask(llm, req())
    llm.clear()
    ask(llm, req())
    assert inner.calls == 2


def test_invalid_mode_is_rejected(path):
    with pytest.raises(ValueError):
        CachingLlm(model="m", mode="sometimes", path=path)


def test_pacing_spaces_real_requests_but_not_cache_hits(path, monkeypatch):
    clock, sleeps = {"t": 100.0}, []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds
    monkeypatch.setattr(llm_cache, "_now", lambda: clock["t"])
    monkeypatch.setattr(llm_cache, "_sleep", fake_sleep)
    inner, llm = make(path, ["a", "b", "c"], min_interval=12.5)
    ask(llm, req(text="1"))                       # first real request: no waiting
    clock["t"] += 1.0
    ask(llm, req(text="2"))                       # 1 s later: must wait 11.5 s
    ask(llm, req(text="1"))                       # cache hit: no waiting, no request
    clock["t"] += 60.0
    ask(llm, req(text="3"))                       # long enough ago: no waiting
    assert sleeps == [pytest.approx(11.5)] and inner.calls == 3


# ---------- the whole pipeline through the cache ----------
def test_a_whole_dashboard_is_free_the_second_time_and_replayable_without_a_model(path, monkeypatch):
    import agents.chart_agent as chart_agent
    import agents.planner as planner
    from agents.dashboard import build_dashboard
    text = lambda p: "\n".join(f"- {c['name']} ({c['role']})" for c in p["columns"])
    monkeypatch.setattr(chart_agent, "profile_to_text", text)
    monkeypatch.setattr(planner, "profile_to_text", text)
    profile = {"dataset": {"rows": 4, "columns": 2}, "columns": [
        {"name": "region", "role": "categorical"}, {"name": "revenue", "role": "numeric"}]}
    df = pd.DataFrame({"region": ["N", "S", "N", "E"], "revenue": [10.0, 20.0, 30.0, 40.0]})
    plan = json.dumps({"title": "T", "summary": "s", "kpis": [{"label": "R", "measure": "revenue", "agg": "sum"}],
                       "charts": [{"type": "bar", "title": "By region", "dimension": "region", "measure": "revenue", "agg": "sum"}]})
    code = ("```python\nd = df.groupby('region', as_index=False)['revenue'].sum()\n"
            "fig = px.bar(d, x='region', y='revenue', title='By region')\n```")
    inner, llm = make(path, [plan, code])
    first = build_dashboard(df, profile, model=llm)
    assert inner.calls == 2 and first.mode == "ai" and [c.source for c in first.charts] == ["ai"]

    second = build_dashboard(df, profile, model=llm)                       # same data, same prompts
    assert inner.calls == 2 and llm.hits == 2 and second.kpis[0].value == 100.0
    assert [c.source for c in second.charts] == ["ai"]

    offline = CachingLlm(model="gemini-3.6-flash", inner=None, mode="replay", path=path)   # no key, no model
    third = build_dashboard(df, profile, model=offline)
    assert third.mode == "ai" and [c.source for c in third.charts] == ["ai"]
